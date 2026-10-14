"""Newton experimental DG/NP + periodic flux v6 + affine constraints.

Single MPI process only. Numerical experiment, NOT COMSOL validation.
Run: python newton_periodique_v9.py
Environment: ED_VOLTAGE=0.006 ED_STEPS=1 ED_MAX_NEWTON=1 ED_KSP_MAX=30
Optional: ED_DIAG_ALL=1 to diagnose the Jacobian at every Newton iteration.
"""
import os
import runpy
from pathlib import Path
import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_matrix
from parameters import F as FARADAY, R, T
from periodic_flux_v6 import PeriodicNPFlux

ROOT = Path(__file__).resolve().parent
p = runpy.run_path(str(ROOT / 'solver_comsol_membranes_v4.py'))
mesh, W, w = p['mesh'], p['W'], p['w']
if mesh.comm.size != 1:
    raise RuntimeError('Ce prototype exige un seul processus MPI.')

pairs = np.asarray(np.load(ROOT / 'periodic_dofs.npy'), dtype=np.int64)
assert pairs.ndim == 2 and pairs.shape[1] == 3
comp, left, right = pairs.T
assert len(set(right)) == len(right) and not (set(left) & set(right))
assert np.all((comp >= 0) & (comp <= 2))
assert np.all(left >= 0) and np.all(right < len(w.x.array))
right_petsc = right.astype(PETSc.IntType)
left_petsc = left.astype(PETSc.IntType)
gauge = int(left[np.where(comp == 2)[0][0]])
assert gauge not in set(right)

concentration = np.unique(np.concatenate([
    W.sub(k).dofmap.list.flatten() for k in (0, 1)
])).astype(np.int64)
F_form, J_form = fem.form(p['residual']), fem.form(p['jacobian'])
operator = PeriodicNPFlux(mesh, W, str(ROOT / 'periodic_pairs.npz'),
                          p['DNa_field'], p['DCl_field'],
                          beta=FARADAY/(R*T), voltage=0.0)

A = create_matrix(J_form)
A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
A.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, True)

ksp = PETSc.KSP().create(mesh.comm)
ksp.setType('gmres')
ksp.getPC().setType('jacobi')
ksp.setTolerances(rtol=1e-7, atol=1e-12,
                  max_it=int(os.getenv('ED_KSP_MAX', '30')))
ksp.setFromOptions()
# Explicit environment override: PETSc command-line options may not be parsed
# if petsc4py was initialized without sys.argv in this environment.
ksp.setTolerances(max_it=int(os.getenv('ED_KSP_MAX', '30')))



def normaliser_lignes_v11(matrix, rhs):
    """Normalisation gauche des lignes PETSc (MPI mono-rang seulement).

    Chaque ligne i est divisee par max_j |A_ij|. Le vecteur residu
    est divise par la meme quantite. Ne change pas le systeme exact.
    """
    start, end = matrix.getOwnershipRange()
    nlocal = end - start
    scales = np.empty(nlocal, dtype=PETSc.ScalarType)
    zero_rows = []
    nonfinite_rows = []
    for local, row in enumerate(range(start, end)):
        cols, vals = matrix.getRow(row)
        values = np.asarray(vals)
        if values.size == 0:
            zero_rows.append(row)
            scales[local] = 1.0
            continue
        if not np.all(np.isfinite(values)):
            nonfinite_rows.append(row)
            scales[local] = 1.0
            continue
        magnitude = float(np.max(np.abs(values)))
        if magnitude == 0.0:
            zero_rows.append(row)
            scales[local] = 1.0
        else:
            scales[local] = 1.0 / magnitude
    if zero_rows or nonfinite_rows:
        raise RuntimeError(
            f'Lignes nulles: {zero_rows[:20]}; non finies: {nonfinite_rows[:20]}')
    scale = matrix.createVecLeft()
    scale.array[:] = scales
    scale.assemblyBegin()
    scale.assemblyEnd()
    matrix.diagonalScale(L=scale)
    rhs.pointwiseMult(rhs, scale)
    matrix.assemble()
    print('  V11 normalisation: lignes =', nlocal,
          '| facteur min =', f'{float(np.min(scales)):.3e}',
          '| facteur max =', f'{float(np.max(scales)):.3e}', flush=True)
    scale.destroy()


def enforce_state(voltage):
    """Project current iterate onto the affine periodic DOF constraints."""
    a = w.x.array
    a[right] = a[left] + np.where(comp == 2, voltage, 0.)
    a[gauge] = gauge_target
    w.x.scatter_forward()


def build(voltage, with_jacobian=True):
    operator.voltage = float(voltage)
    b = assemble_vector(F_form)
    if with_jacobian:
        A.zeroEntries()
        assemble_matrix(A, J_form)
        A.assemble()
        operator.add(w, b, A)
        A.assemblyBegin(); A.assemblyEnd()
    else:
        operator.add(w, b)
    b.assemblyBegin(); b.assemblyEnd()

    a = w.x.array
    b.setValues(right_petsc,
                a[right]-a[left]-np.where(comp == 2, voltage, 0.),
                addv=PETSc.InsertMode.INSERT_VALUES)
    b.setValue(gauge, a[gauge]-gauge_target,
               addv=PETSc.InsertMode.INSERT_VALUES)
    b.assemblyBegin(); b.assemblyEnd()
    if with_jacobian:
        rows = np.r_[right, gauge].astype(PETSc.IntType)
        A.zeroRows(rows, diag=1.0)
        for l, r in zip(left_petsc, right_petsc):
            A.setValue(int(r), int(l), -1., addv=PETSc.InsertMode.INSERT_VALUES)
        A.assemblyBegin(); A.assemblyEnd()
    return b


def diagnose_matrix(matrix, max_examples=20):
    """Inspect actual matrix rows, not just its diagonal.

    A zero diagonal is not equivalent to a zero row or a singular matrix.
    This diagnostic is intended for a single MPI rank.
    """
    diag = matrix.getDiagonal()
    try:
        d = np.asarray(diag.getArray(readonly=True))
        start, end = matrix.getOwnershipRange()
        empty = []
        near_empty = []
        row_max_min = np.inf
        row_max_max = 0.0
        nonfinite_rows = []
        # getRow() returns only structurally stored entries, including possible zeros.
        for i in range(start, end):
            cols, vals = matrix.getRow(i)
            vals = np.asarray(vals)
            if vals.size == 0:
                empty.append(i)
                row_max = 0.0
            elif not np.all(np.isfinite(vals)):
                nonfinite_rows.append(i)
                row_max = np.nan
            else:
                row_max = float(np.max(np.abs(vals)))
                if row_max == 0.0:
                    empty.append(i)
                if row_max < 1e-14:
                    near_empty.append(i)
            if np.isfinite(row_max):
                row_max_min = min(row_max_min, row_max)
                row_max_max = max(row_max_max, row_max)
            # petsc4py getRow: no restoreRow in this environment
        print('\n=== DIAGNOSTIC JACOBIEN APRES CONTRAINTES ===', flush=True)
        print('Dimensions :', matrix.getSize())
        print('Coefficients stockes non nuls :', matrix.getInfo()['nz_used'])
        print('Diagonales exactement nulles :', int(np.count_nonzero(d == 0)))
        print('Diagonales |d| < 1e-14 :', int(np.count_nonzero(np.abs(d) < 1e-14)))
        print('Diagonale min/max abs :', float(np.min(np.abs(d))), float(np.max(np.abs(d))))
        print('Diagonales non finies :', int(np.count_nonzero(~np.isfinite(d))))
        print('Lignes entierement nulles :', len(empty))
        print('Premieres lignes nulles :', empty[:max_examples])
        print('Lignes de norme max < 1e-14 :', len(near_empty))
        print('Premieres lignes quasi nulles :', near_empty[:max_examples])
        print('Lignes contenant NaN/Inf :', len(nonfinite_rows))
        # Analyse des blocs de lignes/colonnes par composante du champ mixte.
        component = np.full(matrix.getSize()[0], -1, dtype=np.int8)
        for k in range(3):
            indices = np.unique(W.sub(k).dofmap.list.flatten()).astype(np.int64)
            component[indices] = k
        if np.any(component < 0):
            raise RuntimeError('DOFs non classes dans les trois composantes')
        counts = np.zeros((3, 3), dtype=np.int64)
        magnitudes = np.zeros((3, 3), dtype=float)
        for i in range(start, end):
            cols, vals = matrix.getRow(i)
            cols = np.asarray(cols, dtype=np.int64)
            vals = np.asarray(vals)
            if len(cols):
                row_block = int(component[i])
                for k in range(3):
                    selected = component[cols] == k
                    nonzero = selected & (vals != 0)
                    counts[row_block, k] += np.count_nonzero(nonzero)
                    if np.any(nonzero):
                        magnitudes[row_block, k] = max(
                            magnitudes[row_block, k], float(np.max(np.abs(vals[nonzero]))))
        print('Blocs lignes/colonnes: Na, Cl, phi')
        print('Nombre de coefficients numeriquement non nuls par bloc:')
        print(counts)
        print('Valeur absolue maximale par bloc:')
        print(magnitudes)
        print('ATTENTION: aucune de ces mesures ne prouve l inversibilite.')

        print('Norme max des lignes, min/max :', row_max_min, row_max_max)
        print('=============================================\n', flush=True)
    finally:
        diag.destroy()


def scaled_norm(b):
    # Raw norm only; the residual equations have different physical units.
    return b.norm()


def try_step(voltage, max_newton):
    for iteration in range(max_newton):
        enforce_state(voltage)
        b = build(voltage)
        n0 = scaled_norm(b)
        print(f'  Newton {iteration:02d} |F|={n0:.6e}', flush=True)
        if not np.isfinite(n0):
            b.destroy()
            return False
        if n0 < 1e-9:
            b.destroy()
            return True

        if iteration == 0 or os.getenv('ED_DIAG_ALL', '0') == '1':
            diagnose_matrix(A)

        # Normaliser A et le residu avant la resolution lineaire.
        normaliser_lignes_v11(A, b)
        ksp.setOperators(A)
        print('KSP:', ksp.getType(), 'PC:', ksp.getPC().getType(),
              'max_it:', ksp.getTolerances()[3], flush=True)
        rhs = b.copy(); rhs.scale(-1.)
        delta = A.createVecRight()
        ksp.solve(rhs, delta)
        reason = ksp.getConvergedReason()
        print(f'    KSP iterations={ksp.getIterationNumber()} reason={reason}', flush=True)
        if reason <= 0:
            print('    Echec du solveur lineaire; arret propre.', flush=True)
            b.destroy(); rhs.destroy(); delta.destroy()
            return False
        previous = w.x.array.copy()
        direction = delta.array.copy()
        accepted = False
        for alpha in (1., .5, .25, .125, .0625, .03125, .015625, .0078125):
            w.x.array[:] = previous + alpha*direction
            enforce_state(voltage)
            if np.min(w.x.array[concentration]) <= 0:
                continue
            bt = build(voltage, with_jacobian=False)
            nt = scaled_norm(bt)
            bt.destroy()
            if np.isfinite(nt) and nt < n0:
                print(f'    alpha={alpha:g} |F_new|={nt:.6e}', flush=True)
                accepted = True
                break
        b.destroy(); rhs.destroy(); delta.destroy()
        if not accepted:
            w.x.array[:] = previous
            w.x.scatter_forward()
            print('    Recherche lineaire echouee; retour au dernier etat.', flush=True)
            return False
    return False


if __name__ == '__main__':
    voltage = float(os.getenv('ED_VOLTAGE', '0.006'))
    steps = int(os.getenv('ED_STEPS', '1'))
    max_newton = int(os.getenv('ED_MAX_NEWTON', '1'))
    if not (0 <= voltage <= 1.5 and steps >= 1):
        raise ValueError('ED_VOLTAGE doit etre entre 0 et 1.5, ED_STEPS >= 1')
    gauge_target = float(w.x.array[gauge])
    print(f'Newton v11 normalise: {len(w.x.array)} DOFs, {len(pairs)} contraintes, gauge={gauge}')
    print('Attention: formulation DG membranes/periodique non validee physiquement.')
    last = w.x.array.copy()
    for v in np.linspace(voltage/steps, voltage, steps):
        print(f'ETAPE TENSION {v:.6g} V', flush=True)
        if not try_step(float(v), max_newton):
            w.x.array[:] = last
            w.x.scatter_forward()
            print('NON CONVERGE : aucun resultat physique exporte.')
            break
        last = w.x.array.copy()
        np.save(ROOT / 'solution_newton_v9_experimentale.npy', last)
        print('Etape convergee; sauvegarde experimentale.')
    A.destroy(); ksp.destroy()
