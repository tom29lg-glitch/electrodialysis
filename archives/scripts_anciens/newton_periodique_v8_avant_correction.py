"""Newton experimental DG/NP + flux periodiques v6 + contraintes affines.

Mono-processus MPI. Essai numerique, PAS validation COMSOL.
Lancer: python newton_periodique_v8.py
Variables: ED_VOLTAGE=0.02 ED_STEPS=3 ED_MAX_NEWTON=8 ED_KSP_MAX=500
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
    raise RuntimeError('Ce prototype exige mpiexec -n 1 (ou python direct).')

pairs = np.asarray(np.load(ROOT / 'periodic_dofs.npy'), dtype=np.int64)
assert pairs.ndim == 2 and pairs.shape[1] == 3
comp, left, right = pairs.T
assert len(set(right)) == len(right) and not (set(left) & set(right))
assert np.all((comp >= 0) & (comp <= 2))
assert np.all(left >= 0) and np.all(right < len(w.x.array))
right_petsc = right.astype(PETSc.IntType)
left_petsc = left.astype(PETSc.IntType)
# Jauge sur un DOF gauche du potentiel, non contraint comme DOF droit.
gauge = int(left[np.where(comp == 2)[0][0]])
assert gauge not in set(right)

concentration = np.unique(np.concatenate([
    W.sub(k).dofmap.list.flatten() for k in (0, 1)
])).astype(np.int64)
F_form, J_form = fem.form(p['residual']), fem.form(p['jacobian'])
operator = PeriodicNPFlux(mesh, W, str(ROOT / 'periodic_pairs.npz'),
                          p['DNa_field'], p['DCl_field'],
                          beta=FARADAY/(R*T), voltage=0.0)

# La matrice UFL seule ne prealloue pas les connexions gauche-droite.
A = create_matrix(J_form)
A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
A.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, True)

ksp = PETSc.KSP().create(mesh.comm)
ksp.setType('gmres')
ksp.getPC().setType('jacobi')
ksp.setTolerances(rtol=1e-7, atol=1e-12,
                  max_it=int(os.getenv('ED_KSP_MAX', '500')))
ksp.setFromOptions()


def enforce_state(voltage):
    """Project current iterate exactly onto periodic affine constraints."""
    a = w.x.array
    a[right] = a[left] + np.where(comp == 2, voltage, 0.)
    # Gauge = initial value on the chosen left potential DOF.
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

    # Row substitution for constraints: F_right = w_right-w_left-offset.
    # This is a collocation/row-replacement approach, not Galerkin reduction.
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


def scaled_norm(b):
    # Norme brute: diagnostic uniquement, les equations ont des unites differentes.
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
        ksp.setOperators(A)
        ksp.setFromOptions()
        print(
            "KSP:", ksp.getType(),
            "PC:", ksp.getPC().getType(),
            "max_it:", ksp.getTolerances()[3],
            flush=True
        )
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
    voltage = float(os.getenv('ED_VOLTAGE', '0.02'))
    steps = int(os.getenv('ED_STEPS', '3'))
    max_newton = int(os.getenv('ED_MAX_NEWTON', '8'))
    if not (0 <= voltage <= 1.5 and steps >= 1):
        raise ValueError('ED_VOLTAGE doit etre entre 0 et 1.5, ED_STEPS >= 1')
    gauge_target = float(w.x.array[gauge])
    print(f'Newton v8 experimental: {len(w.x.array)} DOFs, {len(pairs)} contraintes, gauge={gauge}')
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
        np.save(ROOT / 'solution_newton_v8_experimentale.npy', last)
        print('Etape convergee; sauvegarde experimentale.')
    A.destroy(); ksp.destroy()
