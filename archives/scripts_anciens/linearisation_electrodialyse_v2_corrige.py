"""Linearisation de Taylor DG/Nernst-Planck, test numerique experimental.

Un seul processus MPI :
ED_VOLTAGE=0.006 ED_LINEAR_SOLVER=lu ED_LU_SOLVER=mumps ED_ROW_SCALE=0 \
python linearisation_electrodialyse_v2.py

Ce programme ne valide pas physiquement le modele COMSOL.
"""
import os
import runpy
from pathlib import Path

import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector, create_matrix
from parameters import F as FARADAY, R, T
from periodic_flux_v6 import PeriodicNPFlux

ROOT = Path(__file__).resolve().parent
p = runpy.run_path(str(ROOT / "solver_comsol_membranes_v4.py"))
mesh, W, w = p["mesh"], p["W"], p["w"]
if mesh.comm.size != 1:
    raise RuntimeError("Ce prototype exige un seul processus MPI.")

pairs = np.asarray(np.load(ROOT / "periodic_dofs.npy"), dtype=np.int64)
if pairs.ndim != 2 or pairs.shape[1] != 3 or len(pairs) == 0:
    raise ValueError("periodic_dofs.npy doit etre un tableau non vide (N,3).")
comp, left, right = pairs.T
ndofs = len(w.x.array)
if not np.all((comp >= 0) & (comp <= 2)):
    raise ValueError("Composantes periodiques invalides.")
if not (np.all((left >= 0) & (left < ndofs)) and
        np.all((right >= 0) & (right < ndofs))):
    raise ValueError("Indices periodiques hors limites.")
if len(np.unique(right)) != len(right) or np.intersect1d(left, right).size:
    raise ValueError("DOFs droits dupliques ou chevauchement gauche/droite.")
phi_left = left[comp == 2]
if not len(phi_left):
    raise ValueError("Aucun DOF periodique du potentiel pour la jauge.")
gauge = int(phi_left[0])
if gauge in right:
    raise ValueError("La jauge est un DOF droit contraint.")

right_petsc = right.astype(PETSc.IntType)
concentration = np.unique(np.concatenate(
    [W.sub(k).dofmap.list.flatten() for k in (0, 1)]
)).astype(np.int64)
F_form = fem.form(p["residual"])
J_form = fem.form(p["jacobian"])
operator = PeriodicNPFlux(
    mesh, W, str(ROOT / "periodic_pairs.npz"),
    p["DNa_field"], p["DCl_field"],
    beta=FARADAY / (R * T), voltage=0.0,
)

A = create_matrix(J_form)
A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
A.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, True)
ksp = PETSc.KSP().create(mesh.comm)


def enforce_state(voltage, gauge_target):
    a = w.x.array
    offsets = np.where(comp == 2, voltage, 0.0)
    a[right] = a[left] + offsets
    a[gauge] = gauge_target
    w.x.scatter_forward()


def build(voltage, gauge_target, with_jacobian=True):
    """Assemble le residu/Jacobien et remplace les lignes contraintes."""
    operator.voltage = float(voltage)
    b = assemble_vector(F_form)
    if with_jacobian:
        A.zeroEntries()
        from dolfinx.fem.petsc import assemble_matrix
        assemble_matrix(A, J_form)
        A.assemble()
        operator.add(w, b, A)
        A.assemblyBegin()
        A.assemblyEnd()
    else:
        operator.add(w, b)
    b.assemblyBegin()
    b.assemblyEnd()

    a = w.x.array
    offsets = np.where(comp == 2, voltage, 0.0)
    b.setValues(
        right_petsc, a[right] - a[left] - offsets,
        addv=PETSc.InsertMode.INSERT_VALUES,
    )
    b.setValue(
        gauge, a[gauge] - gauge_target,
        addv=PETSc.InsertMode.INSERT_VALUES,
    )
    b.assemblyBegin()
    b.assemblyEnd()

    if with_jacobian:
        rows = np.unique(np.r_[right, gauge]).astype(PETSc.IntType)
        A.zeroRows(rows, diag=1.0)
        for l, r in zip(left, right):
            A.setValue(int(r), int(l), -1.0,
                       addv=PETSc.InsertMode.INSERT_VALUES)
        A.assemblyBegin()
        A.assemblyEnd()
    return b


def equilibrate_rows(matrix, vector):
    """Applique D*A et D*b, avec D_ii = 1/max_j |A_ij|."""
    indptr, indices, values = matrix.getValuesCSR()
    abs_values = np.abs(values)
    row_max = np.zeros(matrix.getSize()[0], dtype=np.float64)
    for i in range(len(row_max)):
        vals = abs_values[indptr[i]:indptr[i + 1]]
        if vals.size:
            row_max[i] = np.max(vals)
    if not np.all(np.isfinite(row_max)) or np.any(row_max == 0.0):
        raise RuntimeError("Matrice avec ligne nulle ou non finie.")
    factors = 1.0 / row_max
    scale = matrix.createVecLeft()
    scale.array[:] = factors
    matrix.diagonalScale(L=scale)
    vector.pointwiseMult(vector, scale)
    print(f"Equilibrage lignes : min={factors.min():.3e}, max={factors.max():.3e}",
          flush=True)
    return scale


def configure_solver(method):
    if method == "lu":
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        solver = os.getenv("ED_LU_SOLVER", "mumps")
        pc.setFactorSolverType(solver)
        print(f"Factorisation LU : {solver}", flush=True)
    elif method == "gmres":
        ksp.setType("gmres")
        ksp.setGMRESRestart(int(os.getenv("ED_GMRES_RESTART", "100")))
        ksp.getPC().setType("none")
        ksp.setTolerances(rtol=1e-8, atol=1e-12,
                          max_it=int(os.getenv("ED_KSP_MAX", "300")))
        ksp.setMonitor(lambda solver, it, norm:
                       print(f"GMRES {it:03d} : {norm:.6e}", flush=True))
    else:
        raise ValueError("ED_LINEAR_SOLVER doit etre 'lu' ou 'gmres'.")


def linearized_step(voltage):
    gauge_target = float(w.x.array[gauge])
    enforce_state(voltage, gauge_target)
    w0 = w.x.array.copy()
    residual = build(voltage, gauge_target, with_jacobian=True)
    print(f"Initial nonlinear residual: {residual.norm():.6e}", flush=True)
    rhs = residual.copy()
    rhs.scale(-1.0)

    # Copies du systeme original AVANT l'equilibrage des lignes.
    A_original = A.duplicate(copy=True)
    rhs_original = rhs.copy()
    original_rhs_norm = rhs_original.norm()
    row_scale = None
    dx = None
    new_residual = None
    try:
        if os.getenv("ED_ROW_SCALE", "1") == "1":
            row_scale = equilibrate_rows(A, rhs)

        method = os.getenv("ED_LINEAR_SOLVER", "gmres").lower()
        if method == "lu" and len(w0) > 20000 and \
                os.getenv("ED_ALLOW_LARGE_DIRECT", "0") != "1":
            print("STOP : LU sur ce maillage peut demander beaucoup de memoire.")
            print("Utilise un maillage reduit ou ED_ALLOW_LARGE_DIRECT=1 a tes risques.")
            return False

        configure_solver(method)
        ksp.setOperators(A)
        print("=== CONFIGURATION PETSc ===", flush=True)
        print("KSP :", ksp.getType(), flush=True)
        print("PC :", ksp.getPC().getType(), flush=True)
        print("Iterations max :", ksp.getTolerances()[3], flush=True)

        dx = A.createVecRight()
        try:
            ksp.solve(rhs, dx)
        except PETSc.Error as exc:
            print(f"Erreur PETSc pendant la resolution : {exc}", flush=True)
            return False

        reason = ksp.getConvergedReason()
        print(f"Linear KSP: iterations={ksp.getIterationNumber()} reason={reason}",
              flush=True)
        if reason <= 0:
            try:
                print("PC failed reason :", ksp.getPC().getFailedReason())
            except Exception:
                pass
            print("Linear solve failed; no solution exported.")
            return False

        # Residuel du systeme effectivement resolu (eventuellement mis a l'echelle).
        Ax_s = A.createVecLeft()
        res_s = A.createVecLeft()
        A.mult(dx, Ax_s)
        res_s.array[:] = Ax_s.array
        res_s.axpy(-1.0, rhs)
        rel_s = res_s.norm() / max(rhs.norm(), 1e-30)

        # Residuel du systeme original : calcul independant, sans reconstituer
        # le second membre a partir de la matrice normalisee.
        Ax_o = A_original.createVecLeft()
        res_o = A_original.createVecLeft()
        A_original.mult(dx, Ax_o)
        res_o.array[:] = Ax_o.array
        res_o.axpy(-1.0, rhs_original)
        rel_o = res_o.norm() / max(original_rhs_norm, 1e-30)

        print("\n=== VERIFICATION SYSTEME LINEAIRE ===", flush=True)
        print(f"Norme ||rhs original|| : {original_rhs_norm:.6e}", flush=True)
        print(f"Norme ||delta|| : {dx.norm():.6e}", flush=True)
        print(f"Residuel relatif systeme resolu : {rel_s:.6e}", flush=True)
        print(f"Residuel relatif systeme original : {rel_o:.6e}", flush=True)
        print("=====================================\n", flush=True)
        Ax_s.destroy()
        res_s.destroy()
        Ax_o.destroy()
        res_o.destroy()

        if (not np.isfinite(rel_s) or not np.isfinite(rel_o)
                or rel_s > 1e-6 or rel_o > 1e-6):
            print("Linear residual too large; no solution exported.")
            return False

        candidate = w0 + dx.array
        if not np.all(np.isfinite(candidate)):
            print("Non-finite candidate; no export.")
            return False
        if np.min(candidate[concentration]) <= 0.0:
            print("Non-positive concentrations; no export.")
            return False

        w.x.array[:] = candidate
        enforce_state(voltage, gauge_target)
        new_residual = build(voltage, gauge_target, with_jacobian=False)
        print(f"Original nonlinear residual at candidate: {new_residual.norm():.6e}",
              flush=True)

        out = ROOT / "solution_linearisee_experimentale.npy"
        np.save(out, w.x.array.copy())
        print(f"Etat experimental sauvegarde : {out.name}", flush=True)
        print("ATTENTION : etat non valide physiquement; verifier flux, "
              "electroneutralite et interfaces.", flush=True)
        return True
    finally:
        if new_residual is not None:
            new_residual.destroy()
        if dx is not None:
            dx.destroy()
        if row_scale is not None:
            row_scale.destroy()
        A_original.destroy()
        rhs_original.destroy()
        rhs.destroy()
        residual.destroy()


if __name__ == "__main__":
    voltage = float(os.getenv("ED_VOLTAGE", "0.006"))
    if not 0.0 <= voltage <= 1.5:
        raise ValueError("ED_VOLTAGE doit etre dans [0, 1.5].")
    print(f"One-shot Taylor linearization: {len(w.x.array)} DOFs, V={voltage:g} V",
          flush=True)
    print("Periodic DG flux included; membrane treatment remains experimental.",
          flush=True)
    print(f"Normalisation des lignes : {os.getenv('ED_ROW_SCALE', '1')}", flush=True)
    try:
        linearized_step(voltage)
    finally:
        A.destroy()
        ksp.destroy()
