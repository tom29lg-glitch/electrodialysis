
"""Linearisation de Taylor DG/Nernst-Planck, test numerique experimental.

Utilisation (un seul processus MPI) :
  ED_VOLTAGE=0.006 ED_LINEAR_SOLVER=lu ED_LU_SOLVER=mumps \
  ED_ROW_SCALE=0 python linearisation_electrodialyse_v2.py

Ce programme ne constitue pas une validation physique du modele COMSOL.
"""

import os
import runpy
from pathlib import Path

import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_matrix

from parameters import F as FARADAY, R, T
from periodic_flux_v6 import PeriodicNPFlux


ROOT = Path(__file__).resolve().parent

p = runpy.run_path(str(ROOT / "solver_comsol_membranes_v4.py"))
mesh, W, w = p["mesh"], p["W"], p["w"]

if mesh.comm.size != 1:
    raise RuntimeError("Ce prototype exige un seul processus MPI.")

pairs = np.asarray(
    np.load(ROOT / "periodic_dofs.npy"),
    dtype=np.int64,
)

if pairs.ndim != 2 or pairs.shape[1] != 3 or len(pairs) == 0:
    raise ValueError("periodic_dofs.npy doit etre un tableau non vide (N,3).")

comp, left, right = pairs.T
n = len(w.x.array)

if not np.all((comp >= 0) & (comp <= 2)):
    raise ValueError("Composantes periodiques invalides.")

if not (
    np.all((left >= 0) & (left < n))
    and np.all((right >= 0) & (right < n))
):
    raise ValueError("Indices periodiques hors limites.")

if (
    len(np.unique(right)) != len(right)
    or np.intersect1d(left, right).size
):
    raise ValueError("DOFs droits dupliques ou chevauchement gauche/droite.")

phi_left = left[comp == 2]

if not len(phi_left):
    raise ValueError("Aucun DOF periodique du potentiel pour la jauge.")

gauge = int(phi_left[0])

if gauge in right:
    raise ValueError("La jauge est un DOF droit contraint.")

right_petsc = right.astype(PETSc.IntType)

concentration = np.unique(
    np.concatenate([
        W.sub(k).dofmap.list.flatten()
        for k in (0, 1)
    ])
).astype(np.int64)

F_form = fem.form(p["residual"])
J_form = fem.form(p["jacobian"])

operator = PeriodicNPFlux(
    mesh,
    W,
    str(ROOT / "periodic_pairs.npz"),
    p["DNa_field"],
    p["DCl_field"],
    beta=FARADAY / (R * T),
    voltage=0.0,
)

A = create_matrix(J_form)
A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
A.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, True)

ksp = PETSc.KSP().create(mesh.comm)


def enforce_state(voltage, gauge_target):
    """Impose les contraintes affines sur l'etat courant."""
    a = w.x.array

    offsets = np.where(comp == 2, voltage, 0.0)
    a[right] = a[left] + offsets
    a[gauge] = gauge_target

    w.x.scatter_forward()


def build(voltage, gauge_target, with_jacobian=True):
    """Assemble le residu et le Jacobien avec remplacement de lignes."""
    operator.voltage = float(voltage)

    b = assemble_vector(F_form)

    if with_jacobian:
        A.zeroEntries()
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

    # Residus des contraintes periodiques.
    b.setValues(
        right_petsc,
        a[right] - a[left] - offsets,
        addv=PETSc.InsertMode.INSERT_VALUES,
    )

    # Condition de jauge sur le potentiel.
    b.setValue(
        gauge,
        a[gauge] - gauge_target,
        addv=PETSc.InsertMode.INSERT_VALUES,
    )

    b.assemblyBegin()
    b.assemblyEnd()

    if with_jacobian:
        rows = np.r_[right, gauge].astype(PETSc.IntType)

        A.zeroRows(rows, diag=1.0)

        for l, r in zip(left, right):
            A.setValue(
                int(r),
                int(l),
                -1.0,
                addv=PETSc.InsertMode.INSERT_VALUES,
            )

        A.assemblyBegin()
        A.assemblyEnd()

    return b


def equilibrate_rows(matrix, vector):
    """Equilibre les lignes en appliquant D*A et D*b."""
    indptr, indices, values = matrix.getValuesCSR()

    abs_values = np.abs(values)
    nrows = matrix.getSize()[0]
    row_max = np.zeros(nrows, dtype=np.float64)

    for i in range(nrows):
        row = abs_values[indptr[i]:indptr[i + 1]]
        if row.size:
            row_max[i] = np.max(row)

    if (
        not np.all(np.isfinite(row_max))
        or np.any(row_max == 0.0)
    ):
        raise RuntimeError(
            "Matrice avec ligne nulle ou non finie : arret du test."
        )

    factors = 1.0 / row_max

    scale = matrix.createVecLeft()
    scale.array[:] = factors

    matrix.diagonalScale(L=scale)
    vector.pointwiseMult(vector, scale)

    print(
        "Equilibrage lignes : "
        f"facteur min={factors.min():.3e}, "
        f"max={factors.max():.3e}",
        flush=True,
    )

    return scale


def linearized_step(voltage):
    """Resout une seule linearisation de Taylor."""
    gauge_target = float(w.x.array[gauge])

    enforce_state(voltage, gauge_target)
    w0 = w.x.array.copy()

    residual = build(
        voltage,
        gauge_target,
        with_jacobian=True,
    )

    print(
        f"Initial nonlinear residual: {residual.norm():.6e}",
        flush=True,
    )

    rhs = residual.copy()
    rhs.scale(-1.0)

    # Sauvegarde du second membre avant equilibrage.
    original_rhs = rhs.copy()
    original_rhs_norm = original_rhs.norm()

    row_scale = None
    use_row_scale = os.getenv("ED_ROW_SCALE", "1") == "1"

    if use_row_scale:
        row_scale = equilibrate_rows(A, rhs)

    dx = A.createVecRight()

    method = os.getenv("ED_LINEAR_SOLVER", "gmres").lower()

    if method == "lu":
        if (
            len(w0) > 20000
            and os.getenv("ED_ALLOW_LARGE_DIRECT", "0") != "1"
        ):
            print("STOP : LU sur ce maillage peut demander beaucoup de memoire.")
            print("Utilise un maillage reduit ou ED_ALLOW_LARGE_DIRECT=1 a tes risques.")
            original_rhs.destroy()
            rhs.destroy()
            dx.destroy()
            residual.destroy()
            return False

        ksp.setType("preonly")

        pc = ksp.getPC()
        pc.setType("lu")

        lu_solver = os.getenv("ED_LU_SOLVER", "mumps")
        pc.setFactorSolverType(lu_solver)

        print(f"Factorisation LU : {lu_solver}", flush=True)

    elif method == "gmres":
        ksp.setType("gmres")
        ksp.setGMRESRestart(
            int(os.getenv("ED_GMRES_RESTART", "100"))
        )
        ksp.getPC().setType("none")

        ksp.setTolerances(
            rtol=1e-8,
            atol=1e-12,
            max_it=int(os.getenv("ED_KSP_MAX", "300")),
        )

    else:
        raise ValueError(
            "ED_LINEAR_SOLVER doit etre 'lu' ou 'gmres'."
        )

    ksp.setOperators(A)

    print("=== CONFIGURATION PETSc ===", flush=True)
    print("KSP :", ksp.getType(), flush=True)
    print("PC :", ksp.getPC().getType(), flush=True)
    print("Iterations max :", ksp.getTolerances()[3], flush=True)

    if method == "gmres":
        print(
            "GMRES restart :",
            int(os.getenv("ED_GMRES_RESTART", "100")),
            flush=True,
        )

        ksp.setMonitor(
            lambda solver, it, norm:
                print(f"GMRES {it:03d} : {norm:.6e}", flush=True)
        )

    try:
        ksp.solve(rhs, dx)
    except PETSc.Error as exc:
        print(f"Erreur PETSc pendant la resolution : {exc}", flush=True)

        original_rhs.destroy()
        rhs.destroy()
        dx.destroy()
        residual.destroy()
        return False

    reason = ksp.getConvergedReason()

    print(
        f"Linear KSP: iterations={ksp.getIterationNumber()} "
        f"reason={reason}",
        flush=True,
    )

    if reason <= 0:
        print("Linear solve failed; no solution exported.")

        original_rhs.destroy()
        rhs.destroy()
        dx.destroy()
        residual.destroy()
        return False

    # -----------------------------------------------------
    # Verification du systeme effectivement resolu.
    # -----------------------------------------------------
    Ax = A.createVecLeft()
    check = A.createVecLeft()

    A.mult(dx, Ax)

    check.array[:] = Ax.array
    check.axpy(-1.0, rhs)

    norm_Ax = Ax.norm()
    norm_rhs_solved = rhs.norm()
    norm_check_solved = check.norm()

    relative_solved = (
        norm_check_solved / max(norm_rhs_solved, 1e-30)
    )

    print("\n=== VERIFICATION SYSTEME LINEAIRE ===", flush=True)
    print(f"Norme ||rhs resolu||    : {norm_rhs_solved:.6e}", flush=True)
    print(f"Norme ||delta||         : {dx.norm():.6e}", flush=True)
    print(f"Norme ||A delta||       : {norm_Ax:.6e}", flush=True)
    print(f"Norme ||A delta-rhs||   : {norm_check_solved:.6e}", flush=True)
    print(f"Residuel relatif resolu : {relative_solved:.6e}", flush=True)

    # Reconstruction du residu original si les lignes ont ete equilibrees.
    if row_scale is not None:
        check.pointwiseDivide(check, row_scale)

    check.axpy(1.0, original_rhs)

    relative_original = (
        check.norm() / max(original_rhs_norm, 1e-30)
    )

    print(
        f"Residuel relatif original : {relative_original:.6e}",
        flush=True,
    )
    print("=====================================\n", flush=True)

    check.destroy()
    Ax.destroy()

    if (
        not np.isfinite(relative_solved)
        or not np.isfinite(relative_original)
        or relative_solved > 1e-6
        or relative_original > 1e-6
    ):
        print("Linear residual too large; no solution exported.")

        original_rhs.destroy()
        rhs.destroy()
        dx.destroy()
        residual.destroy()

        if row_scale is not None:
            row_scale.destroy()

        return False

    candidate = w0 + dx.array

    if not np.all(np.isfinite(candidate)):
        print("Non-finite candidate; no export.")
        original_rhs.destroy()
        rhs.destroy()
        dx.destroy()
        residual.destroy()
        if row_scale is not None:
            row_scale.destroy()
        return False

    if np.min(candidate[concentration]) <= 0.0:
        print("Non-positive concentrations; no export.")
        original_rhs.destroy()
        rhs.destroy()
        dx.destroy()
        residual.destroy()
        if row_scale is not None:
            row_scale.destroy()
        return False

    # Met a jour l'etat uniquement apres validation du systeme lineaire.
    w.x.array[:] = candidate
    enforce_state(voltage, gauge_target)

    new_residual = build(
        voltage,
        gauge_target,
        with_jacobian=False,
    )

    print(
        "Original nonlinear residual at candidate: "
        f"{new_residual.norm():.6e}",
        flush=True,
    )

    out = ROOT / "solution_linearisee_experimentale.npy"
    np.save(out, w.x.array.copy())

    print(f"Etat experimental sauvegarde : {out.name}", flush=True)
    print(
        "ATTENTION : solution non validee physiquement. "
        "Verifier les flux, l'electroneutralite et les interfaces.",
        flush=True,
    )

    original_rhs.destroy()
    rhs.destroy()
    dx.destroy()
    residual.destroy()
    new_residual.destroy()

    if row_scale is not None:
        row_scale.destroy()

    return True


if __name__ == "__main__":
    voltage = float(os.getenv("ED_VOLTAGE", "0.006"))

    if not 0.0 <= voltage <= 1.5:
        raise ValueError("ED_VOLTAGE doit etre dans [0, 1.5].")

    print(
        f"One-shot Taylor linearization: {len(w.x.array)} DOFs, "
        f"V={voltage:g} V",
        flush=True,
    )

    print(
        "Periodic DG flux included; membrane treatment remains experimental.",
        flush=True,
    )

    try:
        linearized_step(voltage)
    finally:
        A.destroy()
        ksp.destroy()
