"""Linearisation de Taylor DG/Nernst-Planck, test numerique experimental.

Utilisation (un seul processus MPI):
  ED_VOLTAGE=0.006 ED_LINEAR_SOLVER=gmres ED_KSP_MAX=100 \
    python linearisation_electrodialyse_v2.py

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

pairs = np.asarray(np.load(ROOT / "periodic_dofs.npy"), dtype=np.int64)
if pairs.ndim != 2 or pairs.shape[1] != 3 or len(pairs) == 0:
    raise ValueError("periodic_dofs.npy doit etre un tableau non vide (N,3).")
comp, left, right = pairs.T
n = len(w.x.array)
if not np.all((comp >= 0) & (comp <= 2)):
    raise ValueError("Composantes periodiques invalides.")
if not (np.all((left >= 0) & (left < n)) and np.all((right >= 0) & (right < n))):
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
concentration = np.unique(np.concatenate([
    W.sub(k).dofmap.list.flatten() for k in (0, 1)
])).astype(np.int64)

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
    a[right] = a[left] + np.where(comp == 2, voltage, 0.0)
    a[gauge] = gauge_target
    w.x.scatter_forward()


def build(voltage, gauge_target, with_jacobian=True):
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
    b.setValues(right_petsc, a[right] - a[left] - offsets,
                addv=PETSc.InsertMode.INSERT_VALUES)
    b.setValue(gauge, a[gauge] - gauge_target,
               addv=PETSc.InsertMode.INSERT_VALUES)
    b.assemblyBegin()
    b.assemblyEnd()

    if with_jacobian:
        rows = np.r_[right, gauge].astype(PETSc.IntType)
        A.zeroRows(rows, diag=1.0)
        for l, r in zip(left, right):
            A.setValue(int(r), int(l), -1.0,
                       addv=PETSc.InsertMode.INSERT_VALUES)
        A.assemblyBegin()
        A.assemblyEnd()
    return b


def linearized_step(voltage):
    gauge_target = float(w.x.array[gauge])
    enforce_state(voltage, gauge_target)
    w0 = w.x.array.copy()
    residual = build(voltage, gauge_target, with_jacobian=True)
    initial_norm = residual.norm()
    print(f"Initial nonlinear residual: {initial_norm:.6e}", flush=True)
    rhs = residual.copy()
    rhs.scale(-1.0)
    dx = A.createVecRight()

    method = os.getenv("ED_LINEAR_SOLVER", "gmres").lower()
    if method == "lu":
        if len(w0) > 20000 and os.getenv("ED_ALLOW_LARGE_DIRECT", "0") != "1":
            print("STOP: LU sur ce maillage peut demander beaucoup de memoire.")
            print("Utiliser un maillage reduit (ou ED_ALLOW_LARGE_DIRECT=1 a vos risques).")
            return False
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
    elif method == "gmres":
        ksp.setType("gmres")
        ksp.setGMRESRestart(int(os.getenv("ED_GMRES_RESTART", "100")))
        ksp.getPC().setType("none")
        ksp.setTolerances(rtol=1e-8, atol=1e-12,
                          max_it=int(os.getenv("ED_KSP_MAX", "300")))
    else:
        raise ValueError("ED_LINEAR_SOLVER doit etre 'lu' ou 'gmres'.")

    ksp.setOperators(A)
    print("=== CONFIGURATION PETSc ===", flush=True)
    print("KSP :", ksp.getType(), flush=True)
    print("PC  :", ksp.getPC().getType(), flush=True)
    print("Iterations max :", ksp.getTolerances()[3], flush=True)
    if method == "gmres":
        print("GMRES restart : 100", flush=True)
        ksp.setMonitor(lambda solver, it, norm:
                       print(f"GMRES {it:03d} : {norm:.6e}", flush=True))

    try:
        ksp.solve(rhs, dx)
    except PETSc.Error as exc:
        print(f"Erreur PETSc pendant la resolution: {exc}", flush=True)
        return False
    reason = ksp.getConvergedReason()
    print(f"Linear KSP: iterations={ksp.getIterationNumber()} reason={reason}", flush=True)
    if reason <= 0:
        print("Linear solve failed; no solution exported.")
        return False

    check = A.createVecLeft()
    A.mult(dx, check)
    check.axpy(-1.0, rhs)
    relative = check.norm() / max(rhs.norm(), 1e-30)
    print(f"Unscaled relative linear residual: {relative:.6e}", flush=True)
    if not np.isfinite(relative) or relative > 1e-6:
        print("Linear residual too large; no solution exported.")
        return False

    candidate = w0 + dx.array
    if not np.all(np.isfinite(candidate)):
        print("Non-finite candidate; no export.")
        return False
    if np.min(candidate[concentration]) <= 0:
        print("Non-positive concentrations; no export.")
        return False

    w.x.array[:] = candidate
    enforce_state(voltage, gauge_target)
    new_residual = build(voltage, gauge_target, with_jacobian=False)
    print(f"Original nonlinear residual at candidate: {new_residual.norm():.6e}", flush=True)
    out = ROOT / "solution_linearisee_experimentale.npy"
    np.save(out, w.x.array.copy())
    print(f"EXPERIMENTAL linearized state saved: {out.name}", flush=True)
    print("NOT a validated COMSOL solution: check fluxes, electroneutrality and interfaces.")
    return True


if __name__ == "__main__":
    voltage = float(os.getenv("ED_VOLTAGE", "0.006"))
    if not 0.0 <= voltage <= 1.5:
        raise ValueError("ED_VOLTAGE doit etre dans [0, 1.5].")
    print(f"One-shot Taylor linearization: {len(w.x.array)} DOFs, V={voltage:g} V")
    print("Periodic DG flux included; membrane treatment remains experimental.")
    try:
        linearized_step(voltage)
    finally:
        A.destroy()
        ksp.destroy()
