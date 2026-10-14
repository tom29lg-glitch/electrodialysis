"""Etape 1 : conduction electrique stationnaire sur le maillage existant.

Un processus MPI. Modele de verification, pas encore une simulation
physique d'electrodialyse.

Execution : python electrodialyse_simple_v1.py
Variables : ED_VOLTAGE=0.006 ED_SIGMA=1.0
"""
import os
import runpy
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
import ufl

ROOT = Path(__file__).resolve().parent
VOLTAGE = float(os.getenv("ED_VOLTAGE", "0.006"))
SIGMA = float(os.getenv("ED_SIGMA", "1.0"))
if not np.isfinite(VOLTAGE) or not np.isfinite(SIGMA) or SIGMA <= 0:
    raise ValueError("ED_VOLTAGE doit etre fini et ED_SIGMA strictement positif.")

# Reutilise seulement la construction du maillage de l'ancien projet.
# Le module d'origine peut afficher des diagnostics de son ancien modele,
# mais aucun de ses solveurs non lineaires n'est lance ici.
p = runpy.run_path(str(ROOT / "solver_comsol_membranes_v4.py"))
domain = p["mesh"]
if domain.comm.size != 1:
    raise RuntimeError("Ce prototype utilise un seul processus MPI.")

tdim = domain.topology.dim
fdim = tdim - 1
coords = domain.geometry.x
xmin = float(domain.comm.allreduce(np.min(coords[:, 0]), op=MPI.MIN))
xmax = float(domain.comm.allreduce(np.max(coords[:, 0]), op=MPI.MAX))
length = xmax - xmin
if length <= 0:
    raise RuntimeError("Domaine de longueur nulle selon x.")
tol = max(1e-10 * length, 1e-12)

# CG1 : potentiel continu aux interfaces entre sous-domaines.
V = fem.functionspace(domain, ("Lagrange", 1))
left_facets = dmesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], xmin, rtol=0, atol=tol)
)
right_facets = dmesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], xmax, rtol=0, atol=tol)
)
left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
right_dofs = fem.locate_dofs_topological(V, fdim, right_facets)
if not len(left_dofs) or not len(right_dofs):
    raise RuntimeError("Bords x=min ou x=max non detectes.")
if np.intersect1d(left_dofs, right_dofs).size:
    raise RuntimeError("Les DOFs des deux bords se chevauchent.")

bc_left = fem.dirichletbc(PETSc.ScalarType(0.0), left_dofs, V)
bc_right = fem.dirichletbc(PETSc.ScalarType(VOLTAGE), right_dofs, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
sigma = fem.Constant(domain, PETSc.ScalarType(SIGMA))
a = sigma * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx

problem = LinearProblem(
    a, L, bcs=[bc_left, bc_right],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="ed_simple_v1_",
)
phi = problem.solve()
phi.name = "potentiel_electrique"
phi.x.scatter_forward()

values = phi.x.array.real
if not np.all(np.isfinite(values)):
    raise RuntimeError("Le potentiel contient des valeurs non finies.")

# Solution analytique exacte pour sigma constant : phi(x)=V*(x-xmin)/L.
# Pour CG1, les DOFs correspondent aux coordonnees de l'espace.
dof_x = V.tabulate_dof_coordinates()[:, 0]
reference = VOLTAGE * (dof_x - xmin) / length
error_inf = float(np.max(np.abs(values - reference)))

np.savez(
    ROOT / "potentiel_simple_v1.npz",
    x=dof_x, phi=values, phi_analytique=reference,
    voltage=VOLTAGE, sigma=SIGMA, xmin=xmin, xmax=xmax,
)
print("\n=== ELECTRODIALYSE SIMPLE V1 : POTENTIEL ===")
print(f"Cellules : {domain.topology.index_map(tdim).size_local}")
print(f"DOFs potentiel : {len(values)}")
print(f"x gauche / droite : {xmin:.6e} / {xmax:.6e} m")
print(f"Facettes gauche / droite : {len(left_facets)} / {len(right_facets)}")
print(f"Tension : {VOLTAGE:.6e} V")
print(f"Conductivite uniforme : {SIGMA:.6e} S/m")
print(f"Potentiel min / max : {values.min():.6e} / {values.max():.6e} V")
print(f"Erreur maximale vs solution lineaire : {error_inf:.6e} V")
print("Fichier : potentiel_simple_v1.npz")
print("Ce test ne modelise pas encore les membranes ni le transport ionique.")
