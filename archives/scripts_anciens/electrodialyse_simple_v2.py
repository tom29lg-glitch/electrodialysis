"""Test V2 : conduction stationnaire avec 5 bandes de conductivite.

Les interfaces sont des positions GEOMETRIQUES de demonstration, pas les
frontieres reelles des membranes. Remplacer ED_INTERFACES par les quatre
positions x (en metres) pour utiliser la geometrie physique.

Execution : ED_VOLTAGE=0.006 python electrodialyse_simple_v2.py
Option : ED_INTERFACES='-0.0003,-0.0002,0.0002,0.0003'
Option : ED_SIGMAS='1,0.2,1,0.2,1'
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
voltage = float(os.getenv('ED_VOLTAGE', '0.006'))
sigmas = np.fromstring(os.getenv('ED_SIGMAS', '1,0.2,1,0.2,1'), sep=',')
if len(sigmas) != 5 or not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
    raise ValueError('ED_SIGMAS : exactement 5 conductivites positives (S/m).')
if not np.isfinite(voltage):
    raise ValueError('ED_VOLTAGE doit etre fini.')

# L'ancien module sert uniquement a construire/charger le maillage.
p = runpy.run_path(str(ROOT / 'solver_comsol_membranes_v4.py'))
domain = p['mesh']
if domain.comm.size != 1:
    raise RuntimeError('Cette version de demonstration utilise un seul processus MPI.')

tdim = domain.topology.dim
fdim = tdim - 1
coords = domain.geometry.x
xmin = float(domain.comm.allreduce(coords[:, 0].min(), op=MPI.MIN))
xmax = float(domain.comm.allreduce(coords[:, 0].max(), op=MPI.MAX))
length = xmax - xmin
if length <= 0:
    raise RuntimeError('Longueur du domaine nulle.')

specified = os.getenv('ED_INTERFACES', '').strip()
if specified:
    interfaces = np.fromstring(specified, sep=',')
    mode = 'interfaces indiquees par ED_INTERFACES'
else:
    # Bandes artificielles pour tester la conductivite heterogene.
    interfaces = xmin + length * np.array([0.38, 0.42, 0.58, 0.62])
    mode = 'bandes artificielles (38/42/58/62 %), NON membranes reelles'
if len(interfaces) != 4 or not np.all(np.isfinite(interfaces)):
    raise ValueError('ED_INTERFACES doit contenir quatre positions x en metres.')
if not np.all(np.diff(np.r_[xmin, interfaces, xmax]) > 0):
    raise ValueError('Interfaces hors domaine ou non strictement croissantes.')

# DG0 : coefficient constant par cellule. Les cellules traversant une
# interface artificielle sont classees par leur centre de masse.
Q = fem.functionspace(domain, ('DG', 0))
sigma = fem.Function(Q)
sigma.name = 'conductivite_S_par_m'
domain.topology.create_connectivity(tdim, 0)
cell_vertices = domain.topology.connectivity(tdim, 0)
num_cells = domain.topology.index_map(tdim).size_local
mid_x = np.empty(num_cells)
for cell in range(num_cells):
    vertices = cell_vertices.links(cell)
    mid_x[cell] = domain.geometry.x[vertices, 0].mean()

region = np.searchsorted(interfaces, mid_x, side='right')
# DG0 scalaire : un degre de liberte par cellule.
for cell in range(num_cells):
    dof = Q.dofmap.cell_dofs(cell)
    if len(dof) != 1:
        raise RuntimeError('DG0 inattendu : plusieurs DOFs par cellule.')
    sigma.x.array[dof[0]] = sigmas[region[cell]]
sigma.x.scatter_forward()

V = fem.functionspace(domain, ('Lagrange', 1))
tol = max(1e-10 * length, 1e-12)
left_facets = dmesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], xmin, atol=tol, rtol=0))
right_facets = dmesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], xmax, atol=tol, rtol=0))
left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
right_dofs = fem.locate_dofs_topological(V, fdim, right_facets)
if not len(left_dofs) or not len(right_dofs):
    raise RuntimeError('Bords x=min / x=max non trouves.')
if np.intersect1d(left_dofs, right_dofs).size:
    raise RuntimeError('DOFs des bords gauche et droit chevauchants.')
bcs = [
    fem.dirichletbc(PETSc.ScalarType(0.0), left_dofs, V),
    fem.dirichletbc(PETSc.ScalarType(voltage), right_dofs, V),
]

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = sigma * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx
problem = LinearProblem(
    a, L, bcs=bcs,
    petsc_options_prefix='ed_simple_v2_',
    petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
)
phi = problem.solve()
phi.name = 'potentiel_electrique'
phi.x.scatter_forward()
values = np.asarray(phi.x.array.real)
if not np.all(np.isfinite(values)):
    raise RuntimeError('Potentiel non fini.')

# Reference 1D pour les 5 couches de conductivite piecewise constant :
# dphi/dx = J/sigma, J = V / somme(epaisseur_i/sigma_i).
bounds = np.r_[xmin, interfaces, xmax]
widths = np.diff(bounds)
resistances = widths / sigmas
J_1d = voltage / resistances.sum()
dof_x = V.tabulate_dof_coordinates()[:, 0]
ref = np.empty_like(dof_x)
for k, x in enumerate(dof_x):
    i = min(np.searchsorted(interfaces, x, side='right'), 4)
    ref[k] = J_1d * (resistances[:i].sum() + (x - bounds[i]) / sigmas[i])
error = float(np.max(np.abs(values - ref)))

out = ROOT / 'potentiel_simple_v2.npz'
np.savez(out, x=dof_x, phi=values, phi_reference_1d=ref,
         interfaces=interfaces, sigmas=sigmas, voltage=voltage,
         x_min=xmin, x_max=xmax, region_cell=region)

print('\n=== ELECTRODIALYSE SIMPLE V2 : CONDUCTIVITE HETEROGENE ===')
print('Mode :', mode)
print(f'Cellules : {num_cells}; DOFs potentiel : {len(values)}')
print(f'Bords gauche/droit : {len(left_facets)}/{len(right_facets)} facettes')
print(f'Tension : {voltage:.6e} V')
for i in range(5):
    print(f'Bande {i+1}: x=[{bounds[i]:.6e}, {bounds[i+1]:.6e}] m, '
          f'sigma={sigmas[i]:.3g} S/m, cellules={np.count_nonzero(region==i)}')
print(f'Potentiel min/max : {values.min():.6e} / {values.max():.6e} V')
print(f'Courant 1D theorique par surface : {J_1d:.6e} A/m2')
print(f'Ecart max a la reference 1D : {error:.6e} V')
print(f'Fichier : {out.name}')
print('ATTENTION : reference 1D ideale; interfaces en escalier sur le maillage.')
print('Cette V2 ne modelise ni transport ionique ni selectivite des membranes.')
