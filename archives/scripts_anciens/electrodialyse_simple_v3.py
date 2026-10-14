"""Test V3 : conduction stationnaire sur les vrais sous-domaines Gmsh.

Les conductivites sont affectees a partir des marqueurs physiques 1 a 5.

Execution : ED_VOLTAGE=0.006 python electrodialyse_simple_v3.py
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

# Retrouver les vrais marqueurs de cellules produits par le solveur importe.
# Aucun classement artificiel par coordonnee n'est utilise ici.
cell_tags = p.get('cell_tags', None)
if cell_tags is None:
    for key in ('cell_markers', 'subdomains', 'ct'):
        candidate = p.get(key)
        if candidate is not None and hasattr(candidate, 'find') and hasattr(candidate, 'values'):
            cell_tags = candidate
            break
if cell_tags is None:
    available = [k for k, v in p.items()
                 if not k.startswith('__') and hasattr(v, 'find') and hasattr(v, 'values')]
    raise RuntimeError(
        'Le module importe ne fournit pas cell_tags. '
        f'Objets de type meshtags disponibles : {available}. '
        'Il faudra exposer cell_tags depuis solver_comsol_membranes_v4.py.')

Q = fem.functionspace(domain, ('DG', 0))
sigma = fem.Function(Q)
sigma.name = 'conductivite_S_par_m'
num_cells = domain.topology.index_map(tdim).size_local
region = np.full(num_cells, -1, dtype=np.int32)
for physical_id in range(1, 6):
    cells = np.asarray(cell_tags.find(physical_id), dtype=np.int32)
    cells = cells[cells < num_cells]
    if np.any(region[cells] != -1):
        raise RuntimeError(f'Cellules associees a plusieurs marqueurs, ID {physical_id}')
    region[cells] = physical_id - 1
if np.any(region < 0):
    raise RuntimeError(f'{np.count_nonzero(region < 0)} cellules sans marqueur physique 1..5')
for cell in range(num_cells):
    dofs = Q.dofmap.cell_dofs(cell)
    if len(dofs) != 1:
        raise RuntimeError('DG0 inattendu : plusieurs DOFs par cellule.')
    sigma.x.array[dofs[0]] = sigmas[region[cell]]
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
    petsc_options_prefix='ed_simple_v3_',
    petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
)
phi = problem.solve()
phi.name = 'potentiel_electrique'
phi.x.scatter_forward()
values = np.asarray(phi.x.array.real)
if not np.all(np.isfinite(values)):
    raise RuntimeError('Potentiel non fini.')

# La comparaison 1D de V2 n'est plus supposee exacte pour le vrai maillage.
dof_x = V.tabulate_dof_coordinates()[:, 0]
out = ROOT / 'potentiel_simple_v3.npz'
np.savez(out, x=dof_x, phi=values, sigmas=sigmas, voltage=voltage,
         x_min=xmin, x_max=xmax, region_cell=region)

print('\n=== ELECTRODIALYSE SIMPLE V3 : SOUS-DOMAINES REELS ===')
print(f'Cellules : {num_cells}; DOFs potentiel : {len(values)}')
print(f'Bords gauche/droit : {len(left_facets)}/{len(right_facets)} facettes')
print(f'Tension : {voltage:.6e} V')
for i, name in enumerate(('Canal gauche', 'CEM', 'Canal central', 'AEM', 'Canal droit')):
    print(f'ID {i+1} ({name}) : sigma={sigmas[i]:.3g} S/m, '
          f'cellules={np.count_nonzero(region == i)}')
print(f'Potentiel min/max : {values.min():.6e} / {values.max():.6e} V')
print(f'Fichier : {out.name}')
print('Ce modele est purement ohmique : pas de transport ionique ni de selectivite.')
