"""Audit des traces et flux Nernst-Planck sur les 160 paires périodiques.

Diagnostic indépendant de Newton et du résidu DG. Ne crée PAS les termes
numériques inter-facettes manquants dans le solveur.
Exécution : python audit_flux_periodiques_v5.py
"""
import numpy as np
import basix.ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile
from parameters import DNa, DCl, F, R, T, cCl_0, cMem_default, Vtot

comm = MPI.COMM_WORLD
if comm.size != 1:
    raise RuntimeError('Audit mono-processus uniquement')
with XDMFFile(comm, 'electrodialysis_domains.xdmf', 'r') as f:
    mesh = f.read_mesh(name='mesh')
    tags = f.read_meshtags(mesh, name='cell_tags')
mesh.topology.create_entities(mesh.topology.dim - 1)
mesh.topology.create_connectivity(1, 2)
mesh.topology.create_connectivity(1, 0)
facets_to_cells = mesh.topology.connectivity(1, 2)
facets_to_vertices = mesh.topology.connectivity(1, 0)
P = basix.ufl.element('DG', mesh.basix_cell(), 1)
W = fem.functionspace(mesh, basix.ufl.mixed_element([P] * 3))
V = [W.sub(k).collapse()[0] for k in range(3)]
coords = [s.tabulate_dof_coordinates()[:, :2] for s in V]
values = [np.zeros(s.dofmap.index_map.size_local * s.dofmap.index_map_bs) for s in V]
marker_of_cell = dict(zip(map(int, tags.indices), map(int, tags.values)))
for cell, marker in marker_of_cell.items():
    charge = -cMem_default if marker == 2 else cMem_default if marker == 4 else 0.
    na = (-charge + np.sqrt(charge**2 + 4*cCl_0**2))/2
    cl = (charge + np.sqrt(charge**2 + 4*cCl_0**2))/2
    phi = -(R*T/F)*np.log(na/cCl_0)
    for k, val in enumerate((na, cl, phi)):
        values[k][V[k].dofmap.cell_dofs(cell)] = val

pairs = np.load('periodic_pairs.npz')
left_facets, right_facets = pairs['facet_left'], pairs['facet_right']
if len(left_facets) != 160 or len(right_facets) != 160:
    raise RuntimeError('160 paires attendues')

# Reconstitue exactement le polynôme DG1 sur chaque triangle à partir des
# coordonnées de ses trois DOFs; gradient constant sur le triangle.
def cell_trace(cell, midpoint, component):
    dofs = V[component].dofmap.cell_dofs(cell)
    xy = coords[component][dofs]
    matrix = np.column_stack((np.ones(len(dofs)), xy[:, 0], xy[:, 1]))
    if matrix.shape != (3, 3):
        raise RuntimeError('Cet audit nécessite des triangles DG1 (3 DOFs)')
    a, bx, by = np.linalg.solve(matrix, values[component][dofs])
    return a + bx*midpoint[0] + by*midpoint[1], np.array([bx, by])

def side(facet):
    cells = facets_to_cells.links(int(facet))
    if len(cells) != 1:
        raise RuntimeError(f'Facette {facet} non extérieure')
    vertices = facets_to_vertices.links(int(facet))
    midpoint = np.mean(mesh.geometry.x[vertices, :2], axis=0)
    cell = int(cells[0])
    concentrations, gradients = zip(*(cell_trace(cell, midpoint, k) for k in range(3)))
    cna, ccl, phi = concentrations
    gna, gcl, gphi = gradients
    # Normale extérieure : (-1,0) à gauche et (+1,0) à droite.
    nx = -1. if midpoint[0] < 0 else 1.
    flux_na = -DNa*(gna[0] + F/(R*T)*cna*gphi[0])*nx
    flux_cl = -DCl*(gcl[0] - F/(R*T)*ccl*gphi[0])*nx
    return np.array(concentrations), np.array([flux_na, flux_cl]), midpoint

trace_mismatch = np.zeros((160, 3))
flux_sum = np.zeros((160, 2))
for j, (fl, fr) in enumerate(zip(left_facets, right_facets)):
    vl, nl, xl = side(fl)
    vr, nr, xr = side(fr)
    if abs(xl[1]-xr[1]) > 1e-11:
        raise RuntimeError('Facettes appariées à des hauteurs différentes')
    trace_mismatch[j] = vr - vl - np.array([0., 0., Vtot])
    flux_sum[j] = nl + nr  # normales extérieures opposées

print('AUDIT DES FRONTIÈRES PÉRIODIQUES (ÉTAT INITIAL DONNAN)')
print('Paires de facettes :', len(left_facets))
for k, label in enumerate(('Na+ (mol/m³)', 'Cl- (mol/m³)', 'phi (V)')):
    print(f'Écart affine maximal {label}: {np.max(np.abs(trace_mismatch[:,k])):.9e}')
for k, label in enumerate(('Na+', 'Cl-')):
    print(f'Défaut flux normal maximal {label} (mol/m²/s): {np.max(np.abs(flux_sum[:,k])):.9e}')
print('NOTE : un écart de potentiel initial de 1,5 V est normal : la tension n’est pas encore appliquée.')
print('NOTE : les flux physiques ici ne remplacent PAS les flux numériques DG périodiques.')
