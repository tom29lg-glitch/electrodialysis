#!/usr/bin/env python3
"""V12 experimental: 2D NP + correction de potentiel par conservation de charge.

IMPORTANT: ce n'est PAS le modele electroneutre couple complet.
Les facteurs Donnan sont fixes a la concentration de reference.
Le potentiel volumique est ajuste par iteration pour reduire div(i).
Utilise le maillage et les bords de la V3. Convection verticale dans canaux.
"""
import os
import runpy
from pathlib import Path
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

ROOT = Path(__file__).resolve().parent
p = runpy.run_path(str(ROOT / 'electrodialyse_simple_v3.py'))
domain, V, phi, region, tags = (p[k] for k in ('domain', 'V', 'phi', 'region', 'cell_tags'))
if domain.comm.size != 1:
    raise RuntimeError('V12 : execution MPI serie uniquement')
F, R = 96485.33212, 8.314462618
T = float(os.getenv('ED_T', '298.15'))
beta = F/(R*T)
cs = float(os.getenv('ED_C_SALT', '500'))
cin = float(os.getenv('ED_C_INLET', '500'))
cleft = float(os.getenv('ED_C_LEFT', '500'))
cright = float(os.getenv('ED_C_RIGHT', '500'))
uy = float(os.getenv('ED_UY', '1e-6'))
xcem = float(os.getenv('ED_X_CEM', '-1000'))
xaem = float(os.getenv('ED_X_AEM', '1000'))
iterations = int(os.getenv('ED_ITER', '8'))
relax = float(os.getenv('ED_RELAX', '0.2'))
if min(T, cs, cin, cleft, cright) <= 0 or xcem >= 0 or xaem <= 0 or not 0 < relax <= 1 or iterations < 1:
    raise ValueError('Parametres invalides')

def read5(key, default):
    a = np.fromstring(os.getenv(key, default), sep=',')
    if len(a) != 5 or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f'{key}: 5 nombres positifs requis')
    return a

dna = read5('ED_DNA', '1.33e-9,1.33e-9,1.33e-9,1.33e-11,1.33e-9')
dcl = read5('ED_DCL', '2.03e-9,2.03e-11,2.03e-9,2.03e-9,2.03e-9')
offsets = np.array([0., np.arcsinh(xcem/(2*cs))/beta, 0., np.arcsinh(xaem/(2*cs))/beta, 0.])
Q = fem.functionspace(domain, ('DG', 0))
nc = domain.topology.index_map(domain.topology.dim).size_local

def dg0(a, name):
    f = fem.Function(Q)
    f.name = name
    for cell in range(nc):
        f.x.array[Q.dofmap.cell_dofs(cell)[0]] = a[region[cell]]
    f.x.scatter_forward()
    return f

Dna = dg0(dna, 'DNa')
Dcl = dg0(dcl, 'DCl')
offset = dg0(offsets, 'Donnan')
vel = dg0(np.array([uy, 0., uy, 0., uy]), 'vitesse_verticale')
X = dg0(np.array([0., xcem, 0., xaem, 0.]), 'charge_fixe')
xyz = domain.geometry.x
xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
ymin, ymax = xyz[:, 1].min(), xyz[:, 1].max()
tol = max(1e-8 * max(xmax-xmin, ymax-ymin), 1e-12)
# Detection de l'entree en bas des canaux par la position des facettes.
domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
facets = mesh.exterior_facet_indices(domain.topology)
channel_intervals = []
geom_map = domain.geometry.dofmap
for i in (0, 2, 4):
    cells = np.flatnonzero(region == i)
    if not len(cells):
        raise RuntimeError(f'Canal {i+1} vide')
    vertices = np.unique(geom_map[cells].ravel())
    channel_intervals.append((xyz[vertices, 0].min(), xyz[vertices, 0].max()))
inlet, top = [], []
for f in facets:
    verts = mesh.entities_to_geometry(domain, domain.topology.dim-1,
                                      np.array([f], dtype=np.int32), False)[0]
    x, y = xyz[verts, :2].mean(axis=0)
    if abs(y-ymax) < tol:
        top.append(f)
    if abs(y-ymin) < tol and any(lo+tol < x < hi-tol for lo,hi in channel_intervals):
        inlet.append(f)
inlet_dofs = fem.locate_dofs_topological(V, 1, np.asarray(inlet, dtype=np.int32))
inlet_dofs = np.setdiff1d(inlet_dofs, np.union1d(p['left_dofs'], p['right_dofs'])).astype(np.int32)
if not len(inlet_dofs):
    raise RuntimeError('Aucun DOF entree detecte')
top = np.sort(np.asarray(top, dtype=np.int32))
facet_tags = mesh.meshtags(domain, 1, top, np.full(len(top), 99, dtype=np.int32))
ds_top = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags, subdomain_id=99)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
zero = fem.Constant(domain, PETSc.ScalarType(0.))
outer_bcs = [fem.dirichletbc(PETSc.ScalarType(0.), p['left_dofs'], V),
             fem.dirichletbc(PETSc.ScalarType(0.), p['right_dofs'], V)]

def transport(name, z, D):
    weight = ufl.exp(-z*beta*(phi+offset))
    a = (D*weight*ufl.inner(ufl.grad(u), ufl.grad(v))
         - vel*weight*u*ufl.Dx(v, 1))*ufl.dx + vel*weight*u*v*ds_top
    inlet_values = fem.Function(V)
    inlet_values.x.array[:] = cin*np.exp(z*beta*np.asarray(phi.x.array.real))
    inlet_values.x.scatter_forward()
    bcs = [fem.dirichletbc(PETSc.ScalarType(cleft), p['left_dofs'], V),
           fem.dirichletbc(PETSc.ScalarType(cright*np.exp(z*beta*p['voltage'])), p['right_dofs'], V),
           fem.dirichletbc(inlet_values, inlet_dofs)]
    w = LinearProblem(a, zero*v*ufl.dx, bcs=bcs,
                      petsc_options_prefix=f'ed_v12_{name}_',
                      petsc_options={'ksp_type':'preonly', 'pc_type':'lu'}).solve()
    w.x.scatter_forward()
    c = fem.Function(Q)
    c.interpolate(fem.Expression(weight*w, Q.element.interpolation_points))
    c.x.scatter_forward()
    flux = -D*weight*ufl.grad(w) + ufl.as_vector((0., vel*weight*w))
    return c, flux

history = []
for step in range(iterations):
    cna, na = transport('na', 1, Dna)
    ccl, cl = transport('cl', -1, Dcl)
    current = F*(na-cl)
    # Correction de potentiel: delta(i) ~ -kappa grad(delta_phi).
    # Le flux ionique calcule comprend convection et migration.
    kappa = F*beta*(Dna*cna+Dcl*ccl)
    a = kappa*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = ufl.inner(current, ufl.grad(v))*ufl.dx
    delta = LinearProblem(a, L, bcs=outer_bcs,
                          petsc_options_prefix='ed_v12_charge_',
                          petsc_options={'ksp_type':'preonly', 'pc_type':'lu'}).solve()
    delta.x.scatter_forward()
    dmax = float(np.max(np.abs(delta.x.array.real)))
    phi.x.array[:] += relax*delta.x.array
    phi.x.scatter_forward()
    history.append(dmax)
    print(f'Iteration {step+1:02d}: correction max brute={dmax:.4e} V')
    if not np.isfinite(dmax) or dmax > 10.:
        raise RuntimeError('Correction diverge: reduire ED_RELAX ou ED_UY')

# Recalcul des ions sur le dernier potentiel, pour diagnostics coherents.
cna, na = transport('na_final', 1, Dna)
ccl, cl = transport('cl_final', -1, Dcl)
current = F*(na-cl)
ix = fem.Function(Q)
iy = fem.Function(Q)
ix.interpolate(fem.Expression(current[0], Q.element.interpolation_points))
iy.interpolate(fem.Expression(current[1], Q.element.interpolation_points))
ix.x.scatter_forward(); iy.x.scatter_forward()
coords = np.array([xyz[geom_map[k], :2].mean(axis=0) for k in range(nc)])
nbins = int(os.getenv('ED_Y_BINS', '30'))
if nbins < 2: raise ValueError('ED_Y_BINS doit etre >=2')
edges = np.linspace(ymin, ymax, nbins+1)
profiles = np.full((5, nbins), np.nan)
for j in range(5):
    for k in range(nbins):
        m = (region == j) & (coords[:,1]>=edges[k]) & (coords[:,1] < edges[k+1] if k<nbins-1 else coords[:,1]<=edges[k+1])
        if np.any(m): profiles[j,k] = np.mean(ix.x.array[m])
neutrality = cna.x.array.real-ccl.x.array.real+X.x.array.real
print('\n=== V12 : COUPLAGE ITERATIF DE COURANT 2D (EXPERIMENTAL) ===')
print(f'Vitesse verticale={uy:.3e} m/s, tension={p["voltage"]:.6g} V')
print(f'Concentrations min: Na={cna.x.array.real.min():.5g}, Cl={ccl.x.array.real.min():.5g} mol/m3')
print(f'Ecart max electroneutralite NON imposee={np.max(np.abs(neutrality)):.6g} mol/m3')
for j,name in enumerate(('Canal gauche','CEM','Canal central','AEM','Canal droit')):
    arr = profiles[j, np.isfinite(profiles[j])]
    print(f'ID {j+1} {name:14s}: i_x(y)=[{arr.min():+.6e}, {arr.max():+.6e}] A/m2')
print('Correction max derniere iteration=', history[-1], 'V')
print('ATTENTION : iterations fixes, sans preuve de convergence ni conservation exacte.')
print('ATTENTION : Donnan fixe; electroneutralite et interfaces non couplees.')
out = ROOT/'electrodialyse_2d_courant_v12.npz'
np.savez(out, x_cell=coords[:,0], y_cell=coords[:,1], region_cell=region,
         c_na=cna.x.array.real, c_cl=ccl.x.array.real,
         i_x=ix.x.array.real, i_y=iy.x.array.real, phi=phi.x.array.real,
         y_centers=(edges[:-1]+edges[1:])/2, i_x_profile=profiles,
         corrections=np.array(history), electroneutrality=neutrality)
print('Fichier :', out.name)
