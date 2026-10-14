#!/usr/bin/env python3
"""V11: 2D Na/Cl transport avec convection verticale et Donnan impose.

Requiert electrodialyse_simple_v3.py dans le meme dossier. Modele
intermediaire: potentiel gele (V3), PAS de couplage electroneutre V10.
ED_UY=1e-6 ED_C_INLET=500 python electrodialyse_2d_convection_v11.py

La convection est verticale dans les canaux 1,3,5 et nulle dans les membranes.
Concentrations d'entree imposees en bas des canaux (hors bords lateraux).
Les bords hauts ont des conditions de flux diffusif naturel nul.
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
domain, phi, V = p['domain'], p['phi'], p['V']
region, tags = p['region'], p['cell_tags']
if domain.comm.size != 1:
    raise RuntimeError('V11 est prevue pour un seul processus MPI.')
if domain.geometry.dim != 2:
    raise RuntimeError('Maillage 2D requis.')
F, R = 96485.33212, 8.314462618
T = float(os.getenv('ED_T', '298.15'))
cs = float(os.getenv('ED_C_SALT', '500'))
c_in = float(os.getenv('ED_C_INLET', '500'))
c_left = float(os.getenv('ED_C_LEFT', '500'))
c_right = float(os.getenv('ED_C_RIGHT', '500'))
uy = float(os.getenv('ED_UY', '1e-6'))
Xcem = float(os.getenv('ED_X_CEM', '-1000'))
Xaem = float(os.getenv('ED_X_AEM', '1000'))
if min(T, cs, c_in, c_left, c_right) <= 0 or Xcem >= 0 or Xaem <= 0:
    raise ValueError('Parametres physiques invalides.')
beta = F/(R*T)
offsets = np.array([0., np.arcsinh(Xcem/(2*cs))/beta, 0.,
                    np.arcsinh(Xaem/(2*cs))/beta, 0.])

def read_array(key, default):
    a = np.fromstring(os.getenv(key, default), sep=',')
    if len(a) != 5 or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f'{key} doit contenir 5 diffusivites positives.')
    return a

dna = read_array('ED_DNA', '1.33e-9,1.33e-9,1.33e-9,1.33e-11,1.33e-9')
dcl = read_array('ED_DCL', '2.03e-9,2.03e-11,2.03e-9,2.03e-9,2.03e-9')
Q = fem.functionspace(domain, ('DG', 0))
nc = domain.topology.index_map(domain.topology.dim).size_local

def dg0(vals, name):
    out = fem.Function(Q)
    out.name = name
    for k in range(nc):
        out.x.array[Q.dofmap.cell_dofs(k)[0]] = vals[region[k]]
    out.x.scatter_forward()
    return out

offset = dg0(offsets, 'Donnan_offset')
vel = dg0(np.array([uy, 0., uy, 0., uy]), 'vitesse_y')
# Facettes du bord inferieur, en ne gardant que celles des canaux.
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
facets = mesh.exterior_facet_indices(domain.topology)
xyz = domain.geometry.x
xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
ymin, ymax = xyz[:, 1].min(), xyz[:, 1].max()
span = max(xmax-xmin, ymax-ymin)
tol = max(span*1e-8, 1e-12)
# L'intervalle horizontal de chaque domaine est estime par les sommets
# des cellules du domaine (le maillage est une succession de bandes).
channel_intervals = []
for i in (0, 2, 4):
    cells = np.flatnonzero(region == i)
    if len(cells) == 0:
        raise RuntimeError(f'Domaine canal {i+1} vide')
    verts = np.unique(domain.geometry.dofmap[cells].ravel())
    channel_intervals.append((xyz[verts, 0].min(), xyz[verts, 0].max()))

facet_coords = np.empty((len(facets), 2))
for j, f in enumerate(facets):
    vertices = mesh.entities_to_geometry(domain, domain.topology.dim-1,
                                         np.array([f], dtype=np.int32), False)[0]
    facet_coords[j] = xyz[vertices, :2].mean(axis=0)
top_facets = np.asarray([f for f, (_, fy) in zip(facets, facet_coords)
                         if abs(fy-ymax) < tol], dtype=np.int32)
top_tags = mesh.meshtags(domain, domain.topology.dim-1,
                         np.sort(top_facets),
                         np.full(len(top_facets), 99, dtype=np.int32))
ds_top = ufl.Measure('ds', domain=domain, subdomain_data=top_tags, subdomain_id=99)
selected = []
for f, (fx, fy) in zip(facets, facet_coords):
    if abs(fy-ymin) < tol and any(lo+tol < fx < hi-tol
                                 for lo, hi in channel_intervals):
        selected.append(f)
selected = np.asarray(selected, dtype=np.int32)
inlet_dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, selected)
# On exclut les DOFs deja soumis aux conditions gauche/droite.
outer = np.union1d(p['left_dofs'], p['right_dofs'])
inlet_dofs = np.setdiff1d(inlet_dofs, outer).astype(np.int32)
if not len(inlet_dofs):
    raise RuntimeError('Aucun DOF d\'entree trouve. Verifier orientation du maillage.')

results = {}
for name, z, Dvals in (('Na', +1, dna), ('Cl', -1, dcl)):
    D = dg0(Dvals, 'D_'+name)
    weight = ufl.exp(-z*beta*(phi+offset))
    # N = -D*weight*grad(w) + uy*weight*w*e_y
    # div(N)=0, test v: int D weight grad(w).grad(v)
    #                      - int uy weight w d_y(v) = 0
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (D*weight*ufl.inner(ufl.grad(trial), ufl.grad(test))
         - vel*weight*trial*ufl.Dx(test, 1))*ufl.dx \
        + vel*weight*trial*test*ds_top
    L = fem.Constant(domain, PETSc.ScalarType(0))*test*ufl.dx
    # Sur le bas, c=c_in. La variable w varie selon phi(x).
    inlet_values = fem.Function(V)
    inlet_values.x.array[:] = c_in*np.exp(z*beta*np.asarray(phi.x.array.real))
    inlet_values.x.scatter_forward()
    bcs = [
        fem.dirichletbc(PETSc.ScalarType(c_left), p['left_dofs'], V),
        fem.dirichletbc(PETSc.ScalarType(c_right*np.exp(z*beta*p['voltage'])),
                        p['right_dofs'], V),
        fem.dirichletbc(inlet_values, inlet_dofs),
    ]
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options_prefix='ed_v11_'+name.lower()+'_',
                            petsc_options={'ksp_type':'preonly', 'pc_type':'lu'})
    w = problem.solve()
    w.x.scatter_forward()
    c = fem.Function(Q)
    c.name = 'c_'+name+'_DG0'
    expr = fem.Expression(w*weight, Q.element.interpolation_points)
    c.interpolate(expr)
    c.x.scatter_forward()
    if not np.all(np.isfinite(c.x.array)):
        raise RuntimeError(f'{name}: concentrations non finies')
    if np.min(c.x.array) <= 0:
        print(f'AVERTISSEMENT: {name} c_min={np.min(c.x.array):.4g}; '
              'oscillations de convection possibles (CG1 sans SUPG).')
    nx = -D*weight*ufl.Dx(w, 0)
    ny = -D*weight*ufl.Dx(w, 1) + vel*weight*w
    means = []
    for i in range(1,6):
        dx = ufl.Measure('dx', domain=domain, subdomain_data=tags,
                         subdomain_id=i)
        area = fem.assemble_scalar(fem.form(1.0*dx))
        means.append(float(np.real(fem.assemble_scalar(fem.form(nx*dx))/area)))
    results[name] = (w, c, np.asarray(means), nx, ny)

wna, cna, fna, nxna, nyna = results['Na']
wcl, ccl, fcl, nxcl, nycl = results['Cl']
# Profils de i_x(y): moyenne volumique sur bandes horizontales.
# La projection DG0 du flux est realisee aux barycentres.
ix = fem.Function(Q)
ix.name = 'courant_ionique_x_DG0'
ix.interpolate(fem.Expression(F*(nxna-nxcl), Q.element.interpolation_points))
ix.x.scatter_forward()
iy = fem.Function(Q)
iy.name = 'courant_ionique_y_DG0'
iy.interpolate(fem.Expression(F*(nyna-nycl), Q.element.interpolation_points))
iy.x.scatter_forward()
coords = np.empty((nc, 2))
for k in range(nc):
    coords[k] = xyz[domain.geometry.dofmap[k], :2].mean(axis=0)
nbins = int(os.getenv('ED_Y_BINS', '30'))
if nbins < 2:
    raise ValueError('ED_Y_BINS >= 2')
edges = np.linspace(ymin, ymax, nbins+1)
prof = np.full((5, nbins), np.nan)
counts = np.zeros((5, nbins), dtype=int)
for j in range(5):
    for k in range(nbins):
        subset = ((region == j) & (coords[:,1] >= edges[k]) &
                  ((coords[:,1] < edges[k+1]) if k < nbins-1
                   else (coords[:,1] <= edges[k+1])))
        counts[j,k] = int(subset.sum())
        if subset.any():
            prof[j,k] = float(np.mean(ix.x.array[subset]))

print('\n=== V11 : TRANSPORT 2D AVEC CONVECTION VERTICALE ===')
print('Modele decouple: potentiel V3 gele; sauts Donnan imposes.')
print(f'vitesse verticale canaux={uy:.4e} m/s ; c_in={c_in:g} mol/m3')
print(f'Hauteur y=[{ymin:.6g},{ymax:.6g}] m ; '
      f'{len(inlet_dofs)} DOFs entree ; {nbins} bandes')
for j, name in enumerate(('Canal gauche','CEM','Canal central','AEM','Canal droit')):
    vals = prof[j, np.isfinite(prof[j])]
    if len(vals):
        print(f'ID {j+1} {name:14s}: i_x moyen={F*(fna[j]-fcl[j]):+.7e} A/m2; '
              f'profil i_x(y)=[{vals.min():+.7e},{vals.max():+.7e}] A/m2')
    else:
        print(f'ID {j+1} {name}: aucune bande peuplee')
print('ATTENTION : le profil i_x(y) est une moyenne sur bandes, '
      'pas un flux integre sur les interfaces.')
print('ATTENTION : pas de couplage electroneutre, ni SUPG a fort Peclet.')
out = ROOT/'electrodialyse_2d_convection_v11.npz'
np.savez(out, x_cell=coords[:,0], y_cell=coords[:,1], region_cell=region,
         c_na_DG0=cna.x.array.real, c_cl_DG0=ccl.x.array.real,
         i_x_DG0=ix.x.array.real, i_y_DG0=iy.x.array.real,
         y_centers=(edges[:-1]+edges[1:])/2, i_x_profile=prof,
         profile_counts=counts, flux_na_region=fna, flux_cl_region=fcl,
         velocity_y=uy, voltage=p['voltage'])
print('Fichier :', out.name)
