"""V5 : transport stationnaire Na+ (Nernst-Planck) dans les 5 domaines.

Potentiel gele : execute electrodialyse_simple_v3.py, puis resout Na+.
Sans convection, Donnan, selectivite ni couplage retroactif sur phi.

Transformation de Slotboom : w = c * exp(F*phi/(R*T)).
Elle conduit a div(D*exp(-F*phi/(R*T))*grad(w)) = 0,
avec c = w*exp(-F*phi/(R*T)).

Usage : ED_VOLTAGE=0.006 python electrodialyse_simple_v5.py
Options : ED_C_LEFT=500 ED_C_RIGHT=500 ED_D_NA='1.33e-9,1.33e-9,1.33e-9,1.33e-9,1.33e-9'
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
p = runpy.run_path(str(ROOT / 'electrodialyse_simple_v3.py'))
domain, phi, region = p['domain'], p['phi'], p['region']
V = phi.function_space
if domain.comm.size != 1:
    raise RuntimeError('V5 : execution MPI a un processus uniquement.')

R = 8.31446261815324
F = 96485.33212
T = float(os.getenv('ED_TEMPERATURE', '298.15'))
c_left = float(os.getenv('ED_C_LEFT', '500'))
c_right = float(os.getenv('ED_C_RIGHT', '500'))
diff = np.fromstring(os.getenv('ED_D_NA', '1.33e-9,1.33e-9,1.33e-9,1.33e-9,1.33e-9'), sep=',')
if not np.isfinite(T) or T <= 0:
    raise ValueError('ED_TEMPERATURE doit etre positif.')
if not np.all(np.isfinite([c_left, c_right])) or min(c_left, c_right) <= 0:
    raise ValueError('Concentrations aux bords strictement positives requises.')
if len(diff) != 5 or not np.all(np.isfinite(diff)) or np.any(diff <= 0):
    raise ValueError('ED_D_NA : cinq coefficients de diffusion strictement positifs.')
alpha = F / (R * T)

Q = fem.functionspace(domain, ('DG', 0))
D = fem.Function(Q)
D.name = 'diffusion_Na_m2_s'
for cell in range(len(region)):
    dofs = Q.dofmap.cell_dofs(cell)
    D.x.array[dofs[0]] = diff[region[cell]]
D.x.scatter_forward()

xmin, xmax = p['xmin'], p['xmax']
length = xmax - xmin
tol = max(1e-10 * length, 1e-12)
fdim = domain.topology.dim - 1
left_facets = dmesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], xmin, atol=tol, rtol=0))
right_facets = dmesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], xmax, atol=tol, rtol=0))
left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
right_dofs = fem.locate_dofs_topological(V, fdim, right_facets)
if not len(left_dofs) or not len(right_dofs):
    raise RuntimeError('Facettes de Dirichlet introuvables.')
voltage = p['voltage']
if abs(alpha * voltage) > 600:
    raise ValueError('F*U/(RT) trop grand pour la transformation exponentielle.')
bcs = [
    fem.dirichletbc(PETSc.ScalarType(c_left), left_dofs, V),
    fem.dirichletbc(PETSc.ScalarType(c_right * np.exp(alpha * voltage)), right_dofs, V),
]

w_trial, v = ufl.TrialFunction(V), ufl.TestFunction(V)
weight = D * ufl.exp(-alpha * phi)
a = weight * ufl.inner(ufl.grad(w_trial), ufl.grad(v)) * ufl.dx
L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx
problem = LinearProblem(
    a, L, bcs=bcs,
    petsc_options_prefix='ed_simple_v5_',
    petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
)
w = problem.solve()
w.name = 'slotboom_Na'
w.x.scatter_forward()

# CG1 : valeurs nodales (meme espace et ordre de DOFs pour phi et w).
c = fem.Function(V)
c.name = 'concentration_Na_mol_m3'
c.x.array[:] = np.asarray(w.x.array.real) * np.exp(-alpha * np.asarray(phi.x.array.real))
c.x.scatter_forward()
if not np.all(np.isfinite(c.x.array)) or np.min(c.x.array) < -1e-9:
    raise RuntimeError('Concentration Na+ negative ou non finie.')

# Flux Nernst-Planck : N = -D*(grad(c) + alpha*c*grad(phi)).
# Pour la solution transformee, flux coherent : N = -D*exp(-alpha*phi)*grad(w).
flux_x = -weight * ufl.Dx(w, 0)
area = np.zeros(5)
mean_flux = np.zeros(5)
for i in range(5):
    dx_i = ufl.Measure('dx', domain=domain, subdomain_data=p['cell_tags'], subdomain_id=i+1)
    area[i] = fem.assemble_scalar(fem.form(1.0 * dx_i))
    mean_flux[i] = fem.assemble_scalar(fem.form(flux_x * dx_i)) / area[i]

coords = V.tabulate_dof_coordinates()
output = ROOT / 'transport_na_v5.npz'
np.savez(output, x=coords[:, 0], y=coords[:, 1], phi=np.asarray(phi.x.array.real),
         c_na=np.asarray(c.x.array.real), slotboom=np.asarray(w.x.array.real),
         region_cell=region, diffusivities=diff, c_left=c_left, c_right=c_right,
         temperature=T, alpha=alpha, flux_mean_x=mean_flux, areas=area)

print('\n=== V5 : TRANSPORT STATIONNAIRE Na+ ===')
print(f'Temperature : {T:.2f} K ; F/(RT) : {alpha:.6g} 1/V')
print(f'Concentrations imposees : gauche={c_left:.6g}, droite={c_right:.6g} mol/m3')
print(f'Potentiel impose par V3 : 0 a {voltage:.6g} V')
for i, name in enumerate(('Canal gauche', 'CEM', 'Canal central', 'AEM', 'Canal droit')):
    print(f'ID {i+1} {name:14s}: D={diff[i]:.6e} m2/s ; <Nx>={mean_flux[i]:+.8e} mol/(m2 s)')
print(f'Na+ min/max : {c.x.array.real.min():.8f} / {c.x.array.real.max():.8f} mol/m3')
print(f'Ecart relatif max-min des flux moyens : '
      f'{(mean_flux.max()-mean_flux.min())/max(np.max(np.abs(mean_flux)), 1e-300)*100:.6g} %')
print(f'Fichier : {output.name}')
print('Modele decouple : potentiel gele, aucune selectivite membranaire ni Donnan.')
