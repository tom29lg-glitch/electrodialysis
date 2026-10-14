"""V6 : transport Na+ et Cl- decouple, potentiel V3 gele.

Execute V5 (qui execute V3), puis resout Cl- par transformation de Slotboom.
Aucune selectivite membranaire, Donnan, electroneutralite imposee ou
retroaction des concentrations sur le potentiel.

Usage : ED_VOLTAGE=0.006 python electrodialyse_simple_v6.py
Options : ED_D_CL='2.03e-9,2.03e-9,2.03e-9,2.03e-9,2.03e-9'
          ED_C_CL_LEFT=500 ED_C_CL_RIGHT=500
"""
import os
import runpy
from pathlib import Path

import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl

ROOT = Path(__file__).resolve().parent
p = runpy.run_path(str(ROOT / 'electrodialyse_simple_v5.py'))
domain, phi, region = p['domain'], p['phi'], p['region']
V = phi.function_space
Q = fem.functionspace(domain, ('DG', 0))
alpha, F = p['alpha'], p['F']
voltage = p['voltage']

c_left = float(os.getenv('ED_C_CL_LEFT', '500'))
c_right = float(os.getenv('ED_C_CL_RIGHT', '500'))
diff = np.fromstring(os.getenv('ED_D_CL', '2.03e-9,2.03e-9,2.03e-9,2.03e-9,2.03e-9'), sep=',')
if not np.all(np.isfinite([c_left, c_right])) or min(c_left, c_right) <= 0:
    raise ValueError('ED_C_CL_LEFT et ED_C_CL_RIGHT doivent etre positifs et finis.')
if len(diff) != 5 or not np.all(np.isfinite(diff)) or np.any(diff <= 0):
    raise ValueError('ED_D_CL : cinq coefficients positifs et finis requis.')

D = fem.Function(Q)
D.name = 'diffusion_Cl_m2_s'
for cell in range(len(region)):
    D.x.array[Q.dofmap.cell_dofs(cell)[0]] = diff[region[cell]]
D.x.scatter_forward()

# Pour Cl- (z=-1) : w_Cl = c_Cl exp(-alpha*phi)
# N_Cl = -D exp(+alpha*phi) grad(w_Cl).
# phi=0 a gauche, phi=voltage a droite.
if abs(alpha * voltage) > 600:
    raise ValueError('Tension trop elevee pour la transformation exponentielle.')
bcs = [
    fem.dirichletbc(PETSc.ScalarType(c_left), p['left_dofs'], V),
    fem.dirichletbc(PETSc.ScalarType(c_right * np.exp(-alpha * voltage)), p['right_dofs'], V),
]
trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
weight = D * ufl.exp(alpha * phi)
a = weight * ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
L = fem.Constant(domain, PETSc.ScalarType(0.0)) * test * ufl.dx
problem = LinearProblem(a, L, bcs=bcs,
                        petsc_options_prefix='ed_simple_v6_cl_',
                        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'})
w_cl = problem.solve()
w_cl.name = 'slotboom_Cl'
w_cl.x.scatter_forward()

c_cl = fem.Function(V)
c_cl.name = 'concentration_Cl_mol_m3'
c_cl.x.array[:] = np.asarray(w_cl.x.array.real) * np.exp(alpha * np.asarray(phi.x.array.real))
c_cl.x.scatter_forward()
if not np.all(np.isfinite(c_cl.x.array)) or np.min(c_cl.x.array) < -1e-9:
    raise RuntimeError('Concentration Cl- negative ou non finie.')

flux_cl_x = -weight * ufl.Dx(w_cl, 0)
areas = np.zeros(5)
mean_cl = np.zeros(5)
for i in range(5):
    dx_i = ufl.Measure('dx', domain=domain, subdomain_data=p['p']['cell_tags'], subdomain_id=i + 1)
    areas[i] = fem.assemble_scalar(fem.form(1.0 * dx_i))
    mean_cl[i] = fem.assemble_scalar(fem.form(flux_cl_x * dx_i)) / areas[i]

mean_na = p['mean_flux']
current_ionic = F * (mean_na - mean_cl)
current_ohmic = np.array([
    fem.assemble_scalar(fem.form((-p['p']['sigma'] * ufl.Dx(phi, 0)) *
        ufl.Measure('dx', domain=domain, subdomain_data=p['p']['cell_tags'], subdomain_id=i+1))) / areas[i]
    for i in range(5)
])

output = ROOT / 'transport_na_cl_v6.npz'
coords = V.tabulate_dof_coordinates()
np.savez(output, x=coords[:, 0], y=coords[:, 1],
         phi=np.asarray(phi.x.array.real),
         c_na=np.asarray(p['c'].x.array.real), c_cl=np.asarray(c_cl.x.array.real),
         flux_na_x=mean_na, flux_cl_x=mean_cl,
         current_ionic_x=current_ionic, current_ohmic_x=current_ohmic,
         d_cl=diff, d_na=p['diff'], temperature=p['T'],
         region_cell=region, areas=areas)

print('\n=== V6 : TRANSPORT STATIONNAIRE Na+ ET Cl- ===')
print(f'Temperature : {p["T"]:.2f} K ; tension : {voltage:.6g} V')
print(f'Cl- impose : gauche={c_left:g}, droite={c_right:g} mol/m3')
print(f'Cl- min/max : {c_cl.x.array.real.min():.8f} / {c_cl.x.array.real.max():.8f} mol/m3')
for i, name in enumerate(('Canal gauche', 'CEM', 'Canal central', 'AEM', 'Canal droit')):
    print(f'ID {i+1} {name:14s}: N_Na={mean_na[i]:+.8e}, N_Cl={mean_cl[i]:+.8e} mol/(m2 s) ; '
          f'i_ion={current_ionic[i]:+.8e} A/m2 ; i_ohm={current_ohmic[i]:+.8e} A/m2')
print('Ecart relatif Cl entre domaines : '
      f'{100 * np.ptp(mean_cl) / max(np.max(np.abs(mean_cl)), 1e-300):.6g} %')
print(f'Fichier : {output.name}')
print('Attention : i_ion et i_ohm ne sont pas contraints a etre egaux dans ce modele decouple.')
print('Pas de Donnan, selectivite membranaire, electroneutralite imposee ou convection.')
