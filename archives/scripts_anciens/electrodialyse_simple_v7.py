"""V7: transport Na+/Cl- avec diffusion par sous-domaine Gmsh.

Test decouple, sans Donnan, charges fixes, convection ou electroneutralite.
Depend de electrodialyse_simple_v3.py (potentiel et maillage).

ED_VOLTAGE=0.006 ED_SIGMAS='1,0.2,1,0.2,1' python electrodialyse_simple_v7.py
ED_DNA='1.33e-9,1.33e-9,1.33e-9,1.33e-11,1.33e-9'
ED_DCL='2.03e-9,2.03e-11,2.03e-9,2.03e-9,2.03e-9'
"""
import os
import runpy
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

ROOT = Path(__file__).resolve().parent
p = runpy.run_path(str(ROOT / 'electrodialyse_simple_v3.py'))
domain, phi, V = p['domain'], p['phi'], p['V']
cell_tags, region = p['cell_tags'], p['region']
num_cells = domain.topology.index_map(domain.topology.dim).size_local
if domain.comm.size != 1:
    raise RuntimeError('V7 ne supporte que MPI size=1.')

T = float(os.getenv('ED_T', '298.15'))
F, R = 96485.33212, 8.314462618
beta = F / (R * T)
voltage = p['voltage']
c_left = float(os.getenv('ED_C_LEFT', '500'))
c_right = float(os.getenv('ED_C_RIGHT', '500'))
if T <= 0 or min(c_left, c_right) <= 0:
    raise ValueError('Temperature et concentrations aux bords doivent etre positives.')

Q = fem.functionspace(domain, ('DG', 0))
names = ('Canal gauche', 'CEM', 'Canal central', 'AEM', 'Canal droit')

def diffusivities(key, default):
    values = np.fromstring(os.getenv(key, default), sep=',')
    if values.size != 5 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f'{key} doit contenir cinq valeurs strictement positives.')
    return values

D_na = diffusivities('ED_DNA', '1.33e-9,1.33e-9,1.33e-9,1.33e-11,1.33e-9')
D_cl = diffusivities('ED_DCL', '2.03e-9,2.03e-11,2.03e-9,2.03e-9,2.03e-9')

def make_D(values, name):
    D = fem.Function(Q)
    D.name = name
    for cell in range(num_cells):
        D.x.array[Q.dofmap.cell_dofs(cell)[0]] = values[region[cell]]
    D.x.scatter_forward()
    return D

left_dofs, right_dofs = p['left_dofs'], p['right_dofs']
results = {}
for label, charge, values, prefix in (
    ('Na', +1, D_na, 'ed_v7_na_'),
    ('Cl', -1, D_cl, 'ed_v7_cl_'),
):
    D = make_D(values, 'D_' + label)
    # Slotboom: w = c * exp(z*F*phi/(RT)), N = -D*exp(-z*beta*phi)*grad(w).
    weight = ufl.exp(-charge * beta * phi)
    w_left = c_left * np.exp(charge * beta * 0.0)
    w_right = c_right * np.exp(charge * beta * voltage)
    bcs = [
        fem.dirichletbc(PETSc.ScalarType(w_left), left_dofs, V),
        fem.dirichletbc(PETSc.ScalarType(w_right), right_dofs, V),
    ]
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = D * weight * ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
    L = fem.Constant(domain, PETSc.ScalarType(0)) * test * ufl.dx
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options_prefix=prefix,
                            petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'})
    w = problem.solve()
    w.x.scatter_forward()
    c = fem.Function(V)
    c.name = 'concentration_' + label
    c.x.array[:] = np.asarray(w.x.array.real) * np.exp(
        -charge * beta * np.asarray(phi.x.array.real))
    c.x.scatter_forward()
    if not np.all(np.isfinite(c.x.array)) or np.min(c.x.array) <= 0:
        raise RuntimeError(f'{label}: concentration invalide.')
    flux_x = -D * weight * ufl.Dx(w, 0)
    means = []
    for i in range(1, 6):
        dx_i = ufl.Measure('dx', domain=domain, subdomain_data=cell_tags,
                           subdomain_id=i)
        area = fem.assemble_scalar(fem.form(1.0 * dx_i))
        avg = fem.assemble_scalar(fem.form(flux_x * dx_i)) / area
        means.append(float(np.real(avg)))
    results[label] = (c, np.asarray(means), D)

na, na_flux, _ = results['Na']
cl, cl_flux, _ = results['Cl']
ion_current = F * (na_flux - cl_flux)

print('\n=== V7 : NERNST-PLANCK AVEC DIFFUSIVITES PAR DOMAINE ===')
print(f'T={T:g} K ; U={voltage:g} V ; concentrations bords={c_left:g}/{c_right:g} mol/m3')
print(f'Na min/max : {na.x.array.min():.8f} / {na.x.array.max():.8f} mol/m3')
print(f'Cl min/max : {cl.x.array.min():.8f} / {cl.x.array.max():.8f} mol/m3')
for i, name in enumerate(names):
    print(f'ID {i+1} {name:14s}: D_Na={D_na[i]:.3e}, D_Cl={D_cl[i]:.3e} m2/s; '
          f'N_Na={na_flux[i]:+.8e}, N_Cl={cl_flux[i]:+.8e} mol/(m2 s); '
          f'i_ion={ion_current[i]:+.8e} A/m2')
for label, flux in (('Na', na_flux), ('Cl', cl_flux)):
    rel = np.ptp(flux) / max(np.max(np.abs(flux)), 1e-300) * 100
    print(f'Ecart relatif des flux moyens {label} : {rel:.6g} %')

out = ROOT / 'transport_na_cl_v7.npz'
np.savez(out, x=V.tabulate_dof_coordinates()[:, 0],
         c_na=np.asarray(na.x.array.real), c_cl=np.asarray(cl.x.array.real),
         flux_na=na_flux, flux_cl=cl_flux, ion_current=ion_current,
         D_na=D_na, D_cl=D_cl, voltage=voltage, region_cell=region)
print('Fichier :', out.name)
print('Modele decouple, sans Donnan ni selectivite thermodynamique.')
