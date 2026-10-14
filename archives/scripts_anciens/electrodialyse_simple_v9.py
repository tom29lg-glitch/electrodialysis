"""V9: Nernst-Planck avec sauts Donnan prescrits, potentiel V3 gele.

Les concentrations sont DG0, la variable electrochimique Slotboom est CG1.
Limites: pas de couplage electrique ni electroneutralite hors equilibre.
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
v3 = runpy.run_path(str(ROOT / 'electrodialyse_simple_v3.py'))
domain, phi, V = v3['domain'], v3['phi'], v3['V']
region, tags = v3['region'], v3['cell_tags']
if domain.comm.size != 1:
    raise RuntimeError('V9 fonctionne sur un seul processus MPI.')
Q = fem.functionspace(domain, ('DG', 0))
nc = domain.topology.index_map(domain.topology.dim).size_local
F, R = 96485.33212, 8.314462618
T = float(os.getenv('ED_T', '298.15'))
cs = float(os.getenv('ED_C_SALT', '500'))
Xcem = float(os.getenv('ED_X_CEM', '-1000'))
Xaem = float(os.getenv('ED_X_AEM', '1000'))
c_left = float(os.getenv('ED_C_LEFT', '500'))
c_right = float(os.getenv('ED_C_RIGHT', '500'))
if not np.all(np.isfinite([T, cs, Xcem, Xaem, c_left, c_right])):
    raise ValueError('Parametres non finis.')
if min(T, cs, c_left, c_right) <= 0 or Xcem >= 0 or Xaem <= 0:
    raise ValueError('T et concentrations > 0 ; X_CEM < 0 ; X_AEM > 0.')
beta = F / (R * T)
delta_cem = np.arcsinh(Xcem / (2 * cs)) / beta
delta_aem = np.arcsinh(Xaem / (2 * cs)) / beta
offsets = np.array([0., delta_cem, 0., delta_aem, 0.])

def read_diff(key, default):
    values = np.fromstring(os.getenv(key, default), sep=',')
    if len(values) != 5 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f'{key}: cinq valeurs positives requises.')
    return values

dna = read_diff('ED_DNA', '1.33e-9,1.33e-9,1.33e-9,1.33e-11,1.33e-9')
dcl = read_diff('ED_DCL', '2.03e-9,2.03e-11,2.03e-9,2.03e-9,2.03e-9')

def piecewise(values, name):
    f = fem.Function(Q)
    f.name = name
    for cell in range(nc):
        f.x.array[Q.dofmap.cell_dofs(cell)[0]] = values[region[cell]]
    f.x.scatter_forward()
    return f

offset = piecewise(offsets, 'potentiel_Donnan_par_region')
results = {}
for label, z, values in (('Na', 1, dna), ('Cl', -1, dcl)):
    D = piecewise(values, 'D_' + label)
    # Le gradient du saut DG0 est nul dans chaque cellule; la continuite
    # de w impose la continuite du potentiel electrochimique a l'interface.
    weight = ufl.exp(-z * beta * (phi + offset))
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = D * weight * ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
    L = fem.Constant(domain, PETSc.ScalarType(0)) * test * ufl.dx
    bcs = [
        fem.dirichletbc(PETSc.ScalarType(c_left), v3['left_dofs'], V),
        fem.dirichletbc(PETSc.ScalarType(c_right * np.exp(z * beta * v3['voltage'])),
                        v3['right_dofs'], V),
    ]
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options_prefix='ed_v9_' + label.lower() + '_',
                            petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'})
    w = problem.solve()
    w.x.scatter_forward()
    c = fem.Function(Q)
    c.name = 'concentration_' + label + '_DG0'
    expr = fem.Expression(w * weight, Q.element.interpolation_points)
    c.interpolate(expr)
    c.x.scatter_forward()
    if not np.all(np.isfinite(c.x.array)) or np.min(c.x.array) <= 0:
        raise RuntimeError(label + ': concentrations invalides.')
    flux_x = -D * weight * ufl.Dx(w, 0)
    means = []
    for i in range(1, 6):
        dx_i = ufl.Measure('dx', domain=domain, subdomain_data=tags, subdomain_id=i)
        area = fem.assemble_scalar(fem.form(1.0 * dx_i))
        means.append(float(np.real(fem.assemble_scalar(fem.form(flux_x * dx_i)) / area)))
    results[label] = (c, w, np.array(means))

cna, wna, fna = results['Na']
ccl, wcl, fcl = results['Cl']
current = F * (fna - fcl)
fixed = piecewise(np.array([0., Xcem, 0., Xaem, 0.]), 'charge_fixe')
neutrality = np.asarray(cna.x.array.real) - np.asarray(ccl.x.array.real) + np.asarray(fixed.x.array.real)
print('\n=== V9 : TRANSPORT AVEC SAUTS DE DONNAN IMPOSES ===')
print(f'T={T:g} K ; U={v3["voltage"]:g} V ; c_ref={cs:g} mol/m3')
print(f'Donnan CEM={delta_cem*1e3:+.6f} mV ; AEM={delta_aem*1e3:+.6f} mV')
print('Concentrations DG0 discontinues ; variable electrochimique w CG1 continue.')
names = ('Canal gauche', 'CEM', 'Canal central', 'AEM', 'Canal droit')
for i, name in enumerate(names):
    idx = region == i
    print(f'ID {i+1} {name:14s}: '
          f'cNa=[{cna.x.array[idx].min():.5f},{cna.x.array[idx].max():.5f}] ; '
          f'cCl=[{ccl.x.array[idx].min():.5f},{ccl.x.array[idx].max():.5f}] mol/m3 ; '
          f'NNa={fna[i]:+.8e} ; NCl={fcl[i]:+.8e} mol/(m2 s) ; '
          f'i={current[i]:+.8e} A/m2')
for label, flux in (('Na', fna), ('Cl', fcl)):
    rel = 100 * np.ptp(flux) / max(np.max(np.abs(flux)), 1e-300)
    print(f'Conservation des flux moyens {label}: ecart={rel:.6g} %')
print(f'Ecart max a electroneutralite DG0 (NON imposee): {np.max(np.abs(neutrality)):.6g} mol/m3')
out = ROOT / 'transport_na_cl_donnan_v9.npz'
np.savez(out, region_cell=region, c_na_DG0=cna.x.array.real,
         c_cl_DG0=ccl.x.array.real, phi_bulk_CG1=phi.x.array.real,
         w_na_CG1=wna.x.array.real, w_cl_CG1=wcl.x.array.real,
         phi_offset_region=offsets, flux_na=fna, flux_cl=fcl,
         ionic_current=current, D_na=dna, D_cl=dcl, voltage=v3['voltage'])
print('Fichier :', out.name)
print('ATTENTION : potentiel gele et sauts Donnan fixes a l\'equilibre de reference.')
print('Electroneutralite hors equilibre non imposee ; pas de couplage electrique.')
