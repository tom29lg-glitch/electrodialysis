"""Controle du courant aux deux bords exterieurs du modele electrique V3.

Execution : ED_VOLTAGE=0.006 python electrodialyse_flux_v4b.py
La V3 est executee d'abord et fournit mesh, sigma et phi.
"""
import os
import runpy
from pathlib import Path

import numpy as np
from dolfinx import fem, mesh as dmesh
import ufl

root = Path(__file__).resolve().parent
v3 = runpy.run_path(str(root / 'electrodialyse_simple_v3.py'))
domain = v3['domain']
sigma = v3['sigma']
phi = v3['phi']
voltage = float(v3['voltage'])

if domain.comm.size != 1:
    raise RuntimeError('Ce diagnostic est prevu pour un seul processus MPI.')
if abs(voltage) < 1e-15:
    raise ValueError('La tension doit etre non nulle pour comparer P / V.')

tdim = domain.topology.dim
fdim = tdim - 1
xmin, xmax = float(v3['xmin']), float(v3['xmax'])
tol = max(1e-10 * (xmax - xmin), 1e-12)

left = dmesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], xmin, atol=tol, rtol=0))
right = dmesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], xmax, atol=tol, rtol=0))
if not len(left) or not len(right):
    raise RuntimeError('Bords externes non trouves.')
if np.intersect1d(left, right).size:
    raise RuntimeError('Les marqueurs gauche et droit se chevauchent.')

facet_ids = np.r_[left, right].astype(np.int32)
facet_values = np.r_[np.full(len(left), 1, dtype=np.int32),
                     np.full(len(right), 2, dtype=np.int32)]
order = np.argsort(facet_ids)
facet_tags = dmesh.meshtags(domain, fdim, facet_ids[order], facet_values[order])
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags)
dx = ufl.Measure('dx', domain=domain)

J = -sigma * ufl.grad(phi)
n = ufl.FacetNormal(domain)

# Courant sortant par metre de profondeur (A/m) sur chaque bord.
def integrate(expr):
    return float(np.real(fem.assemble_scalar(fem.form(expr))))

I_left = integrate(ufl.dot(J, n) * ds(1))
I_right = integrate(ufl.dot(J, n) * ds(2))
P = integrate(sigma * ufl.inner(ufl.grad(phi), ufl.grad(phi)) * dx)
height_left = integrate(1.0 * ds(1))
height_right = integrate(1.0 * ds(2))
I_energy = P / voltage

scale = max(abs(I_left), abs(I_right), 1e-30)
conservation = abs(I_left + I_right) / scale
balance = abs(I_right + I_energy) / max(abs(I_right), abs(I_energy), 1e-30)

print('\n=== V4B : CONTROLE DES FLUX AUX BORDS ===')
print(f'Hauteur bord gauche / droit : {height_left:.8e} / {height_right:.8e} m')
print(f'Flux sortant bord gauche  : {I_left:+.10e} A/m')
print(f'Flux sortant bord droit   : {I_right:+.10e} A/m')
print(f'Somme des flux sortants   : {I_left + I_right:+.10e} A/m')
print(f'Ecart relatif gauche/droit : {100 * conservation:.6g} %')
print(f'Dissipation P              : {P:.10e} W/m')
print(f'Courant attendu -P/U      : {-I_energy:+.10e} A/m')
print(f'Ecart relatif droit / -P/U : {100 * balance:.6g} %')
print('Note : controle global aux bords, pas verification locale de div(J).')

out = root / 'diagnostic_flux_v4b.npz'
np.savez(out, I_left=I_left, I_right=I_right, P=P,
         conservation_relative=conservation, energy_relative=balance,
         height_left=height_left, height_right=height_right, voltage=voltage)
print(f'Fichier : {out.name}')
