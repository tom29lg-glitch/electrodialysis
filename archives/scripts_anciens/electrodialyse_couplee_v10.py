#!/usr/bin/env python3
"""V10: modele 1D stationnaire electroneutre couple (NaCl, 5 couches).

Les largeurs sont illustratives, PAS extraites du maillage Gmsh.
La tension est imposee entre les deux bords externes. Conditions de Donnan
ideales aux interfaces, avec flux conserves. Pas de convection ni de variation y.
Dependances: numpy, scipy. Sortie: electrodialyse_couplee_v10.npz.
"""
import os
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

F = 96485.33212
R = 8.31446261815324
T = float(os.getenv('ED_T', '298.15'))
beta = F/(R*T)
voltage = float(os.getenv('ED_VOLTAGE', '0.006'))
cl = float(os.getenv('ED_C_LEFT', '500'))
cr = float(os.getenv('ED_C_RIGHT', '500'))
names = ('Canal gauche', 'CEM', 'Canal central', 'AEM', 'Canal droit')

def vector(name, default):
    a = np.fromstring(os.getenv(name, default), sep=',')
    if a.shape != (5,) or not np.all(np.isfinite(a)):
        raise ValueError(f'{name}: exactement 5 nombres requis')
    return a

width = vector('ED_WIDTHS', '0.001,0.0002,0.001,0.0002,0.001')
dna = vector('ED_DNA', '1.33e-9,1.33e-9,1.33e-9,1.33e-11,1.33e-9')
dcl = vector('ED_DCL', '2.03e-9,2.03e-11,2.03e-9,2.03e-9,2.03e-9')
X = np.array([0., float(os.getenv('ED_X_CEM', '-1000')), 0.,
              float(os.getenv('ED_X_AEM', '1000')), 0.])
if (not np.all(np.isfinite([T, voltage, cl, cr, *X])) or T <= 0 or
    min(cl, cr) <= 0 or np.any(width <= 0) or np.any(dna <= 0) or
    np.any(dcl <= 0) or X[1] >= 0 or X[3] <= 0):
    raise ValueError('Parametres physiques invalides')


def donnan(cna, oldx, newx):
    """Continuite des potentiels electrochimiques et neutralite."""
    ccl = cna + oldx
    if cna <= 0 or ccl <= 0:
        raise ValueError('Concentration negative avant interface')
    product = cna*ccl
    root = np.hypot(newx, 2*np.sqrt(product))
    newna = 2*product/(root+newx) if newx >= 0 else (root-newx)/2
    dphi = -np.log(newna/cna)/beta
    return newna, dphi


def deriv(state, flux, i):
    c = state[0]
    if c <= 0 or c + X[i] <= 0:
        raise ValueError('Concentration non positive')
    a, b = -flux[0]/dna[i], -flux[1]/dcl[i]
    denom = 2*c + X[i]
    return np.array([(a*(c+X[i])+b*c)/denom,
                     (a-b)/(beta*denom)])


def simulate(q, detailed=False):
    flux = np.asarray(q)*1e-5  # inconnues adimensionnees pour solveur
    state = np.array([cl, 0.])
    x0 = 0.
    xs, cs, phis, regions, jumps = [], [], [], [], []
    for i in range(5):
        if i:
            before = state.copy()
            newc, dphi = donnan(state[0], X[i-1], X[i])
            state = np.array([newc, state[1]+dphi])
            jumps.append((i, x0, before[0], newc, dphi))
        # RK4, 80 pas/couche, stable pour les parametres proposes
        n = 80
        h = width[i]/n
        for j in range(n+1):
            if detailed:
                xs.append(x0+j*h)
                cs.append(state[0])
                phis.append(state[1])
                regions.append(i+1)
            if j == n:
                break
            k1 = deriv(state, flux, i)
            k2 = deriv(state+0.5*h*k1, flux, i)
            k3 = deriv(state+0.5*h*k2, flux, i)
            k4 = deriv(state+h*k3, flux, i)
            state = state + (h/6)*(k1+2*k2+2*k3+k4)
            if not np.all(np.isfinite(state)):
                raise ValueError('Valeurs non finies')
        x0 += width[i]
    if detailed:
        return state, (np.array(xs), np.array(cs), np.array(phis),
                       np.array(regions), jumps)
    return state


def residual(q):
    try:
        end = simulate(q)
        return np.array([(end[0]-cr)/max(cr, 1.),
                         (end[1]-voltage)/0.0257])
    except (ValueError, OverflowError, FloatingPointError):
        return np.array([1e4, 1e4])

# Plusieurs points de depart pour reduire le risque d'echec du tir.
solutions = []
for guess in ([0., 0.], [-0.1, 0.1], [-1., 1.], [0.1, -0.1]):
    fit = least_squares(residual, guess, max_nfev=100,
                        xtol=1e-11, ftol=1e-11, gtol=1e-11)
    solutions.append(fit)
    if np.max(np.abs(fit.fun)) < 1e-7:
        break
fit = min(solutions, key=lambda z: np.linalg.norm(z.fun))
if np.max(np.abs(fit.fun)) > 1e-6:
    raise RuntimeError(f'Echec convergence: residus={fit.fun}, flux={fit.x*1e-5}')

end, (x, cna, phi, reg, jumps) = simulate(fit.x, detailed=True)
ccl = cna + X[reg-1]
neutral = cna-ccl+X[reg-1]
Nna, Ncl = fit.x*1e-5
current = F*(Nna-Ncl)
print('\n=== V10 : NERNST-PLANCK 1D ELECTRONEUTRE COUPLE ===')
print('Largeurs illustratives : independantes du maillage 2D Gmsh')
print(f'T={T:g} K ; tension={voltage:g} V ; NaCl bords={cl:g}/{cr:g} mol/m3')
print(f'Flux Na={Nna:+.9e} ; Cl={Ncl:+.9e} mol/(m2 s)')
print(f'Courant ionique i_x={current:+.9e} A/m2 (constant en 1D)')
print(f'Bord droit: cNa={end[0]:.9f} mol/m3 ; phi={end[1]:.9f} V')
for i, name in enumerate(names):
    mask = reg == i+1
    print(f'ID {i+1} {name:14s} : cNa=[{cna[mask].min():.3f},{cna[mask].max():.3f}] '
          f'cCl=[{ccl[mask].min():.3f},{ccl[mask].max():.3f}] mol/m3')
for i, pos, before, after, dp in jumps:
    print(f'Interface {i}/{i+1}, x={pos:.6g} m: '
          f'cNa {before:.4f} -> {after:.4f}, saut phi={dp*1e3:+.5f} mV')
print(f'Max |electroneutralite| = {np.max(np.abs(neutral)):.3e} mol/m3')
print(f'Residu conditions de bord = {fit.fun}')
out = Path(__file__).resolve().parent/'electrodialyse_couplee_v10.npz'
np.savez(out, x=x, region=reg, c_na=cna, c_cl=ccl, phi=phi,
         X=X, width=width, D_na=dna, D_cl=dcl,
         flux_na=Nna, flux_cl=Ncl, current=current, voltage=voltage)
print('Fichier :', out.name)
print('Pas de convection ni de variation selon y. Modele 1D, pas encore 2D.')
