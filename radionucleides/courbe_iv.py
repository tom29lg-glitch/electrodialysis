#!/usr/bin/env python3
"""Courbe courant-tension d'une cellule et courant limite, en conditions realistes.

    python courbe_iv.py                      # NaCl 17,1 mol/m3 (1000 ppm), D physiques
    python courbe_iv.py --c0 34.2 --vmax 2   # NaCl 34,2 mol/m3 (2000 ppm)
    python courbe_iv.py --rapide             # maillage 45 x 18

Pour chaque tension (montee adaptative depuis 6 mV), on enregistre :
  - la densite de courant moyenne i (A/m2), identique a travers toute la cellule ;
  - la concentration minimale de sel a la surface de la CEM et de l'AEM, cote diluat.
Le courant limite theorique (Leveque + Peers, Urtenov et al. 2013, eq. 1-2) est trace :
  i_lim = D c0 F / (delta (1 - t)),  delta = 1,02 (L D h / V0)^(1/3)
avec t le nombre de transport du contre-ion en solution et V0 la vitesse moyenne reelle.
Le modele electroneutre n'est valable qu'en dessous de i_lim (pas de dissociation de l'eau,
pas d'electroconvection) : le domaine utile est la partie montante de la courbe.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
F = 96485.33212
R = 8.31446261815324


def geometrie(a):
    """Variables d'environnement de geometrie pour solver.py (demi-canaux exterieurs, comme COMSOL)."""
    w = [a.wch / 2, a.wm, a.wch, a.wm, a.wch / 2]
    return dict(ED_WIDTHS=','.join(repr(x) for x in w), ED_HEIGHT=repr(a.L), ED_U_AVG=repr(a.vavg))


def leveque(c0, DNa, DCl, L=0.01, h=1e-3, V0=0.01):
    """Courant limite de Leveque a la sortie et moyen sur la longueur, pour la CEM et l'AEM."""
    D = 2 * DNa * DCl / (DNa + DCl)                 # coefficient de diffusion du sel
    delta_L = 1.02 * (L * D * h / V0) ** (1 / 3)    # epaisseur de couche limite en sortie
    out = {}
    for name, t in (('CEM', DNa / (DNa + DCl)), ('AEM', DCl / (DNa + DCl))):
        i_out = D * c0 * F / (delta_L * (1 - t))
        out[name] = (i_out, 1.5 * i_out)            # moyenne de 1/delta(y) ~ y^(-1/3) sur L : facteur 3/2
    return out, delta_L, D


def mesures(path):
    """Courant moyen et concentrations minimales en surface (cote diluat) d'une solution."""
    d = np.load(path)
    ifc = d['interface_indices'].astype(int)
    i_mean = -float(np.mean(d['current_x'][:, ifc[1]]))
    salt_L = 0.5 * (d['trace_na_left'] + d['trace_cl_left'])     # cote gauche de chaque interface
    salt_R = 0.5 * (d['trace_na_right'] + d['trace_cl_right'])   # cote droit
    return float(d['voltage']), i_mean, float(salt_R[:, 1].min()), float(salt_L[:, 2].min())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--c0', type=float, default=17.1, help='NaCl en entree, mol/m3 (17,1 = 1000 ppm)')
    ap.add_argument('--vmax', type=float, default=1.5, help='tension maximale par cellule (V)')
    ap.add_argument('--cmem', type=float, default=1000., help='charge fixe des membranes, mol/m3')
    ap.add_argument('--DNa', type=float, default=1.334e-9)
    ap.add_argument('--DCl', type=float, default=2.032e-9)
    ap.add_argument('--vavg', type=float, default=0.005, help='parametre v_avg COMSOL : vitesse moyenne reelle = 2 x v_avg')
    ap.add_argument('--wch', type=float, default=1e-3, help='epaisseur des canaux (m) ; Xie : 0,95e-3 (espaceur)')
    ap.add_argument('--wm', type=float, default=0.25e-3, help='epaisseur des membranes (m) ; Xie : 0,10e-3')
    ap.add_argument('--L', type=float, default=0.01, help='longueur d ecoulement (m) ; Xie : ~0,90 (108,5 cm2 / 1,2 cm)')
    ap.add_argument('--rapide', action='store_true')
    ap.add_argument('--dossier', type=Path, help='refaire seulement les figures a partir d un calcul deja fait (dossier iv_...)')
    a = ap.parse_args()
    if a.dossier:
        folder = a.dossier
        rows = sorted(mesures(f) for f in (folder / 'etapes').glob('U_*.npz') if bool(np.load(f)['converged']))
        nxt, ny = [int(v) for v in folder.name.split('_')[2].split('x')]
        t0 = time.time()
        return finish(np.array(rows), folder, a, nxt, ny, t0)
    nx, ny = ('9,6,15,6,9', 18) if a.rapide else ('20,20,100,20,20', 60)
    nxt = sum(map(int, nx.split(',')))
    folder = ROOT / 'resultats' / f'iv_c{a.c0:g}_{nxt}x{ny}_{datetime.now():%Y-%m-%d_%Hh%Mm%Ss}'
    steps = folder / 'etapes'
    steps.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ED_')}
    env.update(ED_NX=nx, ED_NY=str(ny), ED_X_CEM=str(-a.cmem), ED_X_AEM=str(a.cmem),
               ED_GRADING='10,2,10,2,10', ED_VELOCITY_PROFILE='comsol', ED_D_NA=str(a.DNa),
               ED_D_CL=str(a.DCl), ED_C_INLET=str(a.c0), **geometrie(a),
               ED_NEWTON_STEPS='60', ED_NEWTON_TOL='4e-7')
    rows, V, step, prev, n, t0 = [], 0.0, 0.02, '', 0, time.time()
    while V < a.vmax - 1e-12:
        trial = 0.006 if V == 0 else min(a.vmax, V + step)
        n += 1
        out = steps / f'U_{trial:.4f}.npz'
        e = dict(env, ED_VOLTAGE=str(trial), ED_OUTPUT_NPZ=str(out), ED_INIT_NPZ=str(prev))
        with (steps / 'dernier_calcul.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'solver.py')], cwd=ROOT, env=e, stdout=log, stderr=subprocess.STDOUT)
        ok = out.exists() and bool(np.load(out)['converged'])
        if ok:
            _, i_mean, c_cem, c_aem = mesures(out)
            rows.append((trial, i_mean, c_cem, c_aem))
            V, prev = trial, out
            step = min(0.05, step * 1.4)
            print(f'  U = {V:.3f} V : i = {i_mean:8.2f} A/m2, c_min surface CEM = {c_cem:.3g}, AEM = {c_aem:.3g} mol/m3', flush=True)
        else:
            step /= 2
            print(f'  echec a {trial:.4f} V, pas reduit a {step:.4g}', flush=True)
            if step < 1e-3:
                print('  arret : la montee en tension ne converge plus (voir etapes/dernier_calcul.log)', flush=True)
                break
    finish(np.array(rows), folder, a, nxt, ny, t0)


def finish(rows, folder, a, nxt, ny, t0):
    lev, delta_L, Ds = leveque(a.c0, a.DNa, a.DCl, L=a.L, h=a.wch, V0=2 * a.vavg)
    with (folder / 'courbe_iv.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['tension_V', 'courant_A_m2', 'c_min_surface_CEM_molm3', 'c_min_surface_AEM_molm3'])
        w.writerows(rows.tolist())
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(rows[:, 0], rows[:, 1], 'o-', ms=3, color='C0', label='Python (2D)')
    for name, col in (('CEM', 'C3'), ('AEM', 'C2')):
        ax[0].axhline(lev[name][1], color=col, ls='--', label=f'Lévêque {name}, moyenne sur L : {lev[name][1]:.0f} A/m²')
        ax[0].axhline(lev[name][0], color=col, ls=':', label=f'Lévêque {name}, en sortie : {lev[name][0]:.0f} A/m²')
    ax[0].set(xlabel='Tension par cellule (V)', ylabel='Densité de courant moyenne (A/m²)',
              title=f'Courbe courant-tension, NaCl {a.c0:g} mol/m³, D physiques')
    ax[0].grid(alpha=.25); ax[0].legend(fontsize=8)
    ax[1].semilogy(rows[:, 0], rows[:, 2] / a.c0, 'o-', ms=3, color='C3', label='surface CEM, côté diluat')
    ax[1].semilogy(rows[:, 0], rows[:, 3] / a.c0, 'o-', ms=3, color='C2', label='surface AEM, côté diluat')
    ax[1].axhline(0.05, color='k', lw=.8, ls='--', label='5 % de c₀')
    ax[1].set(xlabel='Tension par cellule (V)', ylabel='c_min / c₀ à la surface (sortie du canal)',
              title='Appauvrissement aux membranes')
    ax[1].grid(alpha=.25, which='both'); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(folder / 'courbe_iv.png', dpi=170); plt.close(fig)
    lines = [f'NaCl c0 = {a.c0:g} mol/m3, D_Na = {a.DNa:g}, D_Cl = {a.DCl:g} m2/s, cMem = {a.cmem:g} mol/m3, maillage {nxt} x {ny}',
             f'Leveque : D_sel = {Ds:.4g} m2/s, delta(sortie) = {delta_L*1e6:.1f} um, V0 = {2*a.vavg*1e3:g} mm/s',
             *(f'  i_lim {k} : {v[0]:.1f} A/m2 en sortie, {v[1]:.1f} A/m2 en moyenne sur L' for k, v in lev.items()),
             f'Dernier point converge : U = {rows[-1,0]:.3f} V, i = {rows[-1,1]:.2f} A/m2 ({rows[-1,1]/lev["CEM"][1]*100:.0f} % de i_lim CEM moyen)',
             f'Duree : {time.time()-t0:.0f} s']
    (folder / 'resume.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines)); print(f'Resultats : {folder}')


if __name__ == '__main__':
    main()
