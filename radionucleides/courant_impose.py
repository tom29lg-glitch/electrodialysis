#!/usr/bin/env python3
"""Calcul a courant impose (comme Xie et al. 2022), en conditions realistes.

    python courant_impose.py --frac 0.6                 # 0,6 x i_lim, NaCl 17,1 mol/m3
    python courant_impose.py --i 30 --c0 34.2           # 30 A/m2, NaCl 34,2 mol/m3
    python courant_impose.py --frac 0.6 --tracer Cs+    # + radionucleide en traces
    python courant_impose.py --frac 0.6 --rapide        # maillage 45 x 18

i_lim : courant limite moyen de Leveque cote CEM (voir courbe_iv.py), sauf si --ilim est donne
(par exemple le plateau lu sur la courbe courant-tension calculee).
Methode : montee en tension par paliers jusqu'a depasser le courant cible, puis un calcul a
courant impose (tension inconnue, ED_CURRENT dans solver.py) depuis la solution juste en dessous.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from courbe_iv import leveque, mesures, geometrie


def solve(env, out, log, **kw):
    e = dict(env, ED_OUTPUT_NPZ=str(out), **{k: str(v) for k, v in kw.items()})
    with log.open('w') as fh:
        subprocess.run([sys.executable, str(ROOT / 'solver.py')], cwd=ROOT, env=e, stdout=fh, stderr=subprocess.STDOUT)
    return out.exists() and bool(np.load(out)['converged'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--i', type=float, help='densite de courant moyenne cible (A/m2)')
    g.add_argument('--frac', type=float, help='fraction du courant limite (0,6 = optimum de Xie)')
    ap.add_argument('--ilim', type=float, help='courant limite a utiliser avec --frac (defaut : Leveque CEM moyen)')
    ap.add_argument('--c0', type=float, default=17.1)
    ap.add_argument('--cmem', type=float, default=1000.)
    ap.add_argument('--DNa', type=float, default=1.334e-9)
    ap.add_argument('--DCl', type=float, default=2.032e-9)
    ap.add_argument('--vavg', type=float, default=0.005, help='parametre v_avg COMSOL : vitesse moyenne reelle = 2 x v_avg')
    ap.add_argument('--wch', type=float, default=1e-3, help='epaisseur des canaux (m) ; Xie : 0,95e-3 (espaceur)')
    ap.add_argument('--wm', type=float, default=0.25e-3, help='epaisseur des membranes (m) ; Xie : 0,10e-3')
    ap.add_argument('--L', type=float, default=0.01, help='longueur d ecoulement (m) ; Xie : ~0,90 (108,5 cm2 / 1,2 cm)')
    ap.add_argument('--tracer', help='espece en traces a calculer ensuite (Cs+, Sr2+, Co2+...)')
    ap.add_argument('--rapide', action='store_true')
    a = ap.parse_args()
    lev, _, _ = leveque(a.c0, a.DNa, a.DCl, L=a.L, h=a.wch, V0=2 * a.vavg)
    ilim = a.ilim or lev['CEM'][1]
    target = a.i if a.i is not None else a.frac * ilim
    nx, ny = ('9,6,15,6,9', 18) if a.rapide else ('20,20,100,20,20', 60)
    nxt = sum(map(int, nx.split(',')))
    folder = ROOT / 'resultats' / f'courant_{target:.3g}Am2_c{a.c0:g}_{nxt}x{ny}_{datetime.now():%Y-%m-%d_%Hh%Mm%Ss}'
    steps = folder / 'etapes'
    steps.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ED_')}
    env.update(ED_NX=nx, ED_NY=str(ny), ED_X_CEM=str(-a.cmem), ED_X_AEM=str(a.cmem),
               ED_GRADING='10,2,10,2,10', ED_VELOCITY_PROFILE='comsol', ED_D_NA=str(a.DNa),
               ED_D_CL=str(a.DCl), ED_C_INLET=str(a.c0), **geometrie(a),
               ED_NEWTON_STEPS='60', ED_NEWTON_TOL='4e-7')
    print(f'Cible : {target:.4g} A/m2 (i_lim = {ilim:.4g} A/m2, NaCl {a.c0:g} mol/m3, maillage {nxt} x {ny})', flush=True)
    # 1) montee en tension (robuste) jusqu'a depasser le courant cible
    prev = steps / 'depart.npz'
    if not solve(env, prev, steps / 'depart.log', ED_VOLTAGE=0.006):
        raise SystemExit('Echec de la solution de depart')
    U, I = mesures(prev)[:2]
    step, n = 0.02, 0
    while I < target:
        trial = U + step
        n += 1
        out = steps / f'U_{n:03d}.npz'
        if solve(env, out, steps / 'dernier_calcul.log', ED_VOLTAGE=trial, ED_INIT_NPZ=prev):
            U_new, I_new = mesures(out)[:2]
            if I_new >= target:
                break
            prev, U, I = out, U_new, I_new
            step = min(0.05, step * 1.4)
            print(f'  U = {U:.4f} V -> i = {I:7.3f} A/m2', flush=True)
        else:
            step /= 2
            if step < 1e-3:
                raise SystemExit(f'Arret : courant cible non atteint (au-dela du courant limite ?). Voir {steps}')
    # 2) un calcul a courant impose, depuis la solution juste en dessous, tension interpolee
    U_guess = U + (target - I) * (U_new - U) / (I_new - I)
    out = steps / 'courant_impose.npz'
    if not solve(env, out, steps / 'courant_impose.log', ED_CURRENT=target, ED_VOLTAGE=U_guess,
                 ED_INIT_NPZ=prev, ED_NEWTON_STEPS=30):
        raise SystemExit(f'Echec du calcul a courant impose, voir {steps}/courant_impose.log')
    prev = out
    print(f'  courant impose {target:.4g} A/m2 -> U = {float(np.load(out)["voltage"]):.4f} V', flush=True)
    final = folder / 'solution.npz'
    shutil.copy2(prev, final)
    Uf, If, c_cem, c_aem = mesures(final)
    d = np.load(final)
    reg = d['region'] - 1
    w = d['velocity_x'] * d['dx']
    m = reg == 2
    salt_out = float(np.sum((0.5 * (d['c_na'][-1] + d['c_cl'][-1]))[m] * w[m]) / np.sum(w[m]))
    res = dict(courant_cible_A_m2=target, courant_calcule_A_m2=If, tension_V=Uf, i_lim_A_m2=ilim,
               fraction_i_lim=If / ilim, c0_molm3=a.c0, sel_sortie_diluat_molm3=salt_out,
               elimination_sel_un_passage=1 - salt_out / a.c0,
               c_min_surface_CEM_molm3=c_cem, c_min_surface_AEM_molm3=c_aem,
               puissance_W_m2=Uf * If, maillage=f'{nxt}x{ny}')
    (folder / 'resume.json').write_text(json.dumps(res, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(res, indent=2, ensure_ascii=False))
    subprocess.run([sys.executable, str(ROOT / 'visualization.py'), '--input', str(final), '--output', str(folder / 'figures')],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    if a.tracer:
        subprocess.run([sys.executable, str(ROOT / 'tracer.py'), str(final), '--species', a.tracer], cwd=ROOT, check=True)
    print(f'Resultats : {folder}')


if __name__ == '__main__':
    main()
