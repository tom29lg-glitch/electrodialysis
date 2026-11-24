#!/usr/bin/env python3
"""Fonctionnement en recirculation (mode batch), comme Sugimoto 1978 et Xie et al. 2022.

Deux reservoirs (diluat V_d, concentrat V_c) passent en boucle dans un empilement de N paires
de membranes de surface A. Le passage dans la cellule (~1 s) est tres court devant l'evolution
des reservoirs (heures) : a chaque pas de temps on calcule la cellule 2D en regime permanent
avec les concentrations actuelles des reservoirs, puis on met a jour les reservoirs :

    d(V_d C_d)/dt = -N A J_d ,   d(V_c C_c)/dt = +N A J_c      (sel, et de meme pour le traceur)

J_d, J_c : flux par m2 de membrane, tires des bilans de matiere des canaux de la cellule.
Transport d'eau (option --hydratation a) : a molecules d'eau par ion transfere (Xie : a = 15).

Modes de fonctionnement :
  --mode courant : courant constant (Xie). Par defaut I = 0,6 x i_lim initial x A.
                   Le calcul s'arrete si le courant limite est atteint (diluat trop appauvri).
  --mode frac    : courant ajuste a chaque pas a --frac x i_lim(C_d) : reste sous la limite.
  --mode tension : tension par cellule constante --U (Sugimoto : 15-16 V pour 11 paires).

Exemples :
  python recirculation.py --rapide                       # conditions de Xie, courant constant
  python recirculation.py --rapide --mode frac --frac 0.6
  python recirculation.py --rapide --mode tension --U 0.4 --VR 10 --Vc 1e-3 --c0 17.1
"""
import argparse
import csv
import json
import os
import shutil
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
sys.path.insert(0, str(ROOT))
from courbe_iv import leveque, mesures, geometrie   # noqa: E402
from tracer import solve_tracer, SPECIES        # noqa: E402

F = 96485.33212
M_EAU = 18.015e-3                                # kg/mol
RHO_EAU = 997.                                   # kg/m3


def run_solver(env, out, log, **kw):
    e = dict(env, ED_OUTPUT_NPZ=str(out), **{k: str(v) for k, v in kw.items()})
    with log.open('w') as fh:
        subprocess.run([sys.executable, str(ROOT / 'solver.py')], cwd=ROOT, env=e, stdout=fh, stderr=subprocess.STDOUT)
    return out.exists() and bool(np.load(out)['converged'])


def cellule(env, work, Cd, Cc, prev, U_guess, i_target=None, U_fixed=None):
    """Solution de la cellule pour les entrees (Cd, Cc). Renvoie le chemin du npz ou None."""
    env = dict(env, ED_C_INLET=repr(Cd), ED_C_INLET_CONC=repr(Cc))
    log = work / 'dernier_calcul.log'
    out = work / 'courant.npz'
    if prev is not None:
        if U_fixed is not None:
            ok = run_solver(env, out, log, ED_VOLTAGE=U_fixed, ED_INIT_NPZ=prev)
        else:
            ok = run_solver(env, out, log, ED_CURRENT=i_target, ED_VOLTAGE=U_guess, ED_INIT_NPZ=prev, ED_NEWTON_STEPS=30)
        if ok:
            return out
    # depart a froid (premier pas, ou echec) : montee en tension par paliers
    p = work / 'rampe_0.npz'
    if not run_solver(env, p, log, ED_VOLTAGE=0.006):
        return None
    U, I = mesures(p)[:2]
    step, n = 0.02, 0
    goal_U = U_fixed if U_fixed is not None else np.inf
    while True:
        if U_fixed is not None and U >= goal_U - 1e-12:
            return p
        trial = min(U + step, goal_U)
        n += 1
        q = work / f'rampe_{n}.npz'
        if run_solver(env, q, log, ED_VOLTAGE=trial, ED_INIT_NPZ=p):
            Un, In = mesures(q)[:2]
            if U_fixed is None and In >= i_target:
                Ug = U + (i_target - I) * (Un - U) / (In - I)
                ok = run_solver(env, out, log, ED_CURRENT=i_target, ED_VOLTAGE=Ug, ED_INIT_NPZ=p, ED_NEWTON_STEPS=30)
                return out if ok else None
            p, U, I = q, Un, In
            step = min(0.05, step * 1.4)
        else:
            step /= 2
            if step < 1e-3:
                return None


def bilans_canaux(d, c_cells, Cd, Cc):
    """Flux par m2 de membrane retire du diluat (J_d) et recu par le concentrat (J_c)."""
    reg = d['region'] - 1
    w = d['velocity_x'] * d['dx']                # debit par unite de profondeur de chaque colonne
    L = float(d['y'][0] + d['y'][-1])            # hauteur de la cellule = surface de membrane par unite de profondeur
    dil = reg == 2
    con = (reg == 0) | (reg == 4)
    J_d = float(np.sum(w[dil] * (Cd - c_cells[-1, dil]))) / L
    J_c = float(np.sum(w[con] * (c_cells[-1, con] - Cc))) / L
    return J_d, J_c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=('courant', 'frac', 'tension'), default='courant')
    ap.add_argument('--I', type=float, help='courant de l empilement (A), mode courant')
    ap.add_argument('--frac', type=float, default=0.6, help='fraction de i_lim (modes courant et frac)')
    ap.add_argument('--U', type=float, default=0.4, help='tension par cellule (V), mode tension')
    ap.add_argument('--N', type=int, default=10, help='nombre de paires de membranes (Xie : 10)')
    ap.add_argument('--A', type=float, default=108.5e-4, help='surface utile d une membrane, m2 (Xie : 108,5 cm2)')
    ap.add_argument('--Vc', type=float, default=0.5e-3, help='volume initial du concentrat, m3 (Xie : 0,5 L)')
    ap.add_argument('--VR', type=float, default=10., help='rapport V_diluat / V_concentrat (Xie : 10 a 20)')
    ap.add_argument('--c0', type=float, default=34.2, help='sel initial, mol/m3 (34,2 = 2000 mg/L de NaCl)')
    ap.add_argument('--species', default='Cs+', help='radionucleide en traces (' + ', '.join(SPECIES) + ')')
    ap.add_argument('--cr0', type=float, default=3.76e-2, help='radionucleide initial, mol/m3 (Xie : Cs 3,76e-2)')
    ap.add_argument('--hydratation', type=float, default=0., help='molecules d eau transportees par ion (Xie : 15)')
    ap.add_argument('--pas', type=float, default=0.03, help='variation relative maximale de C_d par pas de temps')
    ap.add_argument('--tmax', type=float, default=6., help='duree maximale (h)')
    ap.add_argument('--dtmax', type=float, default=5., help='pas de temps maximal (min)')
    ap.add_argument('--saturation', type=float, default=1e-3,
                    help='arret si la decontamination varie de moins de cette valeur en 30 min')
    ap.add_argument('--objectif', type=float, default=0.99, help='arret quand la decontamination atteint cette valeur')
    ap.add_argument('--rendement', type=float, default=1.0,
                    help='fraction du courant de l alimentation qui traverse les cellules (courants de fuite de l empilement) ; mode courant')
    ap.add_argument('--au-dela', choices=('arret', 'plafond'), default='arret',
                    help='mode courant : arret au courant limite, ou plafond (courant utile au sel limite a --plafond x i_lim, '
                         'le reste dissocie l eau)')
    ap.add_argument('--plafond', type=float, default=0.85, help='fraction de i_lim(C_d) gardee pour le sel en mode plafond')
    ap.add_argument('--cmem', type=float, default=1000., help='charge fixe de la CEM (et de l AEM si --cmem-aem absent), mol/m3 d eau de membrane')
    ap.add_argument('--cmem-aem', type=float, default=None, help='charge fixe de l AEM, mol/m3')
    ap.add_argument('--Kcem', type=float, default=1.0, help='coefficient de partage du radionucleide dans la CEM (selectivite / Na+)')
    ap.add_argument('--Kaem', type=float, default=1.0, help='coefficient de partage du radionucleide dans l AEM')
    ap.add_argument('--DNa', type=float, default=1.334e-9)
    ap.add_argument('--DCl', type=float, default=2.032e-9)
    ap.add_argument('--vavg', type=float, default=0.005, help='parametre v_avg COMSOL : vitesse moyenne reelle = 2 x v_avg')
    ap.add_argument('--wch', type=float, default=1e-3, help='epaisseur des canaux (m) ; Xie : 0,95e-3 (espaceur)')
    ap.add_argument('--wm', type=float, default=0.25e-3, help='epaisseur des membranes (m) ; Xie : 0,10e-3')
    ap.add_argument('--L', type=float, default=0.01, help='longueur d ecoulement (m) ; Xie : ~0,90 (108,5 cm2 / 1,2 cm)')
    ap.add_argument('--ilim0', type=float, help='courant limite au debut (A/m2), par ex. mesure (Xie : 0,78 A / 108,5 cm2 = 71,9) ;'
                    ' defaut : Leveque. Ensuite i_lim(C_d) = ilim0 x C_d / C0')
    ap.add_argument('--rapide', action='store_true')
    a = ap.parse_args()

    z, Dr = SPECIES[a.species]
    nx, ny = ('9,6,15,6,9', 18) if a.rapide else ('20,20,100,20,20', 60)
    nxt = sum(map(int, nx.split(',')))
    folder = ROOT / 'resultats' / f'recirculation_{a.mode}_c{a.c0:g}_{nxt}x{ny}_{datetime.now():%Y-%m-%d_%Hh%Mm%Ss}'
    work = folder / 'travail'
    work.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ED_')}
    env.update(ED_NX=nx, ED_NY=str(ny), ED_X_CEM=str(-a.cmem), ED_X_AEM=str(a.cmem_aem or a.cmem),
               ED_GRADING='10,2,10,2,10', ED_VELOCITY_PROFILE='comsol', ED_D_NA=str(a.DNa),
               ED_D_CL=str(a.DCl), ED_NEWTON_STEPS='60', ED_NEWTON_TOL='4e-7', **geometrie(a))

    Vc, Vd = a.Vc, a.VR * a.Vc
    Cd = Cc = a.c0
    Rd = Rc = a.cr0
    nd0, nrd0 = Vd * Cd, Vd * Rd
    ntot0, nrtot0 = Vd * Cd + Vc * Cc, Vd * Rd + Vc * Rc
    ilim0 = a.ilim0 or leveque(a.c0, a.DNa, a.DCl, L=a.L, h=a.wch, V0=2 * a.vavg)[0]['CEM'][1]
    I_stack = a.I if a.I is not None else a.frac * ilim0 * a.A
    print(f'Empilement : N = {a.N}, A = {a.A*1e4:g} cm2, V_d = {Vd*1e3:g} L, V_c = {Vc*1e3:g} L ; '
          f'NaCl {a.c0:g} mol/m3, {a.species} {a.cr0:g} mol/m3 ; i_lim initial = {ilim0:.1f} A/m2', flush=True)
    if a.mode == 'courant':
        print(f'Mode courant constant : I = {I_stack:.4g} A, soit i = {a.rendement*I_stack/a.A:.2f} A/m2 dans les cellules '
              f'(rendement {a.rendement:g})', flush=True)

    t, E, prev, U = 0.0, 0.0, None, 0.1
    rows, raison = [], f'duree maximale ({a.tmax:g} h) atteinte'
    t0 = time.time()
    while True:
        ilim = ilim0 * Cd / a.c0                    # i_lim proportionnel a la concentration du diluat
        if a.mode == 'courant':
            i_t = a.rendement * I_stack / a.A
            if a.au_dela == 'plafond':
                # au-dela du courant limite : le sel ne passe plus qu'au plafond, le reste du courant
                # dissocie l'eau (H+/OH-, non modelise) ; la tension reelle monte, elle n'est pas calculee
                i_t = min(i_t, a.plafond * ilim)
        elif a.mode == 'frac':
            i_t = a.frac * ilim
        else:
            i_t = None
        if i_t is not None and i_t >= 0.97 * ilim:
            raison = f'courant limite atteint (i = {i_t:.1f} A/m2, i_lim(C_d) = {ilim:.1f} A/m2)'
            break
        sol = cellule(env, work, Cd, Cc, prev, U, i_target=i_t, U_fixed=(a.U if a.mode == 'tension' else None))
        if sol is None:
            raison = 'le calcul de la cellule ne converge plus (proche du courant limite ?)'
            break
        keep = work / 'precedent.npz'
        shutil.copy2(sol, keep)
        prev = keep
        d = np.load(prev)
        U, i, c_cem, c_aem = mesures(prev)
        if c_cem < 0.01 * Cd:
            raison = f'courant limite atteint dans le modele (sel en surface de la CEM < 1 % de C_d)'
            break
        salt = 0.5 * (d['c_na'] + d['c_cl'])
        Js_d, Js_c = bilans_canaux(d, salt, Cd, Cc)
        res, fields, _ = solve_tracer(prev, z, Dr, Rd, K_cem=a.Kcem, K_aem=a.Kaem, verbose=False, c_in_conc=Rc)
        Jr_d, Jr_c = bilans_canaux(d, fields['c'], Rd, Rc)
        DF = 1 - Rd / a.cr0                        # decontamination (concentration), comme Sugimoto et Xie
        DS = 1 - Cd / a.c0                         # dessalement
        rows.append(dict(t_h=t / 3600, V_d_L=Vd * 1e3, V_c_L=Vc * 1e3, C_d=Cd, C_c=Cc, R_d=Rd, R_c=Rc,
                         U_cellule_V=U, i_A_m2=i, i_lim_A_m2=ilim, I_A=i * a.A, c_min_CEM=c_cem,
                         dessalement=DS, decontamination=DF, energie_kWh_m3=E / 3.6e6 / (a.VR * a.Vc)))
        print(f'  t = {t/3600:6.3f} h : C_d = {Cd:7.3f}, C_c = {Cc:7.2f} mol/m3, {a.species} retire {DF*100:5.1f} %, '
              f'sel retire {DS*100:5.1f} %, U = {U:.3f} V, i = {i:.1f} A/m2', flush=True)
        if DF >= a.objectif:
            raison = f'objectif de decontamination atteint ({a.objectif*100:g} %)'
            break
        if t >= a.tmax * 3600 - 1e-6:
            break
        # saturation : la decontamination ne progresse plus (migration ~ retrodiffusion)
        old = [r for r in rows if r['t_h'] <= t / 3600 - 0.5]
        if old and abs(DF - old[-1]['decontamination']) < a.saturation:
            raison = (f'saturation : la decontamination varie de moins de {a.saturation*100:g} % en 30 min '
                      '(migration compensee par la retrodiffusion depuis le concentrat)')
            break
        # pas de temps : variation relative limitee du sel et du traceur dans le diluat
        rate_s = a.N * a.A * Js_d
        rate_r = a.N * a.A * Jr_d
        if rate_s <= 0 or rate_r <= 0:
            raison = 'saturation : le flux net vers le concentrat est nul ou negatif (retrodiffusion)'
            break
        dt = a.pas * min(Vd * Cd / rate_s, Vd * Rd / rate_r)
        dt = min(dt, a.dtmax * 60, a.tmax * 3600 - t)
        # mise a jour (Euler explicite) : matiere puis volumes (eau d'hydratation, 2 ions par mole de sel)
        nd, nc = Vd * Cd - rate_s * dt, Vc * Cc + a.N * a.A * Js_c * dt
        nrd, nrc = Vd * Rd - rate_r * dt, Vc * Rc + a.N * a.A * Jr_c * dt
        dV = a.hydratation * 2 * rate_s * dt * M_EAU / RHO_EAU
        Vd, Vc = Vd - dV, Vc + dV
        Cd, Cc, Rd, Rc = nd / Vd, nc / Vc, nrd / Vd, nrc / Vc
        E += a.N * U * i * a.A * dt
        t += dt

    # sorties
    keys = list(rows[0])
    with (folder / 'serie_temporelle.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    last = rows[-1]
    bilan_sel = (last['V_d_L'] * last['C_d'] + last['V_c_L'] * last['C_c']) * 1e-3 / ntot0 - 1
    bilan_r = (last['V_d_L'] * last['R_d'] + last['V_c_L'] * last['R_c']) * 1e-3 / nrtot0 - 1
    res = dict(arret=raison, duree_h=last['t_h'], decontamination=last['decontamination'], dessalement=last['dessalement'],
               facteur_concentration_concentrat=last['R_c'] / a.cr0, tension_finale_V=last['U_cellule_V'],
               energie_kWh_m3=last['energie_kWh_m3'], ecart_bilan_sel=bilan_sel, ecart_bilan_traceur=bilan_r,
               pas_de_temps=len(rows), parametres=vars(a), duree_calcul_s=round(time.time() - t0))
    (folder / 'resume.json').write_text(json.dumps(res, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if prev is not None:
        shutil.copy2(prev, folder / 'solution_finale.npz')

    T = np.array([r['t_h'] for r in rows])
    col = lambda k: np.array([r[k] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(T, col('C_d') / a.c0, 'C0--', label='sel, diluat')
    ax[0, 0].plot(T, col('R_d') / a.cr0, 'C0-', label=f'{a.species}, diluat')
    ax[0, 0].set(xlabel='Temps (h)', ylabel='C / C initial', title='Réservoir du diluat')
    ax[0, 1].plot(T, col('C_c') / a.c0, 'C3--', label='sel, concentrat')
    ax[0, 1].plot(T, col('R_c') / a.cr0, 'C3-', label=f'{a.species}, concentrat')
    ax[0, 1].set(xlabel='Temps (h)', ylabel='C / C initial', title='Réservoir du concentrat')
    ax[1, 0].plot(T, 100 * col('decontamination'), 'C1-', label=f'décontamination {a.species}')
    ax[1, 0].plot(T, 100 * col('dessalement'), 'C1--', label='dessalement')
    ax[1, 0].set(xlabel='Temps (h)', ylabel='%', title='Élimination dans le diluat')
    ax[1, 1].plot(T, col('i_A_m2'), 'C2-', label='i (A/m²)')
    ax[1, 1].plot(T, col('i_lim_A_m2'), 'C2:', label='i_lim(C_d), Lévêque')
    ax2 = ax[1, 1].twinx()
    ax2.plot(T, col('U_cellule_V'), 'k-', lw=1, label='U par cellule (V)')
    ax2.set_ylabel('Tension par cellule (V)')
    ax[1, 1].set(xlabel='Temps (h)', ylabel='Densité de courant (A/m²)', title='Fonctionnement électrique')
    for axx in ax.ravel():
        axx.grid(alpha=.25)
        axx.legend(fontsize=8, loc='best')
    ax2.legend(fontsize=8, loc='lower right')
    fig.suptitle(f'Recirculation, mode {a.mode} : {raison}', fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / 'recirculation.png', dpi=160)
    plt.close(fig)
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({k: v for k, v in res.items() if k != 'parametres'}, indent=2, ensure_ascii=False))
    print(f'Resultats : {folder}')


if __name__ == '__main__':
    main()
