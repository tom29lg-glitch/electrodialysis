#!/usr/bin/env python3
"""Etude d'optimisation en recirculation, autour du cas valide sur Xie et al. 2022.

    python optimisation.py --balayage frac --valeurs 0.3,0.4,0.5,0.6,0.7,0.8 --modes courant,frac
    python optimisation.py --balayage v --valeurs 0.10,0.14,0.19,0.23,0.30
    python optimisation.py --balayage frac --figures-seulement

Un seul parametre varie a la fois, les autres restent au cas de base (essai A2 de Xie) :
  frac  fraction du courant limite (mode courant : I = frac x 0,78 A a l'alimentation ;
        mode frac : courant des cellules = frac x i_lim du moment, qui baisse avec le diluat)
  v     vitesse de circulation (m/s)        VR    rapport V_diluat / V_concentrat
  wch   epaisseur des canaux (m)            cmem  charge fixe des membranes (mol/m3)

Parametres de la validation : rendement de courant 0,82, vitesse effective = 2,23 x 0,82^3 x v,
membranes 0,30 mm, 15 H2O par ion. Le modele s'arrete au courant limite (pas de surlimite).
Energie : par m3 de diluat initial, ramenee au courant de l'alimentation (divisee par le rendement),
sans les surtensions d'electrodes.

Les calculs deja faits sont conserves : on peut relancer la meme commande jusqu'a ce que tout soit calcule
(--budget limite la duree d'un appel).
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
I_LIM_MESURE = 0.78          # A, Xie
BASE = dict(frac=0.6, v=0.14, VR=10., wch=0.95e-3, cmem=1000.)
UNITES = dict(frac='fraction du courant limite', v='vitesse (m/s)', VR='rapport de volumes V_d / V_c',
              wch='epaisseur des canaux (mm)', cmem='charge des membranes (mol/m3)')
OBJECTIFS = (0.5, 0.8, 0.9, 0.95)


def nom(mode, param, val):
    return f'{mode}_{param}_{val:g}'


def lancer(mode, p, dossier, a):
    vavg = p['v'] / 2 * 2.23 * a.rendement ** 3
    args = [sys.executable, str(ROOT / 'recirculation.py'), '--mode', mode, '--N', '10', '--A', '108.5e-4',
            '--Vc', '0.5e-3', '--VR', str(p['VR']), '--c0', '31.2', '--cr0', '3.76e-2', '--hydratation', '15',
            '--wch', str(p['wch']), '--wm', '0.3e-3', '--L', '0.9', '--vavg', str(vavg), '--cmem', str(p['cmem']), '--cmem-aem', str(p['cmem'] * a.ratio_aem),
            '--rendement', str(a.rendement), '--pas', str(a.pas), '--dtmax', str(a.dtmax), '--tmax', str(a.tmax),
            '--objectif', str(a.objectif), '--rapide']
    args += ['--I', str(p['frac'] * I_LIM_MESURE)] if mode == 'courant' else ['--frac', str(p['frac'])]
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith('Resultats : ')]
    if not line:
        print(f'  echec\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}')
        return
    Path(line[-1].split('Resultats : ', 1)[1]).rename(dossier)


def metriques(run, a):
    rows = list(csv.DictReader((run / 'serie_temporelle.csv').open()))
    f = lambda k: np.array([float(r[k]) for r in rows])
    t, DF, E = f('t_h'), f('decontamination'), f('energie_kWh_m3') / a.rendement
    res = json.loads((run / 'resume.json').read_text())
    out = dict(arret=res['arret'], duree_h=t[-1], Cs_retire=DF[-1], dessalement=f('dessalement')[-1],
               energie_kWh_m3=E[-1], facteur_concentrat_Cs=f('R_c')[-1] / f('R_c')[0],
               tension_moy_V=float(np.mean(f('U_cellule_V'))))
    for o in OBJECTIFS:
        ok = DF[-1] >= o
        out[f't{int(o*100)}_h'] = float(np.interp(o, DF, t)) if ok else np.nan
        out[f'E{int(o*100)}_kWh_m3'] = float(np.interp(o, DF, E)) if ok else np.nan
    return out, (t, DF)


def figures(dossier, a, valeurs, modes):
    table, courbes = [], {}
    for m in modes:
        for v in valeurs:
            run = dossier / nom(m, a.balayage, v)
            if (run / 'resume.json').exists():
                out, c = metriques(run, a)
                table.append(dict(mode=m, **{a.balayage: v}, **out))
                courbes[(m, v)] = c
    if not table:
        return
    with (dossier / 'synthese.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    ech = 1e3 if a.balayage == 'wch' else 1
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    cmap = plt.get_cmap('viridis')
    for k, m in enumerate(modes):
        for j, v in enumerate(valeurs):
            if (m, v) in courbes:
                t, DF = courbes[(m, v)]
                ax[0, k].plot(t, DF * 100, color=cmap(j / max(len(valeurs) - 1, 1)), lw=2, label=f'{v * ech:g}')
        ax[0, k].set(title=f'Mode {m}' + (' (I constant)' if m == 'courant' else ' (I suit i_lim)'),
                     xlabel='Temps (h)', ylabel='Cs retire (%)', ylim=(0, 100))
        ax[0, k].legend(title=UNITES[a.balayage], fontsize=9)
        ax[0, k].grid(alpha=.3)
    tw = ax[1, 0].twinx()
    for m, mk in zip(modes, 'os'):
        rows = [r for r in table if r['mode'] == m]
        x = np.array([r[a.balayage] for r in rows]) * ech
        ax[1, 0].plot(x, [r['Cs_retire'] * 100 for r in rows], mk + '-', lw=2, label=f'{m} : Cs retire a l arret')
        ax[1, 1].plot(x, [r['E50_kWh_m3'] for r in rows], mk + '-', lw=2, label=f'{m} : energie pour 50 %')
        if m == 'frac':
            tw.plot(x, [r['t80_h'] for r in rows], mk + '--', color='C2', lw=1.5, label='frac : temps pour 80 %')
            ax[1, 1].plot(x, [r['E80_kWh_m3'] for r in rows], mk + '--', lw=1.5, label='frac : energie pour 80 %')
    tw.set_ylabel('Temps pour retirer 80 % du Cs (h)', color='C2')
    tw.legend(fontsize=9, loc='lower right')
    ax[1, 0].set(xlabel=UNITES[a.balayage], ylabel='Cs retire a l arret (%)', ylim=(0, 100))
    ax[1, 1].set(xlabel=UNITES[a.balayage], ylabel='kWh / m3 de diluat')
    for x in ax[1]:
        x.grid(alpha=.3)
        x.legend(fontsize=9, loc='upper left')
    fig.suptitle(f'Optimisation : balayage de {a.balayage} (cas de base : essai A2 de Xie, rendement {a.rendement:g})')
    fig.tight_layout()
    fig.savefig(dossier / 'synthese.png', dpi=140)
    plt.close(fig)
    lignes = [f'{"mode":8s} {a.balayage:>7s} {"Cs fin":>7s} {"duree":>7s} {"t50":>6s} {"t80":>6s} {"t90":>6s} '
              f'{"E50":>6s} {"E80":>6s} {"Efin":>6s} {"xCs conc":>8s}  arret']
    for r in table:
        g = lambda k, f='6.2f': ('   -  ' if np.isnan(r[k]) else format(r[k], f))
        lignes.append(f'{r["mode"]:8s} {r[a.balayage] * ech:7.3g} {r["Cs_retire"]*100:6.1f}% {r["duree_h"]:6.2f}h '
                      f'{g("t50_h")} {g("t80_h")} {g("t90_h")} {g("E50_kWh_m3")} {g("E80_kWh_m3")} '
                      f'{r["energie_kWh_m3"]:6.2f} {r["facteur_concentrat_Cs"]:8.1f}  {r["arret"][:45]}')
    lignes.append('temps en h, energies en kWh par m3 de diluat initial (courant de l alimentation, sans electrodes)')
    (dossier / 'synthese.txt').write_text('\n'.join(lignes) + '\n', encoding='utf-8')
    print('\n'.join(lignes))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--balayage', choices=list(BASE), default='frac')
    ap.add_argument('--valeurs', default='0.3,0.4,0.5,0.6,0.7,0.8')
    ap.add_argument('--modes', default='courant,frac')
    ap.add_argument('--rendement', type=float, default=0.82)
    ap.add_argument('--ratio-aem', type=float, default=1.0, help='charge AEM / charge CEM (capacites de Xie : 2,70 / 1,85 = 1,46)')
    ap.add_argument('--base', default='', help='modifie le cas de base, ex. cmem=6200,VR=10')
    ap.add_argument('--suffixe', default='', help='ajoute au nom du dossier de resultats')
    ap.add_argument('--objectif', type=float, default=0.95)
    ap.add_argument('--tmax', type=float, default=4.)
    ap.add_argument('--pas', type=float, default=0.08)
    ap.add_argument('--dtmax', type=float, default=10.)
    ap.add_argument('--budget', type=float, default=1e9, help='ne lance plus de calcul apres ce temps (s)')
    ap.add_argument('--figures-seulement', action='store_true')
    a = ap.parse_args()
    for kv in filter(None, a.base.split(',')):
        k, v = kv.split('=')
        BASE[k] = float(v)
    valeurs = [float(x) for x in a.valeurs.split(',')]
    modes = a.modes.split(',')
    dossier = ROOT / 'resultats' / f'optimisation_{a.balayage}{a.suffixe}'
    dossier.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not a.figures_seulement:
        for m in modes:
            for v in valeurs:
                run = dossier / nom(m, a.balayage, v)
                if (run / 'resume.json').exists():
                    continue
                if time.time() - t0 > a.budget:
                    print('budget atteint : relancer la meme commande pour continuer')
                    break
                print(f'{m}, {a.balayage} = {v:g} ...', flush=True)
                lancer(m, dict(BASE, **{a.balayage: v}), run, a)
    figures(dossier, a, valeurs, modes)
    print(f'Resultats : {dossier}')


if __name__ == '__main__':
    main()
