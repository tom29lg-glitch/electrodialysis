#!/usr/bin/env python3
"""Comparaison de plusieurs radionucleides en traces, dans les conditions de reference (essai A2 de Xie).

    python especes.py                         # Cs+, Sr2+, Co2+, Mn2+ ; modes courant (Xie) et frac (recommande)
    python especes.py --figures-seulement

Hypotheses de reference : membranes CEM 6200 / AEM 9000 mol/m3, rendement 0,82, vitesse effective,
NaCl equivalent 31,2 mol/m3, VR 1:10. Les especes sont en traces : elles ne modifient pas le champ du sel,
donc toutes sont calculees dans la meme cellule. Concentrations initiales de Xie (Co, Mn, Cs a 5 mg/L).
Pas de complexation, d'hydrolyse ni de selectivite chimique propre (K = 1) : seul l'effet Donnan et la mobilite.
"""
import argparse, csv, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
C_INIT = {'Cs+': 3.76e-2, 'Sr2+': 5.7e-2, 'Co2+': 8.49e-2, 'Mn2+': 9.11e-2}   # mol/m3 (Xie ; Sr : 5 mg/L)
COUL = {'Cs+': '#1f5fa8', 'Sr2+': '#c0392b', 'Co2+': '#27ae60', 'Mn2+': '#8e44ad'}


def lancer(sp, mode, run, a):
    args = [sys.executable, str(ROOT / 'recirculation.py'), '--mode', mode, '--species', sp, '--cr0', str(C_INIT[sp]),
            '--N', '10', '--A', '108.5e-4', '--Vc', '0.5e-3', '--VR', '10', '--c0', '31.2', '--hydratation', '15',
            '--wch', '0.95e-3', '--wm', '0.3e-3', '--L', '0.9', '--vavg', str(0.07 * 2.23 * 0.82 ** 3),
            '--cmem', '6200', '--cmem-aem', '9000', '--rendement', '0.82', '--pas', '0.08', '--dtmax', '10',
            '--tmax', str(a.tmax), '--objectif', '0.999', '--rapide']
    args += ['--I', '0.468'] if mode == 'courant' else ['--frac', '0.6']
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith('Resultats : ')]
    if not line:
        print(f'  echec {sp} {mode}\n{r.stdout[-1200:]}\n{r.stderr[-1200:]}')
        return
    Path(line[-1].split('Resultats : ', 1)[1]).rename(run)


def lire(run):
    rows = list(csv.DictReader((run / 'serie_temporelle.csv').open()))
    f = lambda k: np.array([float(r[k]) for r in rows])
    return f('t_h'), f('decontamination'), f('dessalement'), f('R_c') / f('R_c')[0]


def figures(dossier, especes):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    lignes = [f'{"espece":6s} {"mode":8s} {"retire":>7s} {"t50 h":>6s} {"t90 h":>6s} {"x conc":>7s} {"S = ln(R)/ln(C)":>16s}  arret']
    for sp in especes:
        for mode, ls in (('courant', '--'), ('frac', '-')):
            run = dossier / f'{mode}_{sp}'
            if not (run / 'resume.json').exists():
                continue
            t, DF, DS, xc = lire(run)
            k = 1 if mode == 'courant' else 0
            ax[k].plot(t, DF * 100, ls, color=COUL[sp], lw=2.5, label=sp)
            if mode == 'courant':
                ax[2].plot(DS * 100, DF * 100, '-', color=COUL[sp], lw=2.5, label=sp)
            S = np.log(1 - DF[-1]) / np.log(1 - DS[-1])
            it = lambda o: f'{np.interp(o, DF, t):6.2f}' if DF[-1] >= o else '   -  '
            res = json.loads((run / 'resume.json').read_text())
            lignes.append(f'{sp:6s} {mode:8s} {DF[-1]*100:6.1f}% {it(.5)} {it(.9)} {xc[-1]:7.1f} {S:16.2f}  {res["arret"][:40]}')
    ax[0].set(title='Courant qui suit i_lim (0,6)', xlabel='Temps (h)', ylabel='Radionucléide retiré (%)', ylim=(0, 100))
    ax[1].set(title='Courant constant (essai A2 de Xie)', xlabel='Temps (h)', ylabel='Radionucléide retiré (%)', ylim=(0, 100))
    ax[2].plot([0, 100], [0, 100], ':', color='0.5', label='même vitesse que le sel')
    ax[2].set(title='Sélectivité (courant constant)', xlabel='Sel retiré (%)', ylabel='Radionucléide retiré (%)', xlim=(0, 50), ylim=(0, 100))
    for x in ax:
        x.grid(alpha=.3)
        x.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(dossier / 'comparaison_especes.png', dpi=140)
    lignes.append('S : vitesse relative au sel (S = 1 : meme vitesse ; S = 2 : le radionucleide restant = (sel restant)^2)')
    (dossier / 'comparaison_especes.txt').write_text('\n'.join(lignes) + '\n', encoding='utf-8')
    print('\n'.join(lignes))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--especes', default='Cs+,Sr2+,Co2+,Mn2+')
    ap.add_argument('--tmax', type=float, default=4.)
    ap.add_argument('--budget', type=float, default=1e9)
    ap.add_argument('--figures-seulement', action='store_true')
    a = ap.parse_args()
    especes = a.especes.split(',')
    dossier = ROOT / 'resultats' / 'especes_REFERENCE'
    dossier.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not a.figures_seulement:
        for mode in ('courant', 'frac'):
            for sp in especes:
                run = dossier / f'{mode}_{sp}'
                if (run / 'resume.json').exists():
                    continue
                if time.time() - t0 > a.budget:
                    print('budget atteint : relancer la meme commande')
                    break
                print(f'{sp}, {mode} ...', flush=True)
                lancer(sp, mode, run, a)
    figures(dossier, especes)


if __name__ == '__main__':
    main()
