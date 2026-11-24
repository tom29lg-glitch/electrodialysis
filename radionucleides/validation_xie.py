#!/usr/bin/env python3
"""Validation du modele en recirculation contre les mesures de Xie et al. 2022 (figures 3 a 5, tableau 3).

    python validation_xie.py --groupes A1,A2,A3 --tag brut                    # canal vide, sans calage
    python validation_xie.py --groupes A1,A2,A3 --tag cale --facteur-vitesse 2.23
    python validation_xie.py --tag cale --figures-seulement                   # refaire les figures

Chaque groupe est simule avec recirculation.py (courant constant, conditions de l'essai) puis compare aux
points mesures numerises (../references/xie2022_cs_numerise.json). Les groupes deja calcules dans un meme
--tag sont conserves : on peut lancer les groupes par paquets.

Hypotheses (a discuter dans le rapport) :
  - sel porteur NaCl + Na2SO4 a 2000 mg/L, moitie-moitie en masse, modelise comme NaCl a 31,2 mol/m3
    (meme concentration de charges positives) ;
  - geometrie : canaux = espaceur de 0,95 mm, membranes de 0,30 mm (tableau 1 de Xie), longueur d'ecoulement 0,90 m
    (108,5 cm2 de membrane / 1,2 cm de largeur de canal) ;
  - transport d'eau : 15 molecules d'eau par ion (valeur de Xie) ;
  - --facteur-vitesse : vitesse effective qui represente le brassage de l'espaceur. 2,23 redonne le courant
    limite mesure par Xie (0,78 A), car i_lim ~ v^(1/3) (Leveque) et le canal vide donne 55 A/m2 au lieu de 71,9.
  - le modele s'arrete au courant limite (pas de dissociation de l'eau) ; Xie poursuit jusqu'a 25 V.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'references' / 'xie2022_cs_numerise.json'
A_MEMBRANE = 108.5e-4
V_CONC = 0.5e-3
TITRES = {'A1': 'I = 0,5 I_lim', 'A2': 'I = 0,6 I_lim', 'A3': 'I = 0,7 I_lim',
          'B1': 'v = 0,14 m/s', 'B2': 'v = 0,19 m/s', 'B3': 'v = 0,23 m/s',
          'C1': 'VR = 1:10', 'C2': 'VR = 1:15', 'C3': 'VR = 1:20'}


def simuler(g, x, folder, a):
    args = [sys.executable, str(ROOT / 'recirculation.py'), '--mode', 'courant', '--I', str(x['I_A']),
            '--N', '10', '--A', str(A_MEMBRANE), '--Vc', str(V_CONC), '--VR', str(x['VR']),
            '--c0', str(a.c0), '--cr0', str(x['C0_mol_L'] * 1e3), '--tmax', str(x['duree_s'] / 3600),
            '--objectif', '1.01', '--saturation', '0', '--wch', '0.95e-3', '--wm', str(a.wm), '--L', '0.9',
            '--vavg', str(x['v_m_s'] / 2 * a.facteur_vitesse), '--rendement', str(a.rendement), '--au-dela', a.au_dela, '--plafond', str(a.plafond), '--hydratation', str(a.hydratation),
            '--pas', str(a.pas), '--dtmax', '2']
    args += (['--cmem-aem', str(a.cmem_aem)] if a.cmem_aem else [])
    args += ['--cmem', str(a.cmem), '--Kcem', str(a.Kcem), '--DCl', str(a.DCl)]
    if a.rapide:
        args.append('--rapide')
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith('Resultats : ')]
    if not line:
        raise SystemExit(f'{g} : echec de recirculation.py\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}')
    src = Path(line[-1].split('Resultats : ', 1)[1])
    dst = folder / g
    if dst.exists():
        for f in dst.iterdir():
            f.unlink()
        dst.rmdir()
    src.rename(dst)
    return dst


def analyser(g, x, run):
    rows = list(csv.DictReader((run / 'serie_temporelle.csv').open()))
    t = np.array([float(r['t_h']) for r in rows]) * 3600
    c = np.array([float(r['R_d']) for r in rows]) * 1e-3            # mol/L
    vd = np.array([float(r['V_d_L']) for r in rows]) * 1e-3          # m3
    res = json.loads((run / 'resume.json').read_text())
    exp = np.array(x['experimental'])
    t_end = t[-1]
    m = exp[:, 0] <= t_end + 1e-6
    ce = exp[m, 1]
    cm = np.interp(exp[m, 0], t, c)
    rms = float(np.sqrt(np.mean(((cm - ce) / x['C0_mol_L']) ** 2)))
    vd0 = x['VR'] * V_CONC
    # flux moyen de Cs sur la fenetre de validite du modele, comme l'eq. 14 de Xie (par m2 d'une membrane)
    n_rem_model = (x['C0_mol_L'] * vd0 - c[-1] * vd[-1]) * 1e3
    c_exp_end = float(np.interp(t_end, exp[:, 0], exp[:, 1]))
    n_rem_exp = (x['C0_mol_L'] - c_exp_end) * vd0 * 1e3                # le volume d'eau transferee est petit
    out = dict(groupe=g, arret_modele=res['arret'], fin_modele_s=t_end, duree_essai_s=x['duree_s'],
               fraction_essai_couverte=t_end / x['duree_s'],
               Cs_retire_modele=1 - c[-1] / x['C0_mol_L'], Cs_retire_exp_meme_instant=1 - c_exp_end / x['C0_mol_L'],
               flux_Cs_modele=n_rem_model / (t_end * A_MEMBRANE) if t_end > 0 else np.nan,
               flux_Cs_exp_meme_fenetre=n_rem_exp / (t_end * A_MEMBRANE) if t_end > 0 else np.nan,
               flux_Cs_exp_essai_complet=x['flux_Cs_experience'], ecart_rms_sur_C0=rms)
    out['ecart_flux'] = out['flux_Cs_modele'] / out['flux_Cs_exp_meme_fenetre'] - 1
    return out, (t, c)


def figures(folder, data):
    runs = sorted(p.name for p in folder.iterdir() if p.is_dir() and p.name in data)
    if not runs:
        return
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    table = []
    for ax, g in zip(axes.ravel(), sorted(data)):
        x = data[g]
        exp, xm = np.array(x['experimental']), np.array(x['modele_xie'])
        ax.plot(exp[:, 0] / 1e3, exp[:, 1] * 1e5, 's', color='0.2', ms=5, label='Xie : mesures')
        ax.plot(xm[:, 0] / 1e3, xm[:, 1] * 1e5, 'o-', color='C3', ms=3, lw=0.8, label='Xie : leur modèle')
        if g in runs:
            out, (t, c) = analyser(g, x, folder / g)
            table.append(out)
            ax.plot(t / 1e3, c * 1e5, '-', color='C0', lw=2, label='notre modèle 2D')
            if out['fin_modele_s'] < x['duree_s'] * 0.98:
                ax.axvline(out['fin_modele_s'] / 1e3, color='C0', ls=':', lw=1)
                ax.text(out['fin_modele_s'] / 1e3, 0.6, ' courant\n limite', color='C0', fontsize=8, va='bottom')
        ax.set(title=f'{g} : {TITRES[g]}', xlabel='Temps (10³ s)', ylabel='Cs diluat (10⁻⁵ mol/L)', ylim=(0, 6))
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle(f'Validation contre Xie et al. 2022 ({folder.name})', fontsize=12)
    fig.tight_layout()
    fig.savefig(folder / 'comparaison_xie.png', dpi=150)
    plt.close(fig)
    with (folder / 'comparaison_xie.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    lines = [f'{"groupe":6s} {"fin modele":>10s} {"Cs modele":>10s} {"Cs exp":>8s} {"flux mod":>9s} {"flux exp":>9s} {"ecart":>7s} {"rms/C0":>7s}  arret']
    for o in table:
        lines.append(f'{o["groupe"]:6s} {o["fin_modele_s"]:9.0f}s {o["Cs_retire_modele"]*100:9.1f}% {o["Cs_retire_exp_meme_instant"]*100:7.1f}% '
                     f'{o["flux_Cs_modele"]*1e6:9.2f} {o["flux_Cs_exp_meme_fenetre"]*1e6:9.2f} {o["ecart_flux"]*100:+6.1f}% {o["ecart_rms_sur_C0"]*100:6.1f}%  {o["arret_modele"][:40]}')
    lines.append('flux en 1e-6 mol/(s.m2) sur la fenetre ou le modele est valide (jusqu au courant limite) ; Cs = fraction retiree a la fin de cette fenetre')
    (folder / 'comparaison_xie.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--groupes', default='A1,A2,A3,B1,B2,B3,C1,C2,C3')
    ap.add_argument('--tag', default='brut', help='nom de la serie de calculs (dossier resultats/validation_xie_<tag>)')
    ap.add_argument('--rendement', type=float, default=1.0,
                    help='fraction du courant qui traverse les cellules ; le courant limite mesure (0,78 A) est alors rendement x 0,78 A dans les cellules')
    ap.add_argument('--facteur-vitesse', type=float, default=None,
                    help='par defaut 2,23 x rendement^3 : redonne le courant limite mesure, vu par les cellules')
    ap.add_argument('--c0', type=float, default=31.2)
    ap.add_argument('--hydratation', type=float, default=15.)
    ap.add_argument('--pas', type=float, default=0.05)
    ap.add_argument('--wm', type=float, default=0.3e-3, help='epaisseur des membranes (m) ; Xie, tableau 1 : 0,25-0,35 (CEM), 0,20-0,30 (AEM)')
    ap.add_argument('--au-dela', choices=('arret', 'plafond'), default='arret')
    ap.add_argument('--plafond', type=float, default=0.85)
    ap.add_argument('--cmem', type=float, default=1000.)
    ap.add_argument('--cmem-aem', type=float, default=None)
    ap.add_argument('--Kcem', type=float, default=1.0, help='partage du Cs dans la CEM')
    ap.add_argument('--DCl', type=float, default=2.032e-9)
    ap.add_argument('--rapide', action='store_true')
    ap.add_argument('--figures-seulement', action='store_true')
    a = ap.parse_args()
    if a.facteur_vitesse is None:
        a.facteur_vitesse = 2.23 * a.rendement ** 3
    data = json.loads(DATA.read_text())
    folder = ROOT / 'resultats' / f'validation_xie_{a.tag}'
    folder.mkdir(parents=True, exist_ok=True)
    if not a.figures_seulement:
        (folder / 'parametres.json').write_text(json.dumps(vars(a), indent=2) + '\n')
        for g in a.groupes.split(','):
            print(f'{g} ({TITRES[g]}) ...', flush=True)
            simuler(g, data[g], folder, a)
    figures(folder, data)
    print(f'Resultats : {folder}')


if __name__ == '__main__':
    main()
