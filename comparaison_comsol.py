#!/usr/bin/env python3
"""V29 -- comparaison du solveur Python au modele COMSOL « Desalination in an Electrodialysis Cell ».

Reference : figures 5 a 9 du PDF COMSOL, numerisees dans references/comsol_figures_numerisees.npz
(script : references/numerisation_figures_comsol.py). Cas COMSOL : NaCl 0,5 M, 1,5 V, cMem = 0, 500
et 1000 mol/m3, D_Na = 2,5e-9, D_Cl = 2e-9 m2/s.

Usage :
  python comparaison_comsol.py --c1000 <solution.npz> [--c500 <npz>] [--c0 <npz>] [--output dossier]

Figures produites (memes coupes que le PDF) :
  fig5_potentiel.png        potentiel a mi-hauteur (cMem = 1000)
  fig6_concentrations.png   Na+ et Cl- a mi-hauteur (cMem = 1000)
  fig7_flux_na.png          flux x de Na+ : total, migration, diffusion
  fig8_flux_cl.png          flux x de Cl-
  fig9_courant_aem.png      courant normal le long de la face diluat/AEM, pour chaque cMem fourni
  comparaison.txt et .csv   valeurs aux points cles et ecarts relatifs
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

F = 96485.33212
R = 8.31446261815324
ROOT = Path(__file__).resolve().parent
GRIS = '0.45'


def load(path):
    d = dict(np.load(path, allow_pickle=False))
    if not bool(d['converged']):
        raise ValueError(f'{path} : solution non convergee')
    reg = d['region'] - 1
    d['reg'] = reg
    if 'dx' not in d:
        d['dx'] = d['widths'][reg] / np.bincount(reg, minlength=5)[reg]
    d['VT'] = R * float(d['temperature']) / F
    return d


def mid_rows(d):
    """Deux rangees encadrant y = L/2 et leurs poids d'interpolation lineaire."""
    y = d['y']
    H = y[0] + y[-1]
    j = int(np.searchsorted(y, H / 2)) - 1
    w = (H / 2 - y[j]) / (y[j + 1] - y[j])
    return j, w


def at_mid(d, a):
    j, w = mid_rows(d)
    return (1 - w) * a[j] + w * a[j + 1]


def potential_offset(d):
    """Jauge COMSOL : phi = 0 au coin (x = 0, y = 0). On extrapole la rangee du bas en x = 0."""
    p = d['phi'][0]
    x = d['x']
    return p[0] - (p[1] - p[0]) / (x[1] - x[0]) * x[0]


def face_fluxes(d, species):
    """Flux x aux faces internes a mi-hauteur : total, diffusion, migration (mol/m2/s).
    Decomposition exacte du schema : diffusion = -D (cR - cL)/h, migration = total - diffusion."""
    c = d['c_na'] if species == 'Na' else d['c_cl']
    z = 1 if species == 'Na' else -1
    D = (d['Dna_region'] if species == 'Na' else d['Dcl_region'])[d['reg']]
    dx = d['dx']
    p = d['phi'] / d['VT']
    h = dx[:-1] / 2 / D[:-1] + dx[1:] / 2 / D[1:]
    dc = np.log(c[:, 1:] / c[:, :-1])
    ratio = np.where(dc == 0, 1., np.expm1(dc) / np.where(dc == 0, 1., dc))
    lm = c[:, :-1] * ratio
    total = -lm * (dc + z * (p[:, 1:] - p[:, :-1])) / h
    diff = -(c[:, 1:] - c[:, :-1]) / h
    xf = np.cumsum(dx)[:-1]
    keep = d['reg'][:-1] == d['reg'][1:]          # faces internes a une region (pas les interfaces)
    tot, dif = at_mid(d, total), at_mid(d, diff)
    tot[~keep] = np.nan
    dif[~keep] = np.nan
    return xf, tot, dif, tot - dif


def by_region(d, x, vals, ax, **kw):
    """Trace region par region pour ne pas relier les sauts de Donnan."""
    reg = d['reg'] if len(x) == len(d['x']) else d['reg'][:-1]
    lab = kw.pop('label', None)
    for r in range(5):
        m = reg == r
        ax.plot(x[m] * 1e3, vals[m], label=lab if r == 0 else None, **kw)


def with_traces(d, cell_vals, left_key, right_key, offset=0.0):
    """Ajoute aux valeurs des cellules les valeurs aux interfaces (traces), pour tracer
    chaque region jusqu'a son bord comme COMSOL. Renvoie x et valeurs, NaN entre regions."""
    ifc = d['interface_indices'].astype(int)
    edges = np.cumsum(d['widths'])
    tl = at_mid(d, d[left_key]) - offset
    tr = at_mid(d, d[right_key]) - offset
    xs, vs = [], []
    for r in range(5):
        m = d['reg'] == r
        xr, vr = list(d['x'][m]), list(cell_vals[m])
        if r > 0:
            xr.insert(0, edges[r - 1]); vr.insert(0, tr[r - 1])
        if r < 4:
            xr.append(edges[r]); vr.append(tl[r])
        xs += xr + [np.nan]; vs += vr + [np.nan]
    return np.array(xs), np.array(vs)


def comsol_at(pts, x0, win=2e-5):
    """Valeur COMSOL numerisee en x0 : droite ajustee sur les points a +/- win
    (sans biais la ou la courbe est en pente, comme le potentiel dans les membranes)."""
    m = np.abs(pts[:, 0] - x0) < win
    if m.sum() >= 3 and np.ptp(pts[m, 0]) > 0:
        return float(np.polyval(np.polyfit(pts[m, 0], pts[m, 1], 1), x0))
    return float(np.median(pts[m, 1])) if m.any() else np.nan


def python_at(d, x, vals, x0):
    """Interpolation lineaire dans la region qui contient x0 (sans traverser une interface)."""
    reg = d['reg'] if len(x) == len(d['x']) else d['reg'][:-1]
    r = int(np.searchsorted(np.cumsum(d['widths']), x0))
    m = (reg == r) & np.isfinite(vals)
    return float(np.interp(x0, x[m], vals[m]))


def bounds(ax, d):
    for b in np.cumsum(d['widths'])[:-1]:
        ax.axvline(b * 1e3, color='k', lw=0.6, ls='--', alpha=0.5)


def main():
    ap = argparse.ArgumentParser(description='Comparaison Python / COMSOL (figures 5 a 9 du PDF)')
    ap.add_argument('--c1000', type=Path, required=True, help='solution.npz a cMem = 1000 mol/m3')
    ap.add_argument('--c500', type=Path)
    ap.add_argument('--c0', type=Path)
    ap.add_argument('--ref', type=Path, default=ROOT / 'references' / 'comsol_figures_numerisees.npz')
    ap.add_argument('--output', type=Path, help='dossier de sortie (defaut : comparaison_comsol a cote de --c1000)')
    a = ap.parse_args()
    ref = np.load(a.ref)
    out = a.output or a.c1000.parent / 'comparaison_comsol'
    out.mkdir(parents=True, exist_ok=True)
    d = load(a.c1000)
    rows = []  # (grandeur, position, COMSOL, Python)

    def save(fig, name):
        fig.savefig(out / name, dpi=170, bbox_inches='tight')
        plt.close(fig)

    # Fig. 5 : potentiel
    x = d['x']
    phi = at_mid(d, d['phi']) - potential_offset(d)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ref['fig5_phi'][:, 0] * 1e3, ref['fig5_phi'][:, 1], '.', color=GRIS, ms=2, label='COMSOL (figure 5, numérisée)')
    xt, vt = with_traces(d, phi, 'trace_phi_left', 'trace_phi_right', potential_offset(d))
    ax.plot(xt * 1e3, vt, color='C0', lw=1.6, label='Python')
    bounds(ax, d)
    ax.set(xlabel='Position x (mm)', ylabel='Potentiel de l\'électrolyte (V)', title='Potentiel à mi-hauteur, cMem = 1000 mol/m³')
    ax.grid(alpha=.25); ax.legend()
    save(fig, 'fig5_potentiel.png')
    for x0, nom in [(0.25e-3, 'canal 1'), (0.625e-3, 'CEM'), (1.25e-3, 'diluat'), (1.875e-3, 'AEM'), (2.25e-3, 'canal 5')]:
        rows.append(('potentiel (V)', nom, comsol_at(ref['fig5_phi'], x0), python_at(d, x, phi, x0)))

    # Fig. 6 : concentrations
    na, cl = at_mid(d, d['c_na']), at_mid(d, d['c_cl'])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ref['fig6_na'][:, 0] * 1e3, ref['fig6_na'][:, 1], '.', color='#7FA7E0', ms=2, label='COMSOL Na⁺')
    ax.plot(ref['fig6_cl'][:, 0] * 1e3, ref['fig6_cl'][:, 1], '.', color='#8FCB8F', ms=2, label='COMSOL Cl⁻')
    xt, vt = with_traces(d, na, 'trace_na_left', 'trace_na_right')
    ax.plot(xt * 1e3, vt, color='C0', lw=1.6, label='Python Na⁺')
    xt, vt = with_traces(d, cl, 'trace_cl_left', 'trace_cl_right')
    ax.plot(xt * 1e3, vt, color='C2', lw=1.6, label='Python Cl⁻')
    bounds(ax, d)
    ax.set(xlabel='Position x (mm)', ylabel='Concentration (mol/m³)', title='Concentrations à mi-hauteur, cMem = 1000 mol/m³')
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    save(fig, 'fig6_concentrations.png')
    for x0, nom, key, val in [(0.625e-3, 'CEM', 'fig6_na', na), (0.625e-3, 'CEM', 'fig6_cl', cl),
                              (1.875e-3, 'AEM', 'fig6_na', na), (1.875e-3, 'AEM', 'fig6_cl', cl),
                              (1.25e-3, 'diluat', 'fig6_cl', cl)]:
        ion = 'Na+' if key.endswith('na') else 'Cl-'
        rows.append((f'c {ion} (mol/m3)', nom, comsol_at(ref[key], x0), python_at(d, x, val, x0)))

    # Fig. 7 et 8 : flux
    for sp, fig_id, ion in [('Na', 'fig7', 'Na⁺'), ('Cl', 'fig8', 'Cl⁻')]:
        xf, tot, dif, mig = face_fluxes(d, sp)
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, col, lab in [('tot', 'C0', 'total'), ('mig', 'C2', 'migration'), ('dif', 'C3', 'diffusion')]:
            pts = ref[f'{fig_id}_{k}']
            ax.plot(pts[:, 0] * 1e3, pts[:, 1], '.', color=GRIS, ms=1.5, label='COMSOL (numérisé)' if k == 'tot' else None)
        for vals, col, lab in [(tot, 'C0', 'total'), (mig, 'C2', 'migration'), (dif, 'C3', 'diffusion')]:
            by_region(d, xf, vals, ax, color=col, lw=1.4, label=f'Python {lab}')
        bounds(ax, d)
        ax.set(xlabel='Position x (mm)', ylabel='Flux x (mol/(m²·s))', title=f'Flux de {ion} à mi-hauteur, cMem = 1000 mol/m³')
        ax.grid(alpha=.25); ax.legend(fontsize=8)
        save(fig, f'{fig_id}_flux_{sp.lower()}.png')
        # Dans le PDF, la courbe de migration recouvre la courbe totale la ou la diffusion est nulle :
        # on reunit les deux jeux de points pour lire le flux total au centre des regions.
        both = np.vstack([ref[f'{fig_id}_tot'], ref[f'{fig_id}_mig']])
        for x0, nom in [(1.25e-3, 'diluat'), (0.625e-3, 'CEM'), (1.875e-3, 'AEM')]:
            rows.append((f'flux {sp} (mol/m2/s)', nom, comsol_at(both, x0), python_at(d, xf, tot, x0)))

    # Fig. 9 : courant normal le long de la face diluat | AEM
    fig, ax = plt.subplots(figsize=(8, 5))
    for cm, path, col in [(0, a.c0, 'C0'), (500, a.c500, 'C2'), (1000, a.c1000, 'C3')]:
        pts = ref[f'fig9_{cm}']
        ax.plot(pts[:, 0] * 1e3, pts[:, 1], '.', color=GRIS, ms=1.5, label='COMSOL (numérisé)' if cm == 0 else None)
        if path is None:
            continue
        dd = load(path)
        if not np.isclose(abs(dd['xreg'][3]), cm):
            raise ValueError(f'{path} : charge fixe {dd["xreg"][3]} au lieu de {cm}')
        ifc = dd['interface_indices'].astype(int)
        i_aem = dd['current_x'][:, ifc[2]]
        ax.plot(dd['y'] * 1e3, i_aem, color=col, lw=1.6, label=f'Python {cm} mol/m³')
        for y0 in (1e-3, 5e-3, 9e-3):
            m = np.abs(pts[:, 0] - y0) < 1e-4
            rows.append(('i AEM (A/m2)', f'cMem={cm}, y={y0*1e3:.0f} mm', float(np.median(pts[m, 1])) if m.any() else np.nan,
                         float(np.interp(y0, dd['y'], i_aem))))
    ax.set(xlabel='Hauteur y (mm)', ylabel='Courant normal (A/m²)', title='Courant normal le long de la face diluat | AEM')
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    save(fig, 'fig9_courant_aem.png')

    # Tableau
    lines = [f'Comparaison Python / COMSOL -- source : {a.c1000}', f'Maillage : {len(d["x"])} x {len(d["y"])}', '',
             f'{"grandeur":24s} {"position":22s} {"COMSOL":>11s} {"Python":>11s} {"ecart":>8s}']
    with (out / 'comparaison.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['grandeur', 'position', 'comsol_numerise', 'python', 'ecart_relatif'])
        for g, pos, c, p in rows:
            e = (p - c) / abs(c) if c not in (0,) and np.isfinite(c) and abs(c) > 1e-12 else np.nan
            w.writerow([g, pos, c, p, e])
            ec = f'{e*100:+7.1f}%' if np.isfinite(e) else '   n.d.'
            cs = f'{c:11.4g}' if np.isfinite(c) else '       n.d.'
            lines.append(f'{g:24s} {pos:22s} {cs} {p:11.4g} {ec}')
    lines += ['', 'Note : valeurs COMSOL lues sur les figures du PDF (precision ~0,3 % de l\'axe).',
              'Le potentiel est compare apres la meme jauge (phi = 0 au coin x = 0, y = 0).']
    (out / 'comparaison.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'\nFigures et tableau : {out}')


if __name__ == '__main__':
    main()
