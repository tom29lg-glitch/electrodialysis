#!/usr/bin/env python3
"""V29 -- transport d'un radionucleide en traces (Cs+, Sr2+, ...) dans une solution NaCl convergee.

Hypothese (justifiee par Sugimoto 1978 et Xie et al. 2022) : le radionucleide est
present a l'etat de traces (1e-5 a 1e-3 mol/m3 face a ~10-500 mol/m3 de sel), il ne
modifie donc ni l'electroneutralite ni le potentiel. On resout UNE equation de
Nernst-Planck pour le traceur dans le champ phi deja calcule par solver.py :

    div( -D_i grad c_i - z_i D_i c_i grad(phi/VT) + u c_i ) = 0

avec la meme discretisation que solver.py (flux_face identique, traces aux interfaces,
periodicite en x, entree Danckwerts, sortie convective).

Interfaces membrane/solution : Donnan generalise avec coefficient de partage K
(selectivite specifique de la membrane, K=1 -> Donnan ideal comme COMSOL) :

    ln(c_m/K_m) + z phi_m/VT = ln(c_s) + z phi_s/VT

Le probleme etant lineaire en c, le resultat est proportionnel a c_in.

Test de coherence : --z 1 --D <D_Na> --c-in <c0> doit redonner c_Na du solveur.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

F = 96485.33212
R = 8.31446261815324

# Coefficients de diffusion a dilution infinie, 25 C (CRC Handbook), m2/s.
SPECIES = {
    'Na+': (1, 1.334e-9), 'K+': (1, 1.957e-9), 'Cs+': (1, 2.056e-9),
    'Sr2+': (2, 0.791e-9), 'Ca2+': (2, 0.792e-9), 'Co2+': (2, 0.732e-9), 'Mn2+': (2, 0.712e-9),
}


def flux_face(cL, cR, pL, pR, DL, DR, hL, hR, z):
    """Identique a solver.py : moyenne logarithmique, potentiel electrochimique."""
    dc = np.log(cR / cL)
    ratio = np.where(dc == 0, 1., np.expm1(dc) / np.where(dc == 0, 1., dc))
    lm = cL * ratio  # = (cR-cL)/ln(cR/cL), sans perte de chiffres
    return -lm * (dc + z * (pR - pL)) / (hL / DL + hR / DR)


def solve_tracer(npz, z, D_bulk, c_in, K_cem=1.0, K_aem=1.0, tol=1e-6, max_steps=40, verbose=True, c_in_conc=None):
    """c_in : entree du diluat ; c_in_conc : entree des canaux de concentrat (defaut : c_in)."""
    d = np.load(npz, allow_pickle=False)
    if not bool(d['converged']):
        raise ValueError('Solution NaCl non convergee : traceur refuse')
    x, y = d['x'], d['y']
    reg = d['region'] - 1
    widths = d['widths']
    dx = d['dx'] if 'dx' in d.files else widths[reg] / np.bincount(reg, minlength=5)[reg]
    nr, ny = len(x), len(y)
    dy = float(y[1] - y[0])
    VT = R * float(d['temperature']) / F
    U = float(d['voltage'])
    v = d['velocity_x']
    p = d['phi'] / VT
    pL = d['trace_phi_left'] / VT
    pR = d['trace_phi_right'] / VT
    ifc = d['interface_indices'].astype(int)
    ni = len(ifc)
    # meme correction de Bruggeman que pour Na (facteur relatif par region)
    factor = d['Dna_region'] / d['Dna_region'][0]
    Dreg = factor * D_bulk
    Dc = Dreg[reg]
    Kreg = np.array([1., K_cem, 1., K_aem, 1.])
    lnK = np.log(Kreg)[reg]
    lnK_L = lnK[ifc][None, :]
    lnK_R = lnK[ifc + 1][None, :]
    N = nr * ny
    M = ny * ni
    inlet = v > 0
    cin_cell = np.where(reg == 2, c_in, c_in if c_in_conc is None else c_in_conc)
    FLUX = D_bulk * c_in / 1e-3
    DIV = FLUX / 1e-3

    def unpack(q):
        c = c_in * np.exp(np.clip(q[:N].reshape(ny, nr), -60, 60))
        sL = q[N:].reshape(ny, ni)
        aL = c_in * np.exp(np.clip(sL, -60, 60))
        # Donnan generalise : c_R = c_L * K_R/K_L * exp(-z (pR - pL))
        aR = aL * np.exp(lnK_R - lnK_L - z * (pR - pL))
        return c, aL, aR

    def residual(q, full=False):
        c, aL, aR = unpack(q)
        div = np.zeros((ny, nr))
        f = flux_face(c[:, :-1], c[:, 1:], p[:, :-1], p[:, 1:], Dc[:-1], Dc[1:], dx[:-1] / 2, dx[1:] / 2, z)
        fr = f.copy()
        fL = flux_face(c[:, ifc], aL, p[:, ifc], pL, Dc[ifc], Dc[ifc], dx[ifc] / 4, dx[ifc] / 4, z)
        fR = flux_face(aR, c[:, ifc + 1], pR, p[:, ifc + 1], Dc[ifc + 1], Dc[ifc + 1], dx[ifc + 1] / 4, dx[ifc + 1] / 4, z)
        f[:, ifc] = fL
        fr[:, ifc] = fR
        div[:, :-1] += f / dx[:-1]
        div[:, 1:] -= fr / dx[1:]
        fper = flux_face(c[:, -1], c[:, 0], p[:, -1], p[:, 0] + U / VT, Dc[-1], Dc[0], dx[-1] / 2, dx[0] / 2, z)
        div[:, -1] += fper / dx[-1]
        div[:, 0] -= fper / dx[0]
        fy = flux_face(c[:-1, :], c[1:, :], p[:-1, :], p[1:, :], Dc[None, :], Dc[None, :], dy / 2, dy / 2, z) + v[None, :] * c[:-1, :]
        div[:-1, :] += fy / dy
        div[1:, :] -= fy / dy
        div[0, inlet] -= v[inlet] * cin_cell[inlet] / dy
        div[-1, inlet] += v[inlet] * c[-1, inlet] / dy
        eq = np.r_[div.ravel() / DIV, (fL - fR).ravel() / FLUX]
        if full:
            return c, aL, aR, f, fL, fR, div
        return eq

    # Structure creuse du Jacobien
    size = N + M
    pat = lil_matrix((size, size), dtype=np.int8)
    for j in range(ny):
        for k in range(nr):
            i0 = j * nr + k
            cols = [i0, j * nr + (k - 1) % nr, j * nr + (k + 1) % nr]
            if j:
                cols.append(i0 - nr)
            if j < ny - 1:
                cols.append(i0 + nr)
            for col in cols:
                pat[i0, col] = 1
        for t, k in enumerate(ifc):
            m = N + j * ni + t
            cells = [j * nr + k, j * nr + k + 1]
            for row in cells + [m]:
                for col in cells + [m]:
                    pat[row, col] = 1
    pat = pat.tocsr()

    # Initialisation de type Boltzmann a partir de Na : c ~ c_in*K*(cNa/c0)^z
    c0 = float(d['c_inlet'])
    q = np.r_[(z * np.log(d['c_na'] / c0) + lnK[None, :]).ravel(),
              (z * np.log(d['trace_na_left'] / c0) + lnK_L * np.ones((ny, 1))).ravel()]
    ok = False
    for it in range(max_steps):
        r = residual(q)
        err = float(np.max(np.abs(r)))
        if verbose:
            print(f'  Newton traceur {it:02d}: max|R|={err:.3e}', flush=True)
        if err < tol:
            ok = True
            break
        J = approx_derivative(residual, q, method='3-point', sparsity=pat)
        step = spsolve(J.tocsc(), -r)
        if not np.all(np.isfinite(step)):
            break
        alpha, nrm = 1.0, np.linalg.norm(r)
        for _ in range(25):
            cand = q + alpha * step
            rc = residual(cand)
            if np.all(np.isfinite(rc)) and np.linalg.norm(rc) < nrm * (1 - 1e-4 * alpha):
                q = cand
                break
            alpha *= 0.5
        else:
            break
    c, aL, aR, f, fL, fR, div = residual(q, True)

    # Indicateurs
    def mixed(rr, row):
        ids = np.where(reg == rr)[0]
        w = v[ids] * dx[ids]
        return float(np.sum(row[ids] * w) / np.sum(w))
    salt = 0.5 * (d['c_na'] + d['c_cl'])
    out = {
        'converged': ok, 'residual_max': err, 'z': z, 'D_bulk': D_bulk, 'c_in': c_in,
        'K_cem': K_cem, 'K_aem': K_aem,
        'tracer_out_diluate': mixed(2, c[-1]),
        'tracer_out_concentrate': 0.5 * (mixed(0, c[-1]) + mixed(4, c[-1])),
        'salt_out_diluate': mixed(2, salt[-1]),
        'salt_in': c0,
    }
    out['removal_tracer_single_pass'] = 1 - out['tracer_out_diluate'] / c_in
    out['removal_salt_single_pass'] = 1 - out['salt_out_diluate'] / c0
    out['ratio_removal_tracer_over_salt'] = out['removal_tracer_single_pass'] / out['removal_salt_single_pass']
    # flux moyen a travers la CEM (interface diluat|CEM = 2e interface, indice 1)
    J_tr = float(np.sum(fL[:, 1] * dy) / (ny * dy))
    i_mean = float(np.mean(F * (d['flux_na_x'][:, ifc[1]] - d['flux_cl_x'][:, ifc[1]])))
    J_na = float(np.mean(d['flux_na_x'][:, ifc[1]]))
    out['flux_tracer_cem_mean'] = J_tr
    out['transport_number_tracer_cem'] = z * F * J_tr / i_mean
    # facteur de separation tracer/Na (flux normalises par concentration d'entree)
    out['separation_factor_vs_Na'] = (J_tr / c_in) / (J_na / c0)
    fields = dict(c=c, trace_left=aL, trace_right=aR, flux_x=f, flux_left=fL, flux_right=fR)
    return out, fields, d


def plot_tracer(res, fields, d, species, folder):
    """Figures du traceur : carte c/c_in, profils axiaux, profil transversal."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=True)
    tag = species.replace('+', 'p').replace('-', 'm')
    c = fields['c'] / res['c_in']
    salt = 0.5 * (d['c_na'] + d['c_cl']) / float(d['c_inlet'])
    x, y, reg, widths = d['x'], d['y'], d['region'] - 1, d['widths']
    dx = d['dx'] if 'dx' in d.files else widths[reg] / np.bincount(reg, minlength=5)[reg]
    v = d['velocity_x']
    edges = np.r_[0., np.cumsum(dx)] * 1e3
    yedges = np.r_[0., (y[:-1] + y[1:]) / 2, 2 * y[-1] - (y[-1] + y[-2]) / 2] * 1e3
    bounds = np.cumsum(widths)[:-1] * 1e3
    names = {0: 'Concentrat (gauche)', 2: 'Diluat', 4: 'Concentrat (droite)'}
    # 1. Carte 2D par sous-domaine (echelle log : membranes tres enrichies ou appauvries)
    fig, ax = plt.subplots(figsize=(11, 5))
    lo, hi = np.log10(c.min()), np.log10(c.max())
    for rr in range(5):
        ids = np.where(reg == rr)[0]
        m = ax.pcolormesh(edges[ids[0]:ids[-1] + 2], yedges, np.log10(c[:, ids]), shading='flat',
                          cmap='viridis', vmin=lo, vmax=hi)
    for b in bounds:
        ax.axvline(b, color='k', lw=.7, ls='--')
    fig.colorbar(m, ax=ax, label=f'log10(c / c_in) {species}')
    ax.set(xlabel='x (mm)', ylabel='y (mm)', title=f'{species} : concentration relative (CEM | diluat | AEM)')
    fig.savefig(folder / f'tracer_{tag}_01_carte.png', dpi=170, bbox_inches='tight'); plt.close(fig)
    # 2. Profils axiaux de concentration de melange, traceur et sel
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for rr, col in ((2, 'C0'), (0, 'C3')):
        ids = np.where(reg == rr)[0]
        w = v[ids] * dx[ids]
        ax.plot(y * 1e3, (c[:, ids] * w).sum(1) / w.sum(), col, label=f'{species} — {names[rr]}')
        ax.plot(y * 1e3, (salt[:, ids] * w).sum(1) / w.sum(), col + '--', label=f'sel — {names[rr]}')
    ax.axhline(1, color='k', lw=.6)
    ax.set(xlabel='Hauteur y (mm)', ylabel='c / c_entree (moyenne de melange)',
           title=f'{species} : elimination {res["removal_tracer_single_pass"]*100:.2f} % '
                 f'(sel {res["removal_salt_single_pass"]*100:.2f} %), rapport {res["ratio_removal_tracer_over_salt"]:.2f}')
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(folder / f'tracer_{tag}_02_profils_axiaux.png', dpi=170, bbox_inches='tight'); plt.close(fig)
    # 3. Profil transversal a mi-hauteur
    mid = len(y) // 2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for rr in range(5):
        ids = np.where(reg == rr)[0]
        ax.semilogy(x[ids] * 1e3, c[mid, ids], 'o-', ms=2.5, color='C0', label=species if rr == 0 else None)
        ax.semilogy(x[ids] * 1e3, salt[mid, ids], 's-', ms=2, color='0.5', label='sel (Na+Cl)/2' if rr == 0 else None)
    for b in bounds:
        ax.axvline(b, color='k', lw=.7, ls='--')
    ax.set(xlabel='x (mm)', ylabel='c / c_entree', title=f'Profil transversal a y = {y[mid]*1e3:.2f} mm')
    ax.grid(alpha=.2, which='both'); ax.legend()
    fig.savefig(folder / f'tracer_{tag}_03_profil_transversal.png', dpi=170, bbox_inches='tight'); plt.close(fig)
    return folder


def main():
    ap = argparse.ArgumentParser(description='Traceur radionucleide dans une solution NaCl convergee (V29)')
    ap.add_argument('npz', type=Path, help='solution.npz issue de main.py')
    ap.add_argument('--species', default='Cs+', help='Cle de SPECIES : ' + ', '.join(SPECIES))
    ap.add_argument('--z', type=int, help='Charge (remplace celle de --species)')
    ap.add_argument('--D', type=float, help='Diffusion en solution m2/s (remplace celle de --species)')
    ap.add_argument('--c-in', type=float, default=3.76e-2, help='Concentration d entree mol/m3 (defaut : Cs de Xie et al. 2022)')
    ap.add_argument('--K-cem', type=float, default=1.0, help='Coefficient de partage dans la CEM (1 = Donnan ideal)')
    ap.add_argument('--K-aem', type=float, default=1.0, help='Coefficient de partage dans l AEM')
    ap.add_argument('--output', type=Path, help='NPZ de sortie (defaut : tracer_<espece>.npz a cote de la solution)')
    ap.add_argument('--no-plot', action='store_true', help='Ne pas tracer les figures')
    a = ap.parse_args()
    z0, D0 = SPECIES.get(a.species, (None, None))
    z = a.z if a.z is not None else z0
    D = a.D if a.D is not None else D0
    if z is None or D is None:
        raise SystemExit('Espece inconnue : preciser --z et --D')
    print(f'Traceur {a.species} : z={z}, D={D:.4g} m2/s, c_in={a.c_in:.4g} mol/m3, K_CEM={a.K_cem:g}, K_AEM={a.K_aem:g}')
    res, fields, d = solve_tracer(a.npz, z, D, a.c_in, a.K_cem, a.K_aem)
    print(json.dumps(res, indent=2))
    tag = a.species.replace('+', 'p').replace('-', 'm')
    outp = a.output or a.npz.with_name(f'tracer_{tag}.npz')
    np.savez(outp, x=d['x'], y=d['y'], region=d['region'], **fields, **{k: v for k, v in res.items()})
    outp.with_suffix('.json').write_text(json.dumps(res, indent=2) + '\n', encoding='utf-8')
    print('Fichiers :', outp, outp.with_suffix('.json'))
    if not a.no_plot:
        print('Figures :', plot_tracer(res, fields, d, a.species, outp.parent / 'figures'))
    if not res['converged']:
        raise SystemExit('NON CONVERGE')


if __name__ == '__main__':
    main()
