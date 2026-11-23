#!/usr/bin/env python3
"""Reproduit le cas COMSOL et le compare au PDF, en une commande.

    python validation_comsol.py            # maillage 180 x 60 (~1 min)
    python validation_comsol.py --rapide   # maillage 45 x 18 (~15 s), pour un essai

Cas COMSOL : NaCl 0,5 M, 1,5 V, D_Na = 2,5e-9, D_Cl = 2e-9 m2/s, cMem = 0, 500 et 1000 mol/m3.
Pour chaque cMem, la tension est montee de 0 a 1,5 V par paliers adaptatifs (comme main.py).
Resultats : resultats/validation_comsol_<maillage>_<date>/ (solutions + figures/ + comparaison.txt).
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def run_case(cmem, nx, ny, folder):
    env = {k: v for k, v in os.environ.items() if not k.startswith('ED_')}
    env.update(ED_NX=nx, ED_NY=str(ny), ED_X_CEM=str(-cmem), ED_X_AEM=str(cmem),
               ED_GRADING='10,2,10,2,10', ED_VELOCITY_PROFILE='comsol',
               ED_D_NA='2.5e-9', ED_D_CL='2e-9', ED_C_INLET='500', ED_U_AVG='0.005',
               ED_NEWTON_STEPS='60', ED_NEWTON_TOL='4e-7')
    steps = folder / f'etapes_cmem{cmem}'
    steps.mkdir(exist_ok=True)
    V, step, prev, n = 0.0, 0.1, '', 0
    while V < 1.5 - 1e-12:
        trial = 0.006 if V == 0 else min(1.5, V + step)
        n += 1
        out = steps / f'etape_{n:03d}.npz'
        e = dict(env, ED_VOLTAGE=str(trial), ED_OUTPUT_NPZ=str(out), ED_INIT_NPZ=str(prev))
        with (steps / 'dernier_calcul.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'solver.py')], cwd=ROOT, env=e, stdout=log, stderr=subprocess.STDOUT)
        ok = out.exists() and bool(np.load(out)['converged'])
        if ok:
            V, prev = trial, out
            step = min(0.2, step * 1.4)
        else:
            step /= 2
            if step < 0.005:
                raise RuntimeError(f'cMem={cmem} : echec a {V:g} V, voir {steps}/dernier_calcul.log')
        print(f'  cMem={cmem:4d} : {V:.3f} V', '(converge)' if ok else '(echec, pas reduit)', flush=True)
    final = folder / f'solution_cmem{cmem}.npz'
    final.write_bytes(Path(prev).read_bytes())
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rapide', action='store_true', help='maillage 45 x 18 au lieu de 180 x 60')
    a = ap.parse_args()
    nx, ny = ('9,6,15,6,9', 18) if a.rapide else ('20,20,100,20,20', 60)
    nxt = sum(map(int, nx.split(',')))
    folder = ROOT / 'resultats' / f'validation_comsol_{nxt}x{ny}_{datetime.now():%Y-%m-%d_%Hh%Mm%Ss}'
    folder.mkdir(parents=True)
    t0 = time.time()
    sols = {}
    for cmem in (0, 500, 1000):
        print(f'Calcul cMem = {cmem} mol/m3, maillage {nxt} x {ny}', flush=True)
        sols[cmem] = run_case(cmem, nx, ny, folder)
    print(f'Calculs termines en {time.time() - t0:.0f} s. Comparaison au PDF COMSOL :\n', flush=True)
    subprocess.run([sys.executable, str(ROOT / 'comparaison_comsol.py'), '--c1000', str(sols[1000]),
                    '--c500', str(sols[500]), '--c0', str(sols[0]), '--output', str(folder / 'figures')],
                   cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
