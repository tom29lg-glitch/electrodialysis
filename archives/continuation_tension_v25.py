#!/usr/bin/env python3
"""V25 : continuation adaptative en tension du solveur V23, sans le modifier.

Exemple : python continuation_tension_v25.py --target 0.1 --initial-step 0.006
Les etapes non convergentes sont conservees dans failed/; les resultats acceptes
sont dans steps/. Le resume et les graphiques sont generes dans --output.
"""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def summarize(path):
    with np.load(path, allow_pickle=False) as d:
        if not bool(d['converged']):
            return None
        reg = d['region']
        na, cl = d['c_na'], d['c_cl']
        v = d['velocity_x']
        current = d['current_x']
        indices = d['interface_indices']
        ci = [float(np.mean(current[:, k])) for k in indices]
        out = {'voltage_V': float(d['voltage']), 'residual': float(d['residual_max']),
               'newton_iterations': int(d['evaluations']),
               'min_na_molm3': float(np.min(na)), 'min_cl_molm3': float(np.min(cl)),
               'current_mean_Am2': float(np.mean(ci)),
               'current_interface_spread_Am2': float(np.ptp(ci))}
        ncell = na.size
        out['max_abs_log_concentration_state'] = float(np.max(np.abs(d['solution_vector'][:ncell])))
        c0 = float(d['c_inlet'])
        for region, label in [(1,'concentrate_left'), (3,'diluate'), (5,'concentrate_right')]:
            mask = reg == region
            weights = v[mask]
            if not np.any(weights > 0):
                raise ValueError('Vitesse nulle dans le canal '+str(region))
            # concentration de sel moyenne ponderée par le debit; cNa=cCl dans les canaux.
            inlet = float(np.average(0.5*(na[0,mask]+cl[0,mask]), weights=weights))
            outlet = float(np.average(0.5*(na[-1,mask]+cl[-1,mask]), weights=weights))
            out[label+'_inlet_molm3'] = inlet
            out[label+'_outlet_molm3'] = outlet
            out[label+'_delta_molm3'] = outlet-inlet
        out['diluate_removal_percent'] = 100*(c0-out['diluate_outlet_molm3'])/c0
        return out


def save_summary(rows, folder):
    if not rows:
        return
    with (folder/'summary_v25.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        volts = [r['voltage_V'] for r in rows]
        plots = [
            ('current_mean_Am2', 'Courant moyen aux membranes (A/m²)', 'courant_vs_tension.png'),
            ('diluate_outlet_molm3', 'Concentration du diluat en sortie (mol/m³)', 'diluat_vs_tension.png'),
            ('diluate_removal_percent', 'Reduction du sel vs alimentation (%)', 'dessalement_vs_tension.png'),
        ]
        for key, ylabel, filename in plots:
            fig, ax = plt.subplots(figsize=(7,4.5))
            ax.plot(volts, [r[key] for r in rows], marker='o')
            ax.set(xlabel='Tension (V)', ylabel=ylabel)
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(folder/filename, dpi=170)
            plt.close(fig)
    except ImportError:
        print('Matplotlib absent : CSV produit, graphiques ignores.')


def main():
    ap = argparse.ArgumentParser(description='V25 : continuation en tension, solveur V23')
    ap.add_argument('--solver', type=Path, default=Path(__file__).with_name('electrodialyse_membranes_v28.py'))
    ap.add_argument('--initial', type=Path, default=Path(__file__).resolve().parent / 'resultats_v28/essai_45x18/solution.npz')
    ap.add_argument('--output', type=Path, default=Path('resultats_v28/continuation'))
    ap.add_argument('--target', type=float, default=1.5)
    ap.add_argument('--initial-step', type=float, default=0.006)
    ap.add_argument('--min-step', type=float, default=0.0001)
    ap.add_argument('--max-step', type=float, default=0.05)
    ap.add_argument('--max-attempts', type=int, default=200)
    args = ap.parse_args()
    if not args.solver.is_file() or not args.initial.is_file():
        ap.error('Fichier solveur V23 ou resultat initial introuvable')
    if not (args.target > 0 and 0 < args.min_step <= args.initial_step <= args.max_step):
        ap.error('Parametres de continuation invalides')
    folder = args.output.resolve()
    steps = folder/'steps'
    failed = folder/'failed'
    steps.mkdir(parents=True, exist_ok=True)
    failed.mkdir(parents=True, exist_ok=True)
    with np.load(args.initial, allow_pickle=False) as init:
        if not bool(init['converged']):
            ap.error('Le fichier initial V23 n est pas converge')
        voltage = float(init['voltage'])
        expected = {'ED_C_INLET': ('c_inlet', float), 'ED_U_AVG': ('u_avg', float),
                    'ED_MEM_POROSITY': ('membrane_porosity', float)}
        # Ne pas accepter silencieusement des parametres physiques contradictoires.
        for env_name, (key, cast) in expected.items():
            if env_name in os.environ and not np.isclose(float(os.environ[env_name]), cast(init[key])):
                ap.error(f'{env_name} differe du fichier initial')
        xreg = np.asarray(init['xreg'])
        for env_name, val in [('ED_X_CEM', xreg[1]), ('ED_X_AEM', xreg[3])]:
            if env_name in os.environ and not np.isclose(float(os.environ[env_name]), val):
                ap.error(f'{env_name} differe du fichier initial')
        # Reconstituer tous les parametres de la solution initiale.
        region = np.asarray(init['region'])
        nx = [int(np.sum(region == k)) for k in range(1,6)]
        env_base = os.environ.copy()
        env_base.update(ED_NX=','.join(map(str,nx)), ED_NY=str(len(init['y'])),
                        ED_WIDTHS=','.join(map(str,np.asarray(init['widths']))),
                        ED_HEIGHT=str(float(np.asarray(init['y'])[-1]+init['y'][0])),
                        ED_C_INLET=str(float(init['c_inlet'])),
                        ED_U_AVG=str(float(init['u_avg'])),
                        ED_MEM_POROSITY=str(float(init['membrane_porosity'])),
                        ED_X_CEM=str(float(xreg[1])), ED_X_AEM=str(float(xreg[3])),
                        ED_T=str(float(init['temperature'])),
                        ED_D_NA=str(float(init['Dna_region'][0])),
                        ED_D_CL=str(float(init['Dcl_region'][0])))
    if args.target < voltage - 1e-12:
        ap.error('La tension cible doit etre superieure a la tension initiale')
    rows = [summarize(args.initial)]
    previous = args.initial.resolve()
    step = args.initial_step
    print(f'V25 : depart {voltage:.6g} V, cible {args.target:.6g} V', flush=True)
    for attempt in range(1, args.max_attempts+1):
        if voltage >= args.target - 1e-12:
            break
        next_v = min(args.target, voltage + step)
        destination = steps/f'v_{next_v:.8f}.npz'
        log = folder/f'last_attempt.log'
        env = env_base.copy()
        env.update(ED_VOLTAGE=str(next_v), ED_INIT_NPZ=str(previous), ED_OUTPUT_NPZ=str(destination))
        print(f'[{attempt}] {voltage:.6f} -> {next_v:.6f} V (pas {step:.6g} V)', flush=True)
        with log.open('w', encoding='utf-8') as f:
            process = subprocess.run([sys.executable, str(args.solver.resolve())],
                                     env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
        record = None
        if process.returncode == 0 and destination.is_file():
            try:
                record = summarize(destination)
            except (ValueError, KeyError, OSError) as exc:
                print('  Fichier resultat illisible :', exc)
        if record is not None and record['residual'] < 1e-6 and record['min_na_molm3'] > 0 and record['min_cl_molm3'] > 0 and record['max_abs_log_concentration_state'] < 14.9:
            rows.append(record)
            previous = destination
            voltage = next_v
            save_summary(rows, folder)
            print(f"  OK : I={record['current_mean_Am2']:.5g} A/m2, diluat sortie={record['diluate_outlet_molm3']:.5g} mol/m3, R={record['residual']:.2e}", flush=True)
            step = min(args.max_step, step*1.4)
        else:
            if destination.is_file():
                destination.rename(failed/destination.name)
            print(f'  ECHEC : voir {log}; reduction du pas', flush=True)
            step /= 2
            if step < args.min_step:
                print('ARRET : pas minimal atteint. Derniere tension valide :', voltage)
                break
    else:
        print('ARRET : nombre maximal de tentatives atteint')
    save_summary(rows, folder)
    print(f'Tension atteinte : {voltage:.6g} V. Resultats : {folder}')
    return 0 if voltage >= args.target-1e-12 else 2

if __name__ == '__main__':
    raise SystemExit(main())
