#!/usr/bin/env python3
"""Continuation adaptative V28 depuis config.json, sans modifier le solveur."""
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def finite_positive(value, name):
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} doit être strictement positif et fini")
    return v


def inspect(path):
    try:
        with np.load(path, allow_pickle=False) as d:
            ok = bool(d['converged']) and float(d['residual_max']) < 1e-6
            ok = ok and np.all(np.isfinite(d['c_na'])) and np.all(np.isfinite(d['c_cl']))
            ok = ok and np.min(d['c_na']) > 0 and np.min(d['c_cl']) > 0
            return ok, float(d['voltage'])
    except (OSError, ValueError, KeyError):
        return False, None


def run_solver(env, output, logfile):
    with logfile.open('w', encoding='utf-8') as log:
        proc = subprocess.run([sys.executable, str(ROOT / 'solver.py')],
                              cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not output.exists():
        return False
    return inspect(output)[0]


def main():
    cfg = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    name = str(cfg['simulation_name'])
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Nom de simulation invalide')
    target = finite_positive(cfg['voltage'], 'voltage')
    nx = list(map(int, cfg['mesh']['nx_regions']))
    ny = int(cfg['mesh']['ny'])
    if len(nx) != 5 or min(nx) < 1 or ny < 1:
        raise ValueError('Maillage invalide')
    c = cfg['membrane']
    porosity = finite_positive(c['porosity'], 'porosity')
    if porosity > 1:
        raise ValueError('La porosité doit être <= 1')
    opts = cfg.get('continuation', {})
    start = finite_positive(opts.get('start_voltage', 0.006), 'start_voltage')
    initial_step = finite_positive(opts.get('initial_step', 0.02), 'initial_step')
    min_step = finite_positive(opts.get('min_step', 0.0001), 'min_step')
    max_step = finite_positive(opts.get('max_step', 0.10), 'max_step')
    max_attempts = int(opts.get('max_attempts', 300))
    if not min_step <= initial_step <= max_step or max_attempts < 1:
        raise ValueError('Réglages de continuation invalides')
    folder = ROOT / 'resultats' / (name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    steps = folder / 'etapes'
    steps.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith('ED_')}
    env.update({
        'ED_NX': ','.join(map(str, nx)), 'ED_NY': str(ny),
        'ED_VOLTAGE': str(target),
        'ED_D_NA': str(finite_positive(cfg['ions']['Na+']['diffusion'], 'D_Na')),
        'ED_D_CL': str(finite_positive(cfg['ions']['Cl-']['diffusion'], 'D_Cl')),
        'ED_C_INLET': str(finite_positive(cfg['concentration_inlet'], 'concentration')),
        'ED_T': str(finite_positive(cfg['temperature'], 'temperature')),
        'ED_U_AVG': str(finite_positive(cfg['velocity'], 'velocity')),
        'ED_MEM_POROSITY': str(porosity),
        'ED_X_CEM': str(float(c['charge_cem'])),
        'ED_X_AEM': str(float(c['charge_aem'])),
        'ED_NEWTON_STEPS': str(int(cfg['newton']['max_iterations'])),
        'ED_NEWTON_TOL': str(finite_positive(cfg['newton']['tolerance'], 'tolerance')),
    })
    # Les valeurs géométriques absentes restent celles par défaut du solveur V28.
    if 'height' in cfg:
        env['ED_HEIGHT'] = str(finite_positive(cfg['height'], 'height'))
    if 'widths' in cfg:
        widths = list(map(float, cfg['widths']))
        if len(widths) != 5 or min(widths) <= 0:
            raise ValueError('widths doit contenir cinq largeurs positives')
        env['ED_WIDTHS'] = ','.join(map(str, widths))

    # Si une solution initiale est fournie, on vérifie sa compatibilité physique.
    previous = None
    voltage = 0.0
    initial = cfg.get('initial_solution')
    if initial:
        previous = Path(initial).expanduser()
        if not previous.is_absolute():
            previous = ROOT / previous
        if not previous.is_file():
            raise FileNotFoundError(previous)
        ok, voltage = inspect(previous)
        if not ok:
            raise ValueError('Solution initiale non convergée')
        with np.load(previous, allow_pickle=False) as d:
            expected = {
                'c_inlet': float(env['ED_C_INLET']),
                'temperature': float(env['ED_T']),
                'u_avg': float(env['ED_U_AVG']),
                'membrane_porosity': float(env['ED_MEM_POROSITY']),
            }
            for key, val in expected.items():
                if not np.isclose(float(d[key]), val, rtol=1e-9, atol=1e-12):
                    raise ValueError(f'Solution initiale incompatible : {key}')
            if [int(np.sum(d['region'] == k)) for k in range(1, 6)] != nx or len(d['y']) != ny:
                raise ValueError('Solution initiale incompatible : maillage')
            if not np.allclose(d['xreg'][[1, 3]], [float(env['ED_X_CEM']), float(env['ED_X_AEM'])]):
                raise ValueError('Solution initiale incompatible : charges membranaires')
            if not np.isclose(float(d['Dna_region'][0]), float(env['ED_D_NA'])) or not np.isclose(float(d['Dcl_region'][0]), float(env['ED_D_CL'])):
                raise ValueError('Solution initiale incompatible : diffusion')
            widths = np.fromstring(env.get('ED_WIDTHS', '0.001,0.0002,0.001,0.0002,0.001'), sep=',')
            if not np.allclose(d['widths'], widths, rtol=0, atol=1e-12):
                raise ValueError('Solution initiale incompatible : largeurs')
            height = float(env.get('ED_HEIGHT', '0.01'))
            if not np.isclose(float(d['y'][0] + d['y'][-1]), height):
                raise ValueError('Solution initiale incompatible : hauteur')
        print(f'Initialisation convergée : {voltage:g} V', flush=True)
    else:
        voltage = min(target, start)
        output = steps / 'etape_0000.npz'
        e = dict(env, ED_VOLTAGE=str(voltage), ED_OUTPUT_NPZ=str(output))
        print(f'Calcul initial à {voltage:g} V', flush=True)
        if not run_solver(e, output, folder / 'etape_0000.log'):
            raise RuntimeError(f'Échec de la solution initiale ; consulter {folder / "etape_0000.log"}')
        previous = output

    if voltage > target + 1e-12:
        raise ValueError('La tension initiale dépasse la tension cible')
    step = initial_step
    for attempt in range(1, max_attempts + 1):
        if voltage >= target - 1e-12:
            break
        trial = min(target, voltage + step)
        output = steps / f'etape_{attempt:04d}.npz'
        e = dict(env, ED_VOLTAGE=str(trial), ED_INIT_NPZ=str(previous), ED_OUTPUT_NPZ=str(output))
        print(f'[{attempt}] {voltage:.6g} -> {trial:.6g} V (pas {step:.5g})', flush=True)
        if run_solver(e, output, folder / f'etape_{attempt:04d}.log'):
            previous, voltage = output, trial
            step = min(max_step, step * 1.4)
            print('  Convergé', flush=True)
        else:
            if output.exists():
                output.rename(steps / f'etape_{attempt:04d}_echec.npz')
            step /= 2
            print('  Échec, réduction du pas', flush=True)
            if step < min_step:
                break
    if voltage < target - 1e-12:
        raise RuntimeError(f'Arrêt à {voltage:g} V avant la cible {target:g} V. Voir {folder}')

    final = folder / 'solution.npz'
    import shutil
    shutil.copy2(previous, final)
    print(f'\nCible atteinte : {voltage:g} V. Solution : {final}', flush=True)
    if cfg.get('generate_plots', True):
        subprocess.run([sys.executable, str(ROOT / 'visualization.py'),
                        '--input', str(final), '--output', str(folder / 'figures')], cwd=ROOT, check=True)
    if cfg.get('run_validation', True):
        subprocess.run([sys.executable, str(ROOT / 'validation.py'), str(final)], cwd=ROOT, check=True)
    print(f'\nRésultats : {folder}', flush=True)


if __name__ == '__main__':
    main()
