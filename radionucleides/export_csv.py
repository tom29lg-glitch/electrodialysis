#!/usr/bin/env python3
"""Exporte les champs cellulaires d'une solution V28 au format CSV."""
import argparse
import csv
from pathlib import Path

import numpy as np


def export_solution(source: Path, destination: Path) -> int:
    with np.load(source, allow_pickle=False) as data:
        x = np.asarray(data['x'], dtype=float)
        y = np.asarray(data['y'], dtype=float)
        region = np.asarray(data['region'], dtype=int)
        na = np.asarray(data['c_na'], dtype=float)
        cl = np.asarray(data['c_cl'], dtype=float)
        phi = np.asarray(data['phi'], dtype=float)
        expected = (len(y), len(x))
        if region.shape != (len(x),) or any(a.shape != expected for a in (na, cl, phi)):
            raise ValueError('Dimensions des champs incompatibles avec le maillage')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['x_m', 'y_m', 'region', 'c_na_molm3', 'c_cl_molm3', 'phi_V'])
            for j, yy in enumerate(y):
                for i, xx in enumerate(x):
                    writer.writerow([xx, yy, int(region[i]), na[j, i], cl[j, i], phi[j, i]])
    return len(x) * len(y)


def main():
    parser = argparse.ArgumentParser(description='Export CSV des champs V28')
    parser.add_argument('input', type=Path, help='Chemin de solution.npz')
    parser.add_argument('output', nargs='?', type=Path, help='Chemin du CSV (par défaut export.csv à côté du NPZ)')
    args = parser.parse_args()
    output = args.output or args.input.with_name('export.csv')
    n = export_solution(args.input, output)
    print(f'Export CSV : {output} ({n} cellules)', flush=True)


if __name__ == '__main__':
    main()
