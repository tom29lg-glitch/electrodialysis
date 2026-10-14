#!/usr/bin/env python3
"""V19 : continuation spatiale V17 et diagnostic de convergence.

Executer dans le meme dossier que electrodialyse_donnan_newton_v19.py.
Chaque grille doit converger avant de passer a la suivante.
Les sorties originales V17 ne sont pas modifiees.
"""
import os
import sys
import subprocess
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
solver=root/'electrodialyse_donnan_newton_v19.py'
levels=[('15x6','3,2,5,2,3',6),('30x12','6,4,10,4,6',12),('45x18','9,6,15,6,9',18)]
max_evals=[int(x) for x in os.getenv('ED_V19_STEPS','35,45,60').split(',')]
if len(max_evals)!=len(levels):
    raise ValueError('ED_V19_STEPS attend trois nombres separes par des virgules')
previous=None
records=[]
for (label,nx,ny),nfev in zip(levels,max_evals):
    out=root/f'electrodialyse_v19_{label}.npz'
    env=os.environ.copy()
    env.update(ED_NX=nx,ED_NY=str(ny),ED_NEWTON_STEPS=str(nfev))
    env['ED_OUTPUT_NPZ']=str(out)
    if previous is not None:
        env['ED_INIT_NPZ']=str(previous)
    else:
        env.pop('ED_INIT_NPZ',None)
    print(f'\n========== V19 : grille {label} ==========\n',flush=True)
    proc=subprocess.run([sys.executable,str(solver)],env=env,cwd=root)
    if proc.returncode or not out.is_file():
        raise RuntimeError(f'Le solveur a echoue sur {label}')
    with np.load(out) as dat:
        converged=bool(dat['converged'])
        resid=float(dat['residual_max'])
        curr=np.asarray(dat['current_x'])
        indices=np.asarray(dat['interface_indices'],dtype=int)
        means=np.array([curr[:,k].mean() for k in indices])
        phiL=np.asarray(dat['trace_phi_left'])
        phiR=np.asarray(dat['trace_phi_right'])
        jumps=(phiR-phiL).mean(axis=0)*1e3
        records.append((label,int(dat['evaluations']),resid,means,jumps))
    print(f'\nResume {label}: residu={resid:.3e}, iterations_Newton={records[-1][1]}, converged={converged}',flush=True)
    if not converged:
        print('ARRET: ce maillage ne converge pas. Augmenter ED_V19_STEPS ou ameliorer le solveur.',flush=True)
        break
    previous=out

print('\n========== COMPARAISON DES MAILLAGES ==========')
for label,ne,resid,means,jumps in records:
    print(f'{label:6s} | pas_Newton={ne:5d} | residu={resid:.3e} | '
          f'courants A/m2={np.array2string(means,precision=6)} | '
          f'Donnan mV={np.array2string(jumps,precision=5)}')
for (lab0,_,_,m0,_),(lab1,_,_,m1,_) in zip(records[:-1],records[1:]):
    pct=100*np.abs((m1-m0)/np.maximum(np.abs(m0),1e-30))
    print(f'Variation {lab0} -> {lab1} (%): {np.array2string(pct,precision=3)}')
print('\nAttention: comparaison des courants moyens seulement, pas une validation physique complete.')
