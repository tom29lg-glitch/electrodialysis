#!/usr/bin/env python3
"""Finalise V27 sans relancer Newton : bilans de maillage, profils et rapport."""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULTS = [
 'resultats_v25_1p5V/steps/v_1.50000000.npz',
 'resultats_v27/test_tolerance_60x24/steps/v_1.50000000.npz',
 'resultats_v27/reprise_75x30/steps/v_1.50000000.npz',
]

def read(path):
 with np.load(path,allow_pickle=False) as d:
  r=np.asarray(d['region']); na=np.asarray(d['c_na']); cl=np.asarray(d['c_cl']); x=np.asarray(d['x']); y=np.asarray(d['y'])
  if na.shape != (len(y),len(x)) or cl.shape!=na.shape: raise ValueError('Dimensions incoherentes')
  if not bool(d['converged']): raise ValueError(f'Solution non convergee : {path}')
  if abs(float(d['voltage'])-1.5)>1e-7: raise ValueError('Tension differente de 1.5 V')
  if np.min(na)<=0 or np.min(cl)<=0: raise ValueError('Concentrations non positives')
  counts=[np.count_nonzero(r==i) for i in range(1,6)]
  widths=np.asarray(d['widths']); dx=np.concatenate([np.full(int(n),w/n) for n,w in zip(counts,widths)])
  v=np.asarray(d['velocity_x']); mask=r==3; ww=v[mask]*dx[mask]
  cout_na=float(np.average(na[-1,mask],weights=ww)); cout_cl=float(np.average(cl[-1,mask],weights=ww))
  c0=float(d['c_inlet']); ii=np.asarray(d['interface_indices'],dtype=int)
  curr=float(np.mean(np.asarray(d['current_x'])[:,ii]))
  xmid=(x[mask]-sum(widths[:2]))/widths[2]
  return {'file':str(path),'nx':len(x),'ny':len(y),'cells':na.size,'residual':float(d['residual_max']),
    'current_Am2':curr,'outlet_na_flow_molm3':cout_na,'outlet_cl_flow_molm3':cout_cl,
    'outlet_salt_flow_molm3':(cout_na+cout_cl)/2,'removal_flow_percent':100*(1-(cout_na+cout_cl)/(2*c0)),
    'diluate_na_min_molm3':float(na[:,mask].min()),'diluate_cl_min_molm3':float(cl[:,mask].min()),
    'xmid':xmid,'y':y,'na':na[:,mask],'cl':cl[:,mask], 'c0':c0}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('files',nargs='*',help='3 fichiers NPZ (sinon chemins par defaut)')
 ap.add_argument('--output',default='resultats_v27/finalisation')
 args=ap.parse_args(); paths=args.files or DEFAULTS
 if len(paths)!=3: ap.error('Fournir exactement 3 fichiers NPZ')
 data=[read(Path(p)) for p in paths]
 if len({(d['nx'],d['ny']) for d in data})!=3: raise ValueError('Les trois maillages doivent etre distincts')
 out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
 keys=['file','nx','ny','cells','residual','current_Am2','outlet_na_flow_molm3','outlet_cl_flow_molm3','outlet_salt_flow_molm3','removal_flow_percent','diluate_na_min_molm3','diluate_cl_min_molm3']
 with (out/'comparaison_v27.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:d[k] for k in keys} for d in data])
 for key,label,filename in [('current_Am2','Courant moyen (A/m²)','courant.png'),('outlet_salt_flow_molm3','Sel en sortie du diluat (mol/m³)','diluat_debit.png'),('diluate_na_min_molm3','Na+ minimum dans le diluat (mol/m³)','minimum_na.png')]:
  fig,ax=plt.subplots(figsize=(7,4.5));ax.plot([d['cells'] for d in data],[d[key] for d in data],'-o');ax.set(xlabel='Nombre de cellules',ylabel=label);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/filename,dpi=170);plt.close(fig)
 for species in ('na','cl'):
  fig,axes=plt.subplots(1,2,figsize=(12,4.5))
  for d in data:
   label=f"{d['nx']}x{d['ny']}"
   axes[0].plot(d['xmid'],d[species][-1],'-o',ms=2,label=label)
   axes[1].plot(d['y'],d[species][:,0],'-o',ms=2,label=label)
  axes[0].set(xlabel='Position relative dans le diluat, x/L',ylabel='Concentration (mol/m³)',title='Profil transversal à la sortie')
  axes[1].set(xlabel='Position verticale y (m)',ylabel='Concentration (mol/m³)',title='Profil le long de la première cellule du diluat')
  for ax in axes: ax.grid(alpha=.3);ax.legend()
  fig.tight_layout();fig.savefig(out/f'profils_{species}.png',dpi=170);plt.close(fig)
 a,b,c=data
 def rel(x,y): return 100*abs(y-x)/max(abs(y),1e-30)
 lines=['# V27 — Synthèse de convergence en maillage','', 'Calculs à 1,5 V. Les concentrations en sortie sont **pondérées par le débit** de Poiseuille.', '',
 '| Maillage | Résidu normalisé | Courant (A/m²) | Sel sortie (mol/m³) | Dessalement (%) | Na+ min. diluat (mol/m³) |', '|---|---:|---:|---:|---:|---:|']
 for d in data: lines.append(f"| {d['nx']}×{d['ny']} | {d['residual']:.3e} | {d['current_Am2']:.6f} | {d['outlet_salt_flow_molm3']:.6f} | {d['removal_flow_percent']:.6f} | {d['diluate_na_min_molm3']:.6f} |")
 lines+=['','## Écarts relatifs successifs (dénominateur : maillage le plus fin de la paire)','', '| Grandeur | 1 → 2 | 2 → 3 |','|---|---:|---:|']
 for key,label in [('current_Am2','Courant'),('outlet_salt_flow_molm3','Sel sortie'),('removal_flow_percent','Dessalement'),('diluate_na_min_molm3','Na+ minimum')]:
  lines.append(f"| {label} | {rel(a[key],b[key]):.5g} % | {rel(b[key],c[key]):.5g} % |")
 lines+=['','## Interprétation','',
 '- Les bilans de conservation et les équilibres de Donnan ont été vérifiés séparément par V26 sur les trois maillages.',
 '- Les valeurs de résidu ont été obtenues avec des tolérances Newton différentes (1e-7, 2e-7, 4e-7) : les erreurs algébriques ne sont donc pas strictement comparables.',
 '- La stabilité des grandeurs globales ne prouve pas la convergence locale. Examiner les profils Na+/Cl- et les minima.',
 '- Le profil vertical de la première cellule du diluat change de position avec le raffinement ; il est qualitatif, pas une comparaison à abscisse physique identique.',
 '- Une convergence asymptotique et un ordre de précision ne sont pas établis par ces seules données.',
 '- Le modèle reste un prototype cartésien : aucune équivalence quantitative à COMSOL n’est démontrée.', '']
 (out/'rapport_v27.md').write_text('\n'.join(lines),encoding='utf-8')
 print('\n'.join(lines[:11]));print(f'\nFichiers générés dans {out.resolve()}')
if __name__=='__main__':main()
