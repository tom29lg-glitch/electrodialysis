#!/usr/bin/env python3
"""V28 : export et comparaison reproductible Python/COMSOL, sans recalcul Newton.

Usage:
 python comparaison_comsol_v28.py
 python comparaison_comsol_v28.py --comsol-csv exports_comsol/champs.csv \
   --comsol-metrics exports_comsol/indicateurs.json

CSV COMSOL attendu : x_m,y_m,c_na_molm3,c_cl_molm3,phi_V (points dans les domaines).
Colonnes optionnelles : region (1..5). Une seule valeur par point; séparateur , ou ;.
Potentiel : la jauge est éliminée par soustraction d'une constante par comparaison.
Les points COMSOL sont interpolés *dans chaque région séparément* vers les centres
Python, afin de ne pas traverser les sauts de Donnan aux membranes.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

FIELDS=('c_na_molm3','c_cl_molm3','phi_V')
DEFAULT='resultats_v28/essai_75x30/solution.npz'

def load_npz(path):
 with np.load(path,allow_pickle=False) as z:
  needed=['x','y','region','c_na','c_cl','phi','velocity_x','widths','c_inlet','voltage','converged','residual_max','current_x','interface_indices']
  missing=[k for k in needed if k not in z]
  if missing: raise ValueError('NPZ incomplet : '+', '.join(missing))
  d={k:np.asarray(z[k]).copy() for k in z.files}
 if not bool(d['converged']): raise ValueError('Solution Python non convergée')
 nx,ny=len(d['x']),len(d['y'])
 if d['c_na'].shape!=(ny,nx) or d['c_cl'].shape!=(ny,nx) or d['phi'].shape!=(ny,nx): raise ValueError('Dimensions NPZ incohérentes')
 if len(d['region'])!=nx or len(d['velocity_x'])!=nx: raise ValueError('Métadonnées de grille incohérentes')
 return d

def metrics(d):
 mask=d['region']==3
 counts=[int(np.count_nonzero(d['region']==i)) for i in range(1,6)]
 dx=np.concatenate([np.full(n,float(w)/n) for n,w in zip(counts,d['widths'])])
 weights=d['velocity_x'][mask]*dx[mask]
 if np.any(weights<0) or weights.sum()<=0: raise ValueError('Poids de débit invalides')
 na=float(np.average(d['c_na'][-1,mask],weights=weights))
 cl=float(np.average(d['c_cl'][-1,mask],weights=weights))
 salt=(na+cl)/2
 current=float(np.mean(d['current_x'][:,np.asarray(d['interface_indices'],dtype=int)]))
 return dict(voltage_V=float(d['voltage']),current_Am2=current,
             outlet_na_flow_molm3=na,outlet_cl_flow_molm3=cl,
             outlet_salt_flow_molm3=salt,desalination_percent=100*(1-salt/float(d['c_inlet'])),
             residual_normalized=float(d['residual_max']),
             min_na_diluate_molm3=float(np.min(d['c_na'][:,mask])))

def export_python(d,path):
 x,y=np.meshgrid(d['x'],d['y'])
 region=np.broadcast_to(d['region'],x.shape)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['x_m','y_m','region',*FIELDS])
  for vals in zip(x.ravel(),y.ravel(),region.ravel(),d['c_na'].ravel(),d['c_cl'].ravel(),d['phi'].ravel()):
   w.writerow([f'{v:.15g}' if i!=2 else int(v) for i,v in enumerate(vals)])

def read_comsol(path):
 with path.open(encoding='utf-8-sig',newline='') as f:
  lines=[s for s in f if s.strip() and not s.lstrip().startswith('%')]
 if not lines: raise ValueError('CSV COMSOL vide')
 delim=';' if lines[0].count(';')>lines[0].count(',') else ','
 reader=csv.DictReader(lines,delimiter=delim)
 required=['x_m','y_m',*FIELDS]
 if not reader.fieldnames or any(k not in reader.fieldnames for k in required):
  raise ValueError('En-têtes attendus : '+', '.join(required)+', éventuellement region. Exporter les unités SI.')
 rows=list(reader)
 if not rows: raise ValueError('Aucun point COMSOL')
 result={k:np.array([float(row[k].replace(',','.')) for row in rows]) for k in required}
 if 'region' in reader.fieldnames: result['region']=np.array([int(row['region']) for row in rows])
 if not all(np.isfinite(v).all() for v in result.values()): raise ValueError('Valeurs non finies dans CSV')
 return result

def compare(d,co,out):
 xx,yy=np.meshgrid(d['x'],d['y'])
 rgrid=np.broadcast_to(d['region'],xx.shape)
 coords=np.column_stack((co['x_m'],co['y_m']))
 # Pas d'interpolation transmembranaire; l'absence de points regionaux est un échec explicite.
 boundaries=np.r_[0,np.cumsum(d['widths'])]
 if 'region' in co: regions=co['region']
 else:
  regions=np.searchsorted(boundaries[1:-1],co['x_m'],side='right')+1
  if np.any(co['x_m']<boundaries[0]) or np.any(co['x_m']>boundaries[-1]):
   raise ValueError('Points COMSOL hors domaine x Python; vérifier la géométrie et les unités')
 interpolated={k:np.full(xx.shape,np.nan) for k in FIELDS}
 for reg in range(1,6):
  source=regions==reg
  target=rgrid==reg
  if source.sum()<3: raise ValueError(f'Région {reg} : moins de 3 points COMSOL')
  for field in FIELDS:
   try: fn=LinearNDInterpolator(coords[source],co[field][source],fill_value=np.nan)
   except Exception as exc: raise ValueError(f'Interpolation impossible en région {reg}: {exc}') from exc
   interpolated[field][target]=fn(xx[target],yy[target])
 rows=[]
 for field,py in [('c_na_molm3',d['c_na']),('c_cl_molm3',d['c_cl']),('phi_V',d['phi'])]:
  for reg in range(1,6):
   mask=(rgrid==reg)&np.isfinite(interpolated[field])
   n=int(mask.sum());total=int(np.count_nonzero(rgrid==reg))
   if n==0:
    rows.append(dict(field=field,region=reg,n_points=0,coverage_percent=0,rmse='',mae='',max_abs='',bias='',gauge_shift_V=''))
    continue
   ref=interpolated[field][mask]; model=py[mask]
   shift=float(np.mean(ref-model)) if field=='phi_V' else 0.
   err=model+shift-ref
   rows.append(dict(field=field,region=reg,n_points=n,coverage_percent=100*n/total,
                    rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(np.abs(err))),
                    max_abs=float(np.max(np.abs(err))),bias=float(np.mean(err)),
                    gauge_shift_V=shift if field=='phi_V' else ''))
  # Afficher une carte d'écart; masque les centres hors enveloppe COMSOL.
  valid=np.isfinite(interpolated[field]);diff=np.full(xx.shape,np.nan)
  if valid.any():
   shift=float(np.mean(interpolated[field][valid]-py[valid])) if field=='phi_V' else 0.
   diff[valid]=py[valid]+shift-interpolated[field][valid]
  fig,ax=plt.subplots(figsize=(9,4))
  mesh=ax.pcolormesh(xx,yy,diff,shading='nearest',cmap='coolwarm')
  fig.colorbar(mesh,ax=ax,label='Python - COMSOL ('+('V' if field=='phi_V' else 'mol/m³')+')')
  for boundary in boundaries[1:-1]: ax.axvline(boundary,color='k',lw=.7,alpha=.5)
  ax.set(xlabel='x (m)',ylabel='y (m)',title='Écart '+field+' (zones non interpolables masquées)')
  fig.tight_layout();fig.savefig(out/f'ecart_{field}.png',dpi=170);plt.close(fig)
 with (out/'ecarts_champs.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return rows

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--python-npz',default=DEFAULT)
 p.add_argument('--comsol-csv',type=Path,help='Export COMSOL des champs sur la géométrie correspondante')
 p.add_argument('--comsol-metrics',type=Path,help='JSON avec current_Am2, outlet_salt_flow_molm3, desalination_percent (facultatifs)')
 p.add_argument('--output',type=Path,default=Path('resultats_v28'))
 args=p.parse_args()
 d=load_npz(Path(args.python_npz));out=args.output;out.mkdir(parents=True,exist_ok=True)
 m=metrics(d)
 export_python(d,out/'champs_python.csv')
 (out/'indicateurs_python.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 notes=['# V28 — Comparaison Python / COMSOL','',
        '## Référence Python','',f"- Maillage : {len(d['x'])} × {len(d['y'])}",
        f"- Tension : {m['voltage_V']:.6f} V",f"- Courant : {m['current_Am2']:.6f} A/m²",
        f"- Sel en sortie (pondération débit) : {m['outlet_salt_flow_molm3']:.6f} mol/m³",
        f"- Dessalement : {m['desalination_percent']:.6f} %",'',
        '## Hypothèses et limites','',
        '- Python : modèle électroneutre Nernst–Planck, membranes avec charges fixes et équilibre de Donnan, convection Poiseuille, conditions périodiques en x et Danckwerts en entrée.',
        '- Diffusivités effectives des membranes : D_eff = porosité × D_bulk (tortuosité supposée égale à 1).',
        '- La correspondance de la géométrie, des propriétés, des conditions aux limites et des définitions de courant/dessalement avec COMSOL doit être contrôlée.',
        '- Les profils aux interfaces présentent des sauts de Donnan : comparer les traces de part et d’autre séparément ; les cartes V28 ne comparent que les centres de cellules.',
        '- La différence de potentiel de jauge est retirée dans les comparaisons de champ phi ; cela ne vérifie pas à soi seul le saut de potentiel périodique.',
        '- Les comparaisons sont descriptives et ne prouvent pas la validité physique du modèle.', '']
 if args.comsol_csv:
  co=read_comsol(args.comsol_csv)
  rows=compare(d,co,out)
  notes+=['## Comparaison des champs','',f"- Points COMSOL fournis : {len(co['x_m'])}",
          '- RMSE, biais, maximum et couverture par région : `ecarts_champs.csv`.',
          '- Les régions peu couvertes par interpolation ne doivent pas être interprétées comme une bonne concordance.', '']
 else:
  notes+=['## Données COMSOL non fournies','',
          'Aucun écart Python–COMSOL n’a été calculé. Fournir un CSV avec les colonnes :',
          '`x_m,y_m,c_na_molm3,c_cl_molm3,phi_V` (option : `region`).',
          'Exporter des points suffisamment denses dans chacune des cinq régions ; unités SI, coordonnées physiques identiques.', '']
 if args.comsol_metrics:
  cm=json.loads(args.comsol_metrics.read_text(encoding='utf-8'))
  notes+=['## Indicateurs globaux COMSOL','', '| Grandeur | Python | COMSOL | Écart relatif à COMSOL |','|---|---:|---:|---:|']
  for key in ('current_Am2','outlet_salt_flow_molm3','desalination_percent'):
   if key not in cm: continue
   val=float(cm[key]);v=m[key];delta=100*abs(v-val)/abs(val) if val!=0 else float('nan')
   notes.append(f'| {key} | {v:.8g} | {val:.8g} | {delta:.5g} % |')
  notes+=['','Vérifier l’orientation du courant et la méthode de pondération du dessalement avant toute interprétation.', '']
 (out/'rapport_v28.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
 print('V28 : référence Python lue, sans recalcul Newton')
 print(json.dumps(m,indent=2,ensure_ascii=False))
 print('Comparaison COMSOL : '+('champs importés' if args.comsol_csv else 'en attente de CSV'))
 print('Fichiers :',out.resolve())

if __name__=='__main__': main()
