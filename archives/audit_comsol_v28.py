#!/usr/bin/env python3
"""V28: audit en lecture seule d'un .mph COMSOL et comparaison aux hypothèses V27.
Usage: python audit_comsol_v28.py electrodialysis.mph
Ne décode pas les fichiers binaires solution*.mphbin.
"""
import argparse, json, zipfile, os
from pathlib import Path

def walk(obj):
    if isinstance(obj,dict):
        yield obj
        for v in obj.values(): yield from walk(v)
    elif isinstance(obj,list):
        for v in obj: yield from walk(v)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mph',type=Path,nargs='?',default=Path('electrodialysis.mph'))
    ap.add_argument('--output',type=Path,default=Path('resultats_v28/audit_comsol'))
    a=ap.parse_args()
    if not a.mph.is_file(): ap.error(f'Fichier absent : {a.mph}')
    with zipfile.ZipFile(a.mph) as z:
        names=z.namelist()
        if 'smodel.json' not in names: ap.error('Pas de smodel.json dans ce MPH')
        model=json.loads(z.read('smodel.json'))
    nodes=list(walk(model))
    params={}
    for n in nodes:
        if isinstance(n,dict) and n.get('apiClass')=='ModelParamGroup':
            for s in n.get('settings',[]):
                if isinstance(s,dict) and s.get('name') and s.get('value') is not None:
                    params[s['name']]=s['value']
    # Les paramètres globaux sont aussi représentés par des noeuds 'Parameter'.
    for n in nodes:
        if isinstance(n,dict) and n.get('name') in ('Vtot','DNa','DCl','T','cCl_0','cMem','v_avg','L') and 'value' in n:
            params[n['name']]=n['value']
    relevant=[]
    for n in nodes:
        if not isinstance(n,dict) or 'settings' not in n: continue
        settings=n['settings']
        if not isinstance(settings,list): continue
        selected=[s for s in settings if isinstance(s,dict) and any(term in str(s.get('name','')).lower() for term in ('epsl','taul','diffusioncorrmodel','rhofix','applydonnan','boundaryconditiontype','periodic'))]
        if selected:
            relevant.append({'label':n.get('label'),'tag':n.get('tag'),'type':n.get('type'),'settings':selected})
    known={k:params.get(k,'non extrait') for k in ('Vtot','DNa','DCl','T','cCl_0','cMem','v_avg','L')}
    result={'source':str(a.mph),'model_version':model.get('lastComputationVersion'),'parameters':known,'physics_settings':relevant,'binary_solution_present':any(s.startswith('solutionstatic') and s.endswith('.mphbin') for s in names),'binary_solution_decoded':False,'python_v27_defaults':{'DNa_m2s':1.33e-9,'DCl_m2s':2.03e-9,'membrane_porosity':0.15,'membrane_diffusion':'epsilon*D_bulk; tortuosite=1'}}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# V28 — Audit des paramètres COMSOL','',f"Source : `{a.mph}`",'', '## Paramètres globaux extraits','']
    for k,v in known.items():lines.append(f'- `{k}` : `{v}`')
    lines+=['','## Réglages physiques pertinents','']
    for n in relevant:
        lines.append(f"### {n['label'] or n['type'] or n['tag']}")
        for s in n['settings']:lines.append(f"- `{s.get('name')}` : `{s.get('value',s.get('apiValue','présent'))}`")
    lines+=['','## Comparaison V27','', '- V27 : `DNa=1.33e-9`, `DCl=2.03e-9` m²/s par défaut.', '- V27 : `D_eff = 0.15 × D_bulk` dans les membranes, tortuosité = 1.', '- Les propriétés membranaires COMSOL doivent être comparées avec leurs réglages effectifs, et non avec la seule porosité.', '- Aucune valeur de courant COMSOL n’est extraite du fichier binaire dans cet audit.', '- Ne pas déclarer les modèles équivalents avant vérification de la géométrie, des conditions aux limites et du calcul de courant.','']
    (a.output/'rapport.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Audit COMSOL terminé :',a.output.resolve())
    print('Paramètres:',json.dumps(known,ensure_ascii=False))
    print('Réglages pertinents:',len(relevant),'blocs')
    print('Solution binaire présente:',result['binary_solution_present'],'— non décodée')
if __name__=='__main__':main()
