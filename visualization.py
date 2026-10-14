#!/usr/bin/env python3
"""V24: figures et diagnostics depuis les resultats V23 (aucun recalcul du solveur)."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

F=96485.33212

def main():
    ap=argparse.ArgumentParser(description='Visualisation V24 des resultats V23')
    ap.add_argument('--input',default='resultats/essai_75x30/solution.npz')
    ap.add_argument('--output',default='figures_v24')
    args=ap.parse_args()
    a=np.load(args.input)
    needed=['x','y','region','widths','c_na','c_cl','phi','velocity_x','interface_indices','flux_na_x','flux_cl_x','trace_phi_left','trace_phi_right']
    missing=[k for k in needed if k not in a.files]
    if missing: raise ValueError('Champs manquants dans NPZ: '+', '.join(missing))
    if 'converged' in a.files and not bool(a['converged']):
        raise ValueError('Le calcul source n a pas converge: graphiques refuses')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    x=np.asarray(a['x']); y=np.asarray(a['y']); reg=np.asarray(a['region']); widths=np.asarray(a['widths']);
    na=np.asarray(a['c_na']);cl=np.asarray(a['c_cl']);phi=np.asarray(a['phi']);v=np.asarray(a['velocity_x'])
    if na.shape != (len(y),len(x)) or cl.shape!=na.shape or phi.shape!=na.shape: raise ValueError('Dimensions incompatibles')
    bounds=np.r_[0.,np.cumsum(widths)]
    def finish(fig,name):
        fig.savefig(out/name,dpi=190,bbox_inches='tight');plt.close(fig)
    def boundaries(ax,orientation='x'):
        for pos in bounds[1:-1]:
            if orientation=='x':ax.axvline(pos*1e3,color='k',lw=.7,ls='--',alpha=.65)
            else:ax.axhline(pos*1e3,color='k',lw=.7,ls='--',alpha=.65)
    def field(field,title,unit,filename):
        fig,ax=plt.subplots(figsize=(11,5))
        # Rendu par sous-domaine : aucune interpolation artificielle aux interfaces Donnan.
        m=None
        for rr in range(5):
            ids=np.where(reg==rr+1)[0]; vals=field[:,ids]
            xe=np.linspace(bounds[rr],bounds[rr+1],len(ids)+1)*1e3
            ye=np.linspace(0,float(y[-1]+(y[1]-y[0])/2),len(y)+1)*1e3
            m=ax.pcolormesh(xe,ye,vals,shading='flat',cmap='viridis',vmin=float(np.nanmin(field)),vmax=float(np.nanmax(field)))
        boundaries(ax);fig.colorbar(m,ax=ax,label=unit)
        ax.set(xlabel='Position transversale x (mm)',ylabel='Hauteur y (mm)',title=title)
        finish(fig,filename)
    field(na,'Concentration de Na+','mol/m³','01_concentration_na.png')
    field(cl,'Concentration de Cl−','mol/m³','02_concentration_cl.png')
    field(phi*1e3,'Potentiel électrique (sauts de Donnan non interpolés)','mV','03_potentiel.png')
    # Profils axiaux de chaque canal : concentrations moyennées en largeur (volumes ponderes).
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharex=True)
    for rr in (0,2,4):
        ids=np.where(reg==rr+1)[0]
        dx=widths[rr]/len(ids)
        for ax,field0,name in [(axs[0],na,'Na+'),(axs[1],cl,'Cl−')]:
            avg=np.average(field0[:,ids],axis=1,weights=np.full(len(ids),dx))
            ax.plot(y*1e3,avg,label=f'Canal {rr+1}')
    for ax,ion in zip(axs,('Na+','Cl−')):
        ax.set(xlabel='Hauteur y (mm)',ylabel='Concentration moyenne (mol/m³)',title=f'{ion} : profils axiaux');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();finish(fig,'04_profils_canaux.png')
    # Courants aux quatre interfaces : les flux sont evalues sur les faces internes.
    idx=np.asarray(a['interface_indices'],dtype=int)
    ix=F*(np.asarray(a['flux_na_x'])-np.asarray(a['flux_cl_x']))
    fig,ax=plt.subplots(figsize=(8,5))
    for t,k in enumerate(idx):
        ax.plot(y*1e3,ix[:,k],label=f'Interface {t+1}/{t+2}')
    ax.set(xlabel='Hauteur y (mm)',ylabel='Courant normal iₓ (A/m²)',title='Densité de courant aux interfaces')
    ax.grid(alpha=.2);ax.legend();finish(fig,'05_courants_interfaces.png')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for rr in (0,2,4):
        ids=np.where(reg==rr+1)[0]
        ax.plot(x[ids]*1e3,v[ids]*1e3,marker='.',label=f'Canal {rr+1}')
    boundaries(ax)
    ax.set(xlabel='Position x (mm)',ylabel='Vitesse verticale (mm/s)',title='Poiseuille : vitesse moyenne par cellule')
    ax.grid(alpha=.2);ax.legend();finish(fig,'06_vitesse_poiseuille.png')
    # Profil de potentiel en travers, au milieu de la hauteur; valeurs distinctes aux membranes.
    mid=len(y)//2
    fig,ax=plt.subplots(figsize=(9,4.5))
    for rr in range(5):
        ids=np.where(reg==rr+1)[0];ax.plot(x[ids]*1e3,phi[mid,ids]*1e3,'o-',ms=3,label=f'Domaine {rr+1}')
    boundaries(ax)
    ax.set(xlabel='Position x (mm)',ylabel='Potentiel (mV)',title=f'Potentiel transversal à y={y[mid]*1e3:.2f} mm')
    ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2);finish(fig,'07_potentiel_transversal.png')
    # Diagnostic des concentrations de sel dans les canaux: entree et sortie
    lines=['V24 — diagnostics extraits du fichier V23',f'Source: {args.input}',f'Grille: {len(x)} x {len(y)}',f'Converge: {bool(a["converged"]) if "converged" in a.files else "non renseigne"}']
    for rr in (0,2,4):
        ids=np.where(reg==rr+1)[0]
        salt=.5*(na[:,ids]+cl[:,ids])
        inlet=float(np.mean(salt[0]));outlet=float(np.mean(salt[-1]));change=outlet-inlet
        lines.append(f'Canal {rr+1}: c_sel entree={inlet:.6f}, sortie={outlet:.6f} mol/m3, variation={change:+.6f} mol/m3')
    lines.append('Courant moyen par interface (A/m2): '+', '.join(f'{np.mean(ix[:,k]):+.8g}' for k in idx))
    lines.append('Sauts moyens de potentiel de Donnan (mV): '+', '.join(f'{z:+.6f}' for z in (np.mean((a['trace_phi_right']-a['trace_phi_left'])*1e3,axis=0))))
    (out/'diagnostic_v24.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    print(f'\n7 figures PNG + diagnostic sauvegardes dans {out.resolve()}')

if __name__=='__main__':main()
