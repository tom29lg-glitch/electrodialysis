#!/usr/bin/env python3
"""V26: audit discret independant des fichiers NPZ V23/V25 (sans relancer Newton)."""
import argparse
from pathlib import Path
import numpy as np

F = 96485.33212
R = 8.31446261815324

def flux_face(cL, cR, pL, pR, DL, DR, hL, hR, z):
    dc = np.log(cR / cL)
    lm = np.where(np.abs(dc) < 1e-7, 0.5 * (cL + cR),
                  (cR - cL) / np.where(np.abs(dc) < 1e-7, 1., dc))
    return -lm * (dc + z * (pR - pL)) / (hL / DL + hR / DR)

def run(path):
    with np.load(path, allow_pickle=False) as d:
        required = ('x','y','region','c_na','c_cl','phi','X','solution_vector',
                    'interface_indices','voltage','temperature','c_inlet','velocity_x',
                    'widths','Dna_region','Dcl_region','converged','residual_max')
        for k in required:
            if k not in d: raise ValueError(f'Champ manquant: {k}')
        x,y,reg = d['x'],d['y'],d['region']-1
        na,cl,phi,X = d['c_na'],d['c_cl'],d['phi'],d['X']
        q=d['solution_vector']; indices=d['interface_indices'].astype(int)
        U=float(d['voltage']); VT=R*float(d['temperature'])/F
        c0=float(d['c_inlet']); v=d['velocity_x']
        widths=d['widths']; dna=d['Dna_region'][reg]; dcl=d['Dcl_region'][reg]
        nx,ny=len(x),len(y); dx=widths[reg]/np.bincount(reg, minlength=5)[reg]
        dy=float(y[1]-y[0]); N=nx*ny; ni=len(indices); M=ny*ni
        assert q.size==2*N+4*M
        p=phi/VT
        sL,sR,pL,pR=q[2*N:].reshape(4,ny,ni)
        def conc(s,fixed):
            prod=c0*np.exp(np.clip(s,-15,15))
            root=np.hypot(fixed,2*prod)
            a=np.where(fixed>=0,2*prod**2/(root+fixed),(root-fixed)/2)
            return a,a+fixed
        aL,bL=conc(sL,X[indices][None,:]); aR,bR=conc(sR,X[indices+1][None,:])
        fnaL=flux_face(na[:,indices],aL,p[:,indices],pL,dna[indices],dna[indices],dx[indices]/4,dx[indices]/4,1)
        fnaR=flux_face(aR,na[:,indices+1],pR,p[:,indices+1],dna[indices+1],dna[indices+1],dx[indices+1]/4,dx[indices+1]/4,1)
        fclL=flux_face(cl[:,indices],bL,p[:,indices],pL,dcl[indices],dcl[indices],dx[indices]/4,dx[indices]/4,-1)
        fclR=flux_face(bR,cl[:,indices+1],pR,p[:,indices+1],dcl[indices+1],dcl[indices+1],dx[indices+1]/4,dx[indices+1]/4,-1)
        mu_na=np.log(aL/aR)+pL-pR
        mu_cl=np.log(bL/bR)-pL+pR
        divs=[]; balances=[]
        print(f'\nV26 | {path} | U={U:.6g} V | converged={bool(d["converged"])} | R_enregistre={float(d["residual_max"]):.3e}')
        print(f'Electroneutralite max: {np.max(np.abs(na-cl+X[None,:])):.3e} mol/m3')
        for label,c,D,z,fleft,fright in [('Na+',na,dna,1,fnaL,fnaR),('Cl-',cl,dcl,-1,fclL,fclR)]:
            fl=flux_face(c[:,:-1],c[:,1:],p[:,:-1],p[:,1:],D[:-1],D[1:],dx[:-1]/2,dx[1:]/2,z)
            fr=fl.copy(); fl[:,indices]=fleft; fr[:,indices]=fright
            div=np.zeros_like(c)
            div[:,:-1]+=fl/dx[:-1]; div[:,1:]-=fr/dx[1:]
            fp=flux_face(c[:,-1],c[:,0],p[:,-1],p[:,0]+U/VT,D[-1],D[0],dx[-1]/2,dx[0]/2,z)
            div[:,-1]+=fp/dx[-1]; div[:,0]-=fp/dx[0]
            fy=flux_face(c[:-1,:],c[1:,:],p[:-1,:],p[1:,:],D[None,:],D[None,:],dy/2,dy/2,z)+v[None,:]*c[:-1,:]
            div[:-1,:]+=fy/dy; div[1:,:]-=fy/dy
            inlet=v>0
            fin=v[inlet]*c0; fout=v[inlet]*c[-1,inlet]
            div[0,inlet]-=fin/dy; div[-1,inlet]+=fout/dy
            # Bilan global: integrale sur les cellules, faces internes et periodiques s'annulent,
            # sauf les petites differences de flux entre les deux cotes des interfaces.
            total=float(np.sum(div*dx[None,:]*dy))
            inlet_total=float(np.sum(fin*dx[inlet])); outlet_total=float(np.sum(fout*dx[inlet]))
            jump_total=float(np.sum((fleft-fright)*dy))
            closure=outlet_total-inlet_total+jump_total
            scale=max(abs(inlet_total),abs(outlet_total),1e-30)
            print(f'\n{label}: min(c)={c.min():.6g}, max(c)={c.max():.6g} mol/m3')
            print(f'  max |div N| = {np.max(np.abs(div)):.6e} mol/(m3.s)')
            print(f'  entree={inlet_total:.9e}, sortie={outlet_total:.9e} mol/(m.s) [par profondeur unitaire]')
            print(f'  ecart sortie-entree={outlet_total-inlet_total:+.6e} mol/(m.s) ({100*(outlet_total-inlet_total)/scale:+.4g} %)')
            print(f'  saut total interfaces={jump_total:+.6e} mol/(m.s)')
            print(f'  somme divergence integree={total:+.6e}; fermeture={closure:+.6e} mol/(m.s)')
            print(f'  saut interface max={np.max(np.abs(fleft-fright)):.6e} mol/(m2.s)')
            divs.append(div); balances.append((inlet_total,outlet_total))
        print(f'\nDonnan max |delta mu_Na/RT|={np.max(np.abs(mu_na)):.6e}, |delta mu_Cl/RT|={np.max(np.abs(mu_cl)):.6e}')
        # La conservation du courant en volume s'obtient en soustrayant les divergences ioniques.
        charge_div=F*(divs[0]-divs[1])
        print(f'max |div i| = {np.max(np.abs(charge_div)):.6e} A/m3')
        print(f'Bilan charge entree={F*(balances[0][0]-balances[1][0]):+.6e}, sortie={F*(balances[0][1]-balances[1][1]):+.6e} A/m')
        # Equation de jauge: le residu charge au premier volume n'est pas impose directement.
        print(f'Cellule de jauge: divNa={divs[0][0,0]:+.6e}, divCl={divs[1][0,0]:+.6e} mol/(m3.s)')
        print('Note: un residu faible valide les equations discretes, pas leur convergence en maillage ni le modele physique.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('npz',nargs='?',type=Path,default=Path('resultats/essai_75x30/solution.npz'))
    args=parser.parse_args()
    run(args.npz)
