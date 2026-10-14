#!/usr/bin/env python3
"""V16: reconstruction diagnostique des traces de Donnan a partir de V13.

PAS un nouveau solveur couple. Flux V13 conserves; reconstruction a partir
 de resistances lineaires en potentiel electrochimique (approximation).
Les traces sont electroneutres et ont des potentiels electrochimiques egaux.
Ne prouve pas la convergence des flux locaux ni la physique de l'interface.
"""
import os
from pathlib import Path
import numpy as np

F=96485.33212
R=8.31446261815324
T=float(os.getenv('ED_T','298.15'))
VT=R*T/F
c0=float(os.getenv('ED_C_INLET','500'))
widths=np.fromstring(os.getenv('ED_WIDTHS','0.001,0.0002,0.001,0.0002,0.001'),sep=',')
nxreg=np.fromstring(os.getenv('ED_NX','9,6,15,6,9'),sep=',',dtype=int)
if len(widths)!=5 or len(nxreg)!=5 or np.any(nxreg<1):
    raise ValueError('ED_WIDTHS et ED_NX doivent avoir 5 elements positifs')
dx=np.repeat(widths/nxreg,nxreg)
Dna=np.repeat([1.33e-9,1.33e-9,1.33e-9,1.33e-11,1.33e-9],nxreg)
Dcl=np.repeat([2.03e-9,2.03e-11,2.03e-9,2.03e-9,2.03e-9],nxreg)
path=Path(os.getenv('ED_V16_INPUT','electrodialyse_electroneutre_v13.npz'))
with np.load(path) as d:
    na=d['c_na'].copy(); cl=d['c_cl'].copy(); phi=d['phi'].copy()
    X=d['X'].copy(); reg=d['region'].copy(); x=d['x'].copy(); y=d['y'].copy()
    fn=d['flux_na_x'].copy(); fc=d['flux_cl_x'].copy()
ny,nx=na.shape
if nx!=len(dx) or nx!=len(X) or len(y)!=ny:
    raise ValueError('Dimensions incompatibles: adapter ED_NX au fichier V13')
expected=np.cumsum(nxreg)[:-1]-1
idx=np.flatnonzero(reg[1:]!=reg[:-1])
if not np.array_equal(idx,expected):
    raise ValueError('Les sous-domaines du fichier ne correspondent pas a ED_NX')
if np.any(na<=0) or np.any(cl<=0):
    raise ValueError('Concentrations non positives')

# Potentiels electrochimiques sans dimension: mu_i/(RT), a constante pres.
mu_n=np.log(na/c0)+phi/VT
mu_c=np.log(cl/c0)-phi/VT

def traces_from_mu(mun,muc,Xval):
    """Pour muNa,muCl donnes: cNa*cCl=c0^2 exp(muNa+muCl), cCl-cNa=X."""
    prod=c0*np.exp(np.clip((mun+muc)/2,-50,50))
    root=np.hypot(Xval,2*prod)
    cn=np.where(Xval>=0,2*prod**2/(root+Xval),(root-Xval)/2)
    cc=cn+Xval
    potential=VT*(mun-np.log(cn/c0))
    return cn,cc,potential

# Face mu weighted by half-cell resistances 1/(D*c_bar).
# Assumes log-mean concentration on each half approximated by cell value.
# This reconstructs a common face mu; exact continuity follows algebraically.
records=[]
for m,j in enumerate(idx):
    dl=dx[j]/2; dr=dx[j+1]/2
    # resistances from center to interface, per species
    rn_l=dl/(Dna[j]*na[:,j]); rn_r=dr/(Dna[j+1]*na[:,j+1])
    rc_l=dl/(Dcl[j]*cl[:,j]); rc_r=dr/(Dcl[j+1]*cl[:,j+1])
    mn=(rn_r*mu_n[:,j]+rn_l*mu_n[:,j+1])/(rn_l+rn_r)
    mc=(rc_r*mu_c[:,j]+rc_l*mu_c[:,j+1])/(rc_l+rc_r)
    cn_l,cc_l,p_l=traces_from_mu(mn,mc,X[j])
    cn_r,cc_r,p_r=traces_from_mu(mn,mc,X[j+1])
    # Flux reconstructed on each half from linear electrochemical gradients.
    n_left=-(mn-mu_n[:,j])/rn_l
    n_right=-(mu_n[:,j+1]-mn)/rn_r
    c_left=-(mc-mu_c[:,j])/rc_l
    c_right=-(mu_c[:,j+1]-mc)/rc_r
    # Since the same linear resistances are used, left/right match by construction.
    # Compare instead with the original V13 flux: this is a meaningful approximation error.
    err_n=np.max(np.abs(n_left-fn[:,j]))
    err_c=np.max(np.abs(c_left-fc[:,j]))
    scale_n=max(np.max(np.abs(fn[:,j])),1e-30)
    scale_c=max(np.max(np.abs(fc[:,j])),1e-30)
    delta=(p_r-p_l)*1e3
    print(f'\nInterface {reg[j]}/{reg[j+1]} (x={sum(widths[:m+1])*1e3:.4f} mm)')
    print(f'  Donnan Delta phi (membrane/canal suivant x): min={delta.min():+.5f}, moyenne={delta.mean():+.5f}, max={delta.max():+.5f} mV')
    print(f'  Max erreur electroneutralite traces: {max(np.max(np.abs(cn_l-cc_l+X[j])),np.max(np.abs(cn_r-cc_r+X[j+1]))):.3e} mol/m3')
    print(f'  Max ecart muNa/muCl aux traces: {np.max(np.abs(np.log(cn_l/c0)+p_l/VT-(np.log(cn_r/c0)+p_r/VT))):.3e} / {np.max(np.abs(np.log(cc_l/c0)-p_l/VT-(np.log(cc_r/c0)-p_r/VT))):.3e}')
    print(f'  Ecart flux reconstruit / V13: Na={err_n:.3e} ({100*err_n/scale_n:.2f}%), Cl={err_c:.3e} ({100*err_c/scale_c:.2f}%) mol/m2/s')
    records.append(dict(interface=f'{reg[j]}/{reg[j+1]}',index=j,
        c_na_left=cn_l,c_na_right=cn_r,c_cl_left=cc_l,c_cl_right=cc_r,
        phi_left=p_l,phi_right=p_r,delta_phi=p_r-p_l,
        flux_na_v13=fn[:,j],flux_cl_v13=fc[:,j],
        flux_na_reconstructed=n_left,flux_cl_reconstructed=c_left))

out=Path('electrodialyse_donnan_v16.npz')
payload={'y':y,'interface_indices':idx,'interfaces':np.array([r['interface'] for r in records])}
for k,r in enumerate(records):
    for key,val in r.items():
        if key not in ('interface','index'):
            payload[f'{key}_{k+1}']=val
np.savez(out,**payload)
print('\nSauvegarde:',out)
print('IMPORTANT: Donnan satisfait par construction pour les traces reconstruites.')
print('Les flux V13 ne sont PAS recalcules avec ces traces: comparer les ecarts ci-dessus.')
print('Pour une V16 couplee, traces et flux devront etre inconnues du meme solveur.')
