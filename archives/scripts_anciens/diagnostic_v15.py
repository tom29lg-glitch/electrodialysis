#!/usr/bin/env python3
"""V15: bilans integres et diagnostics d'interfaces du prototype V13.

A executer avec les memes ED_* que V13. Aucun nouveau solveur.
Le calcul reprend exactement les flux du script V13, y compris sa
convention actuelle pour les faces de reservoir x (a revisiter en V16).
"""
import os
from pathlib import Path
import numpy as np

F = 96485.33212
R = 8.31446261815324
T = float(os.getenv('ED_T', '298.15'))
VT = R*T/F
U = float(os.getenv('ED_VOLTAGE', '0.006'))
c0 = float(os.getenv('ED_C_INLET', '500'))
uy = float(os.getenv('ED_UY', '1e-6'))
H = float(os.getenv('ED_HEIGHT', '0.01'))
nxreg = np.fromstring(os.getenv('ED_NX', '3,2,5,2,3'), sep=',', dtype=int)
widths = np.fromstring(os.getenv('ED_WIDTHS', '0.001,0.0002,0.001,0.0002,0.001'), sep=',')
source = Path(os.getenv('ED_V15_INPUT', 'electrodialyse_electroneutre_v13.npz'))
if len(nxreg)!=5 or len(widths)!=5:
    raise ValueError('ED_NX et ED_WIDTHS doivent avoir cinq valeurs')
with np.load(source) as data:
    na,cl,phi = (data[k].copy() for k in ('c_na','c_cl','phi'))
    region = data['region'].copy()
    x = data['x'].copy()
    y = data['y'].copy()
    X = data['X'].copy()
    old_res = float(data['residual_max'])
ny,nx = na.shape
if nx != sum(nxreg) or region.shape!=(nx,) or len(y)!=ny:
    raise ValueError('La grille du fichier ne correspond pas a ED_NX/ED_NY. Relancer avec les memes ED_*.')
if not np.array_equal(region, np.repeat(np.arange(1,6),nxreg)):
    raise ValueError('Ordre des regions inattendu')
dx = np.repeat(widths/nxreg,nxreg)
dy = H/ny
if not np.allclose(x, np.cumsum(dx)-dx/2) or not np.allclose(y,(np.arange(ny)+.5)*dy):
    raise ValueError('Dimensions incompatibles: verifier ED_WIDTHS et ED_HEIGHT')
Dna=np.array([1.33e-9,1.33e-9,1.33e-9,1.33e-11,1.33e-9])[region-1]
Dcl=np.array([2.03e-9,2.03e-11,2.03e-9,2.03e-9,2.03e-9])[region-1]
v=np.where(np.isin(region,[1,3,5]),uy,0.)
p=phi/VT

def face(cL,cR,pL,pR,DL,DR,hL,hR,z):
    delta=np.log(cR/cL)
    logmean=np.where(np.abs(delta)<1e-7,.5*(cL+cR),
                     (cR-cL)/np.where(np.abs(delta)<1e-7,1.,delta))
    return -logmean*(delta+z*(pR-pL))/(hL/DL+hR/DR)

def calculate(c,D,z):
    fx=np.empty((ny,nx+1))
    fx[:,1:-1]=face(c[:,:-1],c[:,1:],p[:,:-1],p[:,1:],
                    D[:-1],D[1:],dx[:-1]/2,dx[1:]/2,z)
    # Convention V13: hL=hR=dx/2, meme pour un reservoir sur le bord.
    fx[:,0]=face(c0,c[:,0],0.,p[:,0],D[0],D[0],dx[0]/2,dx[0]/2,z)
    fx[:,-1]=face(c[:,-1],c0,p[:,-1],U/VT,D[-1],D[-1],dx[-1]/2,dx[-1]/2,z)
    fy=np.zeros((ny+1,nx))
    fy[1:-1,:]=-D[None,:]*(c[1:,:]-c[:-1,:])/dy + v[None,:]*c[:-1,:]
    active=v>0
    fy[0,active]=-D[active]*(c[0,active]-c0)/(dy/2)+v[active]*c0
    fy[-1,active]=v[active]*c[-1,active]
    # Les membranes sont impermeables sur les faces y externes.
    divergence=(fx[:,1:]-fx[:,:-1])/dx[None,:]+(fy[1:,:]-fy[:-1,:])/dy
    return fx,fy,divergence

fna,gna,dna=calculate(na,Dna,1)
fcl,gcl,dcl=calculate(cl,Dcl,-1)
ix=F*(fna-fcl)
iy=F*(gna-gcl)
print('=== V15 : DIAGNOSTIC DES BILANS ===')
print('Fichier:',source)
print('Maillage:',nx,'x',ny,'; tension:',U,'V ; vitesse:',uy,'m/s')
print('Electroneutralite max:',f'{np.max(np.abs(na-cl+X[None,:])):.6e}','mol/m3')
print('Residuel V13 sauvegarde:',f'{old_res:.6e}')
scale=1.33e-9*c0/0.001**2
print('Residuel reconstruit Na / Cl:',f'{np.max(np.abs(dna))/scale:.6e}',
      f'{np.max(np.abs(dcl))/scale:.6e}')
print('Divergence courant max:',f'{np.max(np.abs(F*(dna-dcl))):.6e}','A/m3')
print('\nBILANS INTEGRES par domaine (mol/s par metre de profondeur):')
print('Region  Ion   x_gauche      x_droite      y_entree       y_sortie       somme')
for r in range(1,6):
    ids=np.flatnonzero(region==r)
    a,b=ids[0],ids[-1]+1
    for ion,fx,fy in [('Na ',fna,gna),('Cl ',fcl,gcl)]:
        L=-np.sum(fx[:,a])*dy
        RR=np.sum(fx[:,b])*dy
        B=-np.sum(fy[0,a:b]*dx[a:b])
        TOP=np.sum(fy[-1,a:b]*dx[a:b])
        total=L+RR+B+TOP
        print(f'{r:>6} {ion} {L:+.5e} {RR:+.5e} {B:+.5e} {TOP:+.5e} {total:+.5e}')
print('\nCOURANT: bilans par domaine (A par metre de profondeur):')
for r in range(1,6):
    ids=np.flatnonzero(region==r); a,b=ids[0],ids[-1]+1
    terms=np.array([-np.sum(ix[:,a])*dy,np.sum(ix[:,b])*dy,
                    -np.sum(iy[0,a:b]*dx[a:b]),np.sum(iy[-1,a:b]*dx[a:b])])
    print(f'ID {r}: gauche={terms[0]:+.5e}, droite={terms[1]:+.5e}, '
          f'bas={terms[2]:+.5e}, haut={terms[3]:+.5e}, somme={sum(terms):+.5e}')
print('\nINTERFACES: les flux sont UNIQUES par face -> conservation discrete exacte.')
print('Les differences de mu suivantes sont entre CENTRES voisins, PAS des traces Donnan.')
for j in np.flatnonzero(region[:-1]!=region[1:]):
    mu_na=np.log(na[:,j+1]/na[:,j])+(p[:,j+1]-p[:,j])
    mu_cl=np.log(cl[:,j+1]/cl[:,j])-(p[:,j+1]-p[:,j])
    print(f'{region[j]}/{region[j+1]}: <i_x>={np.mean(ix[:,j+1]):+.6e} A/m2, '
          f'max|delta_muNa|={np.max(np.abs(mu_na)):.5e}, '
          f'max|delta_muCl|={np.max(np.abs(mu_cl)):.5e}')
print('\nATTENTION: ce diagnostic ne valide pas les conditions exactes de Donnan.')
print('ATTENTION: flux aux reservoirs et conditions verticales restent ceux de V13.')
