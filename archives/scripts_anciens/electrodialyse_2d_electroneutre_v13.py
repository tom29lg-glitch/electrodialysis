#!/usr/bin/env python3
"""V13 -- prototype volumes finis 2D electroneutre, independant de Gmsh.

Deux inconnues par cellule: log(cNa) et potentiel phi (en unites RT/F).
cCl=cNa+X est impose algebraiquement (X charge fixe signee).
Flux Na/Cl conservatifs par faces, interfaces via gradients des potentiels
electrochimiques. Convection verticale dans les canaux seulement.

IMPORTANT : maillage cartesien 2D de demonstration, PAS le Gmsh de V11/V12.
Conditions de Donnan ideales approchees par une resistance de face finie.
Validation necessaire : raffinement, bilan et comparaison 1D/COMSOL.
"""
import os
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

F=96485.33212
R=8.31446261815324
T=float(os.getenv("ED_T","298.15"))
VT=R*T/F
U=float(os.getenv("ED_VOLTAGE","0.006"))
c0=float(os.getenv("ED_C_INLET","500"))
uy=float(os.getenv("ED_UY","1e-6"))
H=float(os.getenv("ED_HEIGHT","0.01"))
nxreg=np.fromstring(os.getenv("ED_NX","3,2,5,2,3"),sep=",",dtype=int)
widths=np.fromstring(os.getenv("ED_WIDTHS","0.001,0.0002,0.001,0.0002,0.001"),sep=",")
ny=int(os.getenv("ED_NY","6"))
Xreg=np.array([0.,float(os.getenv("ED_X_CEM","-1000")),0.,float(os.getenv("ED_X_AEM","1000")),0.])
Dna_reg=np.array([1.33e-9,1.33e-9,1.33e-9,1.33e-11,1.33e-9])
Dcl_reg=np.array([2.03e-9,2.03e-11,2.03e-9,2.03e-9,2.03e-9])
if (len(nxreg)!=5 or len(widths)!=5 or np.any(nxreg<1) or
    np.any(widths<=0) or ny<2 or H<=0 or T<=0 or c0<=0 or uy<0):
    raise ValueError("Parametres de grille invalides")
nr=nxreg.sum()
reg=np.repeat(np.arange(5),nxreg)
dx=np.repeat(widths/nxreg,nxreg)
x=np.cumsum(dx)-dx/2
dy=H/ny
X=Xreg[reg]
Dna=Dna_reg[reg]
Dcl=Dcl_reg[reg]
v=np.where(np.isin(reg,[0,2,4]),uy,0.)
N=nr*ny
# variables: a=log(cNa / c0), p=phi / VT.
# cCl=cNa+X. Parametrisation positive: cNa=0.5*(sqrt(X^2+4 exp(2s))-X)
# avec s=log(sqrt(cNa*cCl)); ainsi cNa,cCl>0 pour tout s.
def state(q):
    s=q[:N].reshape(ny,nr)
    p=q[N:].reshape(ny,nr)
    prod=c0*np.exp(np.clip(s,-15,15))
    root=np.hypot(X[None,:],2*prod)
    na=np.where(X[None,:]>=0,2*prod**2/(root+X[None,:]),(root-X[None,:])/2)
    cl=na+X[None,:]
    return na,cl,p

def flux_face(cL,cR,pL,pR,DL,DR,hL,hR,z):
    # D moyenne harmonique par distance; mobilite log moyenne,
    # potentiel electrochimique mu=ln(c)+z*p.
    d=hL+hR
    resistance=hL/DL+hR/DR
    dc=np.log(cR/cL)
    lm=np.where(np.abs(dc)<1e-7,0.5*(cL+cR),
                (cR-cL)/np.where(np.abs(dc)<1e-7,1.,dc))
    return -lm*(np.log(cR/cL)+z*(pR-pL))/resistance

def residual(q, diagnostics=False):
    na,cl,p=state(q)
    divna=np.zeros((ny,nr))
    divcl=np.zeros((ny,nr))
    # flux x interne, quantites en mol/m2/s
    fna=flux_face(na[:,:-1],na[:,1:],p[:,:-1],p[:,1:],
                  Dna[:-1],Dna[1:],dx[:-1]/2,dx[1:]/2,+1)
    fcl=flux_face(cl[:,:-1],cl[:,1:],p[:,:-1],p[:,1:],
                  Dcl[:-1],Dcl[1:],dx[:-1]/2,dx[1:]/2,-1)
    divna[:,:-1]+=fna/dx[:-1]; divna[:,1:]-=fna/dx[1:]
    divcl[:,:-1]+=fcl/dx[:-1]; divcl[:,1:]-=fcl/dx[1:]
    # reservoir gauche / droite (potentiel impose, concentrations c0)
    for k,pp in [(0,0.),(-1,U/VT)]:
        d=dx[k]/2
        if k==0:
            an=flux_face(c0,na[:,0],pp,p[:,0],Dna[0],Dna[0],d,d,+1)
            ac=flux_face(c0,cl[:,0],pp,p[:,0],Dcl[0],Dcl[0],d,d,-1)
            divna[:,0]-=an/dx[0]; divcl[:,0]-=ac/dx[0]
        else:
            an=flux_face(na[:,-1],c0,p[:,-1],pp,Dna[-1],Dna[-1],d,d,+1)
            ac=flux_face(cl[:,-1],c0,p[:,-1],pp,Dcl[-1],Dcl[-1],d,d,-1)
            divna[:,-1]+=an/dx[-1]; divcl[:,-1]+=ac/dx[-1]
    # diffusion y, convection amont (v >= 0)
    for c,D,div in [(na,Dna,divna),(cl,Dcl,divcl)]:
        fy= -D[None,:]*(c[1:,:]-c[:-1,:])/dy + v[None,:]*c[:-1,:]
        div[:-1,:]+=fy/dy; div[1:,:]-=fy/dy
        # entree y=0: concentration c0 dans les canaux, flux advectif
        inlet=(v>0)
        fin=-D[inlet]*(c[0,inlet]-c0)/(dy/2)+v[inlet]*c0
        div[0,inlet]-=fin/dy
        # sortie y=H: zero gradient diffusif; sortie advective
        fout=v[inlet]*c[-1,inlet]
        div[-1,inlet]+=fout/dy
    # echelle des residus: flux caracteristique D*c / 1mm
    scale=1.33e-9*c0/0.001**2
    if diagnostics:
        return na,cl,p,divna,divcl,fna,fcl
    return np.r_[divna.ravel(),divcl.ravel()]/scale

# dependance locale: cellule et quatre voisins, les deux variables.
pat=lil_matrix((2*N,2*N),dtype=np.int8)
for j in range(ny):
    for k in range(nr):
        idx=j*nr+k
        neighbors=[idx]
        if k: neighbors.append(idx-1)
        if k<nr-1: neighbors.append(idx+1)
        if j: neighbors.append(idx-nr)
        if j<ny-1: neighbors.append(idx+nr)
        for row in (idx,N+idx):
            for col in neighbors:
                pat[row,col]=1
                pat[row,N+col]=1

# Equilibre de Donnan initial: produit des concentrations c0^2,
# potentiel choisi pour continuite electrochimique a tension nulle.
s0=np.zeros((ny,nr))
p0=np.zeros((ny,nr))
p0[:,:]=np.arcsinh(X[None,:]/(2*c0))
p0+=U/VT*(x[None,:]/sum(widths))
q0=np.r_[s0.ravel(),p0.ravel()]
print(f"V13 prototype 2D: {nr} x {ny} cellules, {2*N} inconnues")
print("Attention: grille cartesienne independante du Gmsh, sans SUPG ni periodicite.")
fit=least_squares(residual,q0,jac_sparsity=pat.tocsr(),method="trf",
                  max_nfev=int(os.getenv("ED_MAX_NFEV","80")),
                  ftol=1e-9,xtol=1e-9,gtol=1e-9,
                  verbose=int(os.getenv("ED_VERBOSE","0")))
na,cl,p,dna,dcl,fna,fcl=residual(fit.x,True)
maxres=np.max(np.abs(residual(fit.x)))
charge=np.max(np.abs(na-cl+X[None,:]))
current=F*(fna-fcl)
print("\n=== V13 : PROTOTYPE ELECTRONEUTRE 2D ===")
print("Optimisation:",fit.message)
print("Iterations solveur:",fit.nfev,"; residu relatif max:",f"{maxres:.4e}")
print("Concentrations Na:",na.min(),na.max(),"Cl:",cl.min(),cl.max(),"mol/m3")
print("Max |cNa-cCl+X|:",f"{charge:.4e}","mol/m3")
print("Courant aux faces x internes (A/m2):",current.min(),current.max())
for k in np.cumsum(nxreg)[:-1]-1:
    print(f"Interface {reg[k]+1}/{reg[k+1]+1} : i_x(y) min={current[:,k].min():+.5e}, max={current[:,k].max():+.5e}")
out=Path(__file__).resolve().parent/"electrodialyse_electroneutre_v13.npz"
np.savez(out,x=x,y=(np.arange(ny)+0.5)*dy,region=reg+1,
         c_na=na,c_cl=cl,phi=p*VT,X=X,flux_na_x=fna,flux_cl_x=fcl,
         current_x=current,residual_max=maxres,charge_error=charge,
         converged=bool(fit.success and maxres<1e-4))
print("Fichier:",out.name)
if maxres>=1e-4:
    print("NON VALIDE: bilan local insuffisant. Ne pas interpreter les courants.")
else:
    print("Bilan local sous tolerance; raffinement et comparaison restent necessaires.")
