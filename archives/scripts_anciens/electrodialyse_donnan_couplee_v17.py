#!/usr/bin/env python3
"""V17 -- prototype 2D a traces Donnan couplees, independant de Gmsh.

Deux inconnues par cellule: log(cNa) et potentiel phi (en unites RT/F).
cCl=cNa+X est impose algebraiquement (X charge fixe signee).
Flux Na/Cl conservatifs par faces, interfaces via gradients des potentiels
electrochimiques. Convection verticale dans les canaux seulement.

IMPORTANT : maillage cartesien 2D de demonstration, PAS le Gmsh de V11/V12.
Interfaces avec traces propres et egalite des potentiels electrochimiques.
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
    p=q[N:2*N].reshape(ny,nr)
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


# 4 traces inconnues par interface et ligne : sL, sR, pL, pR.
interfaces=np.where(reg[:-1]!=reg[1:])[0]
ni=len(interfaces)
M=ny*ni
FLUX_SCALE=1.33e-9*c0/0.001
DIV_SCALE=FLUX_SCALE/0.001

def concentration(s, fixed):
    prod=c0*np.exp(np.clip(s,-15,15))
    root=np.hypot(fixed,2*prod)
    na=np.where(fixed>=0,2*prod**2/(root+fixed),(root-fixed)/2)
    return na,na+fixed

def interface_data(q,na,cl,p):
    traces=q[2*N:].reshape(4,ny,ni)
    sL,sR,pL,pR=traces
    XL=X[interfaces][None,:]
    XR=X[interfaces+1][None,:]
    aL,bL=concentration(sL,XL)
    aR,bR=concentration(sR,XR)
    # demi-faces: potentiels et concentrations de part et d'autre
    fnaL=flux_face(na[:,interfaces],aL,p[:,interfaces],pL,
                   Dna[interfaces],Dna[interfaces],dx[interfaces]/4,dx[interfaces]/4,+1)
    fnaR=flux_face(aR,na[:,interfaces+1],pR,p[:,interfaces+1],
                   Dna[interfaces+1],Dna[interfaces+1],dx[interfaces+1]/4,dx[interfaces+1]/4,+1)
    fclL=flux_face(cl[:,interfaces],bL,p[:,interfaces],pL,
                   Dcl[interfaces],Dcl[interfaces],dx[interfaces]/4,dx[interfaces]/4,-1)
    fclR=flux_face(bR,cl[:,interfaces+1],pR,p[:,interfaces+1],
                   Dcl[interfaces+1],Dcl[interfaces+1],dx[interfaces+1]/4,dx[interfaces+1]/4,-1)
    # Conditions thermodynamiques de Donnan aux traces
    muNa=np.log(aL/aR)+pL-pR
    muCl=np.log(bL/bR)-pL+pR
    return (fnaL,fnaR,fclL,fclR,muNa,muCl,aL,aR,bL,bR,pL,pR)

def residual(q,diagnostics=False):
    na,cl,p=state(q)
    divna=np.zeros((ny,nr)); divcl=np.zeros((ny,nr))
    fna=flux_face(na[:,:-1],na[:,1:],p[:,:-1],p[:,1:],
                  Dna[:-1],Dna[1:],dx[:-1]/2,dx[1:]/2,+1)
    fcl=flux_face(cl[:,:-1],cl[:,1:],p[:,:-1],p[:,1:],
                  Dcl[:-1],Dcl[1:],dx[:-1]/2,dx[1:]/2,-1)
    # On utilise un flux different pour chaque cote jusqu'a convergence
    fnar=fna.copy(); fclr=fcl.copy()
    info=interface_data(q,na,cl,p)
    fna[:,interfaces]=info[0]; fnar[:,interfaces]=info[1]
    fcl[:,interfaces]=info[2]; fclr[:,interfaces]=info[3]
    divna[:,:-1]+=fna/dx[:-1]; divna[:,1:]-=fnar/dx[1:]
    divcl[:,:-1]+=fcl/dx[:-1]; divcl[:,1:]-=fclr/dx[1:]
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
    for c,D,div in [(na,Dna,divna),(cl,Dcl,divcl)]:
        fy=-D[None,:]*(c[1:,:]-c[:-1,:])/dy+v[None,:]*c[:-1,:]
        div[:-1,:]+=fy/dy; div[1:,:]-=fy/dy
        inlet=v>0
        fin=-D[inlet]*(c[0,inlet]-c0)/(dy/2)+v[inlet]*c0
        div[0,inlet]-=fin/dy
        fout=v[inlet]*c[-1,inlet]
        div[-1,inlet]+=fout/dy
    eq=np.r_[divna.ravel()/DIV_SCALE,divcl.ravel()/DIV_SCALE,
             (info[0]-info[1]).ravel()/FLUX_SCALE,
             (info[2]-info[3]).ravel()/FLUX_SCALE,
             info[4].ravel(),info[5].ravel()]
    if diagnostics: return na,cl,p,divna,divcl,fna,fcl,info
    return eq

# Matrice de structure du Jacobien: chaque cellule et voisins; traces des faces.
size=2*N+4*M
pat=lil_matrix((size,size),dtype=np.int8)
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
                pat[row,col]=1; pat[row,N+col]=1
for j in range(ny):
    for t,k in enumerate(interfaces):
        m=j*ni+t
        trace_cols=[2*N+b*M+m for b in range(4)]
        cell_cols=[j*nr+k,j*nr+k+1,N+j*nr+k,N+j*nr+k+1]
        rows=[j*nr+k,j*nr+k+1,N+j*nr+k,N+j*nr+k+1]
        rows += [2*N+b*M+m for b in range(4)]
        for row in rows:
            for col in trace_cols+cell_cols: pat[row,col]=1

# Initialisation Donnan par continuite de mu a potentiel nul, puis tension.
s0=np.zeros((ny,nr))
p0=np.broadcast_to(np.arcsinh(X[None,:]/(2*c0)) +
                   U/VT*(x[None,:]/sum(widths)),(ny,nr)).copy()
sL0=np.zeros((ny,ni)); sR0=np.zeros((ny,ni))
# trace: meme produit ionique que le reservoir, potentiel de Donnan local
xface=np.cumsum(widths)[:-1]
face_p=U/VT*xface/sum(widths)
pL0=np.broadcast_to(np.arcsinh(X[interfaces][None,:]/(2*c0))+face_p,(ny,ni)).copy()
pR0=np.broadcast_to(np.arcsinh(X[interfaces+1][None,:]/(2*c0))+face_p,(ny,ni)).copy()
q0=np.r_[s0.ravel(),p0.ravel(),np.stack([sL0,sR0,pL0,pR0]).ravel()]
print(f"V17 Donnan couplee : {nr} x {ny} cellules, {size} inconnues, {ni} interfaces")
print("Prototype cartesien; reservoir lateral, convection verticale; pas de Gmsh.")
fit=least_squares(residual,q0,jac_sparsity=pat.tocsr(),method='trf',
                  max_nfev=int(os.getenv('ED_MAX_NFEV','250')),
                  ftol=1e-10,xtol=1e-10,gtol=1e-10,
                  verbose=int(os.getenv('ED_VERBOSE','0')))
na,cl,p,dna,dcl,fna,fcl,info=residual(fit.x,True)
err=residual(fit.x)
maxerr=np.max(np.abs(err))
print('\n=== V17 : CONTROLES ===')
print('Arret:',fit.message,'; evaluations:',fit.nfev)
print('Residuel global max (normalise):',f'{maxerr:.4e}')
print('Electroneutralite max:',f'{np.max(np.abs(na-cl+X[None,:])):.4e}','mol/m3')
print('Ecart Donnan muNa/muCl:',f'{np.max(np.abs(info[4])):.4e}',f'{np.max(np.abs(info[5])):.4e}')
print('Ecart flux Na/Cl aux interfaces:',f'{np.max(np.abs(info[0]-info[1])):.4e}',f'{np.max(np.abs(info[2]-info[3])):.4e}','mol/m2/s')
print('Concentrations min:',na.min(),cl.min(),'mol/m3')
for t,k in enumerate(interfaces):
    print(f'Interface {reg[k]+1}/{reg[k+1]+1}: '
          f'Delta phi moyen={(info[11][:,t]-info[10][:,t]).mean()*VT*1000:+.5f} mV, '
          f'i_x moyen={np.mean(F*(info[0][:,t]-info[2][:,t])):+.5e} A/m2')
out=Path(__file__).resolve().parent/'electrodialyse_donnan_couplee_v17.npz'
np.savez(out,x=x,y=(np.arange(ny)+0.5)*dy,region=reg+1,
         c_na=na,c_cl=cl,phi=p*VT,X=X,
         current_x=F*(fna-fcl),flux_na_x=fna,flux_cl_x=fcl,
         interface_indices=interfaces,trace_na_left=info[6],trace_na_right=info[7],
         trace_cl_left=info[8],trace_cl_right=info[9],
         trace_phi_left=info[10]*VT,trace_phi_right=info[11]*VT,
         residual_max=maxerr,converged=bool(fit.success and maxerr<1e-6))
print('Fichier:',out.name)
if not(fit.success and maxerr<1e-6):
    print('NON VALIDE : resoudre les residus avant interpretation physique.')
else:
    print('Bilans et Donnan couples satisfaits sur cette grille; etude de convergence requise.')
