#!/usr/bin/env python3
"""V23 -- porosite et charge fixe des membranes, modele periodique Poiseuille.

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
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import spsolve, lsmr
from types import SimpleNamespace
from scipy.sparse import lil_matrix

F=96485.33212
R=8.31446261815324
T=float(os.getenv("ED_T","298.15"))
VT=R*T/F
U=float(os.getenv("ED_VOLTAGE","0.006"))
c0=float(os.getenv("ED_C_INLET","500"))
uy=float(os.getenv("ED_U_AVG",os.getenv("ED_UY","0.005")))
H=float(os.getenv("ED_HEIGHT","0.01"))
nxreg=np.fromstring(os.getenv("ED_NX","3,2,5,2,3"),sep=",",dtype=int)
widths=np.fromstring(os.getenv("ED_WIDTHS","0.001,0.0002,0.001,0.0002,0.001"),sep=",")
ny=int(os.getenv("ED_NY","6"))
Xreg=np.array([0.,float(os.getenv("ED_X_CEM","-1000")),0.,float(os.getenv("ED_X_AEM","1000")),0.])
# Modele effectif V23 : D_eff = epsilon_m * D_bulk dans les membranes.
# Hypothese simplifiee : tortuosite = 1. Les coefficients du tutoriel COMSOL
# peuvent differer ; ce choix n'est PAS une reproduction quantitative validee.
eps_mem=float(os.getenv("ED_MEM_POROSITY","0.15"))
Dna_bulk=float(os.getenv("ED_D_NA","1.33e-9"))
Dcl_bulk=float(os.getenv("ED_D_CL","2.03e-9"))
if not (0 < eps_mem <= 1 and Dna_bulk > 0 and Dcl_bulk > 0):
    raise ValueError("Porosite ou diffusivite invalide")
eps_reg=np.array([1.,eps_mem,1.,eps_mem,1.])
Dna_reg=eps_reg*Dna_bulk
Dcl_reg=eps_reg*Dcl_bulk
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
# Moyenne exacte du profil de Poiseuille sur chaque volume de controle.
# u(x) = 6*u_moy*(x-x_g)*(x_d-x)/(x_d-x_g)^2 dans chaque canal.
# La moyenne sur toute la largeur vaut exactement u_moy.
v=np.zeros(nr)
for rr in (0,2,4):
    mask=np.where(reg==rr)[0]
    left=sum(widths[:rr]); w=widths[rr]
    a=(np.cumsum(dx)-dx)[mask]-left
    b=a+dx[mask]
    # Primitive de 6*u_moy*(t/w)*(1-t/w) :
    # 3*u_moy*t**2/w**1 - 2*u_moy*t**3/w**2.
    primitive=lambda t: uy*(3*t*t/w-2*t*t*t/(w*w))
    v[mask]=(primitive(b)-primitive(a))/(b-a)
if np.any(v< -1e-12):
    raise ValueError('Profil Poiseuille negatif: verifier la geometrie')
v=np.maximum(v,0.)
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
    # Periodicite laterale: meme flux sortant a droite et entrant a gauche.
    # p(x+L)=p(x)+U/VT; concentration periodique.
    fper_na=flux_face(na[:,-1],na[:,0],p[:,-1],p[:,0]+U/VT,
                      Dna[-1],Dna[0],dx[-1]/2,dx[0]/2,+1)
    fper_cl=flux_face(cl[:,-1],cl[:,0],p[:,-1],p[:,0]+U/VT,
                      Dcl[-1],Dcl[0],dx[-1]/2,dx[0]/2,-1)
    divna[:,-1]+=fper_na/dx[-1]; divna[:,0]-=fper_na/dx[0]
    divcl[:,-1]+=fper_cl/dx[-1]; divcl[:,0]-=fper_cl/dx[0]
    for c,D,div,z in [(na,Dna,divna,+1),(cl,Dcl,divcl,-1)]:
        fy=flux_face(c[:-1,:],c[1:,:],p[:-1,:],p[1:,:],
                     D[None,:],D[None,:],dy/2,dy/2,z)+v[None,:]*c[:-1,:]
        div[:-1,:]+=fy/dy; div[1:,:]-=fy/dy
        inlet=v>0
        fin=v[inlet]*c0  # Danckwerts: flux total entrant = u*c0
        div[0,inlet]-=fin/dy
        fout=v[inlet]*c[-1,inlet]
        div[-1,inlet]+=fout/dy
    eq=np.r_[divna.ravel()/DIV_SCALE,divcl.ravel()/DIV_SCALE,
             (info[0]-info[1]).ravel()/FLUX_SCALE,
             (info[2]-info[3]).ravel()/FLUX_SCALE,
             info[4].ravel(),info[5].ravel()]
    # Fixation de jauge: remplace une equation de charge redondante.
    # Somme des equations de charge = 0 car les flux sont periodiques
    # et les entrees/sorties de convection sont electroneutres.
    eq[N] = p[0,0]
    if diagnostics: return na,cl,p,divna,divcl,fna,fcl,info
    return eq

# Matrice de structure du Jacobien: chaque cellule et voisins; traces des faces.
size=2*N+4*M
pat=lil_matrix((size,size),dtype=np.int8)
for j in range(ny):
    for k in range(nr):
        idx=j*nr+k
        neighbors=[idx]
        neighbors.append(j*nr+(k-1)%nr)
        neighbors.append(j*nr+(k+1)%nr)
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
# V18: initialisation par prolongement de la solution d'une grille plus grossiere.
init_file=os.getenv('ED_INIT_NPZ','').strip()
if init_file:
    with np.load(init_file) as prev:
        if not bool(prev['converged']):
            raise ValueError('Initialisation refusee: solution precedente non convergee')
        old_reg=np.asarray(prev['region'],dtype=int)
        old_x=np.asarray(prev['x'],dtype=float)
        old_y=np.asarray(prev['y'],dtype=float)
        old_na=np.asarray(prev['c_na'],dtype=float)
        old_cl=np.asarray(prev['c_cl'],dtype=float)
        old_phi=np.asarray(prev['phi'],dtype=float)
        if 'widths' in prev and not np.allclose(prev['widths'],widths,rtol=0,atol=1e-12):
            raise ValueError('Largeurs incompatibles dans le fichier initial')
        if old_na.shape != (len(old_y),len(old_x)) or ni != 4:
            raise ValueError('Dimensions incompatibles dans le fichier initial')
        def interp_region(values):
            result=np.empty((ny,nr))
            new_y=(np.arange(ny)+0.5)*dy
            for rr in range(1,6):
                ko=np.where(old_reg==rr)[0]
                kn=np.where(reg+1==rr)[0]
                if not len(ko) or not len(kn):
                    raise ValueError('Region manquante dans la solution precedente')
                tmp=np.vstack([np.interp(x[kn],old_x[ko],row[ko]) for row in values])
                for ii,k in enumerate(kn):
                    result[:,k]=np.interp(new_y,old_y,tmp[:,ii])
            return result
        old_s=np.log(np.sqrt(old_na*old_cl)/c0)
        s0=interp_region(old_s)
        p0=interp_region(old_phi/VT)
        def interp_trace(key):
            old=np.asarray(prev[key],dtype=float)
            if old.shape != (len(old_y),ni):
                raise ValueError('Dimensions des traces incompatibles: '+key)
            return np.column_stack([np.interp((np.arange(ny)+0.5)*dy,old_y,old[:,i]) for i in range(ni)])
        aL=interp_trace('trace_na_left'); bL=interp_trace('trace_cl_left')
        aR=interp_trace('trace_na_right'); bR=interp_trace('trace_cl_right')
        sL0=np.log(np.sqrt(aL*bL)/c0)
        sR0=np.log(np.sqrt(aR*bR)/c0)
        pL0=interp_trace('trace_phi_left')/VT
        pR0=interp_trace('trace_phi_right')/VT
        q0=np.r_[s0.ravel(),p0.ravel(),np.stack([sL0,sR0,pL0,pR0]).ravel()]
    print('Initialisation par prolongement:',init_file)
print(f"V23 Newton creux periodique Donnan couplee : {nr} x {ny} cellules, {size} inconnues, {ni} interfaces")
print("Prototype cartesien; periodicite x, entree Danckwerts; Poiseuille; pas de Gmsh.")
print(f"Membranes : porosite={eps_mem:.4g}, DNa_eff={Dna_reg[1]:.6g}, DCl_eff={Dcl_reg[1]:.6g} m2/s")
print(f"Charges fixes : X_CEM={Xreg[1]:.6g}, X_AEM={Xreg[3]:.6g} mol/m3")
print(f"Vitesse moyenne cible : {uy:.6g} m/s; vitesse maximale discrete : {v.max():.6g} m/s")
for rr in (0,2,4):
    kk=np.where(reg==rr)[0]
    print(f"Canal {rr+1}: vitesse moyenne discrete = {np.sum(v[kk]*dx[kk])/widths[rr]:.8g} m/s")
# Newton amorti avec Jacobien creux en differences finies groupees.
# approx_derivative est une API interne SciPy: version testee a controler.
max_steps=int(os.getenv('ED_NEWTON_STEPS','45'))
tol=float(os.getenv('ED_NEWTON_TOL','1e-7'))
q=q0.copy()
pattern=pat.tocsr()
fevals=0
message='Nombre maximal de pas de Newton atteint'
success=False
for iteration in range(max_steps):
    r=residual(q); fevals+=1
    err=float(np.max(np.abs(r)))
    norm=float(np.linalg.norm(r))
    print(f'Newton {iteration:02d}: max|R|={err:.4e}, ||R||={norm:.4e}',flush=True)
    if err<tol:
        success=True; message='Newton converge'; break
    J=approx_derivative(residual,q,method='2-point',sparsity=pattern)
    # Les appels internes de differences finies ne sont pas inclus dans le compteur de pas.
    try:
        step=spsolve(J.tocsc(),-r)
    except Exception as exc:
        message=f'Echec factorisation: {exc}'; break
    if not np.all(np.isfinite(step)):
        message='Pas Newton non fini'; break
    alpha=1.0
    accepted=False
    for trial in range(22):
        candidate=q+alpha*step
        rc=residual(candidate); fevals+=1
        if np.all(np.isfinite(rc)) and np.linalg.norm(rc)<norm*(1-1e-4*alpha):
            q=candidate; accepted=True; break
        alpha*=0.5
    if not accepted:
        # secours pour jacobienne mal conditionnee: Gauss-Newton regularise
        alt=lsmr(J,-r,atol=1e-12,btol=1e-12,maxiter=500)[0]
        alpha=1.
        for trial in range(30):
            candidate=q+alpha*alt
            rc=residual(candidate)
            if np.all(np.isfinite(rc)) and np.linalg.norm(rc)<norm:
                q=candidate; accepted=True; break
            alpha*=.5
        if not accepted:
            message='Recherche lineaire echouee'; break
fit=SimpleNamespace(x=q,success=success,message=message,nfev=iteration+1)
na,cl,p,dna,dcl,fna,fcl,info=residual(fit.x,True)
err=residual(fit.x)
maxerr=np.max(np.abs(err))
print('\n=== V23 MEMBRANES : CONTROLES ===')
print('Arret:',fit.message,'; iterations Newton:',fit.nfev)
print('Residuel global max (normalise):',f'{maxerr:.4e}')
print('Electroneutralite max:',f'{np.max(np.abs(na-cl+X[None,:])):.4e}','mol/m3')
print('Ecart Donnan muNa/muCl:',f'{np.max(np.abs(info[4])):.4e}',f'{np.max(np.abs(info[5])):.4e}')
print('Ecart flux Na/Cl aux interfaces:',f'{np.max(np.abs(info[0]-info[1])):.4e}',f'{np.max(np.abs(info[2]-info[3])):.4e}','mol/m2/s')
print('Concentrations min:',na.min(),cl.min(),'mol/m3')
for t,k in enumerate(interfaces):
    print(f'Interface {reg[k]+1}/{reg[k+1]+1}: '
          f'Delta phi moyen={(info[11][:,t]-info[10][:,t]).mean()*VT*1000:+.5f} mV, '
          f'i_x moyen={np.mean(F*(info[0][:,t]-info[2][:,t])):+.5e} A/m2')
out=Path(os.getenv('ED_OUTPUT_NPZ',str(Path(__file__).resolve().parent/'electrodialyse_membranes_v23.npz')))
np.savez(out,x=x,y=(np.arange(ny)+0.5)*dy,region=reg+1,
         c_na=na,c_cl=cl,phi=p*VT,X=X,
         current_x=F*(fna-fcl),flux_na_x=fna,flux_cl_x=fcl,
         interface_indices=interfaces,trace_na_left=info[6],trace_na_right=info[7],
         trace_cl_left=info[8],trace_cl_right=info[9],
         trace_phi_left=info[10]*VT,trace_phi_right=info[11]*VT,
         residual_max=maxerr,converged=bool(fit.success and maxerr<1e-6),
         solution_vector=fit.x, evaluations=fit.nfev,
         voltage=U, c_inlet=c0, uy=uy, u_avg=uy, velocity_x=v, velocity_profile="poiseuille_cell_average", temperature=T,
         boundary_type='periodic_x_danckwerts_y',
         widths=widths, xreg=Xreg, membrane_porosity=eps_mem,
         epsilon_region=eps_reg, Dna_region=Dna_reg, Dcl_region=Dcl_reg,
         membrane_diffusivity_model="D_eff=epsilon*D_bulk; tortuosity=1")
print('Fichier:',out.name)
if not(fit.success and maxerr<1e-6):
    print('NON VALIDE : resoudre les residus avant interpretation physique.')
else:
    print('Bilans et Donnan couples satisfaits sur cette grille; etude de convergence requise.')
