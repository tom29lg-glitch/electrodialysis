#!/usr/bin/env python3
"""V20: audit independant des flux et bilans du solveur V19.
Ne resout pas un nouveau modele; reconstruit exactement les flux de V19.
"""
import os
from pathlib import Path
import numpy as np

F=96485.33212
R=8.31446261815324
file=Path(os.getenv("ED_INPUT_NPZ","electrodialyse_v19_45x18.npz"))
with np.load(file) as z:
    data={k:np.asarray(z[k]) for k in z.files}
if not bool(data["converged"]):
    raise SystemExit("ERREUR: fichier non converge. Audit refuse.")
na=data["c_na"]; cl=data["c_cl"]; p=data["phi"]/(R*float(data["temperature"])/F)
ny,nx=na.shape
reg=data["region"].astype(int)
widths=data["widths"]
counts=np.array([(reg==i).sum() for i in range(1,6)])
dx=np.repeat(widths/counts,counts)
H=float(os.getenv("ED_HEIGHT","0.01"))
dy=H/ny
T=float(data["temperature"]); VT=R*T/F
U=float(data["voltage"]); c0=float(data["c_inlet"]); uy=float(data["uy"])
Dna=np.array([1.33e-9,1.33e-9,1.33e-9,1.33e-11,1.33e-9])[reg-1]
Dcl=np.array([2.03e-9,2.03e-11,2.03e-9,2.03e-9,2.03e-9])[reg-1]
v=np.where(np.isin(reg,[1,3,5]),uy,0.)
ii=data["interface_indices"].astype(int)
def face(a,b,pa,pb,Da,Db,ha,hb,z):
    delta=np.log(b/a)
    lm=np.where(abs(delta)<1e-7,0.5*(a+b),(b-a)/np.where(abs(delta)<1e-7,1.,delta))
    return -lm*(delta+z*(pb-pa))/(ha/Da+hb/Db)
def fluxes(c,D,z,left_key,right_key,phiL,phiR):
    fx=face(c[:,:-1],c[:,1:],p[:,:-1],p[:,1:],D[:-1],D[1:],dx[:-1]/2,dx[1:]/2,z)
    tl=data[left_key]; tr=data[right_key]
    fl=face(c[:,ii],tl,p[:,ii],phiL,D[ii],D[ii],dx[ii]/4,dx[ii]/4,z)
    fr=face(tr,c[:,ii+1],phiR,p[:,ii+1],D[ii+1],D[ii+1],dx[ii+1]/4,dx[ii+1]/4,z)
    fx[:,ii]=fl
    # y faces: advective upstream and diffusive; only in channels v>0
    fy=-D[None,:]*(c[1:,:]-c[:-1,:])/dy+v[None,:]*c[:-1,:]
    fin=np.zeros(nx); fout=np.zeros(nx)
    mask=v>0
    fin[mask]=-D[mask]*(c[0,mask]-c0)/(dy/2)+v[mask]*c0
    fout[mask]=v[mask]*c[-1,mask]
    left=face(c0,c[:,0],0.,p[:,0],D[0],D[0],dx[0]/2,dx[0]/2,z)
    right=face(c[:,-1],c0,p[:,-1],U/VT,D[-1],D[-1],dx[-1]/2,dx[-1]/2,z)
    # finite volume divergence flux per cell (mol/m3/s)
    div=np.zeros_like(c)
    div[:,:-1]+=fx/dx[:-1]
    div[:,1:]-=fx/dx[1:]
    # Right interface flux differs before convergence; use right-side value.
    div[:,ii+1]+=(fl-fr)/dx[ii+1]
    div[:,0]-=left/dx[0]
    div[:,-1]+=right/dx[-1]
    div[:-1,:]+=fy/dy
    div[1:,:]-=fy/dy
    div[0,:]-=fin/dy
    div[-1,:]+=fout/dy
    return dict(fx=fx,fy=fy,fin=fin,fout=fout,left=left,right=right,div=div,
                interface_left=fl,interface_right=fr)
phiL=data["trace_phi_left"]/VT
phiR=data["trace_phi_right"]/VT
n=fluxes(na,Dna,+1,"trace_na_left","trace_na_right",phiL,phiR)
c=fluxes(cl,Dcl,-1,"trace_cl_left","trace_cl_right",phiL,phiR)
charge=na-cl+data["X"][None,:]
Jx=F*(n["fx"]-c["fx"])
Jy=F*(n["fy"]-c["fy"])
Jin=F*(n["fin"]-c["fin"])
Jout=F*(n["fout"]-c["fout"])
Jleft=F*(n["left"]-c["left"])
Jright=F*(n["right"]-c["right"])
def integrated(flux):
    return float(np.sum(flux)*dy)
def species_report(label,d):
    left=-integrated(d["left"]); right=integrated(d["right"])
    bottom=-float(np.sum(d["fin"]*dx)); top=float(np.sum(d["fout"]*dx))
    boundary=left+right+bottom+top
    vol=float(np.sum(d["div"]*dx[None,:])*dy)
    return boundary,vol
print("=== V20 : AUDIT PHYSIQUE ET CONSERVATION ===")
print("Source:",file)
print(f"Grille: {nx} x {ny}; U={U:.6g} V; uy={uy:.3g} m/s; H={H:.4g} m")
print(f"Residuel V19: {float(data['residual_max']):.4e} (normalise)")
print(f"Max electroneutralite: {np.max(np.abs(charge)):.5e} mol/m3")
for name,d in (("Na+",n),("Cl-",c)):
    balance,vol=species_report(name,d)
    print(f"{name}: max divergence locale={np.max(abs(d['div'])):.6e} mol/m3/s")
    print(f"  bilan global flux sortant={balance:+.8e} mol/(s.m), integrale div={vol:+.8e}")
    print(f"  faces x: gauche={integrated(d['left']):+.8e}, droite={integrated(d['right']):+.8e} mol/(s.m)")
    print(f"  faces y: bas={np.sum(d['fin']*dx):+.8e}, haut={np.sum(d['fout']*dx):+.8e} mol/(s.m)")
print("Courant electrique:")
print(f"  max |div J|={np.max(abs(F*(n['div']-c['div']))):.6e} A/m3")
print(f"  I_x gauche={integrated(Jleft):+.8e} A/m; droite={integrated(Jright):+.8e} A/m")
print(f"  I_y bas={np.sum(Jin*dx):+.8e} A/m; haut={np.sum(Jout*dx):+.8e} A/m")
print(f"  bilan I sortant={-integrated(Jleft)+integrated(Jright)-np.sum(Jin*dx)+np.sum(Jout*dx):+.8e} A/m")
print("Courants moyens aux interfaces:")
for t,k in enumerate(ii):
    il=F*(n["interface_left"][:,t]-c["interface_left"][:,t])
    ir=F*(n["interface_right"][:,t]-c["interface_right"][:,t])
    print(f"  {reg[k]}/{reg[k+1]}: moyenne={il.mean():+.8f}, min={il.min():+.8f}, max={il.max():+.8f}, ecart faces={np.max(abs(il-ir)):.3e} A/m2")
print("Bilans integres par region (Na, Cl, courant), mol/(s.m) et A/m:")
for rr in range(1,6):
    m=reg==rr
    bn=float(np.sum(n["div"][:,m]*dx[None,m])*dy)
    bc=float(np.sum(c["div"][:,m]*dx[None,m])*dy)
    print(f"  region {rr}: Na={bn:+.4e}, Cl={bc:+.4e}, I={F*(bn-bc):+.4e}")
print("NOTE: i_x moyen CEM et AEM peuvent differer si courant y entre/sort et/ou")
print("si des courants lateraux et frontieres compensent. Ce bilan n'etablit pas")
print("l'exactitude des conditions aux limites ni la validite du modele physique.")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=Path(os.getenv("ED_V20_PLOT","diagnostic_physique_v20.png"))
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    xx=data["x"]*1000; yy=data["y"]*1000
    for ax,field,title in zip(axes.ravel(),(na,cl,data["phi"]*1000,F*(n["div"]-c["div"])),
                              ("Na+ (mol/m3)","Cl- (mol/m3)","Potentiel (mV)","div J (A/m3)")):
        im=ax.pcolormesh(xx,yy,field,shading="nearest")
        fig.colorbar(im,ax=ax)
        ax.set(title=title,xlabel="x (mm)",ylabel="y (mm)")
        for k in ii:
            ax.axvline(np.sum(dx[:k+1])*1000,color="k",lw=.5,ls="--")
    fig.savefig(out,dpi=170)
    plt.close(fig)
    print("Figure:",out)
except ImportError:
    print("Matplotlib indisponible: figure non generee.")
