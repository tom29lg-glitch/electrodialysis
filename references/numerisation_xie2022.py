#!/usr/bin/env python3
"""Numerisation des figures 3 a 5 de Xie et al. 2022 (Water Sci. Technol. 86(6) 1613) :
C_Cs(t) mesuree (carres noirs) et modele de Xie (points rouges), groupes A (courant), B (debit), C (rapport de volumes).

Pages rendues a 300 dpi : pdftoppm -r 300 -f 10 -l 11 -png wst086061613.pdf h
Calibration de chaque panneau sur ses graduations (exterieures) ; marqueurs detectes par couleur.
Nettoyage (fait ensuite) : point initial pris sur le modele de Xie (meme point de depart, cache sous le
marqueur rouge) ; points qui remontent ecartes (texte "r = ..., P < 0,01" du graphique).
Controle : le flux moyen de Cs recalcule a partir des points redonne le tableau 3 (A2 : 6,37 contre 6,39e-6 mol/s/m2).
Sortie finale : xie2022_cs_numerise.json.
"""
import numpy as np
from PIL import Image
def load(f): return np.array(Image.open(f).convert('RGB')).astype(int)
def frame(a):
    dark=(a.mean(2)<150)
    cols=np.where(dark.mean(0)>0.6)[0]; rows=np.where(dark.mean(1)>0.6)[0]
    return cols.min(), cols.max(), rows.min(), rows.max(), dark
def ticks(a):
    xL,xR,yT,yB,dark=frame(a)
    left=[r for r in range(yT,yB+1) if dark[r,xL-7:xL-1].sum()>=4]
    bot=[c for c in range(xL,xR+1) if dark[yB+2:yB+8,c].sum()>=4]
    def merge(v):
        out=[]
        for t in v:
            if out and t-out[-1][-1]<=2: out[-1].append(t)
            else: out.append([t])
        return [float(np.mean(o)) for o in out]
    return (xL,xR,yT,yB),merge(left),merge(bot)
import numpy as np, json
from PIL import Image
from scipy import ndimage
pages={10:Image.open('h-10.png'),11:Image.open('h-11.png')}
W,H=pages[10].size
P={ # nom: (page, crop fractions, y tick values, x tick values (1e3 s))
 'A1':(10,(0.10,0.43,0.37,0.62),[1,2,3,4,5],[0,1,2,3,4]),
 'A2':(10,(0.355,0.43,0.62,0.62),[1,2,3,4,5],[0,1,2,3]),
 'A3':(10,(0.615,0.43,0.88,0.62),[1,2,3,4,5],[0,1,2,3]),
 'B1':(11,(0.12,0.105,0.39,0.28),[1,2,3,4,5],[0,1,2,3]),
 'B2':(11,(0.395,0.105,0.655,0.28),[1,2,3,4,5,6],[0,1,2,3]),
 'B3':(11,(0.662,0.105,0.93,0.28),[0,1,2,3,4,5,6],[0,1,2,3]),
 'C1':(11,(0.12,0.695,0.40,0.875),[1,2,3,4,5],[0,1,2,3]),
 'C2':(11,(0.395,0.695,0.67,0.875),[1,2,3,4,5,6],[0,1,2,3,4,5]),
 'C3':(11,(0.665,0.695,0.93,0.875),[1,2,3,4,5],[-1,0,1,2,3,4,5,6]),
}
out={}
for name,(pg,(x0,y0,x1,y1),yt,xt) in P.items():
    im=pages[pg].crop((int(x0*W),int(y0*H),int(x1*W),int(y1*H))); im.save(f'{name}.png')
    a=np.array(im.convert('RGB')).astype(int)
    (xL,xR,yT,yB),lt,bt=ticks(a)
    msg=f'{name}: frame {xL},{xR},{yT},{yB} ; ticks y {len(lt)} (attendu {len(yt)}), x {len(bt)} (attendu {len(xt)})'
    if len(lt)!=len(yt) or len(bt)!=len(xt):
        print(msg,' <-- ECHEC', lt, bt); continue
    fy=np.polyfit(lt,yt[::-1],1); fx=np.polyfit(bt,xt,1)
    g=a.mean(2); neutral=(a.max(2)-a.min(2))<30
    m=(g<120)&neutral
    m[:, :xL+3]=False; m[:, xR-2:]=False; m[:yT+3,:]=False; m[yB-2:,:]=False
    m=ndimage.binary_opening(m,structure=np.ones((5,5)))
    lab,n=ndimage.label(m)
    pts=[]
    for k in range(1,n+1):
        ys,xs=np.where(lab==k)
        if len(xs)<12: continue
        pts.append((np.polyval(fx,xs.mean()),np.polyval(fy,ys.mean())))
    pts=sorted(pts)
    c0=pts[0][1]
    pts=[p for p in pts if not (p[0]>1.1 and p[1]>0.85*c0)]   # marqueurs de legende
    r_=(a[...,0]>170)&(a[...,1]<120)&(a[...,2]<120)
    r_[:, :xL+3]=False; r_[:, xR-2:]=False; r_[:yT+3,:]=False; r_[yB-2:,:]=False
    r_=ndimage.binary_opening(r_,structure=np.ones((4,4)))
    lab2,n2=ndimage.label(r_)
    red=[]
    for k in range(1,n2+1):
        ys,xs=np.where(lab2==k)
        if len(xs)<10: continue
        red.append((np.polyval(fx,xs.mean()),np.polyval(fy,ys.mean())))
    red=sorted(red); red=[p for p in red if not (p[0]>1.1 and p[1]>0.85*red[0][1])]
    out[name]={'experimental':pts,'modele_xie':red}
    print(msg, f'-> exp {len(pts)} pts, modele {len(red)} pts ; C0 modele = {red[0][1]:.2f} a t={red[0][0]:.2f} ; exp :', ' '.join(f'({t:.2f},{c:.2f})' for t,c in pts))
json.dump(out,open('xie_points.json','w'),indent=1)
