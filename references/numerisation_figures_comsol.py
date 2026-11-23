from PIL import Image; import numpy as np
def load(f): return np.array(Image.open(f).convert('RGB')).astype(int)
def grid(a,box):
    x0,y0,x1,y1=box; s=a[y0:y1,x0:x1]
    g=(np.abs(s-s.mean(2,keepdims=True)).max(2)<12)&(s.mean(2)>190)&(s.mean(2)<240)
    cols=np.where(g.mean(0)>0.5)[0]+x0; rows=np.where(g.mean(1)>0.5)[0]+y0
    def merge(v):
        out=[];
        for t in v:
            if out and t-out[-1][-1]<=2: out[-1].append(t)
            else: out.append([t])
        return [np.mean(o) for o in out]
    return merge(cols),merge(rows)
if __name__=='__main__':
    import sys
    a=load(sys.argv[1]); box=list(map(int,sys.argv[2:6]))
    c,r=grid(a,box); print('cols',np.round(c,1)); print('rows',np.round(r,1))
import numpy as np, json
from dig import load, grid
def mask(s,color):
    r,g,b=s[...,0],s[...,1],s[...,2]
    if color=='blue':  return (b>r+80)&(b>g+40)
    if color=='green': return (g>r+50)&(g>b+30)
    if color=='red':   return (r>g+80)&(r>b+80)
    if color=='gray':  return (np.abs(r-g)<15)&(np.abs(g-b)<15)&(r<170)&(r>60)
def curve(a,box,xt,yt,color,step=1,excl=None):
    x0,y0,x1,y1=box
    c,rw=grid(a,box)
    def fit(p,t):
        p=np.array(p); sp=np.median(np.diff(p)); idx=np.round((p-p[0])/sp).astype(int)
        assert idx[-1]==len(t)-1,(idx,len(t))
        return np.polyfit(p,np.array(t)[idx],1)
    fx=fit(c,xt); fy=fit(rw,yt)
    s=a[y0:y1,x0:x1]; m=mask(s,color)
    if excl:
        ex0,ey0,ex1,ey1=excl; m[max(ey0-y0,0):ey1-y0,max(ex0-x0,0):ex1-x0]=False
    pts=[]
    for j in range(0,m.shape[1],step):
        ii=np.where(m[:,j])[0]
        if len(ii)==0: continue
        # segments verticaux separes (sauts) : garder chaque groupe
        groups=np.split(ii,np.where(np.diff(ii)>3)[0]+1)
        for gq in groups:
            pts.append((np.polyval(fx,j+x0),np.polyval(fy,gq.mean()+y0)))
    return np.array(pts)
X5=list(np.arange(0,0.00251,0.0005))
out={}
a=load('hi-08.png'); out['fig5_phi']=curve(a,(400,250,1320,950),X5,list(np.arange(1.5,-0.01,-0.1)),'blue')
a=load('hi-09.png'); box=(400,320,1300,1010)
out['fig6_na']=curve(a,box,X5,list(np.arange(1400,99,-100)),'blue',excl=(1130,350,1280,450))
out['fig6_cl']=curve(a,box,X5,list(np.arange(1400,99,-100)),'green',excl=(1130,350,1280,450))
a=load('hi-10.png'); box=(430,300,1290,925); yt=list(np.round(np.arange(0.006,-0.0161,-0.002),4))
for k,c in [('tot','blue'),('mig','green'),('dif','red')]: out['fig7_'+k]=curve(a,box,X5,yt,c,excl=(450,320,660,445))
box=(430,1140,1290,1775); yt=list(np.round(np.arange(0.016,-0.0041,-0.002),4))
for k,c in [('tot','blue'),('mig','green'),('dif','red')]: out['fig8_'+k]=curve(a,box,X5,yt,c,excl=(445,1175,660,1300))
a=load('hi-11.png'); box=(430,505,1300,1135); xt=list(np.arange(0,0.01001,0.002)); yt=list(np.arange(-1200,-1751,-50))
E=(1060,530,1275,660); out['fig9_500']=curve(a,box,xt,yt,'green',excl=E); out['fig9_1000']=curve(a,box,xt,yt,'red',excl=E); out['fig9_0']=curve(a,box,xt,yt,'blue',excl=E)
for k,v in out.items(): print(k,len(v), v[:2].round(5).tolist() if len(v) else '')
np.savez('/tmp/cpdf/comsol_digitized.npz',**out)
