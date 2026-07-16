import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import deque
import _check as C

def label(m, conn8=True):
    H,W=m.shape; lab=np.zeros((H,W),int); cur=0
    nb=[(-1,0),(1,0),(0,-1),(0,1)]+([(-1,-1),(-1,1),(1,-1),(1,1)] if conn8 else [])
    for i in range(H):
        for j in range(W):
            if m[i,j] and lab[i,j]==0:
                cur+=1;q=deque([(i,j)]);lab[i,j]=cur
                while q:
                    r,c=q.popleft()
                    for dr,dc in nb:
                        nr,nc=r+dr,c+dc
                        if 0<=nr<H and 0<=nc<W and m[nr,nc] and lab[nr,nc]==0:
                            lab[nr,nc]=cur;q.append((nr,nc))
    return lab,cur

def rid(a): return a
def fH(a): return a[:,::-1].copy()
def fV(a): return a[::-1].copy()
def r180(a): return a[::-1,::-1].copy()
def tr(a): return a.T.copy()
def r90(a): return np.rot90(a,-1).copy()
def r270(a): return np.rot90(a,1).copy()
def atr(a): return a[::-1,::-1].T.copy()

def fill_holes(a):
    # background=0; fill enclosed 0-regions (not touching border) with surrounding color
    b=a.copy(); H,W=a.shape
    m=(a==0).astype(int); lab,n=label(m,conn8=False)
    border=set()
    for k in range(1,n+1):
        ys,xs=np.where(lab==k)
        if (ys.min()==0 or ys.max()==H-1 or xs.min()==0 or xs.max()==W-1):
            border.add(k)
    for k in range(1,n+1):
        if k in border: continue
        ys,xs=np.where(lab==k)
        # surrounding color
        nbc=[]
        for r,c in zip(ys,xs):
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=r+dr,c+dc
                if 0<=nr<H and 0<=nc<W and a[nr,nc]!=0: nbc.append(a[nr,nc])
        if nbc:
            col=np.bincount(nbc).argmax()
            for r,c in zip(ys,xs): b[r,c]=col
    return b

def grav(a, axis, sign):
    # move nonzero to one side
    b=np.zeros_like(a); H,W=a.shape
    if axis==0:
        for c in range(W):
            col=a[:,c]; vals=col[col!=0]
            if sign>0: b[H-len(vals):,c]=vals
            else: b[:len(vals),c]=vals
    else:
        for r in range(H):
            row=a[r]; vals=row[row!=0]
            if sign>0: b[r,W-len(vals):]=vals
            else: b[r,:len(vals)]=vals
    return b

TF={'id':rid,'fH':fH,'fV':fV,'r180':r180,'tr':tr,'r90':r90,'r270':r270,'atr':atr,
    'holes':fill_holes,'gravU':lambda a:grav(a,0,-1),'gravD':lambda a:grav(a,0,1),
    'gravL':lambda a:grav(a,1,-1),'gravR':lambda a:grav(a,1,1)}

def best_recolor(t):
    # learn consistent per-color map from train, apply
    data=C.load_all(t); mp={}
    for a,b in data['train']:
        if a.shape!=b.shape: return None
        for ci,co in zip(a.flatten(),b.flatten()):
            if ci in mp and mp[ci]!=co: return None
            mp[ci]=co
    def rule(a):
        out=a.copy()
        for ci,co in mp.items(): out[a==ci]=co
        return out
    return rule

if __name__=="__main__":
    tasks=[int(x) for x in sys.argv[1:]]
    for t in tasks:
        line=f"T{t}: "
        hits=[]
        for name,fn in TF.items():
            try:
                res,_=C.check(t,fn)
                tr_=res.get('train',(0,1)); ag=res.get('arc-gen',(0,1))
                if tr_[0]==tr_[1] and ag[1]>0 and ag[0]==ag[1]:
                    hits.append(f"{name}=FULL")
                elif tr_[0]==tr_[1]:
                    hits.append(f"{name}(train-ok,ag={ag[0]}/{ag[1]})")
            except Exception as e:
                pass
        rc=best_recolor(t)
        if rc:
            res,_=C.check(t,rc); ag=res.get('arc-gen',(0,1)); tr_=res['train']
            if tr_[0]==tr_[1]: hits.append(f"recolor(ag={ag[0]}/{ag[1]})")
        print(line+("; ".join(hits) if hits else "no generic hit"))
