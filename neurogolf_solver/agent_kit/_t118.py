import numpy as np, json,sys,os
sys.path.insert(0,'.');sys.path.insert(0,'..')
from ng_utils_shim import tasks_dir
from collections import deque
import _check as C

def label(m,conn8):
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
                        if 0<=nr<H and 0<=nc<W and m[nr,nc] and lab[nr,nc]==0: lab[nr,nc]=cur;q.append((nr,nc))
    return lab,cur

def make_rule(struct, dst, center='bbox', conn8=False):
    def rule(a):
        b=a.copy(); H,W=a.shape
        m=(a==struct).astype(int); lab,n=label(m,conn8)
        for k in range(1,n+1):
            ys,xs=np.where(lab==k)
            if center=='bbox':
                cr2=ys.min()+ys.max(); cc2=xs.min()+xs.max()
            else:
                cr2=int(round(2*ys.mean())); cc2=int(round(2*xs.mean()))
            for r,c in zip(ys,xs):
                nr=cr2-r; nc=cc2-c   # 180 about center
                if 0<=nr<H and 0<=nc<W and a[nr,nc]!=struct:
                    b[nr,nc]=dst
        return b
    return rule

if __name__=="__main__":
    for struct in [2]:
        for dst in [8]:
            for center in ['bbox','centroid']:
                for conn8 in [False,True]:
                    res,fails=C.check(118, make_rule(struct,dst,center,conn8))
                    print(f"struct={struct} dst={dst} center={center} conn8={conn8}: {res} {fails[:3]}")
