import numpy as np, sys
sys.path.insert(0,'.');sys.path.insert(0,'..')
import _check as C

def rule(a, fill=6, tie='first'):
    H,W=a.shape
    avail=(a==0)
    # content bbox
    rows_c=np.where(a.any(axis=1))[0]; cols_c=np.where(a.any(axis=0))[0]
    if len(rows_c)==0: return a.copy()
    H2=rows_c.max()+1; W2=cols_c.max()+1
    av=np.zeros((H,W),bool)
    av[:H2,:W2]=avail[:H2,:W2]
    best=None  # (area, r0,r1,c0,c1)
    for r0 in range(H2):
        for r1 in range(r0,H2):
            band=av[r0:r1+1,:].all(axis=0)  # cols available across band
            # find runs
            c=0
            while c<W2:
                if band[c]:
                    c2=c
                    while c2<W2 and band[c2]: c2+=1
                    width=c2-c; area=width*(r1-r0+1)
                    cand=(area, r0,r1,c,c2-1)
                    if best is None or area>best[0]:
                        best=cand
                    c=c2
                else:
                    c+=1
    b=a.copy()
    if best and best[0]>0:
        _,r0,r1,c0,c1=best
        b[r0:r1+1,c0:c1+1]=fill
    return b

if __name__=="__main__":
    res,fails=C.check(90, rule)
    print("largest-empty-rect:",res, fails[:6])
