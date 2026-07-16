import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ng_utils_shim import tasks_dir
tdir = tasks_dir()

def load(t):
    ex = json.load(open(tdir / f"task{t:03d}.json"))
    return [(np.array(e["input"],int), np.array(e["output"],int), s, i) for s in ("train","test") for i,e in enumerate(ex.get(s,[]))]

def apply_op(a, cr2, cc2, op):
    H,W=a.shape; out=np.zeros_like(a)
    ys,xs=np.nonzero(a)
    for r,c in zip(ys,xs):
        if op=='id': nr,nc=r,c
        elif op=='v': nr,nc=cr2-r,c
        elif op=='h': nr,nc=r,cc2-c
        elif op=='r': nr,nc=cr2-r,cc2-c
        elif op=='d':  # transpose about center: (r,c)->(cr2/2 + (c-cc2/2), ...) only valid integer if cr2,cc2 same parity
            nr=(cr2-cc2)//2 + c; nc=(cc2-cr2)//2 + r
        elif op=='a':  # anti-transpose
            nr=(cr2+cc2)//2 - c; nc=(cr2+cc2)//2 - r
        else: nr,nc=r,c
        if 0<=nr<H and 0<=nc<W: out[nr,nc]=a[r,c]
    return out

def try_task(t):
    prs=load(t); print(f"TASK {t}")
    OPS=['id','v','h','r','d','a']
    for (a,b,s,i) in prs:
        H,W=a.shape
        found=None
        for cr2 in range(0,2*H-1):
            for cc2 in range(0,2*W-1):
                # union of full D4 ops
                res=np.zeros_like(a)
                ok=True
                for op in OPS:
                    img=apply_op(a,cr2,cc2,op)
                    # consistency: where img!=0 and res!=0 they must match
                    clash=(img!=0)&(res!=0)&(img!=res)
                    if clash.any(): ok=False;break
                    res=np.where(img!=0,img,res)
                if ok and np.array_equal(res,b):
                    found=(cr2,cc2);break
            if found:break
        print(f"  {s}[{i}] shape({H},{W}) D4-union center2=({found})")

if __name__=="__main__":
    for t in [int(x) for x in sys.argv[1:]]:
        try_task(t)
