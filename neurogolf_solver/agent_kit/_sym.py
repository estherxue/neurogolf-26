import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ng_utils_shim import tasks_dir
tdir = tasks_dir()

def load(t):
    ex = json.load(open(tdir / f"task{t:03d}.json"))
    return [(np.array(e["input"],int), np.array(e["output"],int)) for s in ("train","test") for e in ex.get(s,[])]

def reflect_about(a, center_r2, center_c2, mode):
    # center_r2 = 2*center_row (integer). mode in 'h','v','r' (rot180)
    H,W = a.shape
    out = np.zeros_like(a)
    for r in range(H):
        for c in range(W):
            if a[r,c]==0: continue
            if mode=='v': nr,nc = center_r2-r, c
            elif mode=='h': nr,nc = r, center_c2-c
            elif mode=='r': nr,nc = center_r2-r, center_c2-c
            if 0<=nr<H and 0<=nc<W:
                out[nr,nc]=a[r,c]
    return out

def try_task(t):
    prs = load(t)
    print(f"TASK {t}")
    # For each example find center as bbox center of all nonzero
    for idx,(a,b) in enumerate(prs):
        ys,xs = np.nonzero(a)
        cr2 = ys.min()+ys.max(); cc2 = xs.min()+xs.max()
        # try union of operations
        for ops in [('v','h','r'),('r',),('v','h'),('h',),('v',)]:
            res = a.copy()
            for m in ops:
                ref = reflect_about(a, cr2, cc2, m)
                res = np.where(ref!=0, ref, res)
            if np.array_equal(res,b):
                print(f"  ex{idx}: MATCH bbox-center ops={ops}")
                break
        else:
            print(f"  ex{idx}: no bbox match (cr2={cr2},cc2={cc2})")

if __name__=="__main__":
    for t in [int(x) for x in sys.argv[1:]]:
        try_task(t)
