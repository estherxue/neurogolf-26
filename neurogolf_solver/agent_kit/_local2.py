import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ng_utils_shim import tasks_dir
tdir = tasks_dir()

def load(t):
    ex = json.load(open(tdir / f"task{t:03d}.json"))
    return [(np.array(e["input"],int), np.array(e["output"],int)) for s in ("train","test") for e in ex.get(s,[])]

def test_localmask(t, src, dst, struct_color, k):
    """For cells where input==src: predict (output==dst) from KxK binary mask of struct_color.
    Returns (consistent?, n_pos_patterns, n_neg_patterns)."""
    prs=load(t)
    pos=set(); neg=set(); incons=0
    for a,b in prs:
        if a.shape!=b.shape: return None
        H,W=a.shape
        m=(a==struct_color).astype(int)
        mp=np.pad(m,k)
        for r in range(H):
            for c in range(W):
                if a[r,c]!=src: continue
                patch=tuple(mp[r:r+2*k+1,c:c+2*k+1].flatten().tolist())
                is_dst = (b[r,c]==dst)
                if is_dst: pos.add(patch)
                else: neg.add(patch)
    overlap = pos & neg
    return dict(consistent=(len(overlap)==0), npos=len(pos), nneg=len(neg), overlap=len(overlap))

if __name__=="__main__":
    t=int(sys.argv[1]); src=int(sys.argv[2]); dst=int(sys.argv[3]); sc=int(sys.argv[4])
    for k in (1,2,3):
        r=test_localmask(t,src,dst,sc,k)
        print(f"k={k}: {r}")
