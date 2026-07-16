import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ng_utils_shim import tasks_dir
tdir = tasks_dir()

def patches(t, k):
    """Return dict patch_tuple -> set of output centers, plus consistency stats."""
    ex = json.load(open(tdir / f"task{t:03d}.json"))
    m = {}
    incons = 0
    total = 0
    for s in ("train","test"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int); b = np.array(e["output"], int)
            if a.shape != b.shape:
                return None
            H,W = a.shape
            ap = np.pad(a, k, constant_values=0)  # pad with bg 0
            for r in range(H):
                for c in range(W):
                    patch = tuple(ap[r:r+2*k+1, c:c+2*k+1].flatten().tolist())
                    o = int(b[r,c])
                    total += 1
                    if patch in m:
                        if o not in m[patch]:
                            m[patch].add(o); incons += 1
                    else:
                        m[patch] = {o}
    # count patches mapping to >1 output
    ambig = sum(1 for v in m.values() if len(v) > 1)
    return dict(npatch=len(m), ambig=ambig, total=total)

if __name__ == "__main__":
    ts = [int(x) for x in sys.argv[1:]]
    for t in ts:
        print(f"TASK {t}")
        for k in (1,2,3):
            r = patches(t,k)
            if r is None:
                print(f"  k={k}: SHAPE CHANGE"); break
            print(f"  k={k}: npatch={r['npatch']} ambiguous_patches={r['ambig']} (0=>locally expressible)")
