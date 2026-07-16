import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ng_utils_shim import tasks_dir
tdir = tasks_dir()

def nb(a, r, c, k=2):
    H,W = a.shape
    out=[]
    for dr in range(-k,k+1):
        row=[]
        for dc in range(-k,k+1):
            rr,cc=r+dr,c+dc
            if 0<=rr<H and 0<=cc<W:
                row.append(str(a[rr,cc]))
            else:
                row.append('.')
        out.append("".join(row))
    return out

def analyze(t, k=2):
    ex = json.load(open(tdir / f"task{t:03d}.json"))
    print(f"===== TASK {t} (k={k}) =====")
    for s in ("train","test"):
        for i, e in enumerate(ex.get(s, [])):
            a = np.array(e["input"], int); b = np.array(e["output"], int)
            if a.shape!=b.shape:
                print(f"--- {s}[{i}] SHAPE CHANGE {a.shape}->{b.shape}"); continue
            diff = np.argwhere(a != b)
            print(f"--- {s}[{i}] {a.shape} ndiff={len(diff)}")
            for (r,c) in diff:
                rows = nb(a,r,c,k)
                print(f"  ({r},{c}) {a[r,c]}->{b[r,c]}  nbhd:" )
                for rr in rows: print("      "+rr)

if __name__=="__main__":
    args=[x for x in sys.argv[1:]]
    k=2
    ts=[]
    for x in args:
        if x.startswith('k='): k=int(x[2:])
        else: ts.append(int(x))
    for t in ts: analyze(t,k)
