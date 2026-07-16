import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ng_utils_shim import tasks_dir

tdir = tasks_dir()

def grid_str(a):
    a = np.array(a, int)
    return "\n".join("".join(str(v) for v in row) for row in a)

def show(t, sections=("train","test")):
    ex = json.load(open(tdir / f"task{t:03d}.json"))
    print(f"===== TASK {t} =====")
    for s in sections:
        for i, e in enumerate(ex.get(s, [])):
            a = np.array(e["input"], int); b = np.array(e["output"], int)
            print(f"--- {s}[{i}] in {a.shape} -> out {b.shape}")
            ah = a.shape[0]; bh = b.shape[0]
            arows = grid_str(a).split("\n")
            brows = grid_str(b).split("\n")
            n = max(len(arows), len(brows))
            for k in range(n):
                l = arows[k] if k < len(arows) else ""
                r = brows[k] if k < len(brows) else ""
                print(f"{l:<32}  {r}")

if __name__ == "__main__":
    ts = [int(x) for x in sys.argv[1:]]
    for t in ts:
        show(t)
