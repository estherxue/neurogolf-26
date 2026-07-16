import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ng_utils_shim import tasks_dir

tdir = tasks_dir()

def analyze(t):
    ex = json.load(open(tdir / f"task{t:03d}.json"))
    print(f"===== TASK {t} =====")
    for s in ("train","test"):
        for i, e in enumerate(ex.get(s, [])):
            a = np.array(e["input"], int); b = np.array(e["output"], int)
            print(f"--- {s}[{i}] in {a.shape} -> out {b.shape}  in_colors={sorted(set(a.flatten().tolist()))} out_colors={sorted(set(b.flatten().tolist()))}")
            if a.shape == b.shape:
                diff = np.argwhere(a != b)
                print(f"   n_diff={len(diff)}")
                for (r,c) in diff[:60]:
                    print(f"     ({r},{c}): {a[r,c]} -> {b[r,c]}")
            else:
                print("   (shape changed)")

if __name__ == "__main__":
    for t in [int(x) for x in sys.argv[1:]]:
        analyze(t)
