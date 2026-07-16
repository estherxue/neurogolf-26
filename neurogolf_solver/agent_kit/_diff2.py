import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from ng_utils_shim import tasks_dir
tdir = tasks_dir()

def bg_of(a):
    return Counter(a.flatten().tolist()).most_common(1)[0][0]

def show(t, sections=("train","test")):
    ex = json.load(open(tdir / f"task{t:03d}.json"))
    print(f"===== TASK {t} =====")
    for s in sections:
        for i, e in enumerate(ex.get(s, [])):
            a = np.array(e["input"], int); b = np.array(e["output"], int)
            print(f"--- {s}[{i}] in {a.shape} -> out {b.shape}  bg_in={bg_of(a)} bg_out={bg_of(b)}")
            if a.shape == b.shape:
                diff = np.argwhere(a != b)
                print("  changed cells (r,c): in->out")
                for (r,c) in diff:
                    print(f"    ({r},{c}): {a[r,c]} -> {b[r,c]}")
            else:
                print("  shapes differ; non-bg in input:")
                bg=bg_of(a)
                for (r,c) in np.argwhere(a!=bg):
                    print(f"    in ({r},{c})={a[r,c]}")
                print("  output non-bg:")
                bgo=bg_of(b)
                for (r,c) in np.argwhere(b!=bgo):
                    print(f"    out ({r},{c})={b[r,c]}")

if __name__ == "__main__":
    ts = [int(x) for x in sys.argv[1:]]
    for t in ts:
        show(t)
