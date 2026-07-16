"""Score a family module over all 400 tasks via the real grader.
usage: python family_test.py <module_name> [start] [end]
"""
import importlib, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # neurogolf_solver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # agent_kit
from ng_utils_shim import tasks_dir
from evaluate import evaluate

mod = importlib.import_module(sys.argv[1])
start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
end = int(sys.argv[3]) if len(sys.argv) > 3 else 400
tdir = tasks_dir()
solved, total = [], 0.0
for t in range(start, end + 1):
    ex = json.load(open(tdir / f"task{t:03d}.json"))
    best = None
    try:
        cands = mod.candidates(ex) or []
    except Exception as e:
        cands = []
    for name, model in cands:
        try:
            res = evaluate(model, ex, tag=f"{sys.argv[1]}_{t}_{name}")
        except Exception:
            continue
        if res.get("ok") and (best is None or res["points"] > best[1]):
            best = (name, res["points"])
    if best:
        solved.append((t, best[0], round(best[1], 2)))
        total += best[1]
print(f"SOLVED {len(solved)} TOTAL {round(total,1)}")
for s in solved:
    print(s)
