"""Integrator: merge ALL family_*.py modules + base symbolic + local learner into one
candidate stream, score every task with the official grader, keep the best (highest-points)
solution per task, write submission.zip + report.

usage: python combined_solve.py [out_dir]
"""
import importlib, glob, json, os, sys, pathlib, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)   # neurogolf_solver
sys.path.insert(0, HERE)  # agent_kit

from ng_utils_shim import tasks_dir
from evaluate import evaluate
import onnx

# Discover all family modules (each exposes candidates(examples)).
fam_mods = []
for f in sorted(glob.glob(os.path.join(HERE, "family_*.py"))):
    name = pathlib.Path(f).stem
    try:
        fam_mods.append(importlib.import_module(name))
    except Exception as e:
        print(f"[skip] {name}: {e}", flush=True)

# Base symbolic + local learner from the package.
import solve as base
import learner

print(f"[integrator] families: {[m.__name__ for m in fam_mods]}", flush=True)


import math

GATE = os.environ.get("NG_GATE", "1") == "1"


def all_candidates(ex):
    """Anti-overfit held-out gate: families fit/detect on only 70% of arc-gen; the caller
    then grades each candidate on 100% of arc-gen (incl. the held-out 30%). Overfit models
    fail the held-out split and are dropped, so local score tracks the (held-out) public LB.
    Symbolic families detect from train/test only, so they're unaffected."""
    if GATE:
        arc = ex.get("arc-gen", [])
        nfit = max(1, math.ceil(len(arc) * 0.7)) if len(arc) >= 4 else len(arc)
        fit_ex = {"train": ex.get("train", []), "test": ex.get("test", []),
                  "arc-gen": arc[:nfit]}
    else:
        fit_ex = ex
    for fn in (base.candidates,):
        try:
            for c in (fn(fit_ex) or []):
                yield c
        except Exception:
            pass
    for m in fam_mods:
        try:
            for c in (m.candidates(fit_ex) or []):
                yield c
        except Exception:
            pass
    try:
        for c in learner.local_candidates(fit_ex):
            yield c
    except Exception:
        pass


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (pathlib.Path(HERE) / "out_combined")
    onx = out / "onnx"
    onx.mkdir(parents=True, exist_ok=True)
    tdir = tasks_dir()
    report, total, solved = [], 0.0, 0
    for t in range(1, 401):
        ex = json.load(open(tdir / f"task{t:03d}.json"))
        best = None
        for name, model in all_candidates(ex):
            try:
                res = evaluate(model, ex, tag=f"comb_{t}_{name}")
            except Exception:
                continue
            if res.get("ok") and (best is None or res["points"] > best["points"]):
                best = dict(name=name, model=model, points=res["points"],
                            params=res.get("params"), memory=res.get("memory"))
        if best:
            onnx.save(best["model"], str(onx / f"task{t:03d}.onnx"))
            total += best["points"]; solved += 1
            report.append(dict(task=t, name=best["name"], points=round(best["points"], 2),
                               params=best["params"], memory=best["memory"]))
        else:
            report.append(dict(task=t, name=None, points=0.0))
    with zipfile.ZipFile(out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(onx.glob("task*.onnx")):
            z.write(f, f.name)
    json.dump(dict(solved=solved, total_points=round(total, 3), tasks=report),
              open(out / "report.json", "w"), indent=2)
    print(f"COMBINED solved={solved}/400 total_points={total:.2f}", flush=True)
    for r in report:
        if r["name"]:
            print(f"  task{r['task']:03d} {r['name']:<22} {r['points']:.2f}", flush=True)


if __name__ == "__main__":
    main()
