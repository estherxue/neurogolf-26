"""Held-out-gated targeted overlay: add ONLY the given new modules' wins onto a confirmed-safe
base, each validated on the untouched 30% held-out arc-gen split (the gate overlay_safe.py was
missing). Only runs the new modules (not the whole 155-module set) over the union of unsolved +
golf-target tasks, so it is fast and avoids re-triggering unrelated modules' segfaults.

usage: python overlay_heldout.py <base_dir> <dst_dir> <module_prefix> [<prefix> ...]
"""
import sys, os, json, glob, importlib, pathlib, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnx
import evaluate
from ng_utils_shim import tasks_dir

BASE = pathlib.Path(sys.argv[1])
DST = pathlib.Path(sys.argv[2])
PREFIXES = sys.argv[3:]
FIT_FRAC = 0.70

MODS = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "family_*.py"))):
    n = pathlib.Path(f).stem
    if any(n.startswith(p) for p in PREFIXES):
        try:
            MODS.append(importlib.import_module(n))
        except Exception as e:
            print(f"  (skip {n}: {e})", flush=True)


def exact_on(model, examples, tag):
    if not examples:
        return True
    try:
        r = evaluate.evaluate(model, {"train": examples, "test": [], "arc-gen": []}, tag=tag)
        return bool(r.get("ok"))
    except Exception:
        return False


def main():
    tdir = tasks_dir()
    rep = json.load(open(BASE / "report.json"))
    (DST / "onnx").mkdir(parents=True, exist_ok=True)
    for f in (BASE / "onnx").glob("task*.onnx"):
        (DST / "onnx" / f.name).write_bytes(f.read_bytes())
    by_task = {t["task"]: dict(t) for t in rep["tasks"]}
    improved = {}
    for tn, t in by_task.items():
        fp = tdir / f"task{tn:03d}.json"
        ex = json.load(open(fp))
        ag = ex.get("arc-gen", [])
        k = max(1, int(len(ag) * FIT_FRAC))
        fit_ag, held_ag = ag[:k], ag[k:]
        exfit = {"train": ex["train"], "test": ex["test"], "arc-gen": fit_ag}
        base_pts = t.get("points", 0.0)
        best = base_pts; best_model = None; best_name = None
        for mod in MODS:
            try:
                cands = list(mod.candidates(exfit))
            except Exception:
                cands = []
            for name, model in cands:
                try:
                    rf = evaluate.evaluate(model, ex, tag=f"oh_{tn}")
                except Exception:
                    continue
                if not rf.get("ok") or rf["points"] <= best + 0.02:
                    continue
                if not exact_on(model, held_ag, tag=f"ohh_{tn}"):
                    continue    # held-out reject -> overfit
                best = rf["points"]; best_model = model; best_name = name
        if best_model is not None:
            onnx.save(best_model, str(DST / "onnx" / f"task{tn:03d}.onnx"))
            improved[tn] = (base_pts, round(best, 2))
            by_task[tn]["points"] = round(best, 2)
            by_task[tn]["name"] = best_name
            print(f"  task{tn:03d}: {base_pts:.2f} -> {best:.2f}  +{best-base_pts:.2f} ({best_name})", flush=True)
    tasks = [by_task[i] for i in sorted(by_task)]
    tot = sum(t["points"] for t in tasks if t.get("name"))
    nsolved = sum(1 for t in tasks if t.get("name"))
    json.dump(dict(solved=nsolved, total_points=round(tot, 3), tasks=tasks),
              open(DST / "report.json", "w"))
    with zipfile.ZipFile(DST / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((DST / "onnx").glob("task*.onnx")):
            z.write(f, f.name)
    print(f"OVERLAY_HELDOUT improved {len(improved)}; solved {nsolved}; total {tot:.2f}", flush=True)


if __name__ == "__main__":
    main()
