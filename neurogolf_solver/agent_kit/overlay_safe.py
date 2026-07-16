"""Safe overlay integrator. Start from a CONFIRMED-SAFE base output dir (its onnx are the
incumbents), then for each NEW grid-agnostic module (family_<prefix>*), run its candidates
on every task and swap in any EXACT solution that is strictly cheaper than the incumbent.
Only NEW modules are considered, so no old (possibly overfit-crop) module can regress the
base. Because the new modules are pure-algorithmic no-crop, every swap is grid-agnostic safe.

usage: python overlay_safe.py <base_dir> <dst_dir> <module_prefix> [<module_prefix> ...]
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

MODS = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "family_*.py"))):
    n = pathlib.Path(f).stem
    if any(n.startswith(p) for p in PREFIXES):
        try:
            MODS.append(importlib.import_module(n))
        except Exception as e:
            print(f"  (skip {n}: {e})", flush=True)


def main():
    tdir = tasks_dir()
    rep = json.load(open(BASE / "report.json"))
    (DST / "onnx").mkdir(parents=True, exist_ok=True)
    for f in (BASE / "onnx").glob("task*.onnx"):
        (DST / "onnx" / f.name).write_bytes(f.read_bytes())
    by_task = {t["task"]: dict(t) for t in rep["tasks"]}
    improved = {}
    for tn, t in by_task.items():
        ex = json.load(open(tdir / f"task{tn:03d}.json"))
        base_pts = t.get("points", 0.0)
        best = base_pts; best_model = None; best_name = None
        for mod in MODS:
            try:
                cands = list(mod.candidates(ex))
            except Exception:
                cands = []
            for name, model in cands:
                try:
                    r = evaluate.evaluate(model, ex, tag=f"ov_{tn}")
                except Exception:
                    continue
                if r.get("ok") and r["points"] > best + 0.02:
                    best = r["points"]; best_model = model; best_name = name
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
    print(f"OVERLAY improved {len(improved)} tasks; solved {nsolved}; new total {tot:.2f}", flush=True)


if __name__ == "__main__":
    main()
