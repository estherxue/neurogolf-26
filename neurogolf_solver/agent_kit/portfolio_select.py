"""Third-gen per-task solver PORTFOLIO with a STRICT held-out validator (the reliable route).

For every task: split arc-gen into FIT (70%) and HELD-OUT (30%). Gather ALL candidate models
from ALL family modules, each built using ONLY the FIT split (train+test+fit arc-gen). A model
is VALID only if it is EXACT on BOTH the fit split AND the untouched held-out split. Among valid
candidates pick the CHEAPEST (highest points). This catches every overfit mode observed:
  - feature/LUT overfit (perceptron over seen KxK neighborhoods) -> fails held-out neighborhoods
  - crop grid-size overfit on VARIABLE-size tasks -> held-out has larger grids the crop cuts off
  - crops on TRULY fixed-size tasks -> held-out is same size -> pass (correctly kept, they're safe)
The shipped model is the FIT-split model that already generalized to held-out, so it is safe.

usage: python portfolio_select.py <dst_dir>
"""
import sys, os, json, glob, importlib, pathlib, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnx
import evaluate
from ng_utils_shim import tasks_dir

DST = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("out_portfolio")
FIT_FRAC = 0.70

MODS = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "family_*.py"))):
    n = pathlib.Path(f).stem
    if n == "family_test":
        continue
    try:
        MODS.append(importlib.import_module(n))
    except Exception:
        pass


def exact_on(model, examples, tag):
    """True iff model is grader-exact on the given example list (as arc-gen split)."""
    if not examples:
        return True
    try:
        r = evaluate.evaluate(model, {"train": examples, "test": [], "arc-gen": []}, tag=tag)
        return bool(r.get("ok"))
    except Exception:
        return False


def main():
    tdir = tasks_dir()
    (DST / "onnx").mkdir(parents=True, exist_ok=True)
    tasks = []
    files = sorted(glob.glob(str(tdir / "task*.json")))
    for fp in files:
        tn = int(pathlib.Path(fp).stem.replace("task", ""))
        ex = json.load(open(fp))
        ag = ex.get("arc-gen", [])
        k = max(1, int(len(ag) * FIT_FRAC))
        fit_ag, held_ag = ag[:k], ag[k:]
        exfit = {"train": ex["train"], "test": ex["test"], "arc-gen": fit_ag}
        best = None  # (points, name, model)
        for mod in MODS:
            try:
                cands = list(mod.candidates(exfit))
            except Exception:
                cands = []
            for name, model in cands:
                # full-points score on 100% (train+test+all arc-gen) via official grader
                rf = evaluate.evaluate(model, ex, tag=f"ps_{tn}")
                if not rf.get("ok"):
                    continue
                # STRICT: must also be exact on the untouched held-out arc-gen
                if not exact_on(model, held_ag, tag=f"ho_{tn}"):
                    continue
                if best is None or rf["points"] > best[0]:
                    best = (rf["points"], name, model)
        rec = {"task": tn}
        if best is not None:
            rec.update(name=best[1], points=round(best[0], 2))
            onnx.save(best[2], str(DST / "onnx" / f"task{tn:03d}.onnx"))
        tasks.append(rec)
        if tn % 25 == 0:
            done = sum(1 for t in tasks if t.get("name"))
            print(f"  ...task{tn:03d} ({done} solved so far)", flush=True)
    total = sum(t["points"] for t in tasks if t.get("name"))
    nsolved = sum(1 for t in tasks if t.get("name"))
    json.dump(dict(solved=nsolved, total_points=round(total, 3), tasks=tasks),
              open(DST / "report.json", "w"))
    with zipfile.ZipFile(DST / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((DST / "onnx").glob("task*.onnx")):
            z.write(f, f.name)
    print(f"PORTFOLIO: held-out-validated, solved {nsolved}/400, total {total:.2f}", flush=True)


if __name__ == "__main__":
    main()
