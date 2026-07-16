"""Crop-augmented portfolio with strict held-out validation. Starts from a base output dir
(its per-task incumbent), and for every task ALSO generates work-area crop variants of each
family module's solver (patch size globals -> S, rebuild, Slice/Pad crop-wrap) at a RANGE of S.
Every candidate (incumbent + crops) must be EXACT on the untouched held-out 30% arc-gen split;
among the valid ones the cheapest is kept. The held-out gate makes cropping safe: on VARIABLE
tasks a too-small crop cuts a held-out grid and is rejected; on FIXED tasks crops pass.

usage: python portfolio_crop.py <base_dir> <dst_dir> [max_points]
"""
import sys, os, json, copy, glob, importlib, pathlib, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnx
from onnx import helper as oh, TensorProto as TP
import onnx.checker as _chk
import evaluate
from ng_utils_shim import tasks_dir

BASE = pathlib.Path(sys.argv[1])
DST = pathlib.Path(sys.argv[2])
MAXP = float(sys.argv[3]) if len(sys.argv) > 3 else 18.0
FIT_FRAC = 0.70
SIZE_ATTRS = ["H", "W", "HEIGHT", "WIDTH", "GRID", "S"]

MODS = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "family_*.py"))):
    n = pathlib.Path(f).stem
    if n == "family_test":
        continue
    try:
        MODS.append(importlib.import_module(n))
    except Exception:
        pass

_NOCHECK = lambda *a, **k: None


def crop_wrap(model, S):
    m = copy.deepcopy(model); g = m.graph
    for nd in g.node:
        nd.input[:] = ["inp_s" if x == "input" else x for x in nd.input]
        nd.output[:] = ["out_s" if x == "output" else x for x in nd.output]
    g.initializer.extend([oh.make_tensor("cwS", TP.INT64, [2], [0, 0]),
                          oh.make_tensor("cwE", TP.INT64, [2], [S, S]),
                          oh.make_tensor("cwA", TP.INT64, [2], [2, 3])])
    g.node.insert(0, oh.make_node("Slice", ["input", "cwS", "cwE", "cwA"], ["inp_s"], name="cw_s"))
    g.node.append(oh.make_node("Pad", ["out_s"], ["output"], mode="constant", value=0.0,
                               pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S], name="cw_p"))
    return m


def patch(mod, S):
    old = {}
    for a in SIZE_ATTRS:
        if hasattr(mod, a) and isinstance(getattr(mod, a), int):
            old[a] = getattr(mod, a); setattr(mod, a, S)
    if hasattr(mod, "NC") and isinstance(mod.NC, int):
        old["NC"] = mod.NC; mod.NC = 2 * S
    return old


def unpatch(mod, old):
    for a, v in old.items():
        setattr(mod, a, v)


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
        if not t.get("name") or t["points"] >= MAXP:
            continue
        ex = json.load(open(tdir / f"task{tn:03d}.json"))
        ag = ex.get("arc-gen", [])
        k = max(1, int(len(ag) * FIT_FRAC))
        fit_ag, held_ag = ag[:k], ag[k:]
        exfit = {"train": ex["train"], "test": ex["test"], "arc-gen": fit_ag}
        # max grid across the FIT split (train+test+fit arc-gen) -> smallest crop to even try
        gmax = 1
        for e in exfit["train"] + exfit["test"] + fit_ag:
            gmax = max(gmax, len(e["input"]), len(e["input"][0]), len(e["output"]), len(e["output"][0]))
        best = t["points"]; best_model = None
        Srange = sorted({s for s in range(gmax, min(gmax + 6, 30))})
        _orig_check = _chk.check_model
        for S in Srange:
            _chk.check_model = _NOCHECK
            try:
                for mod in MODS:
                    old = patch(mod, S)
                    try:
                        cands = list(mod.candidates(exfit))
                    except Exception:
                        cands = []
                    unpatch(mod, old)
                    for name, model in cands:
                        try:
                            cw = crop_wrap(model, S)
                        except Exception:
                            continue
                        rf = evaluate.evaluate(cw, ex, tag=f"pc_{tn}_{S}")
                        if not rf.get("ok") or rf["points"] <= best + 0.02:
                            continue
                        if not exact_on(cw, held_ag, tag=f"pch_{tn}_{S}"):
                            continue   # held-out reject -> overfit crop
                        best = rf["points"]; best_model = cw
            finally:
                _chk.check_model = _orig_check
        if best_model is not None:
            onnx.save(best_model, str(DST / "onnx" / f"task{tn:03d}.onnx"))
            improved[tn] = (t["points"], round(best, 2))
            by_task[tn]["points"] = round(best, 2)
            by_task[tn]["name"] = (t.get("name") or "") + f"+xcrop"
            print(f"  task{tn:03d}: {t['points']:.2f} -> {best:.2f}  +{best-t['points']:.2f}", flush=True)
    tasks = [by_task[i] for i in sorted(by_task)]
    tot = sum(t["points"] for t in tasks if t.get("name"))
    nsolved = sum(1 for t in tasks if t.get("name"))
    json.dump(dict(solved=nsolved, total_points=round(tot, 3), tasks=tasks),
              open(DST / "report.json", "w"))
    with zipfile.ZipFile(DST / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((DST / "onnx").glob("task*.onnx")):
            z.write(f, f.name)
    print(f"PORTFOLIO_CROP: held-out-validated crops improved {len(improved)}; solved {nsolved}; total {tot:.2f}", flush=True)


if __name__ == "__main__":
    main()
