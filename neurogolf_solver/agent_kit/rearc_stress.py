"""RE-ARC independent-distribution STRESS TEST (the '淘汰脆弱候选' step with a genuinely
different distribution). My tasks are the 400 ARC-1 training tasks (verified: task JSON train
pairs match the ARC-GEN-100K hash files). RE-ARC (Michael Hodel) provides an INDEPENDENTLY
hand-written generator per ARC-1 hash. A solver that passes ARC-GEN held-out but encodes only
ARC-GEN's quirks will FAIL RE-ARC's samples — exactly the ARC-APPX overfit that same-distribution
held-out cannot catch. Passing BOTH is strong evidence of true rule generalization.

Builds task_NNN -> ARC-1 hash map (by matching train pairs to ARC-GEN-100K), generates N fresh
RE-ARC samples per task, runs the shipped ONNX, reports exact-match pass rate. Flags solvers that
pass ARC-GEN but fail RE-ARC (fragile / private-risk).

usage: python rearc_stress.py <dir> [N]
"""
import sys, os, json, glob, pathlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SC = "/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26--claude-worktrees-kaggle-agent-harness/f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad"
sys.path.insert(0, f"{SC}/re-arc")
import onnx, onnxruntime
from ng_utils_shim import ng, tasks_dir
import generators as G

DIR = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("out_p5")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
ARCGEN = f"{SC}/arcgen100k"


def build_map(tdir):
    """task_NNN -> ARC-1 hash via matching each task's train examples to the hash files."""
    hidx = {}
    for hf in glob.glob(f"{ARCGEN}/*.json"):
        h = os.path.basename(hf)[:-5]
        for e in json.load(open(hf))[:8]:
            hidx.setdefault(json.dumps([e["input"], e["output"]]), h)
    m = {}
    for fp in glob.glob(str(tdir / "task*.json")):
        tn = int(pathlib.Path(fp).stem.replace("task", ""))
        t = json.load(open(fp))
        for e in t["train"] + t.get("arc-gen", [])[:15]:
            h = hidx.get(json.dumps([e["input"], e["output"]]))
            if h:
                m[tn] = h
                break
    return m


def _sess(model):
    o = onnxruntime.SessionOptions()
    o.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    return onnxruntime.InferenceSession(ng.sanitize_model(model).SerializeToString(), o)


def _run(sess, grid):
    H, W = len(grid), len(grid[0])
    if H > 30 or W > 30:
        return None
    x = np.zeros((1, 10, 30, 30), np.float32)
    g = np.array(grid)
    for v in range(10):
        x[0, v][:H, :W] = (g == v)
    y = sess.run(None, {"input": x})[0][0]
    lab = y.argmax(0); on = y.max(0) > 0
    return np.where(on[:H, :W], lab[:H, :W], 0)


def main():
    tdir = tasks_dir()
    tmap = build_map(tdir)
    print(f"mapped {len(tmap)}/400 tasks to ARC-1 hashes", flush=True)
    rep = json.load(open(DIR / "report.json"))
    fragile = []; robust = 0; skipped = 0
    for t in rep["tasks"]:
        if not t.get("name"):
            continue
        tn = t["task"]; h = tmap.get(tn)
        gen = getattr(G, f"generate_{h}", None) if h else None
        fp = DIR / "onnx" / f"task{tn:03d}.onnx"
        if gen is None or not fp.exists():
            skipped += 1; continue
        try:
            sess = _sess(onnx.load(str(fp)))
        except Exception:
            skipped += 1; continue
        ok = 0; tot = 0
        for i in range(N):
            try:
                ex = gen(0.0 + 0.9 * (i / N), 0.1 + 0.9 * (i / N))
            except Exception:
                continue
            gi, go = ex["input"], ex["output"]
            if len(gi) > 30 or len(gi[0]) > 30 or len(go) > 30 or len(go[0]) > 30:
                continue
            out = _run(sess, gi)
            if out is None:
                continue
            tot += 1
            if out.shape == np.array(go).shape and np.array_equal(out, go):
                ok += 1
        if tot < 5:
            skipped += 1; continue
        rate = ok / tot
        if rate >= 0.98:
            robust += 1
        else:
            fragile.append((tn, t["name"], round(t["points"], 2), f"{ok}/{tot}"))
            print(f"  FRAGILE task{tn:03d} {t['name']} ({t['points']:.2f}): RE-ARC {ok}/{tot} exact", flush=True)
    print(f"RE-ARC STRESS: robust {robust}, fragile {len(fragile)}, skipped {skipped}", flush=True)
    json.dump([f[0] for f in fragile], open("rearc_fragile.json", "w"))
    json.dump(tmap, open("task_hash_map.json", "w"))


if __name__ == "__main__":
    main()
