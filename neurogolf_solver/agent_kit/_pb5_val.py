"""Validation harness for family_pb_5 candidates.
Runs: local train+test+arc-gen exact, fresh generator samples, and cost.
"""
import sys, json, math, os, random
import numpy as np
import onnx, onnxruntime as ort

GEN = "/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26--claude-worktrees-kaggle-agent-harness/f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad/arc-gen"
sys.path.insert(0, GEN)
import task_list
_TL = task_list.task_list()

from ng_utils_shim import ng, tasks_dir
from evaluate import evaluate

HASHMAP = json.load(open(os.path.join(os.path.dirname(__file__), "task_hash_map.json")))


def to_oh(grid):
    a = np.zeros((1, 10, 30, 30), np.float32)
    g = np.array(grid)
    if g.ndim != 2 or max(g.shape) > 30:
        return None
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            a[0, g[r, c], r, c] = 1.0
    return a


def run(sess, grid):
    x = to_oh(grid)
    if x is None:
        return None
    out = sess.run(["output"], {"input": x})[0]
    return (out > 0.0).astype(np.float32)


def target_oh(grid):
    return to_oh(grid)


def make_sess(model):
    fn = "/tmp/_pb5_tmp.onnx"
    onnx.save(model, fn)
    san = ng.sanitize_model(onnx.load(fn))
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(san.SerializeToString(), opts)


def validate(model, task_num, n_fresh=1000, seed=0):
    """Returns dict(ok, cost, points, gen_fails, gen_n, reason)."""
    h = HASHMAP[str(task_num)]
    gen = _TL[h][0]
    # local score via evaluate (train+test+arc-gen exact + cost)
    ex = json.load(open(tasks_dir() / f"task{task_num:03d}.json"))
    res = evaluate(model, ex, tag=f"pb5_{task_num}")
    if not res.get("ok"):
        return dict(ok=False, reason=res.get("reason"), agi=res.get("agi"), gen=res.get("gen"),
                    cost=None, points=0.0, gen_fails=None, gen_n=0)
    # fresh generator
    try:
        sess = make_sess(model)
    except Exception as e:
        return dict(ok=False, reason=f"load:{e}", cost=res["cost"], points=0.0, gen_fails=None, gen_n=0)
    random.seed(seed)
    import numpy.random as npr
    npr.seed(seed)
    fails = 0
    n = 0
    for _ in range(n_fresh):
        try:
            e = gen()
        except Exception:
            continue
        tgt = target_oh(e["output"])
        got = run(sess, e["input"])
        if tgt is None or got is None:
            continue
        n += 1
        if got.shape != tgt.shape or not np.array_equal(got, tgt):
            fails += 1
    return dict(ok=(fails == 0), reason="ok" if fails == 0 else "gen-fail",
                cost=res["cost"], memory=res["memory"], params=res["params"],
                points=res["points"], gen_fails=fails, gen_n=n)


if __name__ == "__main__":
    import importlib
    mod = importlib.import_module(sys.argv[1])
    tasks = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else None
    for tn in (tasks or []):
        ex = json.load(open(tasks_dir() / f"task{tn:03d}.json"))
        cands = mod.candidates(ex) or []
        best = None
        for name, model in cands:
            r = validate(model, tn, n_fresh=1000)
            print(tn, name, r)
