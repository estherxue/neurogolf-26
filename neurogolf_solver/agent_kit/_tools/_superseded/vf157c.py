"""Independent verify of _cands3/task157.onnx vs blend15 incumbent. My seed 424242, N=1000."""
import sys, os, json, random
import numpy as np
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT + "/_arcgen"); sys.path.insert(0, KIT + "/_arcgen/tasks")
import importlib, onnx, onnxruntime as ort
gen = importlib.import_module("task_6a1e5592")
def mk(p, lvl):
    so = ort.SessionOptions(); so.graph_optimization_level = lvl; so.log_severity_level = 4
    return ort.InferenceSession(onnx.load(p).SerializeToString(), so)
D = ort.GraphOptimizationLevel.ORT_DISABLE_ALL; E = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
inc = mk(KIT + "/out_blend15/onnx/task157.onnx", D)
cand = mk(KIT + "/_cands3/task157.onnx", D)
cande = mk(KIT + "/_cands3/task157.onnx", E)
def to_np(gr):
    b = np.zeros((1, 10, 30, 30), np.float32)
    for r, row in enumerate(gr):
        for c, v in enumerate(row):
            b[0][v][r][c] = 1.0
    return b
SC = "/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26/f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad"
d = json.load(open(SC + "/data/task157.json")); tot = okI = okT = 0
for sp in ("train", "test", "arc-gen"):
    for ex in d.get(sp, []):
        x = to_np(ex["input"]); t = to_np(ex["output"]) > 0
        a = inc.run(["output"], {"input": x})[0] > 0
        b = cand.run(["output"], {"input": x})[0] > 0
        tot += 1; okI += int(np.array_equal(a, b)); okT += int(np.array_equal(b, t))
print(f"official: ident {okI}/{tot}, truth {okT}/{tot}", flush=True)
random.seed(424242); bad = optd = 0; n = 0
for i in range(1000):
    ex = gen.generate(); gi = np.array(ex["input"])
    if max(gi.shape) > 30: continue
    x = to_np(gi); n += 1
    a = inc.run(["output"], {"input": x})[0] > 0
    b = cand.run(["output"], {"input": x})[0] > 0
    if not np.array_equal(a, b):
        bad += 1; print(f"MISMATCH #{i}", flush=True)
    e = cande.run(["output"], {"input": x})[0] > 0
    if not np.array_equal(b, e):
        optd += 1; print(f"OPTDIFF #{i}", flush=True)
    if n % 200 == 0: print(f"... {n} (bad {bad} optd {optd})", flush=True)
print(f"VF157C FINAL: ident-bad {bad}/{n}, optd {optd}", flush=True)
