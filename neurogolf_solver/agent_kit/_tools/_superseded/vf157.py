"""Independent streaming verify of _cands/task157.onnx vs incumbent.
Fresh seed 60606 (agent used 4242). N=1500. Compares (out>0) identity + opt-invariance.
"""
import sys, os, json, random
import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT + "/_arcgen"); sys.path.insert(0, KIT + "/_arcgen/tasks")
import importlib, onnx, onnxruntime as ort
gen = importlib.import_module("task_6a1e5592")

def mk(p, lvl):
    so = ort.SessionOptions(); so.graph_optimization_level = lvl; so.log_severity_level = 4
    return ort.InferenceSession(onnx.load(p).SerializeToString(), so)

D = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
E = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
inc = mk(KIT + "/out_blend13/onnx/task157.onnx", D)
cand = mk(KIT + "/_cands/task157.onnx", D)
cande = mk(KIT + "/_cands/task157.onnx", E)

def to_np(gr):
    b = np.zeros((1, 10, 30, 30), np.float32)
    for r, row in enumerate(gr):
        for c, v in enumerate(row):
            b[0][v][r][c] = 1.0
    return b

SC = "/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26/f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad"
d = json.load(open(SC + "/data/task157.json"))
tot = okc = 0
for split in ("train", "test", "arc-gen"):
    for ex in d.get(split, []):
        x = to_np(ex["input"])
        a = inc.run(["output"], {"input": x})[0] > 0
        b = cand.run(["output"], {"input": x})[0] > 0
        tot += 1; okc += int(np.array_equal(a, b))
print(f"official ident: {okc}/{tot}", flush=True)

random.seed(60606)
bad = optd = n = 0
for i in range(1500):
    ex = gen.generate()
    gi = np.array(ex["input"])
    if max(gi.shape) > 30:
        continue
    x = to_np(ex["input"])
    n += 1
    a = inc.run(["output"], {"input": x})[0] > 0
    b = cand.run(["output"], {"input": x})[0] > 0
    if not np.array_equal(a, b):
        bad += 1
        print(f"MISMATCH at fresh #{i}", flush=True)
    e = cande.run(["output"], {"input": x})[0] > 0
    if not np.array_equal(b, e):
        optd += 1
        print(f"OPTDIFF at fresh #{i}", flush=True)
    if n % 250 == 0:
        print(f"... {n} (bad {bad} optd {optd})", flush=True)
print(f"VF157 FINAL: ident-bad {bad}/{n}, optd {optd}", flush=True)
