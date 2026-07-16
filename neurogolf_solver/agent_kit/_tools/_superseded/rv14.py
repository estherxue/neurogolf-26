"""rv14: high-N reverify of pool candidates vs blend13 incumbents.
For each candidate task: every DISTINCT (md5) pool variant is screened on the SAME
N fresh arc-gen samples as the incumbent. Exceptions count as failures. Each sample
runs under ORT_DISABLE_ALL and ORT_ENABLE_ALL; any output divergence = optdiff.
Adopt gate (reported, not applied here): cand_rate <= inc_rate + 0.002 and optdiff == 0.
usage: python rv14.py [N]
"""
import sys, os, json, random, hashlib, pathlib, traceback
import numpy as np

KIT = "/Users/xingyuanxue1122/Documents/coding/neurogolf-26/.claude/worktrees/kaggle-agent-harness/neurogolf_solver/agent_kit"
sys.path.insert(0, KIT + "/_arcgen")
sys.path.insert(0, KIT + "/_arcgen/tasks")
import task_list, onnx, onnxruntime as ort

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
TASKS = [157, 209, 17, 5, 319, 118, 188, 191, 101, 367, 85, 233, 285]
POOLS = sorted(p for p in os.listdir(KIT + "/_pools") if os.path.isdir(KIT + "/_pools/" + p))
tl = task_list.task_list()
hm = json.load(open(KIT + "/task_hash_map.json"))
rep = {t["task"]: t for t in json.load(open(KIT + "/out_blend13/report.json"))["tasks"]}

def to_np(g):
    b = np.zeros((1, 10, 30, 30), np.float32)
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if 0 <= v < 10 and r < 30 and c < 30:
                b[0][v][r][c] = 1.0
    return b

def ok_shape(e):
    return (len(e["input"]) <= 30 and len(e["input"][0]) <= 30
            and len(e["output"]) <= 30 and len(e["output"][0]) <= 30)

def sessions(path):
    m = onnx.load(path)
    s = []
    for lvl in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
        so = ort.SessionOptions(); so.graph_optimization_level = lvl; so.log_severity_level = 4
        s.append(ort.InferenceSession(m.SerializeToString(), so))
    return s

def screen(path, examples):
    """returns (fails, optdiff, ran) over examples; exceptions are failures."""
    try:
        sd, se = sessions(path)
    except Exception as ex:
        return None, None, f"LOADERR {str(ex)[:80]}"
    fails = optd = 0
    for x, tgt in examples:
        try:
            od = (sd.run(["output"], {"input": x})[0] > 0.0)
        except Exception:
            fails += 1; continue
        if od.shape != tgt.shape or not np.array_equal(od, tgt):
            fails += 1
        try:
            oe = (se.run(["output"], {"input": x})[0] > 0.0)
            if od.shape != oe.shape or not np.array_equal(od, oe):
                optd += 1
        except Exception:
            optd += 1
    return fails, optd, None

results = {}
for tn in TASKS:
    gen = tl[hm[str(tn)]][0]
    random.seed(777000 + tn)
    examples = []
    tries = 0
    while len(examples) < N and tries < N * 8:
        tries += 1
        try:
            e = gen()
        except Exception:
            continue
        if not ok_shape(e):
            continue
        examples.append((to_np(e["input"]), to_np(e["output"]) > 0.0))
    inc_path = f"{KIT}/out_blend13/onnx/task{tn:03d}.onnx"
    inc_fails, inc_optd, inc_err = screen(inc_path, examples)
    inc_pts = rep[tn]["points"]
    print(f"task{tn:03d} inc[{rep[tn]['name'][:30]}] {inc_pts:.2f}pts "
          f"dirt={inc_fails}/{len(examples)} optd={inc_optd} err={inc_err}", flush=True)
    variants = {}
    for pool in POOLS:
        p = f"{KIT}/_pools/{pool}/task{tn:03d}.onnx"
        if not os.path.exists(p):
            continue
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        variants.setdefault(h, (pool, p))
    tres = {"inc": {"pts": inc_pts, "fails": inc_fails, "optd": inc_optd, "n": len(examples), "err": inc_err},
            "cands": []}
    for h, (pool, p) in variants.items():
        fails, optd, err = screen(p, examples)
        verdict = "LOADERR" if err else (
            "ADOPTABLE" if (inc_fails is not None and fails <= inc_fails + max(1, int(0.002 * len(examples))) and optd == 0)
            else "REJECT")
        print(f"  cand {pool:12} md5={h[:8]} dirt={fails}/{len(examples)} optd={optd} -> {verdict} {err or ''}", flush=True)
        tres["cands"].append({"pool": pool, "md5": h, "fails": fails, "optd": optd, "verdict": verdict, "err": err})
    results[tn] = tres
    json.dump(results, open(KIT + "/_tools/rv14_results.json", "w"), indent=1)
print("RV14 DONE", flush=True)
