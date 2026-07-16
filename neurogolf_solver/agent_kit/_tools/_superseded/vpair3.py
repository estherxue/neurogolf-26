"""vpair3: full gate battery for a crop-surgery candidate vs incumbent.
Gates:
  A onnx.checker full_check on the candidate (the SHIPPED file, unpatched)
  B official DATA: cand(out>0) == inc(out>0) on ALL train+test+arc-gen examples
  C fresh gens seed 2468 N=3000 (skip grids >30): cand == inc on ALL (bitwise out>0)
  D opt-invariance on first 300 fresh: ORT_DISABLE_ALL vs ORT_ENABLE_ALL equal, both models
If a model can't run on local ORT (u8 Min/Max kernel gaps), pass --patch: BOTH models get a
LOCAL-TEST-ONLY rewrite u8 Min->(Greater+Where), Max->(Less+Where); shipped file is untouched.
usage: python vpair3.py <task_int> <cand_path> [--inc path] [--n 3000] [--seed 2468] [--patch]
importable: run_battery(tn, cand, inc, n, seed, patch) -> dict
"""
import sys, os, json, random
import numpy as np
import onnx, onnxruntime as ort

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = "/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26/f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad"
sys.path.insert(0, KIT + "/_arcgen")
sys.path.insert(0, KIT + "/_arcgen/tasks")
import task_list


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


def patch_u8_minmax(m):
    """LOCAL-TEST-ONLY: rewrite 2-input Min->(Greater+Where), Max->(Less+Where)."""
    m2 = onnx.ModelProto(); m2.CopyFrom(m)
    new_nodes = []
    k = [0]
    from onnx import helper as oh
    for n in m2.graph.node:
        if n.op_type in ("Min", "Max") and len(n.input) == 2:
            a, b = n.input[0], n.input[1]
            o = n.output[0]
            cmpn = f"__mmpatch_{k[0]}"; k[0] += 1
            if n.op_type == "Min":
                new_nodes.append(oh.make_node("Greater", [a, b], [cmpn]))
                new_nodes.append(oh.make_node("Where", [cmpn, b, a], [o]))
            else:
                new_nodes.append(oh.make_node("Less", [a, b], [cmpn]))
                new_nodes.append(oh.make_node("Where", [cmpn, b, a], [o]))
        else:
            new_nodes.append(n)
    del m2.graph.node[:]
    m2.graph.node.extend(new_nodes)
    return m2


def sessions(m, both=True):
    out = []
    lvls = [ort.GraphOptimizationLevel.ORT_DISABLE_ALL]
    if both:
        lvls.append(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    for lvl in lvls:
        so = ort.SessionOptions(); so.graph_optimization_level = lvl
        so.log_severity_level = 4
        out.append(ort.InferenceSession(m.SerializeToString(), so))
    return out


def run_battery(tn, cand_path, inc_path=None, n=3000, seed=2468, patch=False, optn=300):
    if inc_path is None:
        inc_path = f"{KIT}/out_blend15/onnx/task{tn:03d}.onnx"
    res = {}
    # A: checker on the SHIPPED candidate
    mc_raw = onnx.load(cand_path)
    try:
        onnx.checker.check_model(mc_raw, full_check=True)
        res["A_checker"] = "PASS"
    except Exception as ex:
        res["A_checker"] = f"FAIL {str(ex)[:200]}"
    mi_raw = onnx.load(inc_path)
    mc = patch_u8_minmax(mc_raw) if patch else mc_raw
    mi = patch_u8_minmax(mi_raw) if patch else mi_raw
    cd, ce = sessions(mc)
    idd, ie = sessions(mi)
    # B: official data
    d = json.load(open(f"{SC}/data/task{tn:03d}.json"))
    exs = [e for sp in ("train", "test", "arc-gen") for e in d.get(sp, [])]
    bfail = 0
    for e in exs:
        x = to_np(e["input"])
        oc = cd.run(["output"], {"input": x})[0] > 0.0
        oi = idd.run(["output"], {"input": x})[0] > 0.0
        if not np.array_equal(oc, oi):
            bfail += 1
    res["B_official"] = f"{len(exs)-bfail}/{len(exs)}" + ("" if bfail == 0 else " FAIL")
    # C+D: fresh gens
    hm = json.load(open(KIT + "/task_hash_map.json"))
    gen = task_list.task_list()[hm[str(tn)]][0]
    random.seed(seed)
    cfail = optd = made = skipped = 0
    tries = 0
    while made < n and tries < n * 8:
        tries += 1
        try:
            e = gen()
        except Exception:
            continue
        if not ok_shape(e):
            skipped += 1
            continue
        made += 1
        x = to_np(e["input"])
        oc = cd.run(["output"], {"input": x})[0] > 0.0
        oi = idd.run(["output"], {"input": x})[0] > 0.0
        if not np.array_equal(oc, oi):
            cfail += 1
        if made <= optn:
            oce = ce.run(["output"], {"input": x})[0] > 0.0
            oie = ie.run(["output"], {"input": x})[0] > 0.0
            if not np.array_equal(oc, oce) or not np.array_equal(oi, oie):
                optd += 1
    res["C_fresh"] = f"{made-cfail}/{made} (skipped>{30}:{skipped})" + ("" if cfail == 0 else " FAIL")
    res["D_optinv"] = f"{min(made,optn)-optd}/{min(made,optn)}" + ("" if optd == 0 else " FAIL")
    res["patched_local_only"] = patch
    return res


if __name__ == "__main__":
    tn = int(sys.argv[1]); cand = sys.argv[2]
    inc = None; n = 3000; seed = 2468; patch = "--patch" in sys.argv
    if "--inc" in sys.argv:
        inc = sys.argv[sys.argv.index("--inc") + 1]
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    r = run_battery(tn, cand, inc, n, seed, patch)
    for k, v in r.items():
        print(f"  {k}: {v}")
