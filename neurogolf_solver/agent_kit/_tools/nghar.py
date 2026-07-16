#!/usr/bin/env python
"""nghar — NeuroGolf harness. One module, all the hard-won discipline baked in.

Consolidates the scattered tools (surg_scout/entropy_prof/mkblend/rv14/vf*/probe3)
into a single robust library + CLI. Key correctness rules encoded once:

  * COST = runtime-probe over the task's OWN official examples at MAX shape
    (the grader's method). NEVER file size, NEVER static shape-infer (both lie:
    file size counts externalized initializers; static infer chokes on QLinearConv).
  * u8 Min/Max have no ORT-1.23.2 kernel -> AUTO-PATCH to Less/Greater+Where for
    LOCAL MEASUREMENT ONLY (patch preserves shapes => cost identical; ship unpatched).
  * Gates use an INDEPENDENT seed from whoever built the candidate, and count
    exceptions as failures.
  * one-hot encode/threshold and fresh-gen loops live in one place.

CLI:
  nghar cost   <onnx> <task>                 runtime-probe pts (mem/params breakdown)
  nghar prof   <onnx> <task>                 3-axis entropy profile (u8/crop/mask slack)
  nghar gates  <cand> <inc> <task> [N seed]  A-E battery: exact vs inc + fresh + opt-inv
  nghar truth  <onnx> <task> [N seed]        dirt vs generator truth over N fresh
  nghar scan   <pooldir> <basedir> [tasks]   per-task cost-wins of pool over base (exact-gated)
  nghar merge  <basedir> <spec.json> <out>   assemble bundle (runtime cost), spec=[{task,path,name}]
  nghar audit  <task> [N]                    generator max grid + per-cell value range
"""
import sys, os, json, random, hashlib
import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = "/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26/f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad"
DATA = os.path.join(SC, "data")
sys.path.insert(0, os.path.join(KIT, "_arcgen"))
sys.path.insert(0, os.path.join(KIT, "_arcgen", "tasks"))

import onnx
import onnxruntime as ort
from onnx import helper as oh

_HM = None
def hashmap():
    global _HM
    if _HM is None:
        _HM = json.load(open(os.path.join(KIT, "task_hash_map.json")))
    return _HM

def genmod(task):
    import importlib
    return importlib.import_module("task_" + hashmap()[str(task)])

# --------------------------------------------------------------------------- #
# core encode / patch / session                                               #
# --------------------------------------------------------------------------- #
def to_np(grid):
    b = np.zeros((1, 10, 30, 30), np.float32)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if 0 <= v < 10 and r < 30 and c < 30:
                b[0][int(v)][r][c] = 1.0
    return b

def patch_u8_minmax(m):
    """Rewrite N-input Min/Max -> a chain of Less/Greater + Where (shape-preserving)
    so ORT 1.23.2 (which lacks u8 Min/Max kernels) can run the graph for LOCAL
    cost/exact measurement. Handles any input count (t145 has a 5-input Max).
    Does NOT change the scored cost. Never ship the patched model."""
    g = m.graph
    nn, k = [], [0]
    for n in g.node:
        if n.op_type in ("Min", "Max") and len(n.input) >= 2:
            op = "Less" if n.op_type == "Min" else "Greater"
            acc = n.input[0]
            ins = list(n.input[1:])
            for j, b in enumerate(ins):
                out = n.output[0] if j == len(ins) - 1 else "_nghar_t%d_%d" % (k[0], j)
                cmp = "_nghar_c%d_%d" % (k[0], j)
                nn.append(oh.make_node(op, [acc, b], [cmp]))
                nn.append(oh.make_node("Where", [cmp, acc, b], [out]))
                acc = out
            k[0] += 1
        else:
            nn.append(n)
    del g.node[:]; g.node.extend(nn)
    return m

def _session(model, enable=False):
    so = ort.SessionOptions()
    so.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if enable
                                   else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    so.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), so)

def load(path, patch=True):
    """Load a model for LOCAL measurement. When patch=True, rewrite u8 Min/Max
    (no ORT-1.23.2 kernel) and clamp ir_version to <=10 (ORT 1.23.2 max IR is 11;
    some pool models ship IR 13). Neither changes the scored cost."""
    m = onnx.load(path)
    if patch:
        m = patch_u8_minmax(m)
        if m.ir_version > 10:
            m.ir_version = 10
    return m

def examples(task):
    d = json.load(open(os.path.join(DATA, "task%03d.json" % task)))
    return [e for sp in ("train", "test", "arc-gen") for e in d.get(sp, [])]

# --------------------------------------------------------------------------- #
# cost — runtime probe (grader method)                                        #
# --------------------------------------------------------------------------- #
def cost(path, task, patch=True):
    """Return (pts, mem, params, exact_count, total). mem = sum over named
    intermediates (graph output excluded) of max bytes across the task's own
    official examples. params = sum max(1, numel) over initializers."""
    m = load(path, patch)
    outn = {o.name for o in m.graph.output}
    names = [o for n in m.graph.node for o in n.output if o and o not in outn]
    m2 = onnx.ModelProto(); m2.CopyFrom(m)
    for nm in names:
        m2.graph.output.add().name = nm
    s = _session(m2)
    meta = [o.name for o in s.get_outputs()]
    exs = examples(task)
    mx = {}; ok = 0
    for e in exs:
        rs = s.run(None, {"input": to_np(e["input"])})
        out = rs[meta.index("output")]
        if np.array_equal(out > 0, to_np(e["output"]) > 0):
            ok += 1
        for nm, arr in zip(meta, rs):
            # _nghar_* are patch-introduced temporaries (Min/Max decomposition) —
            # exclude so the measured cost equals the UNPATCHED graph's cost.
            if nm == "output" or nm.startswith("_nghar_") or not arr.size:
                continue
            b = int(np.prod(arr.shape)) * arr.dtype.itemsize
            if b > mx.get(nm, 0):
                mx[nm] = b
    mem = sum(mx.values())
    params = sum(max(1, int(np.prod(i.dims) if list(i.dims) else 1)) for i in m.graph.initializer)
    return 25 - float(np.log(mem + params)), mem, params, ok, len(exs)

# --------------------------------------------------------------------------- #
# 3-axis entropy profile                                                      #
# --------------------------------------------------------------------------- #
def profile(path, task, patch=True):
    m = load(path, patch)
    outn = {o.name for o in m.graph.output}
    prod = {o: n.op_type for n in m.graph.node for o in n.output if o}
    names = [o for n in m.graph.node for o in n.output if o and o not in outn]
    m2 = onnx.ModelProto(); m2.CopyFrom(m)
    for nm in names:
        m2.graph.output.add().name = nm
    s = _session(m2)
    meta = [o.name for o in s.get_outputs()]
    info = {}
    for e in examples(task):
        for nm, arr in zip(meta, s.run(None, {"input": to_np(e["input"])})):
            if nm == "output" or nm.startswith("_nghar_") or not arr.size:
                continue
            b = int(np.prod(arr.shape)) * arr.dtype.itemsize
            a = arr.astype(np.float64)
            if arr.ndim == 4 and arr.shape[2] > 1 and arr.shape[3] > 1:
                bg = a.reshape(-1, a.shape[2], a.shape[3])
                vals, cnts = np.unique(a, return_counts=True)
                bgv = vals[cnts.argmax()]
                nz = np.argwhere((bg != bgv).any(axis=0))
                h = int(nz[:, 0].max()) + 1 if nz.size else 1
                w = int(nz[:, 1].max()) + 1 if nz.size else 1
                H, W = arr.shape[2], arr.shape[3]
            else:
                h = w = H = W = -1
            u = np.unique(a)
            r = info.setdefault(nm, {"b": 0, "isz": arr.dtype.itemsize, "dt": str(arr.dtype),
                                     "vmin": 1e18, "vmax": -1e18, "int": True,
                                     "h": 0, "w": 0, "H": H, "W": W, "nvals": set()})
            r["b"] = max(r["b"], b); r["vmin"] = min(r["vmin"], float(a.min()))
            r["vmax"] = max(r["vmax"], float(a.max())); r["int"] &= bool(np.all(a == np.round(a)))
            r["h"] = max(r["h"], h); r["w"] = max(r["w"], w)
            if r["nvals"] is not None:
                r["nvals"] |= set(u[:80].tolist())
                if len(r["nvals"]) > 80:
                    r["nvals"] = None
    mem = sum(v["b"] for v in info.values())
    params = sum(max(1, int(np.prod(i.dims) if list(i.dims) else 1)) for i in m.graph.initializer)
    u8 = crop = mask = 0
    for v in info.values():
        if v["isz"] > 1 and v["int"] and 0 <= v["vmin"] and v["vmax"] <= 255:
            u8 += v["b"] - v["b"] // v["isz"]
        if v["H"] > 0 and v["h"] > 0 and (v["h"] < v["H"] or v["w"] < v["W"]):
            crop += int(v["b"] * (1 - (v["h"] * v["w"]) / (v["H"] * v["W"])))
        if v["nvals"] is not None and len(v["nvals"]) <= 2:
            mask += v["b"]
    return dict(mem=mem, params=params, pts=25 - float(np.log(mem + params)),
                u8=u8, crop=crop, mask=mask, info=info, prod=prod)

# --------------------------------------------------------------------------- #
# fresh generator loop                                                        #
# --------------------------------------------------------------------------- #
def fresh(task, n, seed, maxdim=30):
    """Yield up to n fresh (input_grid, output_grid) with both dims <= maxdim."""
    g = genmod(task).generate
    random.seed(seed)
    got = 0; tries = 0
    while got < n and tries < n * 10:
        tries += 1
        try:
            ex = g()
        except Exception:
            continue
        gi, go = np.array(ex["input"]), np.array(ex["output"])
        if gi.ndim != 2 or go.ndim != 2 or max(gi.shape) > maxdim or max(go.shape) > maxdim:
            continue
        got += 1
        yield gi, go

# --------------------------------------------------------------------------- #
# gate battery                                                                #
# --------------------------------------------------------------------------- #
def gates(cand_path, inc_path, task, n=3000, seed=None):
    """A-E: byte-identity vs incumbent on official + N fresh + opt-invariance.
    seed defaults to a task-derived value INDEPENDENT of any builder's seed."""
    if seed is None:
        seed = 900000 + task * 7 + 13
    cm = load(cand_path); im = load(inc_path)
    cd = _session(cm); ce = _session(cm, enable=True); ic = _session(im)
    def run(s, x):
        try:
            return s.run(["output"], {"input": x})[0] > 0
        except Exception:
            return None
    # official
    off_ok = off_tot = 0
    for e in examples(task):
        x = to_np(e["input"])
        a, b = run(ic, x), run(cd, x)
        off_tot += 1
        off_ok += int(a is not None and b is not None and np.array_equal(a, b))
    # fresh + opt-inv
    bad = optd = nn = 0
    for gi, go in fresh(task, n, seed):
        x = to_np(gi); nn += 1
        a, b, be = run(ic, x), run(cd, x), run(ce, x)
        if b is None or a is None or not np.array_equal(a, b):
            bad += 1
        if b is not None and be is not None and not np.array_equal(b, be):
            optd += 1
    return dict(official="%d/%d" % (off_ok, off_tot), fresh_n=nn, ident_bad=bad,
                optdiff=optd, seed=seed,
                pass_all=(off_ok == off_tot and bad == 0 and optd == 0))

def truth(path, task, n=3000, seed=None):
    """Dirt of a model vs the generator's own target over N fresh."""
    if seed is None:
        seed = 500000 + task
    s = _session(load(path))
    dirty = nn = 0
    for gi, go in fresh(task, n, seed):
        nn += 1
        try:
            out = s.run(["output"], {"input": to_np(gi)})[0] > 0
        except Exception:
            dirty += 1; continue
        if not np.array_equal(out, to_np(go) > 0):
            dirty += 1
    return dict(dirt=dirty, n=nn, rate=round(dirty / max(nn, 1), 4), seed=seed)

# --------------------------------------------------------------------------- #
# pool scan + merge                                                           #
# --------------------------------------------------------------------------- #
def scan(pooldir, basedir, tasks=None):
    """Per-task: pool file loads + local-exact + runtime-cost beats base -> win."""
    base_rep = json.load(open(os.path.join(basedir, "report.json")))
    base = {t["task"]: t for t in base_rep["tasks"]}
    tasks = tasks or sorted(base)
    wins = []
    for tn in tasks:
        p = os.path.join(pooldir, "task%03d.onnx" % tn)
        q = os.path.join(basedir, "onnx", "task%03d.onnx" % tn)
        if not os.path.exists(p):
            continue
        if hashlib.md5(open(p, "rb").read()).hexdigest() == hashlib.md5(open(q, "rb").read()).hexdigest():
            continue
        try:
            pts, mem, par, ok, tot = cost(p, tn)
        except Exception:
            continue
        if ok == tot and pts > base[tn]["points"] + 0.03:
            wins.append((round(pts - base[tn]["points"], 2), tn, round(base[tn]["points"], 2), round(pts, 2)))
    wins.sort(reverse=True)
    return wins

def merge(basedir, spec, outdir):
    import shutil, zipfile
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    shutil.copytree(basedir, outdir)
    rep = json.load(open(os.path.join(outdir, "report.json")))
    by = {t["task"]: t for t in rep["tasks"]}
    for s in spec:
        tn = s["task"]
        pts, mem, par, ok, tot = cost(s["path"], tn)
        if ok != tot:
            print("SKIP t%03d: not local-exact (%d/%d)" % (tn, ok, tot)); continue
        if pts <= by[tn]["points"] + 0.01:
            print("SKIP t%03d: %.2f <= incumbent %.2f" % (tn, pts, by[tn]["points"])); continue
        shutil.copy(s["path"], os.path.join(outdir, "onnx", "task%03d.onnx" % tn))
        print("SWAP t%03d: %.2f -> %.2f [%s]" % (tn, by[tn]["points"], pts, s.get("name", "?")))
        by[tn]["points"] = round(pts, 2); by[tn]["name"] = s.get("name", by[tn]["name"])
    rep["total_points"] = round(sum(t["points"] for t in rep["tasks"]), 2)
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    zp = os.path.join(outdir, "submission.zip")
    if os.path.exists(zp):
        os.remove(zp)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(os.path.join(outdir, "onnx"))):
            if f.endswith(".onnx"):
                z.write(os.path.join(outdir, "onnx", f), f)
    print("%s total=%.2f" % (os.path.basename(outdir), rep["total_points"]))
    return rep["total_points"]

def audit(task, n=2000):
    g = genmod(task).generate
    random.seed(1)
    mx = 0; vmax = 0
    for _ in range(n):
        try:
            ex = g()
        except Exception:
            continue
        gi, go = np.array(ex["input"]), np.array(ex["output"])
        mx = max(mx, gi.shape[0], gi.shape[1], go.shape[0], go.shape[1])
        vmax = max(vmax, int(gi.max()), int(go.max()))
    return dict(max_dim=mx, max_color=vmax, n=n)

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main(argv):
    if not argv:
        print(__doc__); return
    cmd, a = argv[0], argv[1:]
    if cmd == "cost":
        pts, mem, par, ok, tot = cost(a[0], int(a[1]))
        print("pts=%.3f mem=%d params=%d exact=%d/%d" % (pts, mem, par, ok, tot))
    elif cmd == "prof":
        p = profile(a[0], int(a[1]))
        print("mem=%d params=%d pts=%.2f | u8_slack=%d crop_slack=%d mask=%d"
              % (p["mem"], p["params"], p["pts"], p["u8"], p["crop"], p["mask"]))
        rows = sorted(p["info"].items(), key=lambda kv: -kv[1]["b"])[:8]
        for nm, v in rows:
            nv = len(v["nvals"]) if v["nvals"] is not None else ">80"
            print("   %-13s %-8s %6dB bbox %sx%s/%sx%s nvals=%s rng[%.0f,%.0f]"
                  % (p["prod"].get(nm, "?"), v["dt"], v["b"], v["h"], v["w"], v["H"], v["W"], nv, v["vmin"], v["vmax"]))
    elif cmd == "gates":
        n = int(a[3]) if len(a) > 3 else 3000
        seed = int(a[4]) if len(a) > 4 else None
        print(json.dumps(gates(a[0], a[1], int(a[2]), n, seed)))
    elif cmd == "truth":
        n = int(a[2]) if len(a) > 2 else 3000
        seed = int(a[3]) if len(a) > 3 else None
        print(json.dumps(truth(a[0], int(a[1]), n, seed)))
    elif cmd == "scan":
        tasks = [int(x) for x in a[2].split(",")] if len(a) > 2 else None
        wins = scan(a[0], a[1], tasks)
        for d, tn, ob, nb in wins:
            print("  WIN t%03d +%.2f  %.2f->%.2f" % (tn, d, ob, nb))
        print("TOTAL +%.2f over %d tasks" % (sum(w[0] for w in wins), len(wins)))
    elif cmd == "merge":
        merge(a[0], json.load(open(a[1])), a[2])
    elif cmd == "audit":
        print(json.dumps(audit(int(a[0]), int(a[1]) if len(a) > 1 else 2000)))
    else:
        print("unknown cmd:", cmd); print(__doc__)

if __name__ == "__main__":
    main(sys.argv[1:])
