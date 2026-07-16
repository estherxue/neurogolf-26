"""NeuroGolf 2026 — self-contained GPU solver kernel.

Runs on a Kaggle GPU notebook (competition data attached at /kaggle/input/neurogolf-2026).
Per task it tries: symbolic exact rules -> gated local-rule lookup -> per-task CNN trained on
arc-gen (CUDA). Every candidate is scored through the OFFICIAL neurogolf_utils, and only
solutions that are exact on ALL train+test+arc-gen AND on a held-out arc-gen split (the
anti-overfit gate vs. the private set) are kept. Writes /kaggle/working/submission.zip.
"""
import os, sys, types, json, io, math, time, zipfile, pathlib, tempfile, traceback
import numpy as np

INPUT = os.environ.get("NG_INPUT")
if not INPUT:
    import glob as _glob
    _hits = _glob.glob("/kaggle/input/**/task001.json", recursive=True)
    INPUT = os.path.dirname(_hits[0]) if _hits else "/kaggle/input/neurogolf-2026"
OUT = pathlib.Path(os.environ.get("NG_OUT", "/kaggle/working"))
TSTART = int(os.environ.get("NG_TSTART", "1"))
TEND = int(os.environ.get("NG_TEND", "400"))  # full run

# neurogolf_utils imports onnx_tool/IPython only for its display path; stub onnx_tool so we
# need no internet. (IPython + matplotlib exist on Kaggle images.)
if "onnx_tool" not in sys.modules:
    _m = types.ModuleType("onnx_tool"); _m.model_profile = lambda *a, **k: None
    sys.modules["onnx_tool"] = _m
def _ensure(pkgs):
    import importlib, subprocess
    for mod, pip in pkgs:
        try:
            importlib.import_module(mod)
        except Exception:
            print(f"[bootstrap] pip install {pip}", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip], check=False)

_ensure([("onnx", "onnx"), ("onnxruntime", "onnxruntime")])

import onnx
from onnx import helper as oh
import onnxruntime

def _load_official_utils():
    """Load the competition's neurogolf_utils.py by explicit path (avoid the
    directory-named namespace-package collision that shadows the .py file)."""
    import glob, importlib.util
    cands = (glob.glob(os.path.join(INPUT, "**", "neurogolf_utils.py"), recursive=True)
             or glob.glob("/kaggle/input/**/neurogolf_utils.py", recursive=True))
    if not cands:
        raise FileNotFoundError(f"neurogolf_utils.py not found under {INPUT}")
    spec = importlib.util.spec_from_file_location("neurogolf_utils_official", cands[0])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["neurogolf_utils_official"] = mod
    spec.loader.exec_module(mod)
    print(f"[init] loaded official utils: {cands[0]}", flush=True)
    return mod

ng = _load_official_utils()

DT, GRID = ng._DATA_TYPE, ng._GRID_SHAPE
IR, OPS = ng._IR_VERSION, ng._OPSET_IMPORTS
CH, H, W = ng._CHANNELS, ng._HEIGHT, ng._WIDTH
INT64 = onnx.TensorProto.INT64
_NEG = -(1 << 31)
WORK = pathlib.Path(tempfile.mkdtemp())

def _ensure_torch_cuda():
    """Kaggle's P100 is sm_60; the preinstalled torch only has sm_70+ kernels, so CUDA ops
    crash. Probe in a subprocess (before importing torch here) and reinstall a P100-compatible
    cu118 build if needed."""
    import subprocess
    if not os.path.isdir("/kaggle"):   # only attempt the heavy reinstall on Kaggle
        return
    code = ("import torch\nok=False\n"
            "try:\n"
            " if torch.cuda.is_available():\n"
            "  x=torch.zeros(8,8,device='cuda');_=(x@x).sum().item();ok=True\n"
            "except Exception: ok=False\n"
            "print('CUDAOK' if ok else 'CUDABAD')")
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    except Exception:
        return
    if "CUDAOK" in p.stdout:
        print("[bootstrap] cuda OK", flush=True); return
    print("[bootstrap] cuda unusable on this GPU; installing torch 2.2.2+cu118 (sm_60)...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.2.2",
                    "--index-url", "https://download.pytorch.org/whl/cu118"], check=False)

_ensure_torch_cuda()
try:
    import torch
    import torch.nn as nn
    from numpy.lib.stride_tricks import sliding_window_view
    if torch.cuda.is_available():
        try:
            _t = torch.zeros(8, 8, device="cuda"); _ = (_t @ _t).sum().item()
            DEV = torch.device("cuda")
        except Exception:
            DEV = torch.device("cpu")
    else:
        DEV = torch.device("cpu")
    TORCH_OK = True
except Exception:
    TORCH_OK = False
    DEV = None

print(f"[init] torch={'on' if TORCH_OK else 'off'} device={DEV}", flush=True)


# ----------------------------- builders -------------------------------------
def _model(nodes, inits=()):
    x = oh.make_tensor_value_info("input", DT, GRID)
    y = oh.make_tensor_value_info("output", DT, GRID)
    g = oh.make_graph(nodes, "g", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR, opset_imports=OPS)

def b_identity():
    return _model([oh.make_node("Identity", ["input"], ["output"])])

def b_transpose():
    return _model([oh.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])])

def b_recolor_gather(src):
    idx = oh.make_tensor("idx", INT64, [CH], list(src))
    return _model([oh.make_node("Gather", ["input", "idx"], ["output"], axis=1)], [idx])

def b_recolor_conv(cmap):
    wt = [0.0] * (CH * CH)
    for i, o in enumerate(cmap):
        wt[o * CH + i] = 1.0
    w = oh.make_tensor("W", DT, [CH, CH, 1, 1], wt)
    return _model([oh.make_node("Conv", ["input", "W"], ["output"],
                                kernel_shape=[1, 1], pads=[0, 0, 0, 0])], [w])

def _rev(axes):
    n = len(axes)
    s = oh.make_tensor("ss", INT64, [n], [(H if a == 2 else W) - 1 for a in axes])
    e = oh.make_tensor("se", INT64, [n], [_NEG] * n)
    a = oh.make_tensor("sa", INT64, [n], list(axes))
    st = oh.make_tensor("st", INT64, [n], [-1] * n)
    return _model([oh.make_node("Slice", ["input", "ss", "se", "sa", "st"], ["output"])],
                  [s, e, a, st])

def b_flip_w(): return _rev([3])
def b_flip_h(): return _rev([2])
def b_rot180(): return _rev([2, 3])

def b_translate(dy, dx):
    h0, w0 = max(dy, 0), max(dx, 0)
    h1, w1 = max(-dy, 0), max(-dx, 0)
    pad = oh.make_node("Pad", ["input"], ["padded"], mode="constant", value=0.0,
                       pads=[0, 0, h0, w0, 0, 0, h1, w1])
    s = oh.make_tensor("cs", INT64, [2], [h1, w1])
    e = oh.make_tensor("ce", INT64, [2], [h1 + H, w1 + W])
    a = oh.make_tensor("ca", INT64, [2], [2, 3])
    crop = oh.make_node("Slice", ["padded", "cs", "ce", "ca"], ["output"])
    return _model([pad, crop], [s, e, a])


# ----------------------------- evaluate (faithful) --------------------------
def _verify(session, subset):
    right = wrong = 0
    for ex in subset:
        b = ng.convert_to_numpy(ex)
        if not b:
            continue
        out = ng.run_network(session, b["input"])
        if out.shape == b["output"].shape and (out == b["output"]).all():
            right += 1
        else:
            wrong += 1
    return right, wrong

def evaluate(model, examples, tag):
    fn = WORK / f"{tag}.onnx"
    onnx.save(model, str(fn))
    if not ng.check_network(str(fn)):
        return dict(ok=False, points=0.0)
    try:
        san = ng.sanitize_model(onnx.load(str(fn)))
        if not san:
            return dict(ok=False, points=0.0)
        opt = onnxruntime.SessionOptions()
        opt.enable_profiling = True
        opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        opt.profile_file_prefix = str(WORK / f"{tag}_p")
        sess = onnxruntime.InferenceSession(san.SerializeToString(), opt)
    except Exception:
        return dict(ok=False, points=0.0)
    ar, aw = _verify(sess, examples["train"] + examples["test"])
    if aw:
        sess.end_profiling(); return dict(ok=False, points=0.0)
    gr, gw = _verify(sess, examples.get("arc-gen", []))
    if gw:
        sess.end_profiling(); return dict(ok=False, points=0.0)
    trace = sess.end_profiling()
    mem, par = ng.score_network(san, trace)
    if mem is None or par is None or mem < 0 or par < 0:
        return dict(ok=False, points=0.0)
    cost = mem + par
    return dict(ok=True, points=max(1.0, 25.0 - math.log(max(1.0, cost))),
                params=par, memory=mem, cost=cost)


# ----------------------------- symbolic inference ---------------------------
def _pairs(examples):
    out = []
    for sp in ("train", "test"):
        for e in examples.get(sp, []):
            out.append((np.array(e["input"], int), np.array(e["output"], int)))
    return out

def _colormap(prs):
    m = {}
    for a, b in prs:
        if a.shape != b.shape:
            return None
        for iv, ov in zip(a.ravel().tolist(), b.ravel().tolist()):
            if iv in m and m[iv] != ov:
                return None
            m[iv] = ov
    return m

def symbolic_candidates(examples):
    prs = _pairs(examples)
    if not prs:
        return
    def ok(fn):
        for a, b in prs:
            try:
                t = fn(a)
            except Exception:
                return False
            if t.shape != b.shape or not np.array_equal(t, b):
                return False
        return True
    if ok(lambda a: a): yield ("identity", b_identity())
    if ok(lambda a: a.T): yield ("transpose", b_transpose())
    if ok(lambda a: a[:, ::-1]): yield ("flip_w", b_flip_w())
    if ok(lambda a: a[::-1, :]): yield ("flip_h", b_flip_h())
    if ok(lambda a: a[::-1, ::-1]): yield ("rot180", b_rot180())
    m = _colormap(prs)
    if m is not None and any(k != v for k, v in m.items()):
        cmap = [m.get(i, i) for i in range(CH)]
        if sorted(cmap) == list(range(CH)):
            inv = [0] * CH
            for i, o in enumerate(cmap):
                inv[o] = i
            yield ("recolor_gather", b_recolor_gather(inv))
        yield ("recolor_conv", b_recolor_conv(cmap))
    def shift(a, dy, dx):
        r = np.zeros_like(a); h, w = a.shape
        y0, y1 = max(dy, 0), min(h, h + dy); x0, x1 = max(dx, 0), min(w, w + dx)
        r[y0:y1, x0:x1] = a[y0 - dy:y1 - dy, x0 - dx:x1 - dx]; return r
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            if (dy or dx) and ok(lambda a, dy=dy, dx=dx: shift(a, dy, dx)):
                yield (f"translate{dy}_{dx}", b_translate(dy, dx)); return


# ----------------------------- local-rule learner ---------------------------
MAX_PATTERNS = 3000

def _cells(ex, K):
    b = ng.convert_to_numpy(ex)
    if not b:
        return
    pad = K // 2
    inp, outp = b["input"][0], b["output"][0]
    padded = np.pad(inp, ((0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(padded, (K, K), axis=(1, 2)).transpose(1, 2, 0, 3, 4)
    csum, carg = outp.sum(0), outp.argmax(0)
    for r in range(H):
        for c in range(W):
            m = win[r, c].astype(np.float32)
            yield m.astype(np.int8).tobytes(), m, (int(carg[r, c]) if csum[r, c] > 0 else -1)

def _lookup(example_list, K):
    color, mask, none = {}, {}, set()
    for ex in example_list:
        for key, m, col in _cells(ex, K):
            if col < 0:
                if key in color: return None
                none.add(key)
            else:
                if key in none or (key in color and color[key] != col): return None
                color.setdefault(key, col); mask.setdefault(key, m)
    return color, mask

def _generalizes(fit, hold, K):
    built = _lookup(fit, K)
    if built is None: return False
    color, _ = built
    for ex in hold:
        for key, _m, col in _cells(ex, K):
            if color.get(key, -1) != col: return False
    return True

def _compile_local(color, mask, K):
    pats = [(mask[k], color[k]) for k in color]
    P, pad = len(pats), K // 2
    W1 = np.empty((P, CH, K, K), np.float32); B1 = np.empty((P,), np.float32)
    W2 = np.zeros((CH, P, 1, 1), np.float32)
    for j, (m, col) in enumerate(pats):
        W1[j] = 2 * m - 1; B1[j] = -(float(m.sum()) - 1.0); W2[col, j, 0, 0] = 1.0
    w1 = oh.make_tensor("W1", DT, list(W1.shape), W1.ravel().tolist())
    b1 = oh.make_tensor("B1", DT, [P], B1.tolist())
    w2 = oh.make_tensor("W2", DT, list(W2.shape), W2.ravel().tolist())
    n1 = oh.make_node("Conv", ["input", "W1", "B1"], ["h1"], kernel_shape=[K, K],
                      pads=[pad, pad, pad, pad])
    n2 = oh.make_node("Relu", ["h1"], ["h2"])
    n3 = oh.make_node("Conv", ["h2", "W2"], ["output"], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return _model([n1, n2, n3], [w1, b1, w2]), P

def local_candidates(examples, ks=(3, 5)):
    tt = list(examples.get("train", [])) + list(examples.get("test", []))
    arc = list(examples.get("arc-gen", []))
    if len(arc) < 4:
        return
    nfit = max(1, math.ceil(len(arc) * 0.7))
    fit, hold = tt + arc[:nfit], arc[nfit:]
    for K in ks:
        if not _generalizes(fit, hold, K):
            continue
        built = _lookup(tt + arc, K)
        if built is None:
            continue
        color, mask = built
        if not color or len(color) > MAX_PATTERNS:
            continue
        model, P = _compile_local(color, mask, K)
        yield (f"local_k{K}_p{P}", model); return


# ----------------------------- CNN trainer (CUDA) ---------------------------
def _tensors(example_list):
    X, Y = [], []
    for ex in example_list:
        b = ng.convert_to_numpy(ex)
        if not b:
            continue
        X.append(b["input"][0]); Y.append(b["output"][0])
    if not X:
        return None, None
    return (torch.tensor(np.array(X), dtype=torch.float32, device=DEV),
            torch.tensor(np.array(Y), dtype=torch.float32, device=DEV))

if TORCH_OK:
    class CNN(nn.Module):
        def __init__(self, ch, depth):
            super().__init__()
            self.inp = nn.Conv2d(10, ch, 3, padding=1)
            self.blocks = nn.ModuleList([nn.Conv2d(ch, ch, 3, padding=1) for _ in range(depth)])
            self.out = nn.Conv2d(ch, 10, 1)
        def forward(self, x):
            h = torch.relu(self.inp(x))
            for b in self.blocks:
                h = torch.relu(h + b(h))
            return self.out(h)

def _exact(net, X, Y):
    if X is None:
        return True
    with torch.no_grad():
        return bool(((net(X) > 0).float() == Y).all().item())

def train_cnn(examples, ch, depth, steps=8000, lr=5e-3, time_cap=15):
    torch.manual_seed(0)
    tt = list(examples.get("train", [])) + list(examples.get("test", []))
    arc = list(examples.get("arc-gen", []))
    if len(arc) < 4:
        return None, False, False
    nfit = max(1, math.ceil(len(arc) * 0.7))
    Xf, Yf = _tensors((tt + arc[:nfit])[:96])
    Xh, Yh = _tensors(arc[nfit:])
    if Xf is None:
        return None, False, False
    pos = float(Yf.sum().item()); neg = float(Yf.numel()) - pos
    pw = torch.tensor(min(max(neg / max(pos, 1.0), 1.0), 500.0), device=DEV)
    net = CNN(ch, depth).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    t0 = time.time()
    for i in range(steps):
        opt.zero_grad(); lossf(net(Xf), Yf).backward(); opt.step(); sch.step()
        if i % 25 == 0 and _exact(net, Xf, Yf) and _exact(net, Xh, Yh):
            break
        if time.time() - t0 > time_cap:
            break
    net.eval()
    return net, _exact(net, Xf, Yf), _exact(net, Xh, Yh)

def export_onnx(net):
    net = net.to("cpu").eval()
    buf = io.BytesIO()
    torch.onnx.export(net, torch.zeros(1, 10, 30, 30), buf, input_names=["input"],
                      output_names=["output"], opset_version=10, do_constant_folding=True)
    buf.seek(0); m = onnx.load_model_from_string(buf.read()); m.ir_version = IR
    return m

# tiny -> large: the smallest net that fits+generalizes wins, which also golfs the cost.
LADDER = [(8, 2), (16, 4), (32, 8), (48, 12)]
CNN_ON = os.environ.get("NG_CNN", "1") == "1"
CNN_TIMECAP = float(os.environ.get("NG_CNN_TIMECAP", "15"))
BUDGET_S = float(os.environ.get("NG_BUDGET_S", str(7.5 * 3600)))  # finalize before Kaggle's 9h kill

def cnn_candidates(examples):
    if not TORCH_OK or not CNN_ON:
        return
    for ch, depth in LADDER:
        net, fit, gen = train_cnn(examples, ch, depth, time_cap=CNN_TIMECAP)
        if net is not None and fit and gen:
            yield (f"cnn_c{ch}_d{depth}", export_onnx(net)); return


# ----------------------------- driver ---------------------------------------
def solve_task(tnum, examples):
    best = None
    def consider(name, model):
        nonlocal best
        res = evaluate(model, examples, f"t{tnum}_{name}")
        if res.get("ok") and (best is None or res["points"] > best["points"]):
            best = dict(name=name, model=model, points=res["points"],
                        params=res["params"], memory=res["memory"], cost=res["cost"])
    for n, m in symbolic_candidates(examples) or []:
        consider(n, m)
    if best is None:
        for n, m in local_candidates(examples):
            consider(n, m)
    if best is None:
        for n, m in cnn_candidates(examples):
            consider(n, m)
    return best

def _finalize(onx, report, total, solved, t0, done):
    """Write submission.zip + report.json. Called periodically AND at the end, so a
    timeout/kill still leaves a valid partial submission."""
    with zipfile.ZipFile(OUT / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(onx.glob("task*.onnx")):
            z.write(f, f.name)
    json.dump(dict(solved=solved, total_points=round(total, 3),
                   seconds=round(time.time() - t0, 1), done_through=done, tasks=report),
              open(OUT / "report.json", "w"), indent=2)

def main():
    onx = OUT / "onnx"; onx.mkdir(parents=True, exist_ok=True)
    report, total, solved = [], 0.0, 0
    t0 = time.time()
    last = TSTART - 1
    for tnum in range(TSTART, TEND + 1):
        if time.time() - t0 > BUDGET_S:
            print(f"[budget] stopping at task{tnum:03d} after {time.time()-t0:.0f}s", flush=True)
            break
        try:
            examples = json.load(open(f"{INPUT}/task{tnum:03d}.json"))
            best = solve_task(tnum, examples)
        except Exception:
            best = None
            print(f"task{tnum:03d} ERROR\n{traceback.format_exc()}", flush=True)
        if best:
            onnx.save(best["model"], str(onx / f"task{tnum:03d}.onnx"))
            total += best["points"]; solved += 1
            report.append(dict(task=tnum, name=best["name"], points=best["points"],
                               params=best["params"], memory=best["memory"]))
            print(f"task{tnum:03d}: {best['name']:<16} {best['points']:6.2f}pts "
                  f"| total={total:.1f} solved={solved} ({time.time()-t0:.0f}s)", flush=True)
        else:
            report.append(dict(task=tnum, name=None, points=0.0))
        last = tnum
        if tnum % 10 == 0:                      # checkpoint every 10 tasks
            _finalize(onx, report, total, solved, t0, last)
    _finalize(onx, report, total, solved, t0, last)
    print(f"\n==== DONE solved={solved} total_points={total:.2f} "
          f"through task{last:03d} ({time.time()-t0:.0f}s) ====", flush=True)

if __name__ == "__main__":
    main()
