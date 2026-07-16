"""family_pcrk9_0 -- exact, generalising ARC->ONNX solvers for the hardest
residual slice (indices U[0::4]).

Sub-families
------------
* pcrk9_tmatch (task 182): the grid holds ONE hollow 5-rectangle ("box") whose
  interior is a small shape of colour X (the template).  Elsewhere the grid holds
  several colour-1 objects.  Every colour-1 object whose shape is EXACTLY the
  template (translation-equal, isolated) is recoloured to X; the rest stay 1.

  The template is found without component labelling: the box interior is the set
  of non-5 cells that have a 5 strictly on all four sides (prefix/suffix-OR of the
  5-mask via Hillis-Steele doubling -- grid-agnostic).  Matching is an exact
  template correlation: at every origin require  corr(M1,Tn)==|T|  (all template
  cells are 1) AND  corr(M1,Cn)==0  (the template's bbox complement + 1-cell ring
  is empty -> isolation & no spurious sub-window hit).  Both correlations are
  Conv with the DATA-DERIVED template as the (computed) weight; the recolour is a
  ConvTranspose stamp of the template at every match.  Everything is static-shape
  (Gather-shift to normalise the template, never a dynamic Slice) so the cost is
  measurable.  Fully validated: 267/267 (train+test+arc-gen), ~12.0 pts.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
FLOAT = onnx.TensorProto.FLOAT
G = HEIGHT           # 30
K = 9                # max template bbox + 2-cell margin (data max is 4x5)


# ============================================================ numpy reference
def _enc_prefix(A):
    """Box interior = non-5 cells with a 5 strictly on every side (via doubling)."""
    five = (A == 5).astype(np.int64)

    def pomax(x, axis, rev):
        y = np.flip(x, axis) if rev else x
        y = y.copy()
        for s in (1, 2, 4, 8, 16):
            sh = np.zeros_like(y)
            if axis == 1:
                sh[:, s:] = y[:, :-s]
            else:
                sh[s:, :] = y[:-s, :]
            y = np.maximum(y, sh)
        return np.flip(y, axis) if rev else y

    def strict(x, axis, rev):
        sh = np.zeros_like(x)
        if not rev:
            if axis == 1:
                sh[:, 1:] = x[:, :-1]
            else:
                sh[1:, :] = x[:-1, :]
        else:
            if axis == 1:
                sh[:, :-1] = x[:, 1:]
            else:
                sh[:-1, :] = x[1:, :]
        return pomax(sh, axis, rev)

    L = strict(five, 1, False); R = strict(five, 1, True)
    U = strict(five, 0, False); D = strict(five, 0, True)
    return (L > 0) & (R > 0) & (U > 0) & (D > 0) & (A != 5)


def _solve_tmatch(I):
    H, W = I.shape
    A = np.zeros((G, G), int); A[:H, :W] = I
    enc = _enc_prefix(A)
    xs = [int(v) for v in np.unique(A[enc]) if v not in (0, 1)]
    if len(xs) != 1:
        return None
    X = xs[0]
    Tmask = enc & (A == X)
    ys, xx = np.where(Tmask)
    if len(ys) == 0:
        return None
    r0, c0 = ys.min(), xx.min()
    th = ys.max() - ys.min() + 1
    tw = xx.max() - xx.min() + 1
    if th + 2 > K or tw + 2 > K:
        return None
    Tn = np.zeros((K, K), int)
    for y, x in zip(ys, xx):
        Tn[y - r0 + 1, x - c0 + 1] = 1
    Rect = np.zeros((K, K), int); Rect[0:th + 2, 0:tw + 2] = 1
    Cn = Rect & (~Tn)
    tcnt = Tn.sum()
    M1 = (A == 1).astype(int)
    P = np.zeros((G + K, G + K), int); P[:G, :G] = M1
    recolor = np.zeros((G, G), int)
    for r in range(G):
        for c in range(G):
            win = P[r:r + K, c:c + K]
            if (win * Tn).sum() == tcnt and (win * Cn).sum() == 0:
                for a in range(K):
                    for b in range(K):
                        if Tn[a, b] and r + a < G and c + b < G:
                            recolor[r + a, c + b] = 1
    O = A.copy(); O[recolor > 0] = X
    return O[:H, :W]


# ================================================================ onnx builder
def _build_tmatch():
    nodes, inits = [], []

    def C(name, arr, dt=FLOAT):
        a = np.asarray(arr)
        inits.append(oh.make_tensor(name, dt, list(a.shape), a.flatten().tolist()))
        return name

    def N(op, ins, out, **kw):
        nodes.append(oh.make_node(op, ins, [out], **kw)); return out

    uid = [0]

    def U(p):
        uid[0] += 1; return f"{p}{uid[0]}"

    def shift(x, dy, dx):
        cur = x
        for axis, d in ((2, dy), (3, dx)):
            if d == 0:
                continue
            ad = abs(d)
            s_s, s_e, s_a = U("ss"), U("se"), U("sa")
            if d > 0:
                C(s_s, [0], INT64); C(s_e, [G - ad], INT64); C(s_a, [axis], INT64)
                sl = U("sl"); N("Slice", [cur, s_s, s_e, s_a], sl)
                pd = [0, 0, 0, 0, 0, 0, 0, 0]; pd[axis] = ad
                out = U("sh"); N("Pad", [sl], out, mode="constant", pads=pd, value=0.0)
            else:
                C(s_s, [ad], INT64); C(s_e, [G], INT64); C(s_a, [axis], INT64)
                sl = U("sl"); N("Slice", [cur, s_s, s_e, s_a], sl)
                pd = [0, 0, 0, 0, 0, 0, 0, 0]; pd[4 + axis] = ad
                out = U("sh"); N("Pad", [sl], out, mode="constant", pads=pd, value=0.0)
            cur = out
        return cur

    def maxof(a, b):
        o = U("mx"); N("Max", [a, b], o); return o

    def running_max(x, axis, sign):
        y = x
        for s in (1, 2, 4, 8, 16):
            dy = sign * s if axis == 2 else 0
            dx = sign * s if axis == 3 else 0
            y = maxof(y, shift(y, dy, dx))
        return y

    C("i1", [1], INT64); N("Gather", ["input", "i1"], "M1", axis=1)
    C("i5", [5], INT64); N("Gather", ["input", "i5"], "five", axis=1)

    fL = shift("five", 0, 1); L = running_max(fL, 3, 1)
    fR = shift("five", 0, -1); R = running_max(fR, 3, -1)
    fU = shift("five", 1, 0); Uu = running_max(fU, 2, 1)
    fD = shift("five", -1, 0); Dd = running_max(fD, 2, -1)
    C("half", [0.5])
    N("Greater", [L, "half"], "Lb"); N("Greater", [R, "half"], "Rb")
    N("Greater", [Uu, "half"], "Ub"); N("Greater", [Dd, "half"], "Db")
    N("And", ["Lb", "Rb"], "LR"); N("And", ["Ub", "Db"], "UD"); N("And", ["LR", "UD"], "enc4b")
    N("Greater", ["five", "half"], "fiveb"); N("Not", ["fiveb"], "notfive")
    N("And", ["enc4b", "notfive"], "encb"); N("Cast", ["encb"], "enc", to=FLOAT)

    N("Mul", ["input", "enc"], "ie")
    N("ReduceSum", ["ie"], "chcnt", axes=[2, 3], keepdims=1)
    N("Greater", ["chcnt", "half"], "present"); N("Cast", ["present"], "presentf", to=FLOAT)
    m = np.ones((1, CHANNELS, 1, 1), float); m[0, 0] = 0; m[0, 1] = 0
    C("chmask", m); N("Mul", ["presentf", "chmask"], "selX")

    N("Mul", ["input", "selX"], "isx")
    N("ReduceSum", ["isx"], "Xany", axes=[1], keepdims=1)
    N("Mul", ["Xany", "enc"], "Tmask")

    N("ReduceMax", ["Tmask"], "rowany", axes=[3], keepdims=1)
    N("ReduceMax", ["Tmask"], "colany", axes=[2], keepdims=1)
    C("ridx", np.arange(G, dtype=np.float32).reshape(1, 1, G, 1))
    C("cidx", np.arange(G, dtype=np.float32).reshape(1, 1, 1, G))
    C("big", [1e6]); C("neg", [-1.0]); C("one", [1.0]); C("three", [3.0])
    N("Greater", ["rowany", "half"], "rb")
    N("Where", ["rb", "ridx", "big"], "rmn"); N("ReduceMin", ["rmn"], "r0", axes=[2], keepdims=1)
    N("Where", ["rb", "ridx", "neg"], "rmx"); N("ReduceMax", ["rmx"], "r1", axes=[2], keepdims=1)
    N("Greater", ["colany", "half"], "cb")
    N("Where", ["cb", "cidx", "big"], "cmn"); N("ReduceMin", ["cmn"], "c0", axes=[3], keepdims=1)
    N("Where", ["cb", "cidx", "neg"], "cmx"); N("ReduceMax", ["cmx"], "c1", axes=[3], keepdims=1)

    # normalise template via data-dependent Gather-shift (static output shape)
    N("Sub", ["r0", "one"], "srf"); N("Sub", ["c0", "one"], "scf")
    C("shp1", [1], INT64)
    N("Reshape", ["srf", "shp1"], "sr1f"); N("Reshape", ["scf", "shp1"], "sc1f")
    C("baseR", np.arange(G, dtype=np.float32)); C("baseC", np.arange(G, dtype=np.float32))
    C("f29", [float(G - 1)]); C("f0", [0.0])
    N("Add", ["baseR", "sr1f"], "ridxA"); N("Min", ["ridxA", "f29"], "ridxB"); N("Max", ["ridxB", "f0"], "ridxSf")
    N("Add", ["baseC", "sc1f"], "cidxA"); N("Min", ["cidxA", "f29"], "cidxB"); N("Max", ["cidxB", "f0"], "cidxSf")
    N("Cast", ["ridxSf"], "ridxS", to=INT64); N("Cast", ["cidxSf"], "cidxS", to=INT64)
    N("Gather", ["Tmask", "ridxS"], "Tshr", axis=2)
    N("Gather", ["Tshr", "cidxS"], "Tsh", axis=3)
    C("stK", [0, 0], INT64); C("enK", [K, K], INT64); C("axKp", [2, 3], INT64)
    N("Slice", ["Tsh", "stK", "enK", "axKp"], "Tn")

    C("aK", np.arange(K, dtype=np.float32).reshape(1, 1, K, 1))
    C("bK", np.arange(K, dtype=np.float32).reshape(1, 1, 1, K))
    N("Sub", ["r1", "r0"], "dr"); N("Add", ["dr", "three"], "thr")
    N("Sub", ["c1", "c0"], "dc"); N("Add", ["dc", "three"], "twr")
    N("Less", ["aK", "thr"], "rowok"); N("Less", ["bK", "twr"], "colok")
    N("And", ["rowok", "colok"], "rectb"); N("Cast", ["rectb"], "Rect", to=FLOAT)
    N("Sub", ["one", "Tn"], "invTn"); N("Mul", ["Rect", "invTn"], "Cn")
    N("ReduceSum", ["Tn"], "tcnt", axes=[2, 3], keepdims=1)

    padM = [0, 0, 0, 0, 0, 0, K, K]
    N("Pad", ["M1"], "M1pad", mode="constant", pads=padM, value=0.0)
    N("Conv", ["M1pad", "Tn"], "corrTf", kernel_shape=[K, K], pads=[0, 0, 0, 0])
    N("Conv", ["M1pad", "Cn"], "corrCf", kernel_shape=[K, K], pads=[0, 0, 0, 0])
    C("st00", [0, 0], INT64); C("en3030", [G, G], INT64); C("ax23b", [2, 3], INT64)
    N("Slice", ["corrTf", "st00", "en3030", "ax23b"], "corrT")
    N("Slice", ["corrCf", "st00", "en3030", "ax23b"], "corrC")
    N("Sub", ["corrT", "tcnt"], "dT"); N("Abs", ["dT"], "adT"); N("Less", ["adT", "half"], "mT")
    N("Less", ["corrC", "half"], "mC")
    N("And", ["mT", "mC"], "matchb"); N("Cast", ["matchb"], "match", to=FLOAT)

    N("ConvTranspose", ["match", "Tn"], "stampf", kernel_shape=[K, K], pads=[0, 0, 0, 0])
    N("Slice", ["stampf", "st00", "en3030", "ax23b"], "stamp")
    N("Greater", ["stamp", "half"], "recb"); N("Cast", ["recb"], "recolor", to=FLOAT)

    e1 = np.zeros((1, CHANNELS, 1, 1), float); e1[0, 1] = 1.0; C("e1", e1)
    N("Sub", ["selX", "e1"], "selD"); N("Mul", ["selD", "recolor"], "delta")
    N("Add", ["input", "delta"], "output")

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "pcrk9_tmatch", [x], [y], inits)
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# ===================================================================== registry
_CACHE = {}
_SOLVERS = [
    ("pcrk9_tmatch", _solve_tmatch, _build_tmatch),
]


def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return []
    out = []
    for name, solve, build in _SOLVERS:
        good = True
        for I, O in prs:
            if I.shape != O.shape:
                good = False; break
            pred = solve(I)
            if pred is None or not np.array_equal(pred, O):
                good = False; break
        if good:
            if name not in _CACHE:
                _CACHE[name] = build()
            out.append((name, _CACHE[name]))
    return out
