"""family_pb_1 -- minimal-cost ONNX recompilations for tasks 278,254,57,145,172,83,317.

Each rule has a numpy `_ref` (exact reproduction of the verifier rule, used only for
gating: fire iff it reproduces every train+test pair exactly) and a `build_*` that emits
a minimal ONNX graph writing straight to the free `output`.
"""
from __future__ import annotations
import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

FLOAT = TP.FLOAT
INT64 = TP.INT64
BOOL = TP.BOOL
UINT8 = TP.UINT8

GRID = [1, 10, 30, 30]
IR = 10
OPS = [oh.make_opsetid("", 13)]


class G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0

    def u(self, p="t"):
        self.n += 1
        return f"{p}{self.n}"

    def const(self, arr, dt, name=None):
        name = name or self.u("c")
        arr = np.asarray(arr)
        self.inits.append(oh.make_tensor(name, dt, list(arr.shape), arr.flatten().tolist()))
        return name

    def node(self, op, ins, out=None, **attrs):
        out = out or self.u()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out

    def model(self, out_dtype=FLOAT):
        x = oh.make_tensor_value_info("input", FLOAT, GRID)
        y = oh.make_tensor_value_info("output", out_dtype, GRID)
        g = oh.make_graph(self.nodes, "g", [x], [y], self.inits)
        return oh.make_model(g, ir_version=IR, opset_imports=OPS)


def _grids(examples):
    out = []
    for e in examples.get("train", []) + examples.get("test", []):
        out.append((np.array(e["input"]), np.array(e["output"])))
    return out


# ============================ helpers ============================
def _conv_plus(z):
    o = np.zeros_like(z, dtype=np.int32)
    o[1:, :] += z[:-1, :]; o[:-1, :] += z[1:, :]
    o[:, 1:] += z[:, :-1]; o[:, :-1] += z[:, 1:]
    return o


# ============================ T278: domino outbox ============================
# objects size==2 (dominoes) of color 2 -> draw color-3 outbox ring around each.
# All fg comps are size 1 or 2, so domino cell = fg cell with exactly one ortho
# fg neighbor. ring = dilate3x3(D) & ~D & in-grid.
def _ref278(i):
    if not set(np.unique(i)).issubset({0, 2}):
        return None
    fg = (i == 2).astype(np.int32)
    n = _conv_plus(fg)
    D = (fg == 1) & (n == 1)
    Df = D.astype(np.int32)
    dil = np.zeros_like(Df)
    H, W = Df.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ys0 = max(0, dy); ys1 = min(H, H + dy)
            xs0 = max(0, dx); xs1 = min(W, W + dx)
            dil[ys0:ys1, xs0:xs1] |= Df[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    ring = (dil == 1) & (~D)
    out = i.copy()
    out[ring] = 3
    return out


def build278():
    g = G()
    # crop channel 2 and channel 0 to 18x18
    s2 = g.const([2, 0, 0], INT64); e2 = g.const([3, 18, 18], INT64)
    ax = g.const([1, 2, 3], INT64)
    fg = g.node("Slice", ["input", s2, e2, ax])               # [1,1,18,18] f32
    s0 = g.const([0, 0, 0], INT64); e0 = g.const([1, 18, 18], INT64)
    ch0 = g.node("Slice", ["input", s0, e0, ax])              # [1,1,18,18] f32
    plus = g.const(np.array([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], np.float32), FLOAT)
    n = g.node("Conv", [fg, plus], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    one = g.const([1.0], FLOAT)
    half = g.const([0.5], FLOAT)
    nis1 = g.node("Equal", [n, one])                          # bool
    fgb = g.node("Greater", [fg, half])                       # bool
    Db = g.node("And", [fgb, nis1])                           # bool domino
    Du = g.node("Cast", [Db], to=UINT8)
    dil = g.node("MaxPool", [Du], kernel_shape=[3, 3], pads=[1, 1, 1, 1])  # uint8
    z8 = g.const(np.array([0], np.uint8), UINT8)
    dilb = g.node("Greater", [dil, z8])                       # bool
    notD = g.node("Not", [Db])
    ch0b = g.node("Greater", [ch0, half])
    occ = g.node("Or", [fgb, ch0b])
    r1 = g.node("And", [dilb, notD])
    ring = g.node("And", [r1, occ])                           # [1,1,18,18] bool
    pads = g.const([0, 0, 0, 0, 0, 0, 12, 12], INT64)
    fzero = g.const([False], BOOL)
    ringp = g.node("Pad", [ring, pads, fzero], mode="constant")  # [1,1,30,30] bool
    e3 = np.zeros((1, 10, 1, 1), np.float32); e3[0, 3, 0, 0] = 1.0
    e3c = g.const(e3, FLOAT)
    g.node("Where", [ringp, e3c, "input"], out="output")
    return g.model()


# ============================ T254: bar recolor by height ============================
# 5-bars (bottom-anchored single-column). tallest col -> 1, shortest col -> 2, rest -> 0.
def _ref254(i):
    if not set(np.unique(i)).issubset({0, 5}):
        return None
    H, W = i.shape
    m5 = (i == 5).astype(np.int32)
    cnt = m5.sum(0)
    nz = cnt[cnt > 0]
    if nz.size == 0:
        return None
    mx = cnt.max(); mn = nz.min()
    out = np.zeros_like(i)
    for w in range(W):
        if cnt[w] == 0:
            continue
        col = m5[:, w] > 0
        out[col, w] = 1 if cnt[w] == mx else (2 if cnt[w] == mn else 0)
    return out


def build254():
    g = G()
    # channel-5 count per column, straight off input (no [.,.,30,30] materialized)
    e5 = np.zeros((10,), np.float32); e5[5] = 1.0
    e5c = g.const(e5, FLOAT)
    cnt30 = g.node("Einsum", ["input", e5c], equation="nchw,c->nw")   # [1,30]
    s = g.const([0], INT64); e = g.const([9], INT64); ax = g.const([1], INT64)
    cnt = g.node("Slice", [cnt30, s, e, ax])                          # [1,9]
    cntc = g.node("Reshape", [cnt, g.const([1, 1, 1, 9], INT64)])     # [1,1,1,9] f32
    mx = g.node("ReduceMax", [cntc], axes=[3], keepdims=1)            # [1,1,1,1]
    big = g.const([[[[99.0]]]], FLOAT)
    zero = g.const([[[[0.0]]]], FLOAT)
    cntpos = g.node("Where", [g.node("Greater", [cntc, zero]), cntc, big])
    mn = g.node("ReduceMin", [cntpos], axes=[3], keepdims=1)          # [1,1,1,1]
    # per-column color code (uint8): 1 at maxcol, 2 at mincol, else 0
    u0 = g.const(np.array([0], np.uint8), UINT8)
    u1 = g.const(np.array([1], np.uint8), UINT8)
    u2 = g.const(np.array([2], np.uint8), UINT8)
    ismax = g.node("Equal", [cntc, mx])                              # [1,1,1,9] bool
    ismin = g.node("Equal", [cntc, mn])
    valcol = g.node("Where", [ismax, u1, g.node("Where", [ismin, u2, u0])])  # [1,1,1,9] u8
    # bottom-anchored bar: lit iff cnt[w] > 8-r  (one Greater, no Add)
    rampd = g.const(np.arange(8, -1, -1, dtype=np.float32).reshape(1, 1, 9, 1), FLOAT)  # [1,1,9,1]
    bar = g.node("Greater", [cntc, rampd])                           # [1,1,9,9] bool
    idx9 = g.node("Where", [bar, valcol, u0])                        # [1,1,9,9] u8 {0,1,2}
    ramp3 = g.const(np.arange(3, dtype=np.uint8).reshape(3, 1, 1), UINT8)
    oh3 = g.node("Equal", [idx9, ramp3])                             # [1,3,9,9] bool
    pads = g.const([0, 0, 0, 0, 0, 7, 21, 21], INT64)
    fzero = g.const([False], BOOL)
    g.node("Pad", [oh3, pads, fzero], out="output", mode="constant")
    return g.model(out_dtype=BOOL)


# --------------------------- registry ---------------------------
RULES = [
    ("t278", _ref278, build278),
    ("t254", _ref254, build254),
]


def candidates(examples):
    prs = _grids(examples)
    if not prs:
        return []
    res = []
    for name, ref, build in RULES:
        ok = True
        for i, o in prs:
            if max(i.shape) > 30:
                ok = False; break
            p = ref(i)
            if p is None or p.shape != o.shape or not np.array_equal(p, o):
                ok = False; break
        if ok:
            try:
                res.append((name, build()))
            except Exception:
                pass
    return res
