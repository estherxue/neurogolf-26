"""family_golf8_0 -- aggressive re-golf of the lowest-scoring solvers.

Strategy: keep every intermediate SINGLE-CHANNEL [1,1,30,30] (or [1,1,1,30]);
expand to the 10-channel one-hot only via a FREE data-dependent MatMul into
`output`.  Periods / shifts / selections are computed as [1,1,1,30] column
signatures (120 bytes) rather than full [1,1,30,30] planes, then realised as a
single [1,1,30,30] selection matrix applied with MatMul (output is free).

Each solver detects its rule in numpy, reproduces EVERY train+test+arc-gen pair
exactly, and only then emits.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE
G = HEIGHT  # 30


# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, F, list(dims),
                          [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        return n

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _check(m):
    onnx.checker.check_model(m, full_check=True)


def _pairs(ex, splits):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > G or max(b.shape) > G:
                continue
            out.append((a, b))
    return out


# =========================================================================== #
# TASK 343 -- horizontal period tiling.                                        #
#   out[r][c] = in[r][ c mod p ]   for c < W  (p = smallest horizontal period) #
# =========================================================================== #
HT_QMAX = 12   # observed max period 8 + safety buffer for held-out
HT_BASE = 10.0


def _htile_sim(a):
    H, W = a.shape
    P = np.zeros((G, G), np.float64); P[:H, :W] = a
    real = np.zeros((G, G), np.float64); real[:H, :W] = 1.0
    cidx = np.arange(G).astype(np.float64)
    colhas = (P > 0.5).max(0)
    if colhas.sum() < 0.5:
        return None
    L = (cidx * colhas).max()
    wr = (HT_BASE ** np.arange(G)).astype(np.float64)[:, None]
    sig = (P * wr).sum(0)                                   # [G]
    remaining, p = 1.0, 0.0
    for d in range(1, HT_QMAX + 1):
        sigsh = np.zeros(G); sigsh[d:] = sig[:G - d]
        mism = (np.abs(sig - sigsh) > 0.5).astype(np.float64)
        valid = ((cidx >= d) & (cidx <= L + 0.5)).astype(np.float64)
        zero = 1.0 if (mism * valid).sum() < 0.5 else 0.0
        p += zero * remaining * d
        remaining *= (1.0 - zero)
    if p < 0.5:
        return None
    modp = np.mod(cidx, p)
    T = (np.abs(cidx[:, None] - modp[None, :]) < 0.5).astype(np.float64)
    T *= real.max(0)[None, :]
    out = np.zeros((H, W), int)
    for r in range(H):
        for c in range(W):
            out[r, c] = int(a[r, int(c % p)])
    return out


def _build_htile():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    # signature per column via a full-height Conv:  sig[c] = sum_{ch,r} ch*base^r*input[ch,r,c]
    sw = [0.0] * (CHANNELS * G)
    for ch in range(CHANNELS):
        for r in range(G):
            sw[ch * G + r] = ch * (HT_BASE ** r)
    sigW = oh.make_tensor(g.nm("W"), F, [1, CHANNELS, G, 1], sw)
    g.inits.append(sigW)
    sig = g.nd("Conv", ["input", sigW.name], kernel_shape=[G, 1], pads=[0, 0, 0, 0])  # [1,1,1,30]
    cidx = g.f([1, 1, 1, G], list(range(G)))
    colhas = g.nd("Cast", [g.nd("Greater", [sig, half])], to=F)                       # [1,1,1,30] col has colour
    L = g.nd("ReduceMax", [g.nd("Mul", [cidx, colhas])], axes=[3], keepdims=1)        # [1,1,1,1]
    colcnt = g.nd("ReduceSum", ["input"], axes=[1, 2], keepdims=1)                    # [1,1,1,30] grid-cells/col
    colreal = g.nd("Cast", [g.nd("Greater", [colcnt, half])], to=F)                   # [1,1,1,30] col in grid
    Ladd = g.nd("Add", [L, half])
    cleL = g.nd("Cast", [g.nd("Less", [cidx, Ladd])], to=F)                          # c<=L
    one = g.f([1, 1, 1, 1], [1.0])
    remaining = one
    p = g.f([1, 1, 1, 1], [0.0])
    for d in range(1, HT_QMAX + 1):
        sl = g.nd("Slice", [sig, g.i64([0]), g.i64([G - d]), g.i64([3])])
        sigsh = g.nd("Pad", [sl], mode="constant", value=0.0,
                     pads=[0, 0, 0, d, 0, 0, 0, 0])                                  # shift right d
        diff = g.nd("Abs", [g.nd("Sub", [sig, sigsh])])
        mism = g.nd("Cast", [g.nd("Greater", [diff, half])], to=F)
        ge = g.f([1, 1, 1, G], [1.0 if c >= d else 0.0 for c in range(G)])
        valid = g.nd("Mul", [ge, cleL])
        score = g.nd("ReduceSum", [g.nd("Mul", [mism, valid])], axes=[3], keepdims=1)
        zero = g.nd("Cast", [g.nd("Less", [score, half])], to=F)                     # [1,1,1,1]
        gate = g.nd("Mul", [zero, remaining])
        p = g.nd("Add", [p, g.nd("Mul", [gate, g.f([1, 1, 1, 1], [float(d)])])])
        remaining = g.nd("Mul", [remaining, g.nd("Sub", [one, zero])])
    modp = g.nd("Mod", [cidx, p], fmod=1)                                            # [1,1,1,30] in [0,p)
    # out-of-grid columns (c>=W) gather from the padding column G-1 (guaranteed empty when W<G).
    one = g.f([1, 1, 1, 1], [1.0])
    lastc = g.f([1, 1, 1, 1], [float(G - 1)])
    idxf = g.nd("Add", [g.nd("Mul", [modp, colreal]),
                        g.nd("Mul", [lastc, g.nd("Sub", [one, colreal])])])
    idx = g.nd("Cast", [g.nd("Reshape", [idxf, g.i64([G])])], to=INT64)              # [30]
    g.nd("Gather", ["input", idx], "output", axis=3)                                 # column tiling (free)
    return _model(g)


def _try_htile(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or any(a.shape != b.shape for a, b in det):
        return None
    if all(np.array_equal(a, b) for a, b in det):
        return None
    for a, b in det + allp:
        if a.shape != b.shape:
            return None
        pred = _htile_sim(a)
        if pred is None or not np.array_equal(pred, b):
            return None
    try:
        m = _build_htile(); _check(m)
    except Exception:
        return None
    return ("htile", m)



# =========================================================================== #
# TASK 84 -- square grid, uniform colour C in column 0.  Draw the outer anti-  #
# diagonal (r+c==N-1, c>=1) in colour 2 and the bottom row (r==N-1, c>=1) in   #
# colour 4; keep column 0 = C, everything else background.                     #
#   output = Where(bottom, e4, Where(antidiag, e2, input))   (one intermediate)#
# =========================================================================== #
def _diagbar_sim(a):
    H, W = a.shape
    if H != W:
        return None
    N = H
    c0 = a[:, 0]
    if c0[0] == 0 or not (c0 == c0[0]).all():
        return None
    out = np.zeros((H, W), int)
    out[:, 0] = c0[0]
    for r in range(N - 1):
        out[r, N - 1 - r] = 2
    out[N - 1, 1:] = 4
    return out


def _e(ch):
    return [1.0 if k == ch else 0.0 for k in range(CHANNELS)]


def _build_diagbar():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    colcnt = g.nd("ReduceSum", ["input"], axes=[1, 2], keepdims=1)                    # [1,1,1,30]
    colreal = g.nd("Cast", [g.nd("Greater", [colcnt, half])], to=F)                   # [1,1,1,30] c<W
    N = g.nd("ReduceSum", [colreal], axes=[3], keepdims=1)                            # [1,1,1,1]
    maxidx = g.nd("Sub", [N, one])                                                    # N-1
    # anti-diagonal mask: r+c == N-1 and c>=1  (col0 pre-set to -100 so it never matches)
    rc = g.f([1, 1, G, G], [(r + c if c >= 1 else -100.0) for r in range(G) for c in range(G)])
    diagb = g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rc, maxidx])]), half])            # bool [1,1,30,30]
    e2 = g.f([1, CHANNELS, 1, 1], _e(2))
    tmp = g.nd("Where", [diagb, e2, "input"])                                         # [1,10,30,30]
    # bottom row mask: r == N-1 and c in [1, W-1]
    rowidx = g.f([1, 1, G, 1], list(range(G)))
    reqF = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, maxidx])]), half])], to=F)  # [1,1,30,1]
    cge1 = g.f([1, 1, 1, G], [0.0] + [1.0] * (G - 1))
    colkeep = g.nd("Mul", [colreal, cge1])                                            # [1,1,1,30]
    botb = g.nd("Cast", [g.nd("Mul", [reqF, colkeep])], to=onnx.TensorProto.BOOL)     # bool [1,1,30,30]
    e4 = g.f([1, CHANNELS, 1, 1], _e(4))
    g.nd("Where", [botb, e4, tmp], "output")
    return _model(g)


def _try_diagbar(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    for a, b in det + allp:
        if a.shape != b.shape:
            return None
        pred = _diagbar_sim(a)
        if pred is None or not np.array_equal(pred, b):
            return None
    try:
        m = _build_diagbar(); _check(m)
    except Exception:
        return None
    return ("diagbar", m)


# --------------------------------------------------------------------------- #
_SOLVERS = [_try_htile, _try_diagbar]


def candidates(examples):
    out = []
    for fn in _SOLVERS:
        try:
            r = fn(examples)
        except Exception:
            r = None
        if r:
            out.append(r)
    return out
