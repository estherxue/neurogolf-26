"""crk5_4: data-dependent periodic-hole reconstruction (opset-10, origin-anchored).

Targeted task family (e.g. task 394):
  * The input grid is a fully-coloured PERIODIC texture with one rectangular
    block of background (colour 0) punched out of it ("the hole").  The grid is
    zero-padded TOP-LEFT to 30x30, so the hole is exactly channel-0 of the
    one-hot input (the rest of the grid is non-zero, the padding is all-zero).
  * The OUTPUT is the reconstructed contents of the hole, cropped to the hole's
    bounding box and placed at the top-left.

Static-graph recipe (all intermediates STATIC [1,1,30,30]):
  1. V  = sum_c c*channel_c            (colour-value grid; 0 on hole+padding)
     P  = (V>0)                        (presence of a colour)
     Hole = channel-0 of the input     (1 exactly on the hole cells)
  2. Detect smallest row period pr in 1..6: pr = min{ p : shifting V up by p
     agrees with V on every overlapping coloured cell AND the overlap is
     non-empty }.  Likewise the column period pc.  (static Slice/Pad shifts,
     argmin via masked Min of {p or BIG}.)
  3. Fill the hole separably with data-dependent period shifts realised by
     MatMul shift matrices  S[i,k]=1 iff (i-k)==dy :
        fr = max_k  shift_rows(V , k*pr)      k in -3..3
        filled = max_l shift_cols(fr, l*pc)   l in -3..3
     (every valid period multiple carries the same colour, so an elementwise
     MAX recovers the unique value and leaves coloured cells untouched.)
  4. Crop: read the hole bbox (r0,c0,r1,c1) off `Hole`, MatMul-shift `filled`
     by (-r0,-c0) to the origin, and mask to rows<=r1-r0 & cols<=c1-c0.
  5. Re-expand the cropped colour-value grid to a one-hot [1,10,30,30]
     (channel 0 is identically zero because the hole contents are never 0).

A numpy mirror reproduces the exact float-then-threshold semantics; the
candidate is emitted only when it matches every available train+test+arc-gen
pair, so it never fires on unrelated tasks.
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
H, W = HEIGHT, WIDTH
_CBIG = 1000.0
_BIG = 50.0          # invalid-period sentinel
MAXP = 6             # candidate periods 1..6
KR = (-3, -2, -1, 1, 2, 3)   # period multiples used to fill


# --------------------------------------------------------------------------- #
# graph accumulator                                                           #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0
        self._cache = {}

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(
            n, F, list(dims), [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        return n

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out

    def scal(self, key, v):
        if key not in self._cache:
            self._cache[key] = self.f([1, 1, 1, 1], [v])
        return self._cache[key]


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# ONNX builder                                                                #
# --------------------------------------------------------------------------- #
def build():
    g = _G()
    half = g.scal("half", 0.5)
    cbig = g.scal("cbig", _CBIG)

    rowidx = g.f([1, 1, H, 1], list(range(H)))      # i along axis2
    colidx = g.f([1, 1, 1, W], list(range(W)))      # k/j along axis3
    DR = g.nd("Sub", [rowidx, colidx])              # [1,1,30,30]  DR[i,k]=i-k
    DC = g.nd("Sub", [colidx, rowidx])              # DC[k,j]=j-k

    # colour-value grid V and presence P
    wv = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    V = g.nd("Conv", [g_in := "input", wv], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    P = g.nd("Cast", [g.nd("Greater", [V, half])], to=F)        # 1 where coloured
    Hole = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])  # channel 0

    def shift_static_up(plane, p):
        """plane shifted UP by static p rows (zero fill)."""
        sl = g.nd("Slice", [plane, g.i64([p]), g.i64([H]), g.i64([2])])
        return g.nd("Pad", [sl], mode="constant", value=0.0,
                    pads=[0, 0, 0, 0, 0, 0, p, 0])

    def shift_static_left(plane, p):
        sl = g.nd("Slice", [plane, g.i64([p]), g.i64([W]), g.i64([3])])
        return g.nd("Pad", [sl], mode="constant", value=0.0,
                    pads=[0, 0, 0, 0, 0, 0, 0, p])

    def period(axis):
        """Return smallest valid period scalar [1,1,1,1] for axis (0 row / 1 col)."""
        score = None
        for p in range(1, MAXP + 1):
            Vs = shift_static_up(V, p) if axis == 0 else shift_static_left(V, p)
            Ps = g.nd("Cast", [g.nd("Greater", [Vs, half])], to=F)
            overlap = g.nd("Mul", [P, Ps])
            ov_cnt = g.nd("ReduceSum", [overlap], axes=[2, 3], keepdims=1)   # [1,1,1,1]
            diff = g.nd("Abs", [g.nd("Sub", [V, Vs])])
            mism = g.nd("Cast", [g.nd("Greater", [diff, half])], to=F)
            mm_cnt = g.nd("ReduceSum", [g.nd("Mul", [overlap, mism])], axes=[2, 3], keepdims=1)
            has_ov = g.nd("Cast", [g.nd("Greater", [ov_cnt, half])], to=F)
            no_mm = g.nd("Cast", [g.nd("Less", [mm_cnt, half])], to=F)
            valid = g.nd("Mul", [has_ov, no_mm])
            inv = g.nd("Sub", [g.scal("one", 1.0), valid])
            sc = g.nd("Add", [g.nd("Mul", [valid, g.scal(f"p{p}", float(p))]),
                              g.nd("Mul", [inv, cbig])])     # p or CBIG
            score = sc if score is None else g.nd("Min", [score, sc])
        return score

    pr = period(0)
    pc = period(1)

    def shift_rows(plane, dy):
        S = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [DR, dy])]), half])], to=F)
        return g.nd("MatMul", [S, plane])

    def shift_cols(plane, dx):
        S = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [DC, dx])]), half])], to=F)
        return g.nd("MatMul", [plane, S])

    # separable fill
    fr = V
    for k in KR:
        dy = g.nd("Mul", [pr, g.scal(f"k{k}", float(k))])
        fr = g.nd("Max", [fr, shift_rows(V, dy)])
    filled = fr
    for k in KR:
        dx = g.nd("Mul", [pc, g.scal(f"k{k}", float(k))])
        filled = g.nd("Max", [filled, shift_cols(fr, dx)])

    # hole bbox
    rowhas = g.nd("ReduceMax", [Hole], axes=[3], keepdims=1)     # [1,1,30,1]
    colhas = g.nd("ReduceMax", [Hole], axes=[2], keepdims=1)     # [1,1,1,30]
    r1 = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    r0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    c1 = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    c0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])

    # crop hole to origin: shift up by r0, left by c0  -> dy=-r0, dx=-c0
    neg_r0 = g.nd("Sub", [g.scal("zero", 0.0), r0])
    neg_c0 = g.nd("Sub", [g.scal("zero", 0.0), c0])
    cropped = shift_cols(shift_rows(filled, neg_r0), neg_c0)

    # mask to rows<=r1-r0 and cols<=c1-c0
    rr = g.nd("Sub", [r1, r0])
    cc = g.nd("Sub", [c1, c0])
    rowmask = g.nd("Cast", [g.nd("Less", [rowidx, g.nd("Add", [rr, half])])], to=F)
    colmask = g.nd("Cast", [g.nd("Less", [colidx, g.nd("Add", [cc, half])])], to=F)
    mask = g.nd("Mul", [rowmask, colmask])
    Vout = g.nd("Mul", [cropped, mask])

    # one-hot expansion
    zero_plane = g.nd("Mul", [Vout, g.scal("zero", 0.0)])
    planes = [zero_plane]
    for c in range(1, CHANNELS):
        diff = g.nd("Abs", [g.nd("Sub", [Vout, g.scal(f"col{c}", float(c))])])
        planes.append(g.nd("Cast", [g.nd("Less", [diff, half])], to=F))
    g.nd("Concat", planes, "output", axis=1)
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy mirror (exact float-then-threshold semantics)                         #
# --------------------------------------------------------------------------- #
def _to30(a):
    g = np.zeros((H, W), np.float64)
    g[:a.shape[0], :a.shape[1]] = a
    return g


def _shift(Vg, dy, dx):
    r = np.zeros_like(Vg)
    ys, xs = np.where(Vg != 0)
    for i, j in zip(ys, xs):
        ni, nj = i + dy, j + dx
        if 0 <= ni < H and 0 <= nj < W:
            r[ni, nj] = Vg[i, j]
    return r


def _period(Vg, P, axis):
    for p in range(1, MAXP + 1):
        Vs = _shift(Vg, -p, 0) if axis == 0 else _shift(Vg, 0, -p)
        m = (Vg != 0) & (Vs != 0)
        if m.sum() == 0:
            continue
        if np.all(Vg[m] == Vs[m]):
            return p
    return _BIG


def _predict_onehot(a):
    """Return predicted one-hot bool [10,30,30] or None."""
    hh, ww = a.shape
    Vg = _to30(a)
    G = np.zeros((H, W), np.float64)
    G[:hh, :ww] = 1.0
    Hole = ((Vg == 0) & (G == 1)).astype(np.float64)
    if Hole.sum() == 0:
        return None
    P = (Vg > 0).astype(np.float64)
    pr = _period(Vg, P, 0)
    pc = _period(Vg, P, 1)

    fr = Vg.copy()
    for k in KR:
        fr = np.maximum(fr, _shift(Vg, int(k * pr), 0))
    filled = fr.copy()
    for k in KR:
        filled = np.maximum(filled, _shift(fr, 0, int(k * pc)))

    ys, xs = np.where(Hole == 1)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    cropped = _shift(filled, -r0, -c0)
    rowmask = (np.arange(H) <= (r1 - r0)).astype(np.float64)[:, None]
    colmask = (np.arange(W) <= (c1 - c0)).astype(np.float64)[None, :]
    Vout = cropped * rowmask * colmask

    oh10 = np.zeros((CHANNELS, H, W), bool)
    for c in range(1, CHANNELS):
        oh10[c] = np.abs(Vout - c) < 0.5
    return oh10


def _target_onehot(b):
    oh10 = np.zeros((CHANNELS, H, W), bool)
    for c in range(CHANNELS):
        for r in range(b.shape[0]):
            for cc in range(b.shape[1]):
                if b[r, cc] == c:
                    oh10[c, r, cc] = True
    return oh10


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def candidates(ex):
    prs = _pairs(ex)
    if len(prs) < 2:
        return []
    for a, b in prs:
        pred = _predict_onehot(a)
        if pred is None:
            return []
        if not np.array_equal(pred, _target_onehot(b)):
            return []
    try:
        m = build()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("periodhole", m)]
