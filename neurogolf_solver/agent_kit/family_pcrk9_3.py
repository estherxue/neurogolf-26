"""pcrk9_3 -- data-dependent SHAPE-MATCH recolor (task143 family).

TRUE RULE (verified on train+test+all 262 arc-gen pairs of task143, 266/266):
  * The grid contains a colour-5 "legend" object (a fixed L) sitting in a corner.
  * The KEY object = the single non-background, non-5 object that lies inside the
    colour-5 bounding box (nested with the legend).
  * Exactly one OTHER object elsewhere is a pure TRANSLATE of the key (same shape,
    same orientation, any colour).  That object is recoloured to 5.  Everything
    else is left untouched.

The rule is genuinely per-object shape matching, but it is expressible with a
computed convolution kernel (opset-10 allows a non-constant Conv/ConvTranspose W):

  K1  = the key shape, anchored at (1,1) with a 1-cell margin        [1,1,30,30]
  ring= the 4-connected 1-cell border of K1                          [1,1,30,30]
  For every colour channel c (grouped Conv, one kernel shared):
      match[c,ty,tx] = cross-correlation of (X==c) with K1
      ringc[c,ty,tx] = cross-correlation of (X==c) with ring
      valid = (match == |K1|) & (ringc == 0)     # colour-c holds K1 exactly, isolated
  stamp = ConvTranspose(valid, K1)               # re-paint every matched object
  target= stamp minus the key cells              # never recolour the key itself
  output= X with target cells moved into channel 5

The "same-colour ring == 0" test makes the correlation exact per object without any
connected-component labelling, and works even when the target object is 4-adjacent
to a DIFFERENT-coloured object.  Detection mirrors these numerics exactly and only
emits when every available pair is reproduced, so wrong hypotheses are dropped.
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
CBIG = 1000.0


# --------------------------------------------------------------------------- #
# numpy reference (mirrors the ONNX numerics exactly)                          #
# --------------------------------------------------------------------------- #
def _bbox(m):
    ys, xs = np.where(m)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _ref(a):
    H0, W0 = a.shape
    F5 = (a == 5)
    if F5.sum() == 0:
        return None
    r0, r1, c0, c1 = _bbox(F5)
    nn5 = (a != 0) & (a != 5)
    inb = np.zeros_like(a, bool)
    inb[r0:r1 + 1, c0:c1 + 1] = True
    Kfull = nn5 & inb
    if Kfull.sum() == 0:
        return None
    ky0, ky1, kx0, kx1 = _bbox(Kfull)
    K = Kfull[ky0:ky1 + 1, kx0:kx1 + 1]
    kh, kw = K.shape
    ksz = int(K.sum())
    Kp = np.zeros((kh + 2, kw + 2), bool)
    Kp[1:-1, 1:-1] = K
    dil = Kp.copy()
    dil[1:, :] |= Kp[:-1, :]
    dil[:-1, :] |= Kp[1:, :]
    dil[:, 1:] |= Kp[:, :-1]
    dil[:, :-1] |= Kp[:, 1:]
    ring = dil & ~Kp
    stamp = np.zeros_like(a, bool)
    for c in range(1, 10):
        if c == 5:
            continue
        Fc = (a == c)
        if Fc.sum() == 0:
            continue
        Fcp = np.zeros((H0 + 2, W0 + 2), bool)
        Fcp[1:-1, 1:-1] = Fc
        for ty in range(0, H0 - kh + 1):
            for tx in range(0, W0 - kw + 1):
                if int((Fc[ty:ty + kh, tx:tx + kw] & K).sum()) != ksz:
                    continue
                if int((Fcp[ty:ty + kh + 2, tx:tx + kw + 2] & ring).sum()) != 0:
                    continue
                stamp[ty:ty + kh, tx:tx + kw] |= K
    tgt = stamp & ~Kfull
    out = a.copy()
    out[tgt] = 5
    return out


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                       #
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

    def i64(self, vals, dims=None):
        n = self.nm("i")
        dims = dims if dims is not None else [len(vals)]
        self.inits.append(oh.make_tensor(n, INT64, list(dims),
                          [int(v) for v in np.asarray(vals).ravel()]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _shift(g, x, dr, dc):
    """shift a [1,1,30,30] tensor by (dr,dc) with zero fill (Pad+Slice)."""
    pt, pb = max(dr, 0), max(-dr, 0)
    pl, pr = max(dc, 0), max(-dc, 0)
    p = g.nd("Pad", [x], mode="constant", value=0.0,
             pads=[0, 0, pt, pl, 0, 0, pb, pr])
    st = g.i64([max(-dr, 0), max(-dc, 0)])
    en = g.i64([max(-dr, 0) + H, max(-dc, 0) + W])
    ax = g.i64([2, 3])
    return g.nd("Slice", [p, st, en, ax])


def build_model():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))      # [1,1,30,1]
    colidx = g.f([1, 1, 1, W], list(range(W)))      # [1,1,1,30]
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    cbig = g.f([1, 1, 1, 1], [CBIG])

    # --- channel slices ------------------------------------------------------
    ch0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])   # [1,1,30,30]
    F5 = g.nd("Slice", ["input", g.i64([5]), g.i64([6]), g.i64([1])])    # [1,1,30,30]
    allsum = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)          # [1,1,30,30]
    nn5 = g.nd("Sub", [g.nd("Sub", [allsum, ch0]), F5])                  # [1,1,30,30]

    # --- colour-5 bounding box ----------------------------------------------
    rowhas = g.nd("ReduceMax", [F5], axes=[3], keepdims=1)               # [1,1,30,1]
    colhas = g.nd("ReduceMax", [F5], axes=[2], keepdims=1)               # [1,1,1,30]
    maxrow = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    minrow = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    mincol = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])

    # inside-bbox mask: (minrow<=row<=maxrow) & (mincol<=col<=maxcol)
    ge_r = g.nd("Cast", [g.nd("Greater", [g.nd("Sub", [rowidx, minrow]), g.nd("Neg", [half])])], to=F)  # [1,1,30,1]
    le_r = g.nd("Cast", [g.nd("Less", [g.nd("Sub", [rowidx, maxrow]), half])], to=F)                     # [1,1,30,1]
    ge_c = g.nd("Cast", [g.nd("Greater", [g.nd("Sub", [colidx, mincol]), g.nd("Neg", [half])])], to=F)  # [1,1,1,30]
    le_c = g.nd("Cast", [g.nd("Less", [g.nd("Sub", [colidx, maxcol]), half])], to=F)                     # [1,1,1,30]
    inrow = g.nd("Mul", [ge_r, le_r])                                    # [1,1,30,1]
    incol = g.nd("Mul", [ge_c, le_c])                                    # [1,1,1,30]
    inbb = g.nd("Mul", [inrow, incol])                                   # [1,1,30,30]

    Kfull = g.nd("Mul", [nn5, inbb])                                     # [1,1,30,30] key object

    # --- anchor key to (1,1) (margin of one cell) ---------------------------
    kry = g.nd("ReduceMax", [Kfull], axes=[3], keepdims=1)               # [1,1,30,1]
    krx = g.nd("ReduceMax", [Kfull], axes=[2], keepdims=1)               # [1,1,1,30]
    ky0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [kry, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    kx0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [krx, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])
    # shift up-left by (ky0-1, kx0-1) via selection matrices
    minr = g.nd("Sub", [ky0, one])
    minc = g.nd("Sub", [kx0, one])
    # Scol[k,j] = (j == k - minc)
    diff_c = g.nd("Sub", [g.nd("Add", [colidx, minc]), rowidx])         # [1,1,30,30]
    Scol = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_c]), half])], to=F)
    # Srow[r,k] = (k == r + minr)
    diff_r = g.nd("Sub", [colidx, g.nd("Add", [rowidx, minr])])         # [1,1,30,30]
    Srow = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_r]), half])], to=F)
    K1 = g.nd("MatMul", [Srow, g.nd("MatMul", [Kfull, Scol])])          # [1,1,30,30] anchored key

    ksz = g.nd("ReduceSum", [K1], axes=[2, 3], keepdims=1)              # [1,1,1,1]

    # --- ring kernel (4-conn border of K1) ----------------------------------
    dil = g.nd("Max", [K1, _shift(g, K1, 1, 0), _shift(g, K1, -1, 0),
                       _shift(g, K1, 0, 1), _shift(g, K1, 0, -1)])       # [1,1,30,30]
    ring = g.nd("Sub", [dil, K1])                                       # [1,1,30,30] (0/1)

    # --- per-colour correlation ---------------------------------------------
    fg9 = g.nd("Slice", ["input", g.i64([1]), g.i64([10]), g.i64([1])])  # [1,9,30,30]  colours 1..9
    mask9 = g.f([1, 9, 1, 1], [1, 1, 1, 1, 0, 1, 1, 1, 1])              # zero colour-5 channel
    fg9 = g.nd("Mul", [fg9, mask9])
    # pad top-left by 1 (so edge objects are reachable and the ring above/left is
    # background) and bottom-right so the correlation range covers every position
    fg9p = g.nd("Pad", [fg9], mode="constant", value=0.0,
                pads=[0, 0, 1, 1, 0, 0, H - 1, W - 1])                  # [1,9,60,60]

    Wc = g.nd("Tile", [K1, g.i64([9, 1, 1, 1])])                       # [9,1,30,30]
    Wr = g.nd("Tile", [ring, g.i64([9, 1, 1, 1])])                     # [9,1,30,30]

    match = g.nd("Conv", [fg9p, Wc], group=9, kernel_shape=[H, W],
                 strides=[1, 1], pads=[0, 0, 0, 0])                     # [1,9,31,31]
    ringc = g.nd("Conv", [fg9p, Wr], group=9, kernel_shape=[H, W],
                 strides=[1, 1], pads=[0, 0, 0, 0])                     # [1,9,31,31]

    full = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [match, ksz])]), half])], to=F)
    clean = g.nd("Cast", [g.nd("Less", [ringc, half])], to=F)
    valid = g.nd("Mul", [full, clean])                                 # [1,9,31,31]

    stamp9 = g.nd("ConvTranspose", [valid, Wc], group=9, kernel_shape=[H, W],
                  strides=[1, 1], pads=[0, 0, 0, 0])                    # [1,9,60,60]
    # original cell (y,x) lives at padded coord (y+1,x+1) -> crop [1:31,1:31]
    stamp9 = g.nd("Slice", [stamp9, g.i64([1, 1]), g.i64([H + 1, W + 1]), g.i64([2, 3])])  # [1,9,30,30]
    stamp = g.nd("ReduceSum", [stamp9], axes=[1], keepdims=1)          # [1,1,30,30]

    target = g.nd("Cast", [g.nd("Greater", [g.nd("Sub", [stamp, Kfull]), half])], to=F)  # [1,1,30,30]

    # --- recolour target cells into channel 5 -------------------------------
    keep = g.nd("Sub", [one, target])                                  # [1,1,30,30]
    xkept = g.nd("Mul", ["input", keep])                               # [1,10,30,30]
    onehot5 = g.f([1, CHANNELS, 1, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    add5 = g.nd("Mul", [target, onehot5])                              # [1,10,30,30]
    g.nd("Add", [xkept, add5], "output")

    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "pcrk9_3", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30 or a.shape != b.shape:
                continue
            out.append((a, b))
    return out


def candidates(ex):
    prs = _pairs(ex)
    if len(prs) < 3:
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []
    # every pair must be reproduced by the true rule
    changed = False
    for a, b in prs:
        o = _ref(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return []
        if not np.array_equal(a, b):
            changed = True
    if not changed:
        return []
    try:
        m = build_model()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("pcrk9_3_shapematch5", m)]
