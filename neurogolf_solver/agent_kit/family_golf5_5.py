"""family_golf5_5 -- CHEAPER exact re-solvers for a slice of golf targets.

Each candidate re-derives the rule from train+test+arc-gen pairs (numpy mirror of
the exact ONNX semantics), validates it on EVERY available pair, and only then
emits a minimal opset-10 graph. The integrator auto-picks the cheapest correct
solver, so we only need exactness + lower (params + intermediate_memory) than the
incumbent.

Golf levers used here:
  * SINGLE-channel [1,1,30,30] intermediates for the heavy lifting (10x cheaper
    than [1,10,30,30]);
  * counts via ReduceMax/ReduceSum collapsing to [1,1,1,1] scalars;
  * the FREE output tensor assembled with a single Concat / Where (no extra
    [1,10,30,30] intermediate beyond the one that is unavoidable);
  * geometry baked as small INITIALIZERS (index grids, triangular shift matrices).
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

    def i(self, dims, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, list(dims),
                          [int(v) for v in np.asarray(vals, np.int64).ravel()]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out

    def chan(self, src, c):
        """Slice channel c (-> [1,1,30,30])."""
        s = self.i([1], [c]); e = self.i([1], [c + 1]); ax = self.i([1], [1])
        return self.nd("Slice", [src, s, e, ax])


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# pairs                                                                        #
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


# ===========================================================================
# 274  countbar : a 5-walled container is partly filled from the bottom with 8.
#      Output is a fixed 3x3 grid holding N eights laid out in BOUSTROPHEDON
#      (snake) reading order, where N = (#rows containing a 5) - (#rows with an
#      8) - 1  (= number of still-empty interior rows).               (13.56)
# ===========================================================================
_SNAKE = np.array([[0, 1, 2], [5, 4, 3], [6, 7, 8]], float)


def _t274_mirror(a):
    m5 = (a == 5); m8 = (a == 8)
    rows5 = int(m5.any(1).sum()); rows8 = int(m8.any(1).sum())
    N = rows5 - rows8 - 1
    return np.where(_SNAKE < N, 8, 0).astype(int)


def _t274_build(g):
    m5 = g.chan("input", 5)
    m8 = g.chan("input", 8)
    rows5 = g.nd("ReduceSum", [g.nd("ReduceMax", [m5], axes=[3], keepdims=1)],
                 axes=[2], keepdims=1)                              # [1,1,1,1]
    rows8 = g.nd("ReduceSum", [g.nd("ReduceMax", [m8], axes=[3], keepdims=1)],
                 axes=[2], keepdims=1)
    one = g.f([1, 1, 1, 1], [1.0])
    N = g.nd("Sub", [g.nd("Sub", [rows5, rows8]), one])            # [1,1,1,1]

    ramp = np.full((H, W), 100.0, np.float32)
    ramp[:3, :3] = _SNAKE
    rampc = g.f([1, 1, H, W], ramp)
    real = np.zeros((H, W), np.float32); real[:3, :3] = 1.0
    realc = g.f([1, 1, H, W], real)
    zero = g.f([1, 1, H, W], np.zeros((H, W), np.float32))

    is8 = g.nd("Cast", [g.nd("Less", [rampc, N])], to=F)           # [1,1,30,30]
    is0 = g.nd("Sub", [realc, is8])
    g.nd("Concat", [is0, zero, zero, zero, zero, zero, zero, zero, is8, zero],
         "output", axis=1)
    return _model(g)


# ===========================================================================
# 293  crk2_5_swap : a full-height vertical band (colour V) crosses a
#      full-width horizontal band (colour Hc). At the crossing rectangle the
#      band that is currently on top is swapped for the other -> the whole
#      crossing becomes the single colour Y = V + Hc - X (X = current).  (13.66)
# ===========================================================================
def _t293_mirror(a):
    H_, W_ = a.shape
    nz = (a != 0)
    colcount = nz.sum(0); rowcount = nz.sum(1)
    vband = (colcount == H_); hband = (rowcount == W_)
    if vband.sum() == 0 or hband.sum() == 0:
        return None
    cross = np.outer(hband, vband)
    vcells = np.outer(~hband, vband) & nz
    hcells = np.outer(hband, ~vband) & nz
    if vcells.sum() == 0 or hcells.sum() == 0:
        return None
    V = int(a[vcells].max()); Hc = int(a[hcells].max()); X = int(a[cross].max())
    out = a.copy(); out[cross] = V + Hc - X
    return out


def _t293_build(g):
    half = g.f([1, 1, 1, 1], [0.5]); one = g.f([1, 1, 1, 1], [1.0])
    cidx = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    Wcol = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))      # conv -> colour value
    real = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)   # [1,1,30,30]
    gval = g.nd("Conv", ["input", Wcol], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    nz = g.nd("Sub", [real, g.chan("input", 0)])               # non-bg mask

    def band(axes):
        cnt = g.nd("ReduceSum", [nz], axes=axes, keepdims=1)
        rc = g.nd("ReduceSum", [real], axes=axes, keepdims=1)
        full = g.nd("Cast", [g.nd("Less", [g.nd("Sub", [rc, cnt]), half])], to=F)
        nonempty = g.nd("Cast", [g.nd("Greater", [rc, half])], to=F)
        return g.nd("Mul", [full, nonempty])

    vband = band([2])                                          # [1,1,1,30]
    hband = band([3])                                          # [1,1,30,1]
    cross = g.nd("Mul", [hband, vband])                        # [1,1,30,30]
    # every real cell of a band column/row is non-bg, so the extra nz mask is
    # redundant (gval is already 0 on bg / padding).
    vcell = g.nd("Mul", [vband, g.nd("Sub", [one, hband])])    # [1,1,30,30]
    hcell = g.nd("Mul", [hband, g.nd("Sub", [one, vband])])
    Vval = g.nd("ReduceMax", [g.nd("Mul", [gval, vcell])], axes=[2, 3], keepdims=1)
    Hval = g.nd("ReduceMax", [g.nd("Mul", [gval, hcell])], axes=[2, 3], keepdims=1)
    Xval = g.nd("ReduceMax", [g.nd("Mul", [gval, cross])], axes=[2, 3], keepdims=1)
    Yval = g.nd("Sub", [g.nd("Add", [Vval, Hval]), Xval])      # [1,1,1,1]
    onehotY = g.nd("Relu", [g.nd("Sub", [one,
                   g.nd("Abs", [g.nd("Sub", [Yval, cidx])])])])  # [1,10,1,1]
    A = g.nd("Mul", [cross, onehotY])                          # [1,10,30,30]
    cond = g.nd("Greater", [cross, half])
    g.nd("Where", [cond, A, "input"], "output")
    return _model(g)


# ===========================================================================
# 136  diagrays : each solid colour-1 block shoots a colour-1 ray UP-LEFT from
#      its top-left corner; each colour-2 block shoots a colour-2 ray
#      DOWN-RIGHT from its bottom-right corner. Rays are diagonal half-lines.
#      Implemented with ONE diagonal-kernel Conv per direction.        (13.02)
# ===========================================================================
def _shift(m, dr, dc):
    o = np.zeros_like(m); Hn, Wn = m.shape
    for r in range(Hn):
        for c in range(Wn):
            rr, cc = r - dr, c - dc
            if 0 <= rr < Hn and 0 <= cc < Wn:
                o[r, c] = m[rr, cc]
    return o


def _diag_conv(seed, down_right):
    Hn, Wn = seed.shape
    out = np.zeros((Hn, Wn), int)
    for r in range(Hn):
        for c in range(Wn):
            s = 0
            for t in range(30):
                rr, cc = (r - t, c - t) if down_right else (r + t, c + t)
                if 0 <= rr < Hn and 0 <= cc < Wn:
                    s += seed[rr, cc]
            if s > 0:
                out[r, c] = 1
    return out


def _t136_mirror(a):
    Hn, Wn = a.shape
    m1 = (a == 1).astype(int); m2 = (a == 2).astype(int)
    ctl = ((m1 == 1) & (_shift(m1, 1, 0) == 0) & (_shift(m1, 0, 1) == 0)).astype(int)
    cbr = ((m2 == 1) & (_shift(m2, -1, 0) == 0) & (_shift(m2, 0, -1) == 0)).astype(int)
    ray1 = _diag_conv(ctl, False)
    ray2 = _diag_conv(cbr, True)
    final1 = np.maximum(m1, ray1); final2 = np.maximum(m2, ray2)
    final1 = final1 * (1 - final2)
    out = np.zeros((Hn, Wn), int)
    out[final1.astype(bool)] = 1
    out[final2.astype(bool)] = 2
    return out


def _t136_build(g):
    one = g.f([1, 1, 1, 1], [1.0]); half = g.f([1, 1, 1, 1], [0.5])
    subd = (np.arange(W)[None, :] == np.arange(H)[:, None] - 1).astype(np.float32)
    supd = (np.arange(W)[None, :] == np.arange(H)[:, None] + 1).astype(np.float32)
    SUBD = g.f([H, W], subd)              # shift-down (left-mul) / shift-left (right-mul)
    SUPD = g.f([H, W], supd)              # shift-up   (left-mul) / shift-right (right-mul)
    Kdiag = g.f([1, 1, H, W], np.eye(H, W, dtype=np.float32))
    zero = g.f([1, 1, H, W], np.zeros((H, W), np.float32))

    m1 = g.chan("input", 1); m2 = g.chan("input", 2)
    real = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    # top-left corners of colour-1 blocks
    down = g.nd("MatMul", [SUBD, m1]); right = g.nd("MatMul", [m1, SUPD])
    ctl = g.nd("Mul", [g.nd("Mul", [m1, g.nd("Sub", [one, down])]),
                       g.nd("Sub", [one, right])])
    ray1 = g.nd("Cast", [g.nd("Greater", [g.nd("Conv", [ctl, Kdiag],
                kernel_shape=[H, W], pads=[0, 0, H - 1, W - 1]), half])], to=F)
    # bottom-right corners of colour-2 blocks
    up = g.nd("MatMul", [SUPD, m2]); left = g.nd("MatMul", [m2, SUBD])
    cbr = g.nd("Mul", [g.nd("Mul", [m2, g.nd("Sub", [one, up])]),
                       g.nd("Sub", [one, left])])
    ray2 = g.nd("Cast", [g.nd("Greater", [g.nd("Conv", [cbr, Kdiag],
                kernel_shape=[H, W], pads=[H - 1, W - 1, 0, 0]), half])], to=F)
    final1 = g.nd("Max", [m1, ray1])
    final2 = g.nd("Max", [m2, g.nd("Mul", [ray2, real])])
    final1 = g.nd("Mul", [final1, g.nd("Sub", [one, final2])])
    ch0 = g.nd("Sub", [g.nd("Sub", [real, final1]), final2])
    g.nd("Concat", [ch0, final1, final2, zero, zero, zero, zero, zero, zero, zero],
         "output", axis=1)
    return _model(g)


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
def _check(name, prs, mirror, build, out):
    try:
        for a, b in prs:
            o = mirror(a)
            if o is None or o.shape != b.shape or not np.array_equal(o, b):
                return
        g = _G()
        m = build(g)
        onnx.checker.check_model(m, full_check=True)
        out.append((name, m))
    except Exception:
        pass


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []
    _check("g55_countbar", prs, _t274_mirror, _t274_build, out)
    _check("g55_crossswap", prs, _t293_mirror, _t293_build, out)
    _check("g55_diagrays", prs, _t136_mirror, _t136_build, out)
    return out
