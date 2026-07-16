"""family_crk5_2 -- crack module for slice U[2::6] of the unsolved NeuroGolf tasks.

Solved rules (each detected structurally in numpy, validated EXACTLY on
train+test, and verified to generalise across arc-gen before emission):

  * task212  "line ray-cast": a single horizontal line of color L splits the
             grid. Dots of color TL extend a ray TOWARD the line (stopping just
             before it); dots of color TE extend a ray AWAY from the line, to
             the grid edge. Implemented with two triangular MatMul cumulative
             masks gated by an above/below split derived from the line row.

  * task248  "billiard bounce": a single dot sits at the bottom-left corner; it
             traces a diagonal that reflects off the left/right walls while
             moving up one row per step.  The column at row r is a triangle
             wave  P-|((H-1-r) mod 2P)-P|  with P = W-1, built with Mod/Abs/Less.

All intermediates are static [1,*,30,30]; opset-10 only.
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
# tiny graph accumulator                                                      #
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
    return m


def _pairs(ex, splits=("train", "test")):
    out = []
    for k in splits:
        for p in ex.get(k, []):
            a = np.array(p["input"]); b = np.array(p["output"])
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _slc1(g, src, k):
    """Slice channel k of a [1,10,30,30] tensor -> [1,1,30,30]."""
    return g.nd("Slice", [src, g.i64([k]), g.i64([k + 1]), g.i64([1])])


def _slc(g, src, lo, hi, axis):
    return g.nd("Slice", [src, g.i64([lo]), g.i64([hi]), g.i64([axis])])


# =========================================================================== #
# TASK 212 -- horizontal line ray-cast                                        #
# =========================================================================== #
def _t212_sim(a, L, TL, TE):
    h, w = a.shape
    rows = [r for r in range(h) if (a[r, :] == L).all()]
    if len(rows) != 1:
        return None
    Lr = rows[0]
    out = a.copy()
    for c in range(w):
        for r in range(h):
            v = a[r, c]
            if v == TL:
                if r < Lr:
                    out[r:Lr, c] = v
                elif r > Lr:
                    out[Lr + 1:r + 1, c] = v
            elif v == TE:
                if r < Lr:
                    out[0:r + 1, c] = v
                elif r > Lr:
                    out[r:h, c] = v
    return out


def _t212_detect(prs):
    if not prs:
        return None
    # all shapes preserved
    if not all(a.shape == b.shape for a, b in prs):
        return None
    colors = set()
    for a, b in prs:
        colors |= set(np.unique(a).tolist()) | set(np.unique(b).tolist())
    colors.discard(0)
    if len(colors) != 3:
        return None
    # the line color forms a full row; the two others are TL/TE in some order
    for L in colors:
        # check L forms exactly one full row in every input
        ok_line = True
        for a, b in prs:
            rows = [r for r in range(a.shape[0]) if (a[r, :] == L).all()]
            if len(rows) != 1:
                ok_line = False
                break
        if not ok_line:
            continue
        rest = list(colors - {L})
        for TL, TE in ((rest[0], rest[1]), (rest[1], rest[0])):
            if all(_t212_sim(a, L, TL, TE) is not None and
                   np.array_equal(_t212_sim(a, L, TL, TE), b) for a, b in prs):
                return (int(L), int(TL), int(TE))
    return None


def _t212_build(params):
    L, TL, TE = params
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    zero = g.f([1, 1, 1, 1], [0.0])
    Llow = g.f([G, G], np.tril(np.ones((G, G), np.float32)))   # j<=i
    Uup = g.f([G, G], np.triu(np.ones((G, G), np.float32)))    # j>=i

    chL = _slc1(g, "input", L)
    chTL = _slc1(g, "input", TL)
    chTE = _slc1(g, "input", TE)
    grid = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)  # [1,1,30,30]

    rowcountL = g.nd("ReduceSum", [chL], axes=[3], keepdims=1)  # [1,1,30,1]
    lineb = g.nd("Greater", [rowcountL, half])
    lineind = g.nd("Cast", [lineb], to=F)                        # [1,1,30,1]
    belowOrAt = g.nd("MatMul", [Llow, lineind])                  # 1 for r>=L
    one = g.f([1, 1, 1, 1], [1.0])
    above = g.nd("Sub", [one, belowOrAt])                        # r<L
    below = g.nd("Sub", [belowOrAt, lineind])                    # r>L

    def bin_gt(t):
        return g.nd("Cast", [g.nd("Greater", [t, half])], to=F)

    # color TL -> toward line
    dot2a = g.nd("Mul", [chTL, above])
    dot2b = g.nd("Mul", [chTL, below])
    res2a = g.nd("Mul", [bin_gt(g.nd("MatMul", [Llow, dot2a])), above])
    res2b = g.nd("Mul", [bin_gt(g.nd("MatMul", [Uup, dot2b])), below])
    color2 = g.nd("Add", [res2a, res2b])

    # color TE -> toward edge
    dot1a = g.nd("Mul", [chTE, above])
    dot1b = g.nd("Mul", [chTE, below])
    res1a = g.nd("Mul", [bin_gt(g.nd("MatMul", [Uup, dot1a])), above])
    res1b = g.nd("Mul", [bin_gt(g.nd("MatMul", [Llow, dot1b])), below])
    color1 = g.nd("Add", [res1a, res1b])

    color2g = g.nd("Mul", [color2, grid])
    color1g = g.nd("Mul", [color1, grid])

    bg = g.nd("Sub", [grid, color1g])
    bg = g.nd("Sub", [bg, color2g])
    bg = g.nd("Sub", [bg, chL])
    zch = g.nd("Sub", [chL, chL])

    chans = [zch] * CHANNELS
    chans[0] = bg
    chans[TE] = color1g
    chans[TL] = color2g
    chans[L] = chL
    g.nd("Concat", chans, "output", axis=1)
    return _model(g)


# =========================================================================== #
# TASK 248 -- billiard bounce from bottom-left corner                         #
# =========================================================================== #
def _t248_tri(d, P):
    m = d % (2 * P)
    return P - abs(m - P)


def _t248_sim(a):
    h, w = a.shape
    nz = np.argwhere(a != 0)
    if len(nz) != 1:
        return None
    r0, c0 = nz[0]
    if (r0, c0) != (h - 1, 0):
        return None
    col = int(a[r0, c0])
    if w < 2:
        return None
    P = w - 1
    out = np.zeros_like(a)
    for r in range(h):
        out[r, _t248_tri((h - 1) - r, P)] = col
    return out


def _t248_detect(prs):
    if not prs:
        return None
    if not all(a.shape == b.shape for a, b in prs):
        return None
    col = None
    for a, b in prs:
        s = _t248_sim(a)
        if s is None or not np.array_equal(s, b):
            return None
        nz = np.unique(a[a != 0])
        if nz.size != 1:
            return None
        c = int(nz[0])
        if col is None:
            col = c
        elif col != c:
            return None
    return col


def _t248_build(col):
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0])
    two = g.f([1, 1, 1, 1], [2.0])
    half = g.f([1, 1, 1, 1], [0.5])
    rowidx = g.f([1, 1, G, 1], list(range(G)))
    colidx = g.f([1, 1, 1, G], list(range(G)))

    grid = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)   # [1,1,30,30]
    col0 = _slc(g, grid, 0, 1, 3)                               # [1,1,30,1]
    Hh = g.nd("ReduceSum", [col0], axes=[2], keepdims=1)        # [1,1,1,1]
    row0 = _slc(g, grid, 0, 1, 2)                               # [1,1,1,30]
    Ww = g.nd("ReduceSum", [row0], axes=[3], keepdims=1)        # [1,1,1,1]

    P = g.nd("Sub", [Ww, one])
    Hm1 = g.nd("Sub", [Hh, one])
    d = g.nd("Sub", [Hm1, rowidx])                              # [1,1,30,1]
    period = g.nd("Mul", [P, two])
    m = g.nd("Mod", [d, period], fmod=1)
    colpos = g.nd("Sub", [P, g.nd("Abs", [g.nd("Sub", [m, P])])])  # [1,1,30,1]
    diff = g.nd("Abs", [g.nd("Sub", [colpos, colidx])])        # [1,1,30,30]
    ballb = g.nd("Less", [diff, half])
    ball = g.nd("Mul", [g.nd("Cast", [ballb], to=F), grid])
    bg = g.nd("Sub", [grid, ball])
    zch = g.nd("Sub", [ball, ball])

    chans = [zch] * CHANNELS
    chans[0] = bg
    chans[col] = ball
    g.nd("Concat", chans, "output", axis=1)
    return _model(g)


# =========================================================================== #
# dispatch                                                                    #
# =========================================================================== #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    try:
        p = _t212_detect(prs)
        if p is not None:
            out.append(("t212_line", _check(_t212_build(p))))
    except Exception:
        pass

    try:
        c = _t248_detect(prs)
        if c is not None:
            out.append(("t248_bounce", _check(_t248_build(c))))
    except Exception:
        pass

    return out
