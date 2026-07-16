"""family_golf4_5 -- CHEAPER exact re-solvers for a slice of golf targets.

Each candidate re-derives the rule from train+test+arc-gen pairs, validates a
numpy mirror of the *exact* ONNX semantics on every available pair, and only then
emits a minimal opset-10 graph. The integrator auto-picks the cheapest correct
solver, so we only need exactness + lower (params + intermediate_memory) than the
incumbent.

Golf levers used here:
  * SINGLE-channel [1,1,30,30] intermediates for all the heavy lifting (color value
    grid, masks, span fills) -- 10x cheaper than [1,10,30,30];
  * span fill via Hillis-Steele MAX doubling with shift matrices baked as
    INITIALIZERS (params are 4x cheaper than the equivalent intermediate bytes);
  * dynamic background (most-frequent colour) handled with [1,10,1,1] reductions;
  * one-hot reconstruction only at the very end.

Targets (rule -> incumbent points):
  333  connect-dots-to-box : a solid >=2x2 block is the "target"; every other
       non-bg cell that shares the target's row-range or col-range shoots a ray of
       its own colour into the gap toward the target.            (10.55)
  64   same rule (block colour varies).                          (12.08)
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

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


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
# connect-dots-to-box  (tasks 333, 64)
# ===========================================================================
def _cbx_block_mask(nz):
    """cells that belong to a solid >=2x2 nz square."""
    Hn, Wn = nz.shape
    s2 = np.zeros((Hn, Wn), bool)            # top-left corner of a solid 2x2
    s2[:-1, :-1] = nz[:-1, :-1] & nz[1:, :-1] & nz[:-1, 1:] & nz[1:, 1:]
    bl = np.zeros((Hn, Wn), bool)
    bl[:-1, :-1] |= s2[:-1, :-1]
    bl[1:, :-1] |= s2[:-1, :-1]
    bl[:-1, 1:] |= s2[:-1, :-1]
    bl[1:, 1:] |= s2[:-1, :-1]
    return bl


def _cbx_mirror(a):
    Hn, Wn = a.shape
    vals, counts = np.unique(a, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    nz = (a != bg)
    if nz.sum() == 0:
        return None
    block = _cbx_block_mask(nz)
    rows = np.where(block.any(1))[0]
    cols = np.where(block.any(0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    src = nz & ~block
    color = a.astype(float)
    ri = np.arange(Hn)[:, None] * np.ones((1, Wn))
    ci = np.ones((Hn, 1)) * np.arange(Wn)[None, :]
    above = (np.arange(Hn) < r0)[:, None] & np.ones((Hn, Wn), bool)
    below = (np.arange(Hn) > r1)[:, None] & np.ones((Hn, Wn), bool)
    left = (np.arange(Wn) < c0)[None, :] & np.ones((Hn, Wn), bool)
    right = (np.arange(Wn) > c1)[None, :] & np.ones((Hn, Wn), bool)
    rowrange = ((np.arange(Hn) >= r0) & (np.arange(Hn) <= r1))[:, None] & np.ones((Hn, Wn), bool)
    colrange = ((np.arange(Wn) >= c0) & (np.arange(Wn) <= c1))[None, :] & np.ones((Hn, Wn), bool)
    BIG = 99.0
    Fg = np.zeros((Hn, Wn))
    # vertical down: source above, fill below the TOPMOST source toward block
    sA = src & above & colrange
    has = sA.max(0); col = (color * sA).max(0)
    srcrow = np.where(sA, ri, BIG).min(0)
    fm = above & colrange & (ri > srcrow[None, :]) & has[None, :].astype(bool)
    Fg = np.maximum(Fg, fm * col[None, :])
    # vertical up: source below, fill above the BOTTOMMOST source toward block
    sA = src & below & colrange
    has = sA.max(0); col = (color * sA).max(0)
    srcrow = (ri * sA).max(0)
    fm = below & colrange & (ri < srcrow[None, :]) & has[None, :].astype(bool)
    Fg = np.maximum(Fg, fm * col[None, :])
    # horizontal right: source left, fill right of LEFTMOST source toward block
    sA = src & left & rowrange
    has = sA.max(1); col = (color * sA).max(1)
    srccol = np.where(sA, ci, BIG).min(1)
    fm = left & rowrange & (ci > srccol[:, None]) & has[:, None].astype(bool)
    Fg = np.maximum(Fg, fm * col[:, None])
    # horizontal left: source right, fill left of RIGHTMOST source toward block
    sA = src & right & rowrange
    has = sA.max(1); col = (color * sA).max(1)
    srccol = (ci * sA).max(1)
    fm = right & rowrange & (ci < srccol[:, None]) & has[:, None].astype(bool)
    Fg = np.maximum(Fg, fm * col[:, None])
    bgcell = ~nz
    Fm = Fg * bgcell
    out = a.copy()
    m = Fm > 0.5
    out[m] = np.round(Fm[m]).astype(int)
    return out


def _cbx_build(g):
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    big = g.f([1, 1, 1, 1], [99.0])
    cidx10 = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    Wcolor = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))   # 1x1 conv -> colour value
    ri = g.f([1, 1, H, W], np.arange(H)[:, None] * np.ones((1, W), np.float32))
    ci = g.f([1, 1, H, W], np.ones((H, 1), np.float32) * np.arange(W)[None, :])
    riB = g.f([1, 1, H, W], (np.arange(H)[:, None] - 99.0) * np.ones((1, W), np.float32))
    ciB = g.f([1, 1, H, W], np.ones((H, 1), np.float32) * (np.arange(W)[None, :] - 99.0))
    # shift-by-1 (block) and triangular (range) matrices
    Sd1 = g.f([H, W], (np.arange(H)[:, None] - np.arange(W)[None, :] == 1).astype(np.float32))
    Su1 = g.f([H, W], (np.arange(H)[:, None] - np.arange(W)[None, :] == -1).astype(np.float32))
    L = g.f([H, W], (np.arange(W)[None, :] <= np.arange(H)[:, None]).astype(np.float32))  # k<=i
    U = g.f([H, W], (np.arange(W)[None, :] >= np.arange(H)[:, None]).astype(np.float32))  # k>=i

    # ---- basic planes -----------------------------------------------------
    realcell = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)        # [1,1,30,30]
    cv0 = g.nd("Conv", ["input", Wcolor], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    cvp1 = g.nd("Add", [cv0, one])

    chsum = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)         # [1,10,1,1]
    maxsum = g.nd("ReduceMax", [chsum], axes=[1], keepdims=1)             # [1,1,1,1]
    isbg = g.nd("Cast", [g.nd("Greater", [chsum, g.nd("Sub", [maxsum, half])])], to=F)
    bgval = g.nd("ReduceSum", [g.nd("Mul", [isbg, cidx10])], axes=[1], keepdims=1)
    nzraw = g.nd("Cast", [g.nd("Greater",
                  [g.nd("Abs", [g.nd("Sub", [cv0, bgval])]), half])], to=F)
    nz = g.nd("Mul", [nzraw, realcell])                                  # [1,1,30,30]

    # ---- solid >=2x2 block mask ------------------------------------------
    hp = g.nd("Mul", [nz, g.nd("MatMul", [nz, Sd1])])    # nz(i,j)&nz(i,j+1)
    s2tl = g.nd("Mul", [hp, g.nd("MatMul", [Su1, hp])])  # top-left of solid 2x2
    br = g.nd("MatMul", [s2tl, Su1])
    block = g.nd("Max", [g.nd("Max", [s2tl, g.nd("MatMul", [Sd1, s2tl])]),
                         g.nd("Max", [br, g.nd("MatMul", [Sd1, br])])])
    src = g.nd("Sub", [nz, block])                       # source cells

    # ---- target row/col ranges (triangular MatMul cummax) ----------------
    brow = g.nd("ReduceMax", [block], axes=[3], keepdims=1)               # [1,1,30,1]
    bcol = g.nd("ReduceMax", [block], axes=[2], keepdims=1)               # [1,1,1,30]
    cd = g.nd("Cast", [g.nd("Greater", [g.nd("MatMul", [L, brow]), half])], to=F)
    cu = g.nd("Cast", [g.nd("Greater", [g.nd("MatMul", [U, brow]), half])], to=F)
    cr = g.nd("Cast", [g.nd("Greater", [g.nd("MatMul", [bcol, U]), half])], to=F)
    cl = g.nd("Cast", [g.nd("Greater", [g.nd("MatMul", [bcol, L]), half])], to=F)
    above = g.nd("Sub", [one, cd]); below = g.nd("Sub", [one, cu])
    left = g.nd("Sub", [one, cr]); right = g.nd("Sub", [one, cl])
    rowrange = g.nd("Mul", [cd, cu]); colrange = g.nd("Mul", [cr, cl])

    # ---- directional gap fills via axis reductions -----------------------
    # NOTE: `col` (= colour+1 of the line's source, 0 when none) already zeroes
    # empty lines, so no separate `has` mask is needed.
    def vfill(region_row, take_min):
        region = g.nd("Mul", [region_row, colrange])     # [1,1,30,30]
        sA = g.nd("Mul", [src, region])
        col = g.nd("ReduceMax", [g.nd("Mul", [cvp1, sA])], axes=[2], keepdims=1)  # [1,1,1,30]
        if take_min:                                     # topmost source row
            srow = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [riB, sA]), big])],
                        axes=[2], keepdims=1)
            gt = g.nd("Cast", [g.nd("Greater", [ri, srow])], to=F)
        else:                                            # bottommost source row
            srow = g.nd("ReduceMax", [g.nd("Mul", [ri, sA])], axes=[2], keepdims=1)
            gt = g.nd("Cast", [g.nd("Less", [ri, srow])], to=F)
        return g.nd("Mul", [g.nd("Mul", [gt, region]), col])

    def hfill(region_col, take_min):
        region = g.nd("Mul", [rowrange, region_col])
        sA = g.nd("Mul", [src, region])
        col = g.nd("ReduceMax", [g.nd("Mul", [cvp1, sA])], axes=[3], keepdims=1)  # [1,1,30,1]
        if take_min:                                     # leftmost source col
            scol = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [ciB, sA]), big])],
                         axes=[3], keepdims=1)
            gt = g.nd("Cast", [g.nd("Greater", [ci, scol])], to=F)
        else:                                            # rightmost source col
            scol = g.nd("ReduceMax", [g.nd("Mul", [ci, sA])], axes=[3], keepdims=1)
            gt = g.nd("Cast", [g.nd("Less", [ci, scol])], to=F)
        return g.nd("Mul", [g.nd("Mul", [gt, region]), col])

    vfd = vfill(above, True)
    vfu = vfill(below, False)
    hfr = hfill(left, True)
    hfl = hfill(right, False)
    Fsum = g.nd("Add", [g.nd("Add", [vfd, vfu]), g.nd("Add", [hfr, hfl])])

    bgmask = g.nd("Sub", [realcell, nz])
    Fm = g.nd("Mul", [Fsum, bgmask])                     # (colour+1) on bg fills
    newmask = g.nd("Cast", [g.nd("Greater", [Fm, half])], to=F)

    # ---- one-hot reconstruction ------------------------------------------
    val = g.nd("Sub", [Fm, one])                         # colour at fills, -1 else
    diff = g.nd("Sub", [val, cidx10])                    # [1,10,30,30]
    ind = g.nd("Relu", [g.nd("Sub", [one, g.nd("Abs", [diff])])])
    keep = g.nd("Mul", ["input", g.nd("Sub", [one, newmask])])
    g.nd("Add", [keep, ind], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []
    out = []

    def emit(name, mirror, build):
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

    emit("g45_connectbox", _cbx_mirror, _cbx_build)
    return out
