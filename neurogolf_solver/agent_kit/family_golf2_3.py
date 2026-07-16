"""family_golf2_3 -- cheaper exact solvers for a stride slice of golf targets.

Each candidate re-derives the task rule from train+test+arc-gen pairs with a numpy
reference, verifies EXACT equality on every available pair, and only then emits a
minimal opset-10 ONNX graph.  The integrator auto-picks the cheapest correct model,
so we just need these to be exact and cheaper than the incumbent.

Targets golfed here (rules verified exact on 100% of provided examples):
  * 167 crk2_4_t167  -> #distinct-nonzero-colors selects one of 3 constant 3x3 grids
  * 371 crk2_plusmid -> stamp a colour-3 plus at the midpoint of two colour-1 markers
  * 329 midcol       -> keep only the (data-dependent) centre column, rest -> bg
  * 126 cupdrop_m4   -> each '|_|'-cup drops a colour-4 marker to the bottom row

Cost levers: write the final tensor straight into the FREE "output" via Where/Pad,
keep every working tensor single-channel [1,1,30,30] (3600B) or smaller, use boolean
masks (1B/elem), and replace [1,9,30,30] colour slices with [1,1,*] reductions.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def iconst(self, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)], list(vals)))
        return nm

    def fconst(self, vals, shape):
        nm = self.name("f")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(shape),
                                         [float(v) for v in vals]))
        return nm

    def scalar(self, v):
        return self.fconst([v], [1])

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _slice(g, src, starts, ends, axes, steps=None):
    ins = [src, g.iconst(starts), g.iconst(ends), g.iconst(axes)]
    if steps is not None:
        ins.append(g.iconst(steps))
    return g.node("Slice", ins)


def _onehot_channel(g, ch):
    """const [1,10,1,1] one-hot for a single colour channel."""
    vals = [1.0 if i == ch else 0.0 for i in range(CHANNELS)]
    return g.fconst(vals, [1, CHANNELS, 1, 1])


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
# 167  crk2_4_t167:  #distinct nonzero colours -> one of 3 fixed 3x3 grids
# ===========================================================================
_T167_TOP = np.array([[5, 5, 5], [0, 0, 0], [0, 0, 0]])
_T167_MD = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])
_T167_AD = np.array([[0, 0, 5], [0, 5, 0], [5, 0, 0]])


def _ref_167(a):
    if a.shape != (3, 3):
        return None
    c = len(set(a[a > 0].tolist()))
    return {1: _T167_TOP, 2: _T167_MD, 3: _T167_AD}.get(c)


def _grid_onehot(g, grid):
    """const [1,10,3,3] one-hot for a fixed 3x3 grid."""
    H, W = grid.shape
    arr = np.zeros((1, CHANNELS, H, W), np.float32)
    for r in range(H):
        for col in range(W):
            arr[0, int(grid[r, col]), r, col] = 1.0
    return g.fconst(arr.ravel().tolist(), [1, CHANNELS, H, W])


def _build_167():
    g = _G()
    # presence of each channel anywhere, then count nonzero colours
    pres = g.node("ReduceMax", ["input"], axes=[2, 3], keepdims=1)   # [1,10,1,1]
    pres9 = _slice(g, pres, [1], [CHANNELS], [1])                    # [1,9,1,1]
    c = g.node("ReduceSum", [pres9], axes=[1], keepdims=1)           # [1,1,1,1]
    top = _grid_onehot(g, _T167_TOP)
    md = _grid_onehot(g, _T167_MD)
    ad = _grid_onehot(g, _T167_AD)
    is1 = g.node("Less", [g.node("Abs", [g.node("Sub", [c, g.scalar(1.0)])]),
                          g.scalar(0.5)])
    is2 = g.node("Less", [g.node("Abs", [g.node("Sub", [c, g.scalar(2.0)])]),
                          g.scalar(0.5)])
    inner = g.node("Where", [is2, md, ad])                          # [1,10,3,3]
    sel = g.node("Where", [is1, top, inner])                        # [1,10,3,3]
    g.node("Pad", [sel], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - 3, WIDTH - 3])
    return _model(g.nodes, g.inits)


# ===========================================================================
# 371  crk2_plusmid:  two colour-1 markers -> colour-3 plus at their midpoint
# ===========================================================================
def _ref_371(a):
    H, W = a.shape
    ones = list(zip(*np.where(a == 1)))
    if len(ones) != 2:
        return None
    (r1, c1), (r2, c2) = ones
    if (r1 + r2) % 2 or (c1 + c2) % 2:
        return None
    rc, cc = (r1 + r2) // 2, (c1 + c2) // 2
    o = a.copy()
    for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, ccc = rc + dr, cc + dc
        if 0 <= rr < H and 0 <= ccc < W:
            o[rr, ccc] = 3
    return o


def _build_371():
    g = _G()
    ridx = g.fconst(list(range(HEIGHT)), [1, 1, HEIGHT, 1])
    cidx = g.fconst(list(range(WIDTH)), [1, 1, 1, WIDTH])
    mask1 = _slice(g, "input", [1], [2], [1])                       # [1,1,30,30] colour-1
    rproj = g.node("ReduceSum", [mask1], axes=[3], keepdims=1)      # [1,1,30,1]
    cproj = g.node("ReduceSum", [mask1], axes=[2], keepdims=1)      # [1,1,1,30]
    rsum = g.node("ReduceSum", [g.node("Mul", [rproj, ridx])], axes=[2], keepdims=1)
    csum = g.node("ReduceSum", [g.node("Mul", [cproj, cidx])], axes=[3], keepdims=1)
    half = g.scalar(0.5)
    cr = g.node("Mul", [rsum, half])                               # [1,1,1,1]
    cc = g.node("Mul", [csum, half])
    dr = g.node("Abs", [g.node("Sub", [ridx, cr])])               # [1,1,30,1]
    dc = g.node("Abs", [g.node("Sub", [cidx, cc])])               # [1,1,1,30]
    manh = g.node("Add", [dr, dc])                                # [1,1,30,30]
    near = g.node("Cast", [g.node("Less", [manh, g.scalar(1.5)])], to=DATA_TYPE)
    realcell = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)  # [1,1,30,30]
    plusf = g.node("Mul", [near, realcell])
    plusb = g.node("Greater", [plusf, half])                       # [1,1,30,30] bool
    g.node("Where", [plusb, _onehot_channel(g, 3), "input"], "output")
    return _model(g.nodes, g.inits)


# ===========================================================================
# 329  midcol:  keep only the centre column (odd width); rest -> background
# ===========================================================================
def _ref_329(a):
    H, W = a.shape
    o = np.zeros_like(a)
    cc = (W - 1) // 2
    o[:, cc] = a[:, cc]
    return o


def _build_329():
    g = _G()
    realcell = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)   # [1,1,30,30]
    realcol = g.node("ReduceMax", [realcell], axes=[2], keepdims=1)   # [1,1,1,30]
    W = g.node("ReduceSum", [realcol], axes=[3], keepdims=1)          # [1,1,1,1]
    wm1 = g.node("Sub", [W, g.scalar(1.0)])
    twoidx = g.fconst([2 * i for i in range(WIDTH)], [1, 1, 1, WIDTH])
    diff = g.node("Abs", [g.node("Sub", [twoidx, wm1])])             # [1,1,1,30]
    centerf = g.node("Cast", [g.node("Less", [diff, g.scalar(0.5)])], to=DATA_TYPE)
    noncenter = g.node("Sub", [g.scalar(1.0), centerf])             # [1,1,1,30]
    condf = g.node("Mul", [realcell, noncenter])                    # [1,1,30,30]
    condb = g.node("Greater", [condf, g.scalar(0.5)])              # bool
    g.node("Where", [condb, _onehot_channel(g, 0), "input"], "output")
    return _model(g.nodes, g.inits)


# ===========================================================================
# 126  cupdrop_m4:  each downward-open cup drops colour-4 to the bottom row
# ===========================================================================
def _ref_126(a):
    H, W = a.shape
    occ = (a > 0).astype(int)
    o = a.copy()
    cols = set()
    for r in range(H):
        for c in range(W):
            if a[r, c] == 0:
                left = c - 1 >= 0 and occ[r, c - 1]
                right = c + 1 < W and occ[r, c + 1]
                up = r - 1 >= 0 and occ[r - 1, c]
                if left and right and up:
                    cols.add(c)
    for c in cols:
        o[H - 1, c] = 4
    return o


def _build_126():
    g = _G()
    realcell = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)   # [1,1,30,30]
    bg = _slice(g, "input", [0], [1], [1])                           # [1,1,30,30] ch0
    occ = g.node("Sub", [realcell, bg])                             # [1,1,30,30]
    # neighbour count: up + left + right
    ker = g.fconst([0, 1, 0, 1, 0, 1, 0, 0, 0], [1, 1, 3, 3])
    nsum = g.node("Conv", [occ, ker], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    gapf = g.node("Cast", [g.node("Greater", [nsum, g.scalar(2.5)])], to=DATA_TYPE)
    gap = g.node("Mul", [gapf, bg])                                # [1,1,30,30]
    gapcol = g.node("ReduceMax", [gap], axes=[2], keepdims=1)       # [1,1,1,30]
    realrow = g.node("ReduceMax", [realcell], axes=[3], keepdims=1)  # [1,1,30,1]
    nextrow = g.node("Pad", [_slice(g, realrow, [1], [HEIGHT], [2])],
                     mode="constant", value=0.0,
                     pads=[0, 0, 0, 0, 0, 0, 1, 0])                 # realrow[r+1]
    lastrow = g.node("Mul", [realrow,
                             g.node("Sub", [g.scalar(1.0), nextrow])])
    markerpos = g.node("Mul", [lastrow, gapcol])                    # [1,1,30,30]
    dropb = g.node("Greater", [markerpos, g.scalar(0.5)])         # bool
    g.node("Where", [dropb, _onehot_channel(g, 4), "input"], "output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    def emit(name, fn):
        try:
            out.append((name, fn()))
        except Exception:
            pass

    same_size = all(a.shape == b.shape for a, b in prs)
    changes = any(not np.array_equal(a, b) for a, b in prs)

    # ---- 167 -----------------------------------------------------------------
    if all(a.shape == (3, 3) and b.shape == (3, 3) for a, b in prs):
        ok = True
        for a, b in prs:
            r = _ref_167(a)
            if r is None or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            emit("t167", _build_167)

    # ---- 371 -----------------------------------------------------------------
    if changes and same_size:
        ok = True
        for a, b in prs:
            r = _ref_371(a)
            if r is None or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            emit("plusmid", _build_371)

    # ---- 329 -----------------------------------------------------------------
    if changes and same_size and all(a.shape[1] % 2 == 1 for a, _ in prs):
        ok = all(np.array_equal(_ref_329(a), b) for a, b in prs)
        if ok:
            emit("midcol", _build_329)

    # ---- 126 -----------------------------------------------------------------
    if changes and same_size:
        ok = all(np.array_equal(_ref_126(a), b) for a, b in prs)
        if ok:
            emit("cupdrop", _build_126)

    return out
