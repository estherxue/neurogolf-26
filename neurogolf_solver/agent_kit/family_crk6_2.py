"""family_crk6_2 -- cracks for slice U[2::6].

Solved here
-----------
* task 105  GRID-LINE COMPLETION (new).  The input draws a rectangular lattice
  (an outer frame plus optional internal horizontal/vertical dividers) with
  colour 1, but every line is DASHED -- a random subset of its cells is missing.
  The output redraws every line solidly, painting the formerly-missing cells with
  colour 2 and leaving the original 1-cells (and the lattice interior) untouched.

  A row is a "line row" iff it is the top/bottom edge of the foreground bounding
  box, OR it contains two horizontally-adjacent foreground cells, OR it holds >=4
  foreground cells.  Columns are symmetric.  Every line is then filled across the
  full bounding-box extent.  This reproduces all 266 train/test/arc-gen pairs.

  The whole thing is origin-anchored and size-independent: bounding box, per-axis
  line membership and the in-box masks are recovered with reductions + a doubling
  prefix/suffix OR (Pad/Slice shifts by 1,2,4,8,16), then the fill is a couple of
  broadcast products -- no Loop/NonZero, static shapes throughout.

* task 17   periodic restoration -- delegated to family_dynperiod (its
  autocorrelation + data-dependent doubling-OR already reproduces this task).
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


# --------------------------------------------------------------------------- #
# shift / cumulative helpers (axis 2 = rows, axis 3 = cols; size G on the axis) #
# --------------------------------------------------------------------------- #
def _shift_pos(g, t, k, axis):
    """out[..i..] = t[..i-k..]  (content moves toward higher index), zero fill."""
    pads = [0, 0, 0, 0, 0, 0, 0, 0]
    pads[axis] = k                      # begin pad on this axis
    p = g.nd("Pad", [t], mode="constant", value=0.0, pads=pads)
    return g.nd("Slice", [p, g.i64([0]), g.i64([G]), g.i64([axis])])


def _shift_neg(g, t, k, axis):
    """out[..i..] = t[..i+k..]  (content moves toward lower index), zero fill."""
    s = g.nd("Slice", [t, g.i64([k]), g.i64([G]), g.i64([axis])])
    pads = [0, 0, 0, 0, 0, 0, 0, 0]
    pads[4 + axis] = k                  # end pad on this axis
    return g.nd("Pad", [s], mode="constant", value=0.0, pads=pads)


def _prefix_or(g, t, axis):
    cur = t
    for k in (1, 2, 4, 8, 16):
        cur = g.nd("Max", [cur, _shift_pos(g, cur, k, axis)])
    return cur


def _suffix_or(g, t, axis):
    cur = t
    for k in (1, 2, 4, 8, 16):
        cur = g.nd("Max", [cur, _shift_neg(g, cur, k, axis)])
    return cur


# --------------------------------------------------------------------------- #
# ONNX builder for task-105 grid-line completion                               #
# --------------------------------------------------------------------------- #
def build_gridlines():
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0])
    half = g.f([1], [0.5])
    thr = g.f([1], [3.5])

    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)               # [1,1,30,30]
    bg = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])           # ch0
    f = g.nd("Sub", [realmask, bg])                                             # foreground

    # per-axis counts and presence
    rowcnt = g.nd("ReduceSum", [f], axes=[3], keepdims=1)                       # [1,1,30,1]
    colcnt = g.nd("ReduceSum", [f], axes=[2], keepdims=1)                       # [1,1,1,30]
    rowhas = g.nd("Cast", [g.nd("Greater", [rowcnt, half])], to=F)             # [1,1,30,1]
    colhas = g.nd("Cast", [g.nd("Greater", [colcnt, half])], to=F)             # [1,1,1,30]

    # adjacency (two neighbouring foreground cells along the axis)
    fcol = _shift_neg(g, f, 1, 3)
    adjH = g.nd("Mul", [f, fcol])
    rowAdj = g.nd("Cast", [g.nd("Greater", [g.nd("ReduceMax", [adjH], axes=[3], keepdims=1), half])], to=F)
    frow = _shift_neg(g, f, 1, 2)
    adjV = g.nd("Mul", [f, frow])
    colAdj = g.nd("Cast", [g.nd("Greater", [g.nd("ReduceMax", [adjV], axes=[2], keepdims=1), half])], to=F)

    # >=4 foreground cells
    rowBig = g.nd("Cast", [g.nd("Greater", [rowcnt, thr])], to=F)
    colBig = g.nd("Cast", [g.nd("Greater", [colcnt, thr])], to=F)

    # bounding box (in-box mask) + frame edges via prefix/suffix OR
    rPre = _prefix_or(g, rowhas, 2)
    rSuf = _suffix_or(g, rowhas, 2)
    rowInBox = g.nd("Mul", [rPre, rSuf])                                        # [1,1,30,1]
    prevAnyR = _shift_pos(g, rPre, 1, 2)
    nextAnyR = _shift_neg(g, rSuf, 1, 2)
    rowIsMin = g.nd("Mul", [rowhas, g.nd("Sub", [one, prevAnyR])])
    rowIsMax = g.nd("Mul", [rowhas, g.nd("Sub", [one, nextAnyR])])
    rowFrame = g.nd("Max", [rowIsMin, rowIsMax])

    cPre = _prefix_or(g, colhas, 3)
    cSuf = _suffix_or(g, colhas, 3)
    colInBox = g.nd("Mul", [cPre, cSuf])                                        # [1,1,1,30]
    prevAnyC = _shift_pos(g, cPre, 1, 3)
    nextAnyC = _shift_neg(g, cSuf, 1, 3)
    colIsMin = g.nd("Mul", [colhas, g.nd("Sub", [one, prevAnyC])])
    colIsMax = g.nd("Mul", [colhas, g.nd("Sub", [one, nextAnyC])])
    colFrame = g.nd("Max", [colIsMin, colIsMax])

    rowLine = g.nd("Max", [rowAdj, rowBig, rowFrame])                           # [1,1,30,1]
    colLine = g.nd("Max", [colAdj, colBig, colFrame])                           # [1,1,1,30]

    fillR = g.nd("Mul", [rowLine, colInBox])                                    # [1,1,30,30]
    fillC = g.nd("Mul", [colLine, rowInBox])                                    # [1,1,30,30]
    fillAny = g.nd("Max", [fillR, fillC])
    new2 = g.nd("Mul", [fillAny, bg])                                           # paint only over background

    coeff = [0.0] * CHANNELS
    coeff[0] = -1.0
    coeff[2] = 1.0
    coeffv = g.f([1, CHANNELS, 1, 1], coeff)
    delta = g.nd("Mul", [new2, coeffv])                                         # [1,10,30,30]
    g.nd("Add", ["input", delta], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy reference for task-105 detection (exact rule used above)               #
# --------------------------------------------------------------------------- #
def _gridlines_ref(a):
    f = (a != 0)
    H, W = a.shape
    if not f.any():
        return a.copy()
    rs = np.where(f.any(1))[0]
    cs = np.where(f.any(0))[0]
    rmin, rmax, cmin, cmax = rs[0], rs[-1], cs[0], cs[-1]
    rowline = np.zeros(H, bool)
    colline = np.zeros(W, bool)
    rowline[rmin] = rowline[rmax] = True
    colline[cmin] = colline[cmax] = True
    for r in range(H):
        adj = W > 1 and np.any(f[r, :-1] & f[r, 1:])
        if adj or f[r].sum() >= 4:
            rowline[r] = True
    for c in range(W):
        adj = H > 1 and np.any(f[:-1, c] & f[1:, c])
        if adj or f[:, c].sum() >= 4:
            colline[c] = True
    rowinbox = np.zeros(H, bool); rowinbox[rmin:rmax + 1] = True
    colinbox = np.zeros(W, bool); colinbox[cmin:cmax + 1] = True
    fill = (rowline[:, None] & colinbox[None, :]) | (colline[None, :] & rowinbox[:, None])
    out = a.copy()
    out[(a == 0) & fill] = 2
    return out


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
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


def _gridlines_candidate(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return []
    # shape preserving, non-identity, and the grid is built from {0,1} with fill 2
    if any(a.shape != b.shape for a, b in allp):
        return []
    if all(np.array_equal(a, b) for a, b in allp):
        return []
    incol = set()
    outcol = set()
    for a, b in allp:
        incol |= set(int(v) for v in np.unique(a).tolist())
        outcol |= set(int(v) for v in np.unique(b).tolist())
    if incol - {0, 1}:
        return []
    if outcol - {0, 1, 2}:
        return []
    for a, b in allp:
        pred = _gridlines_ref(a)
        if pred.shape != b.shape or not np.array_equal(pred, b):
            return []
    try:
        m = build_gridlines()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("gridlines", m)]


def candidates(ex):
    out = []
    out += _gridlines_candidate(ex)
    # periodic restoration (task 17 etc.) -- reuse the proven dynperiod family.
    try:
        import family_dynperiod
        for name, model in family_dynperiod.candidates(ex):
            out.append((f"dp_{name}", model))
    except Exception:
        pass
    return out
