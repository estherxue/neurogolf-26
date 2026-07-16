"""family_golf4_4 -- cheaper exact re-solvers for a slice of golf targets.

Each candidate re-derives the task rule from train+test+arc-gen pairs, validates a
numpy mirror of the *exact* ONNX op semantics on the true one-hot representation of
every provided pair, and only then emits a minimal opset-10 graph.  The integrator
auto-picks the cheapest correct solver, so we only need to be exact and cheaper
than the incumbent.

Golf levers used here:
  * single-channel [1,1,30,30] intermediates for the heavy work;
  * per-row span fill via two triangular MatMuls (prefix/suffix max) instead of a
    [900,900] reachability matrix;
  * recolour by broadcasting per-row colour vectors ([1,9,30,1]) against
    column/row masks, then writing straight into the FREE `output` via Concat.

Targets (rule -> incumbent points):
  41   hspan: per-row horizontal span-fill between the leftmost and rightmost
       non-background cell of each row, keeping that row's colour.        (13.11)
  197  crk2_5_template: one fully-populated "template" row encodes a 2-colour
       column pattern; every partial content row supplies two colours (at its
       first and second distinct positions) and the pattern is re-coloured and
       tiled across the row.                                              (12.16)
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


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _slice(g, src, starts, ends, axes, steps=None):
    ins = [src, g.i64(starts), g.i64(ends), g.i64(axes)]
    if steps is not None:
        ins.append(g.i64(steps))
    return g.nd("Slice", ins)


# --------------------------------------------------------------------------- #
# pairs + one-hot helpers                                                      #
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


def _oh(a):
    """grid -> [10,30,30] one-hot, top-left anchored (padding all-zero)."""
    X = np.zeros((CHANNELS, H, W), np.float32)
    h, w = a.shape
    for r in range(h):
        for c in range(w):
            X[int(a[r, c]), r, c] = 1.0
    return X


def _eq_oh(Y, b):
    """EXACT grader comparison: (Y>0) per channel must match one-hot(b)."""
    return np.array_equal(Y > 0, _oh(b) > 0)


# ===========================================================================
# 41  hspan -- per-row span fill between min/max non-bg column, keep colour
# ===========================================================================
def _mir_41(a):
    X = _oh(a)
    inmask = X.sum(0)                         # [30,30]
    Xc = X[1:10]                              # [9,30,30] per-colour presence
    leftcum = np.cumsum(Xc, axis=2)           # sum_{a<=c} along width
    rightcum = np.cumsum(Xc[:, :, ::-1], axis=2)[:, :, ::-1]   # sum_{a>=c}
    colored19 = np.minimum(leftcum, rightcum)  # >0 == per-colour row span
    out0 = inmask - colored19.sum(0)
    return np.concatenate([out0[None], colored19], axis=0)


def _build_41(g):
    # U[a,c]=1 if a<=c  (prefix);  L[a,c]=1 if a>=c (suffix) -- per-row counts
    U = g.f([W, W], np.triu(np.ones((W, W), np.float32)))
    L = g.f([W, W], np.tril(np.ones((W, W), np.float32)))

    inmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)        # [1,1,30,30]
    Xc = _slice(g, "input", [1], [10], [1])                          # [1,9,30,30]
    leftcum = g.nd("MatMul", [Xc, U])                                # [1,9,30,30]
    rightcum = g.nd("MatMul", [Xc, L])                               # [1,9,30,30]
    colored19 = g.nd("Min", [leftcum, rightcum])                     # >0 == span
    colsum = g.nd("ReduceSum", [colored19], axes=[1], keepdims=1)
    out0 = g.nd("Sub", [inmask, colsum])
    g.nd("Concat", [out0, colored19], "output", axis=1)
    return _model(g)


def _detect_41(prs):
    """Same H/W; each row is single-colour; output == per-row span fill."""
    for a, b in prs:
        if a.shape != b.shape:
            return False
    ok = False
    for a, b in prs:
        if not _eq_oh(_mir_41(a), b):
            return False
        if not np.array_equal(a, b):
            ok = True
    return ok


# ===========================================================================
# 197 crk2_5_template -- template row's 2-colour pattern, recoloured per row
# ===========================================================================
def _mir_197(a):
    X = _oh(a)
    inmask = X.sum(0)                                  # [30,30]
    nonbg = inmask - X[0]
    rowfill = nonbg.sum(axis=1)                        # [30]
    rowsel = (rowfill > rowfill.max() - 0.5).astype(np.float32)   # [30]
    templ = (X * rowsel[None, :, None]).sum(axis=1)    # [10,30] template row pattern
    Pcol = templ[:, 0]                                 # [10] one-hot(P)
    maskP = (templ * Pcol[:, None]).sum(0)             # [30]
    tmplnz = templ.sum(0)                              # [30]  (in-grid template cols)
    maskQ = tmplnz - maskP                             # [30]
    colP_c = X[1:10, :, 0]                             # [9,30] col0 colour per row
    rp_c = X[1:10].max(axis=2)                         # [9,30] colours present per row
    colQ_c = rp_c - colP_c                             # [9,30] second colour per row
    coloredP = colP_c[:, :, None] * maskP[None, None, :]   # [9,30,30]
    coloredQ = colQ_c[:, :, None] * maskQ[None, None, :]
    colored19 = coloredP + coloredQ
    out0 = inmask - colored19.sum(0)
    return np.concatenate([out0[None], colored19], axis=0)


def _build_197(g):
    half = g.f([1, 1, 1, 1], [0.5])

    inmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)        # [1,1,30,30]
    bg = _slice(g, "input", [0], [1], [1])
    nonbg = g.nd("Sub", [inmask, bg])
    rowfill = g.nd("ReduceSum", [nonbg], axes=[3], keepdims=1)          # [1,1,30,1]
    maxfill = g.nd("ReduceMax", [rowfill], axes=[2], keepdims=1)        # [1,1,1,1]
    thr = g.nd("Sub", [maxfill, half])
    rowsel = g.nd("Cast", [g.nd("Greater", [rowfill, thr])], to=F)      # [1,1,30,1]
    msel = g.nd("Mul", ["input", rowsel])                              # [1,10,30,30]
    templ = g.nd("ReduceSum", [msel], axes=[2], keepdims=1)             # [1,10,1,30]
    Pcol = _slice(g, templ, [0], [1], [3])                            # [1,10,1,1]
    maskP = g.nd("ReduceSum", [g.nd("Mul", [templ, Pcol])], axes=[1], keepdims=1)  # [1,1,1,30]
    tmplnz = g.nd("ReduceSum", [templ], axes=[1], keepdims=1)           # [1,1,1,30]
    maskQ = g.nd("Sub", [tmplnz, maskP])                              # [1,1,1,30]

    colP_full = _slice(g, "input", [0], [1], [3])                     # [1,10,30,1]
    colP_c = _slice(g, colP_full, [1], [10], [1])                    # [1,9,30,1]
    rp_full = g.nd("ReduceMax", ["input"], axes=[3], keepdims=1)       # [1,10,30,1]
    rp_c = _slice(g, rp_full, [1], [10], [1])                        # [1,9,30,1]
    colQ_c = g.nd("Sub", [rp_c, colP_c])                             # [1,9,30,1]

    coloredP = g.nd("Mul", [colP_c, maskP])                          # [1,9,30,30]
    coloredQ = g.nd("Mul", [colQ_c, maskQ])                          # [1,9,30,30]
    colored19 = g.nd("Add", [coloredP, coloredQ])                    # [1,9,30,30]
    colsum = g.nd("ReduceSum", [colored19], axes=[1], keepdims=1)
    out0 = g.nd("Sub", [inmask, colsum])
    g.nd("Concat", [out0, colored19], "output", axis=1)
    return _model(g)


def _detect_197(prs):
    for a, b in prs:
        if a.shape != b.shape:
            return False
    ok = False
    for a, b in prs:
        if not _eq_oh(_mir_197(a), b):
            return False
        if not np.array_equal(a, b):
            ok = True
    return ok


# ===========================================================================
# 202 crk2_4_t202 -- full-span monochrome bands; each line through a hole,
#                    perpendicular to the band's long axis, is cleared to bg.
# ===========================================================================
def _mir_202(a):
    X = _oh(a)
    inmask = X.sum(0)                                  # [30,30]
    Xc = X[1:10]                                       # [9,30,30] colour masks
    bandrow = Xc.max(axis=2)                           # [9,30] colour k present in row r
    bandcol = Xc.max(axis=1)                           # [9,30] colour k present in col c
    colcount = Xc.sum(axis=1)                          # [9,30] # of colour k in col c
    rowcount = Xc.sum(axis=2)                          # [9,30] # of colour k in row r
    height = bandrow.sum(axis=1)                       # [9] band height
    width = bandcol.sum(axis=1)                        # [9] band width
    gridW = inmask.max(axis=0).sum()                   # full-grid width
    gridH = inmask.max(axis=1).sum()                   # full-grid height
    horiz = (width >= gridW - 0.5).astype(np.float32)  # [9] full-width band -> vertical slits
    vert = (height >= gridH - 0.5).astype(np.float32)  # [9] full-height band -> horizontal slits
    holecol = bandcol * (colcount < height[:, None]).astype(np.float32)   # [9,30]
    holerow = bandrow * (rowcount < width[:, None]).astype(np.float32)    # [9,30]
    hc_h = holecol * horiz[:, None]                    # [9,30] col-indexed
    hr_nh = holerow * vert[:, None]                    # [9,30] row-indexed
    punchV = bandrow[:, :, None] * hc_h[:, None, :]    # [9,30,30]
    punchH = bandcol[:, None, :] * hr_nh[:, :, None]   # [9,30,30]
    punch = punchV + punchH
    keep = 1.0 - punch.sum(0)                          # [30,30]
    colored19 = Xc * keep[None]                        # [9,30,30]
    out0 = inmask - colored19.sum(0)
    return np.concatenate([out0[None], colored19], axis=0)


def _build_202(g):
    one = g.f([1, 1, 1, 1], [1.0])

    inmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)        # [1,1,30,30]
    Xc = _slice(g, "input", [1], [10], [1])                          # [1,9,30,30]
    bandrow = g.nd("ReduceMax", [Xc], axes=[3], keepdims=1)            # [1,9,30,1]
    bandcol = g.nd("ReduceMax", [Xc], axes=[2], keepdims=1)            # [1,9,1,30]
    colcount = g.nd("ReduceSum", [Xc], axes=[2], keepdims=1)           # [1,9,1,30]
    rowcount = g.nd("ReduceSum", [Xc], axes=[3], keepdims=1)           # [1,9,30,1]
    height = g.nd("ReduceSum", [bandrow], axes=[2], keepdims=1)         # [1,9,1,1]
    width = g.nd("ReduceSum", [bandcol], axes=[3], keepdims=1)          # [1,9,1,1]
    half = g.f([1, 1, 1, 1], [0.5])
    colany = g.nd("ReduceMax", [inmask], axes=[2], keepdims=1)          # [1,1,1,30]
    rowany = g.nd("ReduceMax", [inmask], axes=[3], keepdims=1)          # [1,1,30,1]
    gridW = g.nd("Sub", [g.nd("ReduceSum", [colany], axes=[3], keepdims=1), half])  # [1,1,1,1]
    gridH = g.nd("Sub", [g.nd("ReduceSum", [rowany], axes=[2], keepdims=1), half])  # [1,1,1,1]
    horiz = g.nd("Cast", [g.nd("Greater", [width, gridW])], to=F)       # [1,9,1,1]
    vert = g.nd("Cast", [g.nd("Greater", [height, gridH])], to=F)       # [1,9,1,1]
    holecol = g.nd("Mul", [bandcol, g.nd("Cast", [g.nd("Less", [colcount, height])], to=F)])  # [1,9,1,30]
    holerow = g.nd("Mul", [bandrow, g.nd("Cast", [g.nd("Less", [rowcount, width])], to=F)])   # [1,9,30,1]
    hc_h = g.nd("Mul", [holecol, horiz])                              # [1,9,1,30]
    hr_nh = g.nd("Mul", [holerow, vert])                            # [1,9,30,1]
    punchV = g.nd("Mul", [bandrow, hc_h])                            # [1,9,30,30]
    punchH = g.nd("Mul", [bandcol, hr_nh])                           # [1,9,30,30]
    punch = g.nd("Add", [punchV, punchH])                            # [1,9,30,30]
    punch_total = g.nd("ReduceSum", [punch], axes=[1], keepdims=1)      # [1,1,30,30]
    keep = g.nd("Sub", [one, punch_total])
    colored19 = g.nd("Mul", [Xc, keep])                             # [1,9,30,30]
    colsum = g.nd("ReduceSum", [colored19], axes=[1], keepdims=1)
    out0 = g.nd("Sub", [inmask, colsum])
    g.nd("Concat", [out0, colored19], "output", axis=1)
    return _model(g)


def _detect_202(prs):
    for a, b in prs:
        if a.shape != b.shape:
            return False
    ok = False
    for a, b in prs:
        if not _eq_oh(_mir_202(a), b):
            return False
        if not np.array_equal(a, b):
            ok = True
    return ok


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    def emit(name, detect, build):
        try:
            if not detect(prs):
                return
            g = _G()
            m = build(g)
            onnx.checker.check_model(m, full_check=True)
            out.append((name, m))
        except Exception:
            pass

    emit("g44_hspan41", _detect_41, _build_41)
    emit("g44_template197", _detect_197, _build_197)
    emit("g44_bandpunch202", _detect_202, _build_202)
    return out
