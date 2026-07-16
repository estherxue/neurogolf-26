"""family_crk10_3 -- HARD tasks, slice U[3::8] = [23,80,133,159,205,285].

Solved: task 205  ("marker cross inside a solid rectangle, cropped").

TASK 205 rule (verified EXACT on all 266 train+test+arc-gen pairs)
------------------------------------------------------------------
The input is a full 30x30-ish field of RANDOM digits 0..9.  Hidden inside sits a
SOLID axis-aligned rectangle of a single colour C (>=6x6; every border row/col is
100% C; a handful of interior cells carry a second colour M -- the "markers").
The output is that rectangle CROPPED to the top-left, with every marker turned
into a full CROSS: for each marker at (r,c) the whole block-row r and whole
block-column c become M; all other block cells stay C.

Detection that is exactly ONNX-expressible (no connected components, robust to
the heavy digit noise, incl. C==0):
  * For every colour channel compute EH = "this cell starts a run of >=6
    horizontal C" (6-fold Min of column-shifts) and EV likewise vertically.
  * block colour C = ArgMax over channels of (sum EH + sum EV).  Noise never
    forms a 6-run, so only the real block scores -> C is picked reliably.
  * block ROWS = rows that contain any EH cell; block COLS = cols with any EV.
    The solid borders guarantee the min/max of these equal the true bbox, so the
    bbox is exact (interior rows broken by markers don't affect min/max).
  * crop to that bbox with the dyncrop "selection-matrix + MatMul" shift trick.
  * inside the crop: blockmask = any channel set; Mmask = blockmask - C-channel;
    cross = (row-has-M) OR (col-has-M), clipped to blockmask; C-cells = block
    minus cross.  Route C-cells -> channel C, cross -> channel M (both one-hot
    channel indices from ArgMax), everything else 0.

All tensors stay exactly 0/1 so the grader's (output>0) threshold is exact.

The other five slice tasks (23/80/133/159/285) encode global 2x2-tiling /
meta-grid propagation / template-mirror rules that are not expressible as a
static opset-10 graph; no candidate is emitted for them.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
INT32 = onnx.TensorProto.INT32
F = DATA_TYPE
H, W = HEIGHT, WIDTH
_CBIG = 1000.0
RUN = 6  # solid-run length used to find the block


# --------------------------------------------------------------------------- #
# graph accumulator                                                            #
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


def _shift(g, x, dr, dc):
    """result[i,j] = x[i+dr, j+dc]  (zero fill), keeping 30x30, anchored."""
    pt, pb = max(-dr, 0), max(dr, 0)
    pl, pr = max(-dc, 0), max(dc, 0)
    p = g.nd("Pad", [x], mode="constant", value=0.0,
             pads=[0, 0, pt, pl, 0, 0, pb, pr])
    st = g.i64([max(dr, 0), max(dc, 0)])
    en = g.i64([max(dr, 0) + H, max(dc, 0) + W])
    ax = g.i64([2, 3])
    return g.nd("Slice", [p, st, en, ax])


def build_205():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    cbig = g.f([1, 1, 1, 1], [_CBIG])

    # --- per-channel solid-run maps (all 10 channels at once) ---------------
    # EH[i,j] = min over k=0..RUN-1 of input[i, j+k]  (1 iff RUN horizontal C)
    eh = "input"
    for k in range(1, RUN):
        eh = g.nd("Min", [eh, _shift(g, "input", 0, k)])
    ev = "input"
    for k in range(1, RUN):
        ev = g.nd("Min", [ev, _shift(g, "input", k, 0)])

    # --- pick block colour C = ArgMax over channels of (sumEH + sumEV) -------
    sh = g.nd("ReduceSum", [eh], axes=[2, 3], keepdims=1)     # [1,10,1,1]
    sv = g.nd("ReduceSum", [ev], axes=[2, 3], keepdims=1)     # [1,10,1,1]
    score = g.nd("Add", [sh, sv])                             # [1,10,1,1]
    cAmax = g.nd("ArgMax", [score], axis=1, keepdims=1)       # int64 [1,1,1,1]
    chidx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    gateC = g.nd("Cast", [g.nd("Equal", [cAmax, chidx])], to=F)   # [1,10,1,1]

    ehC = g.nd("ReduceSum", [g.nd("Mul", [eh, gateC])], axes=[1], keepdims=1)  # [1,1,30,30]
    evC = g.nd("ReduceSum", [g.nd("Mul", [ev, gateC])], axes=[1], keepdims=1)  # [1,1,30,30]

    # block rows / cols existence -> bbox
    rowhas = g.nd("ReduceMax", [ehC], axes=[3], keepdims=1)   # [1,1,30,1]
    colhas = g.nd("ReduceMax", [evC], axes=[2], keepdims=1)   # [1,1,1,30]

    maxrow = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    minrow = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    mincol = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])
    bbox_h = g.nd("Add", [g.nd("Sub", [maxrow, minrow]), one])
    bbox_w = g.nd("Add", [g.nd("Sub", [maxcol, mincol]), one])

    # --- crop-to-origin selection matrices (dyncrop trick) ------------------
    diff_c = g.nd("Sub", [g.nd("Add", [colidx, mincol]), rowidx])
    match_c = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_c]), half])], to=F)
    trunc_c = g.nd("Cast", [g.nd("Less", [colidx, bbox_w])], to=F)
    Scol = g.nd("Mul", [match_c, trunc_c])                    # [1,1,30,30]

    diff_r = g.nd("Sub", [colidx, g.nd("Add", [rowidx, minrow])])
    match_r = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_r]), half])], to=F)
    trunc_r = g.nd("Cast", [g.nd("Less", [rowidx, bbox_h])], to=F)
    Srow = g.nd("Mul", [match_r, trunc_r])                    # [1,1,30,30]

    shift1 = g.nd("MatMul", ["input", Scol])                  # [1,10,30,30]
    cropped = g.nd("MatMul", [Srow, shift1])                  # [1,10,30,30]

    # --- inside the crop: cross-expand markers ------------------------------
    blockmask = g.nd("ReduceSum", [cropped], axes=[1], keepdims=1)     # [1,1,30,30]
    Ccrop = g.nd("ReduceSum", [g.nd("Mul", [cropped, gateC])], axes=[1], keepdims=1)
    Mmask = g.nd("Sub", [blockmask, Ccrop])                            # [1,1,30,30]

    rowM = g.nd("ReduceMax", [Mmask], axes=[3], keepdims=1)            # [1,1,30,1]
    colM = g.nd("ReduceMax", [Mmask], axes=[2], keepdims=1)            # [1,1,1,30]
    crossfull = g.nd("Max", [rowM, colM])                             # broadcast [1,1,30,30]
    crossM = g.nd("Mul", [crossfull, blockmask])                      # clip to block
    Cfinal = g.nd("Sub", [blockmask, crossM])

    # marker colour channel M = ArgMax of per-channel marker counts (suppress C)
    counts = g.nd("ReduceSum", [cropped], axes=[2, 3], keepdims=1)     # [1,10,1,1]
    supp = g.nd("Mul", [gateC, g.f([1, 1, 1, 1], [_CBIG])])
    selM = g.nd("Sub", [counts, supp])
    mAmax = g.nd("ArgMax", [selM], axis=1, keepdims=1)
    gateM = g.nd("Cast", [g.nd("Equal", [mAmax, chidx])], to=F)        # [1,10,1,1]

    outC = g.nd("Mul", [gateC, Cfinal])                               # [1,10,30,30]
    outM = g.nd("Mul", [gateM, crossM])
    g.nd("Add", [outC, outM], "output")

    return _model(g)


# --------------------------------------------------------------------------- #
# TASK 159:  crop the hollow box, fill its interior with the 3x3 marker upscaled
#            by k = (interior)/marker to exactly fit.  Data-dependent crop AND a
#            data-dependent upscale, both via MatMul selection matrices.
# --------------------------------------------------------------------------- #
def _sel_from_mask(g, mask, rowidx, colidx, half, one, cbig):
    """Return (Srow, Scol, minrow, maxrow, mincol, maxcol) for the bbox of mask."""
    rowhas = g.nd("ReduceMax", [mask], axes=[3], keepdims=1)   # [1,1,30,1]
    colhas = g.nd("ReduceMax", [mask], axes=[2], keepdims=1)   # [1,1,1,30]
    maxrow = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    minrow = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    mincol = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])
    bbox_h = g.nd("Add", [g.nd("Sub", [maxrow, minrow]), one])
    bbox_w = g.nd("Add", [g.nd("Sub", [maxcol, mincol]), one])
    diff_c = g.nd("Sub", [g.nd("Add", [colidx, mincol]), rowidx])
    match_c = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_c]), half])], to=F)
    trunc_c = g.nd("Cast", [g.nd("Less", [colidx, bbox_w])], to=F)
    Scol = g.nd("Mul", [match_c, trunc_c])
    diff_r = g.nd("Sub", [colidx, g.nd("Add", [rowidx, minrow])])
    match_r = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_r]), half])], to=F)
    trunc_r = g.nd("Cast", [g.nd("Less", [rowidx, bbox_h])], to=F)
    Srow = g.nd("Mul", [match_r, trunc_r])
    return Srow, Scol, minrow, maxrow, mincol, maxcol


def build_159():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    cbig = g.f([1, 1, 1, 1], [_CBIG])
    chidx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    bgneg = g.f([1, CHANNELS, 1, 1], [-_CBIG] + [0.0] * (CHANNELS - 1))
    gate0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))

    # colour selection: box = most cells (frame), marker = the other non-bg
    cnt = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)     # [1,10,1,1]
    cnt0 = g.nd("Add", [cnt, bgneg])
    boxAmax = g.nd("ArgMax", [cnt0], axis=1, keepdims=1)
    gateBox = g.nd("Cast", [g.nd("Equal", [boxAmax, chidx])], to=F)
    supp = g.nd("Mul", [gateBox, g.f([1, 1, 1, 1], [_CBIG])])
    cntM = g.nd("Sub", [cnt0, supp])
    markAmax = g.nd("ArgMax", [cntM], axis=1, keepdims=1)
    gateMark = g.nd("Cast", [g.nd("Equal", [markAmax, chidx])], to=F)

    boxmask = g.nd("ReduceSum", [g.nd("Mul", ["input", gateBox])], axes=[1], keepdims=1)
    markmask = g.nd("ReduceSum", [g.nd("Mul", ["input", gateMark])], axes=[1], keepdims=1)

    # crop the whole field to the box bbox -> frame + interior background (chan 0)
    Srb, Scb, minr_b, maxr_b, minc_b, maxc_b = _sel_from_mask(g, boxmask, rowidx, colidx, half, one, cbig)
    BOX = g.nd("MatMul", [Srb, g.nd("MatMul", ["input", Scb])])     # [1,10,30,30]

    # crop the marker channel to its bbox -> Pmask (pattern at origin)
    Srm, Scm, minr_m, maxr_m, minc_m, maxc_m = _sel_from_mask(g, markmask, rowidx, colidx, half, one, cbig)
    Pmask = g.nd("MatMul", [Srm, g.nd("MatMul", [markmask, Scm])])  # [1,1,30,30]

    # interior extent (h-2, w-2) and marker extent (mh, mw)
    hm2 = g.nd("Sub", [g.nd("Sub", [maxr_b, minr_b]), one])         # (maxr-minr+1)-2
    wm2 = g.nd("Sub", [g.nd("Sub", [maxc_b, minc_b]), one])
    mh = g.nd("Add", [g.nd("Sub", [maxr_m, minr_m]), one])
    mw = g.nd("Add", [g.nd("Sub", [maxc_m, minc_m]), one])

    # Arow[i,m] = (i*mh >= m*hm2) & (i*mh < (m+1)*hm2)   (== floor(i/kh)==m, no div)
    imh = g.nd("Mul", [rowidx, mh])                                # [1,1,30,1]
    mhm2 = g.nd("Mul", [colidx, hm2])                             # [1,1,1,30]
    Ar = g.nd("Sub", [imh, mhm2])                                # [1,1,30,30]
    ge_r = g.nd("Cast", [g.nd("Greater", [Ar, g.nd("Sub", [g.f([1, 1, 1, 1], [0.0]), half])])], to=F)
    lt_r = g.nd("Cast", [g.nd("Less", [g.nd("Sub", [Ar, hm2]), g.nd("Sub", [g.f([1, 1, 1, 1], [0.0]), half])])], to=F)
    Arow = g.nd("Mul", [ge_r, lt_r])

    # AcolT[n,j] = (j*mw >= n*wm2) & (j*mw < (n+1)*wm2)
    jmw = g.nd("Mul", [colidx, mw])                              # [1,1,1,30]
    nwm2 = g.nd("Mul", [rowidx, wm2])                            # [1,1,30,1]
    Ac = g.nd("Sub", [jmw, nwm2])                                # [1,1,30,30]
    neghalf = g.nd("Sub", [g.f([1, 1, 1, 1], [0.0]), half])
    ge_c = g.nd("Cast", [g.nd("Greater", [Ac, neghalf])], to=F)
    lt_c = g.nd("Cast", [g.nd("Less", [g.nd("Sub", [Ac, wm2]), neghalf])], to=F)
    AcolT = g.nd("Mul", [ge_c, lt_c])

    scaled = g.nd("MatMul", [Arow, g.nd("MatMul", [Pmask, AcolT])])  # [1,1,30,30]
    interior = _shift(g, scaled, -1, -1)                            # placed at (1,1)

    # output = BOX  + (gateMark - gate0) * interior
    delta = g.nd("Sub", [gateMark, gate0])                         # [1,10,1,1]
    g.nd("Add", [BOX, g.nd("Mul", [delta, interior])], "output")
    return _model(g)


def _solve_159(grid):
    counts = np.array([(grid == c).sum() for c in range(CHANNELS)], np.int64)
    counts0 = counts.copy(); counts0[0] = -(10 ** 9)
    nz = [c for c in range(1, CHANNELS) if counts[c] > 0]
    if len(nz) != 2:
        return None
    boxc = int(np.argmax(counts0))
    mc = nz[0] if nz[1] == boxc else nz[1]
    if mc == boxc:
        return None
    by, bx = np.where(grid == boxc)
    r0, r1, c0, c1 = int(by.min()), int(by.max()), int(bx.min()), int(bx.max())
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = grid[r0:r1 + 1, c0:c1 + 1].copy()
    my, mx = np.where(grid == mc)
    mr0, mr1, mc0, mc1 = int(my.min()), int(my.max()), int(mx.min()), int(mx.max())
    mh, mw = mr1 - mr0 + 1, mc1 - mc0 + 1
    P = (grid[mr0:mr1 + 1, mc0:mc1 + 1] == mc)
    ih, iw = h - 2, w - 2
    if ih <= 0 or iw <= 0 or ih % mh or iw % mw:
        return None
    scaled = np.kron(P, np.ones((ih // mh, iw // mw), bool))
    out[1:h - 1, 1:w - 1] = np.where(scaled, mc, out[1:h - 1, 1:w - 1])
    return out


# --------------------------------------------------------------------------- #
# numpy mirror of the exact ONNX numerics (used for detection)                 #
# --------------------------------------------------------------------------- #
def _onehot(grid):
    h, w = grid.shape
    oh_ = np.zeros((CHANNELS, H, W), np.float32)
    for c in range(CHANNELS):
        oh_[c, :h, :w] = (grid == c)
    return oh_


def _erode_run(m, axis):
    """m [30,30] 0/1 -> cell is 1 iff RUN consecutive 1s start here along axis."""
    e = m.copy()
    for k in range(1, RUN):
        s = np.zeros_like(m)
        if axis == 1:
            s[:, :W - k] = m[:, k:]
        else:
            s[:H - k, :] = m[k:, :]
        e = np.minimum(e, s)
    return e


def _solve(grid):
    oh_ = _onehot(grid)
    eh = np.stack([_erode_run(oh_[c], 1) for c in range(CHANNELS)])
    ev = np.stack([_erode_run(oh_[c], 0) for c in range(CHANNELS)])
    score = eh.reshape(CHANNELS, -1).sum(1) + ev.reshape(CHANNELS, -1).sum(1)
    C = int(np.argmax(score))
    rowhas = eh[C].max(axis=1)
    colhas = ev[C].max(axis=0)
    rr = np.where(rowhas > 0)[0]
    cc = np.where(colhas > 0)[0]
    if rr.size == 0 or cc.size == 0:
        return None
    r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    sub = grid[r0:r1 + 1, c0:c1 + 1]
    nonC = (sub != C)
    if nonC.sum() == 0:
        return None
    mv, mc = np.unique(sub[nonC], return_counts=True)
    M = int(mv[np.argmax(mc)])
    marker = (sub == M)
    rowm = marker.any(axis=1, keepdims=True)
    colm = marker.any(axis=0, keepdims=True)
    cross = (rowm | colm)
    out = np.full(sub.shape, C, int)
    out[cross] = M
    return out


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
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def candidates(ex):
    prs = _pairs(ex)
    if len(prs) < 2:
        return []
    # must genuinely crop (output strictly smaller) on some pair
    if not any(b.shape[0] < a.shape[0] or b.shape[1] < a.shape[1] for a, b in prs):
        return []

    out = []

    # ---- task 205: solid rectangle w/ marker crosses ----
    if all((o := _solve(a)) is not None and o.shape == b.shape and np.array_equal(o, b)
           for a, b in prs):
        try:
            m = build_205()
            onnx.checker.check_model(m, full_check=True)
            out.append(("crk10_3_blockcross", m))
        except Exception:
            pass

    # ---- task 159: hollow box + upscaled marker fill ----
    if all((o := _solve_159(a)) is not None and o.shape == b.shape and np.array_equal(o, b)
           for a, b in prs):
        try:
            m = build_159()
            onnx.checker.check_model(m, full_check=True)
            out.append(("crk10_3_boxfill", m))
        except Exception:
            pass

    return out
