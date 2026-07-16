"""crk6_3 -- a grab-bag of exact, structure-inferred ARC->ONNX solvers.

Each sub-rule is verified EXACTLY (numpy mirror of the ONNX semantics) on every
train+test+arc-gen pair before its model is emitted, so wrong hypotheses are
never scored.  All graphs are opset-10, static-shape, origin-anchored.

Rules implemented
-----------------
* task305 -- "cyclic diagonal restore":  out[r,c] = ((r+c) mod K) + 1 where
  K = number of colours (= max colour present).  The input is the same Latin /
  diagonal pattern with cells corrupted to 0; the model reconstructs it from the
  cell positions (Mod with a data-dependent divisor) masked to the real grid.

* task033 -- "cell-union completion":  the grid is a 3x3 array of 5x5 cells split
  by full divider lines (rows/cols 5,11) of a single colour D.  The union over
  all cells of their (non-background, non-divider) content is the "master" shape;
  every cell is completed so that each master position it lacks is filled with the
  divider colour D, keeping each cell's own colours.  Realised by OR-ing the nine
  cells together via cell-aligned shifts and stamping D into the still-background
  master positions.
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


def _pairs(ex, splits):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            out.append((np.array(e["input"], int), np.array(e["output"], int)))
    return out


def _shift(g, t, dy, dx):
    """out[...,r,c] = t[...,r-dy,c-dx], zero-filled (origin-anchored translate)."""
    if dy == 0 and dx == 0:
        return t
    h0, w0 = max(dy, 0), max(dx, 0)
    h1, w1 = max(-dy, 0), max(-dx, 0)
    p = g.nd("Pad", [t], mode="constant", value=0.0,
             pads=[0, 0, h0, w0, 0, 0, h1, w1])
    return g.nd("Slice", [p, g.i64([h1, w1]), g.i64([h1 + G, w1 + G]), g.i64([2, 3])])


# =========================================================================== #
# task 305 : cyclic diagonal restore                                          #
# =========================================================================== #
def _sol305(a):
    H, W = a.shape
    K = int(a.max())
    if K < 1:
        return None
    pred = np.fromfunction(lambda r, c: ((r + c) % K) + 1, (H, W), dtype=int)
    return pred


def build305():
    g = _G()
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)          # [1,1,30,30]
    # K = max colour present
    chvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    haschan = g.nd("ReduceMax", ["input"], axes=[2, 3], keepdims=1)        # [1,10,1,1]
    K = g.nd("ReduceMax", [g.nd("Mul", [chvec, haschan])], axes=[1], keepdims=1)  # [1,1,1,1]
    # P[r,c] = r + c
    P = g.f([1, 1, G, G], [[r + c for c in range(G)] for r in range(G)])
    M = g.nd("Mod", [P, K], fmod=1)                                        # [1,1,30,30]  0..K-1
    # one-hot: out[:,j] = relu(1 - |M - (j-1)|) ; channel j wins iff M == j-1
    cidx = g.f([1, CHANNELS, 1, 1], [j - 1 for j in range(CHANNELS)])
    absd = g.nd("Abs", [g.nd("Sub", [M, cidx])])                           # [1,10,30,30]
    one = g.f([1, 1, 1, 1], [1.0])
    ind = g.nd("Relu", [g.nd("Sub", [one, absd])])
    g.nd("Mul", [ind, realmask], "output")
    return _model(g)


# =========================================================================== #
# task 033 : 3x3-cell union completion (fixed 17x17 geometry)                  #
# =========================================================================== #
_BANDS = [(0, 5), (6, 11), (12, 17)]


def _sol033(a):
    H, W = a.shape
    if (H, W) != (17, 17):
        return None
    # divider colour
    D = None
    for r in range(H):
        row = a[r]
        if row[0] != 0 and (row == row[0]).all():
            D = int(row[0])
            break
    if D is None:
        return None
    for r in (5, 11):
        if not (a[r] == D).all():
            return None
    for c in (5, 11):
        if not (a[:, c] == D).all():
            return None
    master = np.zeros((5, 5), bool)
    for (r0, r1) in _BANDS:
        for (c0, c1) in _BANDS:
            master |= (a[r0:r1, c0:c1] != 0)
    out = a.copy()
    for (r0, r1) in _BANDS:
        for (c0, c1) in _BANDS:
            cell = out[r0:r1, c0:c1]
            fill = master & (cell == 0)
            cell[fill] = D
    return out


def _cell_region_mask():
    m = np.zeros((G, G), np.float32)
    valid = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16]
    for r in valid:
        for c in valid:
            m[r, c] = 1.0
    return m


def build033():
    g = _G()
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)          # [1,1,30,30]
    bg = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])      # [1,1,30,30]
    crm = g.f([1, 1, G, G], _cell_region_mask())
    content = g.nd("Mul", [g.nd("Sub", [realmask, bg]), crm])              # coloured cell cells
    # OR over the 9 cells via cell-aligned shifts (cell pitch 6)
    hs = [_shift(g, content, 0, dc) for dc in (-12, -6, 0, 6, 12)]
    hmax = g.nd("Max", hs)
    vs = [_shift(g, hmax, dr, 0) for dr in (-12, -6, 0, 6, 12)]
    master = g.nd("Max", vs)                                               # [1,1,30,30]
    fillmask = g.nd("Mul", [g.nd("Mul", [master, crm]), bg])               # background master pos
    # divider colour one-hot, read at a divider cell (row 5, col 0)
    Donehot = g.nd("Slice", ["input", g.i64([5, 0]), g.i64([6, 1]), g.i64([2, 3])])  # [1,10,1,1]
    e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    dminus = g.nd("Sub", [Donehot, e0])                                    # +1 @ D, -1 @ 0
    fill = g.nd("Mul", [dminus, fillmask])                                 # [1,10,30,30]
    g.nd("Add", ["input", fill], "output")
    return _model(g)


# =========================================================================== #
# task 348 : upward light-cone from a vertical 7-line                          #
# =========================================================================== #
def _sol348(a):
    H, W = a.shape
    ys, xs = np.where(a == 7)
    if len(ys) == 0:
        return None
    A = int((ys + xs).max())
    B = int((ys - xs).max())
    c0 = int(xs.max())
    R = np.arange(H)[:, None]
    C = np.arange(W)[None, :]
    lit = ((R + C) <= A) & ((R - C) <= B)
    par = np.abs((C % 2) - (c0 % 2))
    out = np.zeros_like(a)
    out[lit & (par == 0)] = 7
    out[lit & (par == 1)] = 8
    return out


def build348():
    g = _G()
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)          # [1,1,30,30]
    seven = g.nd("Slice", ["input", g.i64([7]), g.i64([8]), g.i64([1])])   # [1,1,30,30]
    notseven = g.nd("Sub", [g.f([1, 1, 1, 1], [1.0]), seven])
    pen = g.nd("Mul", [notseven, g.f([1, 1, 1, 1], [-1000.0])])            # -1000 off the line
    RpC = g.f([1, 1, G, G], [[r + c for c in range(G)] for r in range(G)])
    RmC = g.f([1, 1, G, G], [[r - c for c in range(G)] for r in range(G)])
    Cg = g.f([1, 1, G, G], [[c for c in range(G)] for _ in range(G)])
    A = g.nd("ReduceMax", [g.nd("Add", [g.nd("Mul", [RpC, seven]), pen])], axes=[2, 3], keepdims=1)
    Bv = g.nd("ReduceMax", [g.nd("Add", [g.nd("Mul", [RmC, seven]), pen])], axes=[2, 3], keepdims=1)
    c0 = g.nd("ReduceMax", [g.nd("Add", [g.nd("Mul", [Cg, seven]), pen])], axes=[2, 3], keepdims=1)
    half = g.f([1, 1, 1, 1], [0.5])
    c1 = g.nd("Cast", [g.nd("Less", [RpC, g.nd("Add", [A, half])])], to=F)  # r+c <= A
    c2 = g.nd("Cast", [g.nd("Less", [RmC, g.nd("Add", [Bv, half])])], to=F)  # r-c <= B
    lit = g.nd("Mul", [g.nd("Mul", [c1, c2]), realmask])                    # [1,1,30,30]
    cpar = g.f([1, 1, 1, G], [c % 2 for c in range(G)])
    c0par = g.nd("Mod", [c0, g.f([1, 1, 1, 1], [2.0])], fmod=1)
    par = g.nd("Abs", [g.nd("Sub", [cpar, c0par])])                         # 0->7, 1->8
    m7 = g.nd("Mul", [lit, g.nd("Sub", [g.f([1, 1, 1, 1], [1.0]), par])])
    m8 = g.nd("Mul", [lit, par])
    bgm = g.nd("Mul", [realmask, g.nd("Sub", [g.f([1, 1, 1, 1], [1.0]), lit])])  # real & unlit
    e0 = g.f([1, CHANNELS, 1, 1], [1.0 if j == 0 else 0.0 for j in range(CHANNELS)])
    e7 = g.f([1, CHANNELS, 1, 1], [1.0 if j == 7 else 0.0 for j in range(CHANNELS)])
    e8 = g.f([1, CHANNELS, 1, 1], [1.0 if j == 8 else 0.0 for j in range(CHANNELS)])
    s = g.nd("Add", [g.nd("Mul", [e0, bgm]), g.nd("Mul", [e7, m7])])
    g.nd("Add", [s, g.nd("Mul", [e8, m8])], "output")
    return _model(g)


# =========================================================================== #
# task 213 : line colours -> NxN block (period-3 compaction, both orientations) #
# =========================================================================== #
_WVEC = [0, 1, 2, 3, 4, 0, 6, 7, 8, 9]   # colour 5 -> 0 (the occluder/marker)


def _sol213(a):
    H, W = a.shape
    if H > 30 or W > 30:
        return None
    G = np.zeros((30, 30), int)
    G[:H, :W] = a
    cg = np.array(_WVEC)[G]
    rowmax = cg.max(1)
    colmax = cg.max(0)
    rb = rowmax.reshape(10, 3).max(1)
    cb = colmax.reshape(10, 3).max(1)
    Nh = int((rb > 0).sum())
    Nv = int((cb > 0).sum())
    horiz = int(rowmax.astype(bool).sum()) < int(colmax.astype(bool).sum())
    if horiz:
        N = Nh
        out = np.zeros((N, N), int)
        for i in range(N):
            out[i, :] = rb[i]
    else:
        N = Nv
        out = np.zeros((N, N), int)
        for j in range(N):
            out[:, j] = cb[j]
    if N < 1:
        return None
    return out


def build213():
    g = _G()
    wvec = g.f([1, CHANNELS, 1, 1], _WVEC)
    cg = g.nd("ReduceSum", [g.nd("Mul", ["input", wvec])], axes=[1], keepdims=1)  # [1,1,30,30]
    rowmax = g.nd("ReduceMax", [cg], axes=[3], keepdims=1)                 # [1,1,30,1]
    colmax = g.nd("ReduceMax", [cg], axes=[2], keepdims=1)                 # [1,1,1,30]
    rb = g.nd("ReduceMax", [g.nd("Reshape", [rowmax, g.i64([1, 1, 10, 3])])], axes=[3], keepdims=1)  # [1,1,10,1]
    cb = g.nd("ReduceMax", [g.nd("Reshape", [colmax, g.i64([1, 1, 10, 3])])], axes=[3], keepdims=1)  # [1,1,10,1]
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    Nh = g.nd("ReduceSum", [g.nd("Cast", [g.nd("Greater", [rb, half])], to=F)], axes=[2, 3], keepdims=1)
    Nv = g.nd("ReduceSum", [g.nd("Cast", [g.nd("Greater", [cb, half])], to=F)], axes=[2, 3], keepdims=1)
    rowsW = g.nd("ReduceSum", [g.nd("Cast", [g.nd("Greater", [rowmax, half])], to=F)], axes=[2, 3], keepdims=1)
    colsW = g.nd("ReduceSum", [g.nd("Cast", [g.nd("Greater", [colmax, half])], to=F)], axes=[2, 3], keepdims=1)
    horiz = g.nd("Cast", [g.nd("Less", [rowsW, colsW])], to=F)             # [1,1,1,1]
    rbPad = g.nd("Pad", [rb], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 20, 0])  # [1,1,30,1]
    cb_t = g.nd("Reshape", [cb, g.i64([1, 1, 1, 10])])
    cbPad = g.nd("Pad", [cb_t], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 0, 20])  # [1,1,1,30]
    jvec = g.f([1, 1, 1, G], list(range(G)))
    ivec = g.f([1, 1, G, 1], list(range(G)))
    colmask_h = g.nd("Cast", [g.nd("Less", [jvec, Nh])], to=F)             # [1,1,1,30]
    rowmask_v = g.nd("Cast", [g.nd("Less", [ivec, Nv])], to=F)             # [1,1,30,1]
    Vh = g.nd("Mul", [rbPad, colmask_h])                                   # [1,1,30,30]
    Vv = g.nd("Mul", [rowmask_v, cbPad])
    V = g.nd("Add", [g.nd("Mul", [horiz, Vh]),
                     g.nd("Mul", [g.nd("Sub", [one, horiz]), Vv])])
    cidx = g.f([1, CHANNELS, 1, 1], [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    absd = g.nd("Abs", [g.nd("Sub", [V, cidx])])
    g.nd("Relu", [g.nd("Sub", [one, absd])], "output")
    return _model(g)


# =========================================================================== #
# task 383 : framed-rectangle ray projection from border anomalies             #
# =========================================================================== #
def _sol383(a):
    H, W = a.shape
    nz = np.argwhere(a != 0)
    if len(nz) == 0:
        return None
    r0, c0 = nz.min(0)
    r1, c1 = nz.max(0)
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None
    B = a[r0, c0]
    I = a[(r0 + r1) // 2, (c0 + c1) // 2]
    if I == B or B == 0 or I == 0:
        return None
    fg = set(int(v) for v in np.unique(a)) - {0}
    if fg != {int(B), int(I)}:
        return None
    out = a.copy()
    markcols, markrows = set(), set()
    for k in range(c0, c1 + 1):
        if I in (a[r0, k], a[r0 + 1, k], a[r1, k], a[r1 - 1, k]):
            markcols.add(k)
    for k in range(r0, r1 + 1):
        if I in (a[k, c0], a[k, c0 + 1], a[k, c1], a[k, c1 - 1]):
            markrows.add(k)
    # restore border anomalies (I -> B)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            border = rr in (r0, r0 + 1, r1, r1 - 1) or cc in (c0, c0 + 1, c1, c1 - 1)
            if border and out[rr, cc] == I:
                out[rr, cc] = B
    for k in markcols:
        for rr in range(H):
            if rr < r0 or rr > r1:
                out[rr, k] = I
            elif r0 + 2 <= rr <= r1 - 2:
                out[rr, k] = B
    for k in markrows:
        for cc in range(W):
            if cc < c0 or cc > c1:
                out[k, cc] = I
            elif c0 + 2 <= cc <= c1 - 2:
                out[k, cc] = B
    return out


def _shiftax(g, t, axis, d):
    """Shift a vector tensor (30 on `axis`, 1 on the other spatial axis) by d."""
    lo, hi = max(d, 0), max(-d, 0)
    if axis == 2:
        p = g.nd("Pad", [t], mode="constant", value=0.0, pads=[0, 0, lo, 0, 0, 0, hi, 0])
        return g.nd("Slice", [p, g.i64([hi]), g.i64([hi + G]), g.i64([2])])
    p = g.nd("Pad", [t], mode="constant", value=0.0, pads=[0, 0, 0, lo, 0, 0, 0, hi])
    return g.nd("Slice", [p, g.i64([hi]), g.i64([hi + G]), g.i64([3])])


def build383():
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0])
    gridmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)          # [1,1,30,30]
    bg = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])
    content = g.nd("Sub", [gridmask, bg])                                  # nonzero indicator
    rowhas = g.nd("ReduceMax", [content], axes=[3], keepdims=1)            # [1,1,30,1]
    colhas = g.nd("ReduceMax", [content], axes=[2], keepdims=1)            # [1,1,1,30]
    inrect = g.nd("Mul", [rowhas, colhas])
    introw = g.nd("Min", [_shiftax(g, rowhas, 2, d) for d in (-2, -1, 0, 1, 2)])
    intcol = g.nd("Min", [_shiftax(g, colhas, 3, d) for d in (-2, -1, 0, 1, 2)])
    interior = g.nd("Mul", [introw, intcol])
    border = g.nd("Mul", [inrect, g.nd("Sub", [one, interior])])
    exterior = g.nd("Mul", [gridmask, g.nd("Sub", [one, inrect])])
    tbborder = g.nd("Mul", [g.nd("Mul", [rowhas, g.nd("Sub", [one, introw])]), colhas])
    lrborder = g.nd("Mul", [g.nd("Mul", [colhas, g.nd("Sub", [one, intcol])]), rowhas])
    # colours
    haschan = g.nd("ReduceMax", ["input"], axes=[2, 3], keepdims=1)        # [1,10,1,1]
    mask01 = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    fg = g.nd("Mul", [haschan, mask01])
    rowabove = _shiftax(g, rowhas, 2, 1)                                   # rowhas[r-1]
    colleft = _shiftax(g, colhas, 3, 1)
    firstrow = g.nd("Mul", [rowhas, g.nd("Sub", [one, rowabove])])
    firstcol = g.nd("Mul", [colhas, g.nd("Sub", [one, colleft])])
    corner = g.nd("Mul", [firstrow, firstcol])                            # [1,1,30,30] single cell
    Bonehot = g.nd("ReduceSum", [g.nd("Mul", ["input", corner])], axes=[2, 3], keepdims=1)
    Ionehot = g.nd("Sub", [fg, Bonehot])
    Imask = g.nd("ReduceSum", [g.nd("Mul", ["input", Ionehot])], axes=[1], keepdims=1)
    markcol = g.nd("ReduceMax", [g.nd("Mul", [Imask, tbborder])], axes=[2], keepdims=1)  # [1,1,1,30]
    markrow = g.nd("ReduceMax", [g.nd("Mul", [Imask, lrborder])], axes=[3], keepdims=1)  # [1,1,30,1]
    marked = g.nd("Max", [markcol, markrow])                              # [1,1,30,30]
    notm = g.nd("Sub", [one, marked])
    Bregion = g.nd("Add", [border, g.nd("Mul", [interior, marked])])
    Iregion = g.nd("Add", [g.nd("Mul", [interior, notm]), g.nd("Mul", [exterior, marked])])
    bgregion = g.nd("Mul", [exterior, notm])
    e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    s = g.nd("Add", [g.nd("Mul", [Bonehot, Bregion]), g.nd("Mul", [Ionehot, Iregion])])
    g.nd("Add", [s, g.nd("Mul", [e0, bgregion])], "output")
    return _model(g)


# =========================================================================== #
# dispatch                                                                     #
# =========================================================================== #
def candidates(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []

    out = []

    # ---- task 305 ----
    def ok305(plist):
        for a, b in plist:
            if a.shape != b.shape:
                return False
            p = _sol305(a)
            if p is None or not np.array_equal(p, b):
                return False
        return True

    if ok305(det) and ok305(allp):
        try:
            m = build305()
            onnx.checker.check_model(m, full_check=True)
            out.append(("crk6_3_cyclic", m))
        except Exception:
            pass

    # ---- task 033 ----
    def ok033(plist):
        for a, b in plist:
            if a.shape != b.shape:
                return False
            p = _sol033(a)
            if p is None or not np.array_equal(p, b):
                return False
        return True

    if ok033(det) and ok033(allp):
        try:
            m = build033()
            onnx.checker.check_model(m, full_check=True)
            out.append(("crk6_3_cellunion", m))
        except Exception:
            pass

    # ---- task 348 ----
    def ok348(plist):
        for a, b in plist:
            if a.shape != b.shape:
                return False
            p = _sol348(a)
            if p is None or not np.array_equal(p, b):
                return False
        return True

    if ok348(det) and ok348(allp):
        try:
            m = build348()
            onnx.checker.check_model(m, full_check=True)
            out.append(("crk6_3_cone", m))
        except Exception:
            pass

    # ---- task 213 ----
    def ok213(plist):
        for a, b in plist:
            p = _sol213(a)
            if p is None or p.shape != b.shape or not np.array_equal(p, b):
                return False
        return True

    if ok213(det) and ok213(allp):
        try:
            m = build213()
            onnx.checker.check_model(m, full_check=True)
            out.append(("crk6_3_linecolors", m))
        except Exception:
            pass

    # ---- task 383 ----
    def ok383(plist):
        for a, b in plist:
            if a.shape != b.shape:
                return False
            p = _sol383(a)
            if p is None or not np.array_equal(p, b):
                return False
        return True

    if ok383(det) and ok383(allp):
        try:
            m = build383()
            onnx.checker.check_model(m, full_check=True)
            out.append(("crk6_3_rayframe", m))
        except Exception:
            pass

    return out
