"""family_crk3_0 -- crack module for slice IDX=0 of the unsolved NeuroGolf tasks.

Each detected task gets its own structural detector (validated EXACTLY against the
provided train/test/arc-gen pairs in numpy) plus a static opset-10 ONNX builder.
All intermediates are static-shape; data-dependent geometry uses computed index
grids + ReduceMin/Max/Mod/Abs/Less + MatMul shift matrices, never dynamic
Resize/Pad.
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
BIG = 1000.0


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


def _slc(g, src, lo, hi, axis):
    return g.nd("Slice", [src, g.i64([lo]), g.i64([hi]), g.i64([axis])])


def _onehot(a):
    o = np.zeros((CHANNELS,) + a.shape, np.float64)
    for c in range(CHANNELS):
        o[c] = (a == c)
    return o


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


# =========================================================================== #
# TASK 30 -- align every coloured object's TOP row to the anchor colour's top  #
#            (columns unchanged); anchor colour is fixed per task (=1).        #
# =========================================================================== #
def _t30_ref(a, A):
    h, w = a.shape
    if not (a == A).any():
        return None
    atop = int(np.where((a == A).any(axis=1))[0].min())
    o = np.zeros_like(a)
    for c in range(1, 10):
        if not (a == c).any():
            continue
        ct = int(np.where((a == c).any(axis=1))[0].min())
        sh = atop - ct
        ys, xs = np.where(a == c)
        for y, x in zip(ys, xs):
            ny = y + sh
            if 0 <= ny < h:
                o[ny, x] = c
    return o


def _t30_detect(prs):
    if any(a.shape != b.shape for a, b in prs):
        return None
    # anchor must be a single fixed colour consistent across every pair
    for A in range(1, 10):
        ok = True
        moved = False
        for a, b in prs:
            m = _t30_ref(a, A)
            if m is None or not np.array_equal(m, b):
                ok = False
                break
            if not np.array_equal(a, b):
                moved = True
        if ok and moved:
            return A
    return None


def _t30_build(A):
    g = _G()
    colors = _slc(g, "input", 1, 10, 1)                      # [1,9,30,30]
    Ih = g.f([1, 1, G, 1], list(range(G)))                   # row index
    Iw = g.f([1, 1, 1, G], list(range(G)))                   # col index (used as k)
    big = g.f([1, 1, 1, 1], [BIG])
    half = g.f([1, 1, 1, 1], [0.5])
    # top row per colour
    pres = g.nd("ReduceMax", [colors], axes=[3], keepdims=1)  # [1,9,30,1]
    idx = g.nd("Add", [big, g.nd("Mul", [pres, g.nd("Sub", [Ih, big])])])
    t = g.nd("ReduceMin", [idx], axes=[2], keepdims=1)        # [1,9,1,1]
    t1 = _slc(g, t, A - 1, A, 1)                              # anchor top
    shift = g.nd("Sub", [t1, t])                             # [1,9,1,1]
    # shift matrix S[c,i,k]=1 iff i-k==shift_c
    IminusK = g.nd("Sub", [Ih, Iw])                          # [1,1,30,30]
    diff = g.nd("Sub", [IminusK, shift])                    # [1,9,30,30]
    S = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff]), half])], to=F)
    shifted = g.nd("MatMul", [S, colors])                    # [1,9,30,30]
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    csum = g.nd("ReduceSum", [shifted], axes=[1], keepdims=1)
    bg = g.nd("Sub", [realmask, csum])
    g.nd("Concat", [bg, shifted], "output", axis=1)
    return _model(g)


# =========================================================================== #
# TASK 226 -- grid of 5-lines; fill the TOP-LEFT cell with 1, the CENTER cell  #
#             with 2, the BOTTOM-RIGHT cell with 3 (their background 0-cells).  #
# =========================================================================== #
def _t226_bands(is_div, n):
    res = []
    i = 0
    while i < n:
        if not is_div[i]:
            j = i
            while j < n and not is_div[j]:
                j += 1
            res.append((i, j))
            i = j
        else:
            i += 1
    return res


def _t226_ref(a):
    h, w = a.shape
    o = a.copy()
    fr = np.array([(a[r] == 5).all() for r in range(h)])
    fc = np.array([(a[:, c] == 5).all() for c in range(w)])
    rb = _t226_bands(fr, h)
    cb = _t226_bands(fc, w)
    if len(rb) < 2 or len(cb) < 2:
        return None

    def fill(rband, cband, col):
        (r0, r1), (c0, c1) = rband, cband
        sub = o[r0:r1, c0:c1]
        sub[sub == 0] = col

    fill(rb[0], cb[0], 1)
    fill(rb[(len(rb) - 1) // 2], cb[(len(cb) - 1) // 2], 2)
    fill(rb[-1], cb[-1], 3)
    return o


def _t226_detect(prs):
    if any(a.shape != b.shape for a, b in prs):
        return False
    moved = False
    for a, b in prs:
        if set(np.unique(a).tolist()) - {0, 5}:
            return False
        m = _t226_ref(a)
        if m is None or not np.array_equal(m, b):
            return False
        if not np.array_equal(a, b):
            moved = True
    return moved


def _t226_build():
    g = _G()
    five = _slc(g, "input", 5, 6, 1)                         # [1,1,30,30]
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    bg = _slc(g, "input", 0, 1, 1)                           # [1,1,30,30] real 0-cells
    half = g.f([1, 1, 1, 1], [0.5])
    Lr = g.f([1, 1, G, G], [[1.0 if k < r else 0.0 for k in range(G)] for r in range(G)])
    Uc = g.f([1, 1, G, G], [[1.0 if k < c else 0.0 for c in range(G)] for k in range(G)])

    def eqf(x, v):
        return g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [x, v])]), half])], to=F)

    def gtf(x):
        return g.nd("Cast", [g.nd("Greater", [x, half])], to=F)

    # full-5 rows / cols
    rs5 = g.nd("ReduceSum", [five], axes=[3], keepdims=1)     # [1,1,30,1]
    rsR = g.nd("ReduceSum", [realmask], axes=[3], keepdims=1)
    fr = g.nd("Mul", [eqf(rs5, rsR), gtf(rsR)])              # [1,1,30,1]
    cs5 = g.nd("ReduceSum", [five], axes=[2], keepdims=1)     # [1,1,1,30]
    csR = g.nd("ReduceSum", [realmask], axes=[2], keepdims=1)
    fc = g.nd("Mul", [eqf(cs5, csR), gtf(csR)])             # [1,1,1,30]

    # band indices (cumulative dividers before)
    bandr = g.nd("MatMul", [Lr, fr])                         # [1,1,30,1]
    bandc = g.nd("MatMul", [fc, Uc])                        # [1,1,1,30]
    nfr = g.nd("ReduceSum", [fr], axes=[2], keepdims=1)       # [1,1,1,1]
    nfc = g.nd("ReduceSum", [fc], axes=[3], keepdims=1)
    crb = g.nd("Floor", [g.nd("Mul", [nfr, half])])
    ccb = g.nd("Floor", [g.nd("Mul", [nfc, half])])
    zero = g.f([1, 1, 1, 1], [0.0])

    notfr = g.nd("Sub", [g.f([1, 1, 1, 1], [1.0]), fr])
    notfc = g.nd("Sub", [g.f([1, 1, 1, 1], [1.0]), fc])
    rowTL = g.nd("Mul", [eqf(bandr, zero), notfr])
    rowC = g.nd("Mul", [eqf(bandr, crb), notfr])
    rowBR = g.nd("Mul", [eqf(bandr, nfr), notfr])
    colTL = g.nd("Mul", [eqf(bandc, zero), notfc])
    colC = g.nd("Mul", [eqf(bandc, ccb), notfc])
    colBR = g.nd("Mul", [eqf(bandc, nfc), notfc])

    tlfill = g.nd("Mul", [g.nd("Mul", [rowTL, colTL]), bg])
    cfill = g.nd("Mul", [g.nd("Mul", [rowC, colC]), bg])
    brfill = g.nd("Mul", [g.nd("Mul", [rowBR, colBR]), bg])

    def vec(col):
        v = [0.0] * CHANNELS
        v[col] = 1.0
        v[0] = -1.0
        return g.f([1, CHANNELS, 1, 1], v)

    add = g.nd("Add", [g.nd("Mul", [tlfill, vec(1)]),
                       g.nd("Add", [g.nd("Mul", [cfill, vec(2)]),
                                    g.nd("Mul", [brfill, vec(3)])])])
    g.nd("Add", ["input", add], "output")
    return _model(g)


# =========================================================================== #
# TASK 284 -- connect two coloured dots (same row/col) with a line through a    #
#             hollow 4x5 box centred at the midpoint; each half coloured by the #
#             nearer dot.                                                       #
# =========================================================================== #
def _t284_vert(a, r1, r2, cc, C1, C2):
    h, w = a.shape
    o = np.zeros_like(a)
    mf = (r1 + r2) // 2
    for r in range(r1, mf):
        o[r, cc] = C1
    for r in range(mf + 2, r2 + 1):
        o[r, cc] = C2
    for c in range(cc - 2, cc + 3):
        if 0 <= c < w:
            o[mf - 1, c] = C1
            o[mf + 2, c] = C2
    for c2 in (cc - 2, cc + 2):
        if 0 <= c2 < w:
            o[mf, c2] = C1
            o[mf + 1, c2] = C2
    return o


def _t284_ref(a):
    nz = np.argwhere(a != 0)
    if len(nz) != 2:
        return None
    (y1, x1), (y2, x2) = nz
    c1, c2 = int(a[y1, x1]), int(a[y2, x2])
    if x1 == x2:
        if y1 < y2:
            return _t284_vert(a, y1, y2, x1, c1, c2)
        return _t284_vert(a, y2, y1, x1, c2, c1)
    if y1 == y2:
        m = _t284_ref(a.T)
        return None if m is None else m.T
    return None


def _t284_detect(prs):
    if any(a.shape != b.shape for a, b in prs):
        return False
    moved = False
    for a, b in prs:
        m = _t284_ref(a)
        if m is None or not np.array_equal(m, b):
            return False
        if not np.array_equal(a, b):
            moved = True
    return moved


def _t284_grids(g):
    ROW = g.f([1, 1, G, 1], list(range(G)))
    COL = g.f([1, 1, 1, G], list(range(G)))
    return ROW, COL


def _t284_solve_vert(g, src, ROW, COL, half, big, e0):
    """Build the vertical-orientation output [1,10,30,30] for source `src`."""
    def geq(x, v):
        return g.nd("Cast", [g.nd("Greater", [x, g.nd("Sub", [v, half])])], to=F)

    def leq(x, v):
        return g.nd("Cast", [g.nd("Less", [x, g.nd("Add", [v, half])])], to=F)

    def eqf(x, v):
        return g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [x, v])]), half])], to=F)

    cl = g.nd("Slice", [src, g.i64([1]), g.i64([10]), g.i64([1])])  # [1,9,30,30]
    pres = g.nd("ReduceSum", [cl], axes=[1], keepdims=1)             # [1,1,30,30]
    rowhas = g.nd("ReduceMax", [pres], axes=[3], keepdims=1)         # [1,1,30,1]
    colhas = g.nd("ReduceMax", [pres], axes=[2], keepdims=1)         # [1,1,1,30]
    cc = g.nd("ReduceMax", [g.nd("Mul", [colhas, COL])], axes=[3], keepdims=1)
    r2 = g.nd("ReduceMax", [g.nd("Mul", [rowhas, ROW])], axes=[2], keepdims=1)
    r1 = g.nd("Sub", [big, g.nd("ReduceMax",
              [g.nd("Mul", [rowhas, g.nd("Sub", [big, ROW])])], axes=[2], keepdims=1)])
    # dot colour vectors
    topc = g.nd("Mul", [eqf(ROW, r1), eqf(COL, cc)])                 # [1,1,30,30]
    botc = g.nd("Mul", [eqf(ROW, r2), eqf(COL, cc)])
    C1 = g.nd("ReduceSum", [g.nd("Mul", [src, topc])], axes=[2, 3], keepdims=1)
    C2 = g.nd("ReduceSum", [g.nd("Mul", [src, botc])], axes=[2, 3], keepdims=1)
    mf = g.nd("Floor", [g.nd("Mul", [g.nd("Add", [r1, r2]), half])])
    one = g.nd("Add", [half, half])
    two = g.nd("Add", [one, one])
    m1 = g.nd("Sub", [mf, one])
    mp1 = g.nd("Add", [mf, one])
    m2 = g.nd("Add", [mf, two])
    ccm = g.nd("Sub", [cc, two])
    ccp = g.nd("Add", [cc, two])

    def uni(*parts):
        s = parts[0]
        for p in parts[1:]:
            s = g.nd("Add", [s, p])
        return g.nd("Cast", [g.nd("Greater", [s, half])], to=F)

    colcc = eqf(COL, cc)
    colspan = g.nd("Mul", [geq(COL, ccm), leq(COL, ccp)])            # [1,1,1,30]
    colside = g.nd("Cast", [g.nd("Greater",
              [g.nd("Add", [eqf(COL, ccm), eqf(COL, ccp)]), half])], to=F)
    line1 = g.nd("Mul", [colcc, g.nd("Mul", [geq(ROW, r1), leq(ROW, m1)])])
    topedge = g.nd("Mul", [eqf(ROW, m1), colspan])
    topinner = g.nd("Mul", [eqf(ROW, mf), colside])
    C1mask = uni(line1, topedge, topinner)                          # [1,1,30,30]
    line2 = g.nd("Mul", [colcc, g.nd("Mul", [geq(ROW, m2), leq(ROW, r2)])])
    botedge = g.nd("Mul", [eqf(ROW, m2), colspan])
    botinner = g.nd("Mul", [eqf(ROW, mp1), colside])
    C2mask = uni(line2, botedge, botinner)

    colored = g.nd("Add", [g.nd("Mul", [C1mask, C1]), g.nd("Mul", [C2mask, C2])])
    realmask = g.nd("ReduceSum", [src], axes=[1], keepdims=1)
    colored = g.nd("Mul", [colored, realmask])
    csum = g.nd("ReduceSum", [colored], axes=[1], keepdims=1)
    bgo = g.nd("Sub", [realmask, csum])
    return g.nd("Add", [colored, g.nd("Mul", [bgo, e0])])


def _t284_build():
    g = _G()
    ROW, COL = _t284_grids(g)
    half = g.f([1, 1, 1, 1], [0.5])
    big = g.f([1, 1, 1, 1], [BIG])
    e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    vert = _t284_solve_vert(g, "input", ROW, COL, half, big, e0)
    tin = g.nd("Transpose", ["input"], perm=[0, 1, 3, 2])
    ht = _t284_solve_vert(g, tin, ROW, COL, half, big, e0)
    horiz = g.nd("Transpose", [ht], perm=[0, 1, 3, 2])
    # orientation: vertical iff the two dots share a column
    cl = g.nd("Slice", ["input", g.i64([1]), g.i64([10]), g.i64([1])])
    pres = g.nd("ReduceSum", [cl], axes=[1], keepdims=1)
    colhas = g.nd("ReduceMax", [pres], axes=[2], keepdims=1)
    cmax = g.nd("ReduceMax", [g.nd("Mul", [colhas, COL])], axes=[3], keepdims=1)
    cmin = g.nd("Sub", [big, g.nd("ReduceMax",
               [g.nd("Mul", [colhas, g.nd("Sub", [big, COL])])], axes=[3], keepdims=1)])
    vf = g.nd("Cast", [g.nd("Less", [g.nd("Sub", [cmax, cmin]), half])], to=F)  # [1,1,1,1]
    one = g.f([1, 1, 1, 1], [1.0])
    nvf = g.nd("Sub", [one, vf])
    g.nd("Add", [g.nd("Mul", [vert, vf]), g.nd("Mul", [horiz, nvf])], "output")
    return _model(g)


# =========================================================================== #
# TASK 131 -- slide a coloured blob (3) toward a full straight wall-line (2)    #
#             until adjacent, then draw a full 8-line just beyond its far edge. #
# =========================================================================== #
def _t131_detect(prs):
    if any(a.shape != b.shape for a, b in prs):
        return False
    moved = False
    for a, b in prs:
        ina = set(np.unique(a).tolist()) - {0}
        inb = set(np.unique(b).tolist()) - {0}
        line = list(inb - ina)
        if line != [8]:
            return False
        h, w = a.shape
        wall = None
        wo = None
        for c in ina:
            ys, xs = np.where(a == c)
            if xs.min() == xs.max() and (a[:, xs[0]] == c).all():
                wall = c
                wo = ('col', int(xs[0]))
            elif ys.min() == ys.max() and (a[ys[0]] == c).all():
                wall = c
                wo = ('row', int(ys[0]))
        if wall != 2:
            return False
        shape = list(ina - {wall})
        if shape != [3]:
            return False
        m = _t131_ref(a, wo)
        if m is None or not np.array_equal(m, b):
            return False
        if not np.array_equal(a, b):
            moved = True
    return moved


def _t131_ref(a, wo):
    h, w = a.shape
    o = np.zeros_like(a)
    o[a == 2] = 2
    ys, xs = np.where(a == 3)
    if len(ys) == 0:
        return None
    sr0, sr1, sc0, sc1 = ys.min(), ys.max(), xs.min(), xs.max()
    kind, pos = wo
    if kind == 'col':
        cw = pos
        if sc0 > cw:
            shift = (cw + 1) - sc0
            far = sc1 + shift + 1
        else:
            shift = (cw - 1) - sc1
            far = sc0 + shift - 1
        for y, x in zip(ys, xs):
            if 0 <= x + shift < w:
                o[y, x + shift] = 3
        if 0 <= far < w:
            o[:, far] = 8
    else:
        rw = pos
        if sr0 > rw:
            shift = (rw + 1) - sr0
            far = sr1 + shift + 1
        else:
            shift = (rw - 1) - sr1
            far = sr0 + shift - 1
        for y, x in zip(ys, xs):
            if 0 <= y + shift < h:
                o[y + shift, x] = 3
        if 0 <= far < h:
            o[far, :] = 8
    return o


def _t131_solve_col(g, src, ROW, COL, half, big):
    """Wall is a vertical full column; shape (3) slides horizontally toward it."""
    def gtf(x):
        return g.nd("Cast", [g.nd("Greater", [x, half])], to=F)

    def eqf(x, v):
        return g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [x, v])]), half])], to=F)

    pS = g.nd("Slice", [src, g.i64([3]), g.i64([4]), g.i64([1])])    # shape plane
    pW = g.nd("Slice", [src, g.i64([2]), g.i64([3]), g.i64([1])])    # wall plane
    realmask = g.nd("ReduceSum", [src], axes=[1], keepdims=1)
    colW = g.nd("ReduceMax", [pW], axes=[2], keepdims=1)             # [1,1,1,30]
    cw = g.nd("ReduceMax", [g.nd("Mul", [colW, COL])], axes=[3], keepdims=1)
    colS = g.nd("ReduceMax", [pS], axes=[2], keepdims=1)
    sc1 = g.nd("ReduceMax", [g.nd("Mul", [colS, COL])], axes=[3], keepdims=1)
    sc0 = g.nd("Sub", [big, g.nd("ReduceMax",
              [g.nd("Mul", [colS, g.nd("Sub", [big, COL])])], axes=[3], keepdims=1)])
    one = g.nd("Add", [half, half])
    right = gtf(g.nd("Sub", [sc0, cw]))                              # sc0>cw
    left = gtf(g.nd("Sub", [cw, sc1]))                              # cw>sc1
    sh_r = g.nd("Sub", [g.nd("Add", [cw, one]), sc0])               # (cw+1)-sc0
    sh_l = g.nd("Sub", [g.nd("Sub", [cw, one]), sc1])               # (cw-1)-sc1
    shift = g.nd("Add", [g.nd("Mul", [right, sh_r]), g.nd("Mul", [left, sh_l])])
    far_r = g.nd("Add", [g.nd("Add", [sc1, shift]), one])
    far_l = g.nd("Sub", [g.nd("Add", [sc0, shift]), one])
    far = g.nd("Add", [g.nd("Mul", [right, far_r]), g.nd("Mul", [left, far_l])])
    # horizontal shift of shape plane: Scol[k,j]=1 iff j-k==shift
    Kc = ROW  # [1,1,30,1] as k index
    Jc = COL  # [1,1,1,30] as j index
    Scol = g.nd("Cast", [g.nd("Less",
           [g.nd("Abs", [g.nd("Sub", [g.nd("Sub", [Jc, Kc]), shift])]), half])], to=F)
    shifted = g.nd("Mul", [g.nd("MatMul", [pS, Scol]), realmask])    # [1,1,30,30]
    line = g.nd("Mul", [eqf(COL, far), realmask])                   # [1,1,30,30]
    # assemble one-hot
    def ev(ch):
        v = [0.0] * CHANNELS
        v[ch] = 1.0
        return g.f([1, CHANNELS, 1, 1], v)

    colored = g.nd("Add", [g.nd("Mul", [shifted, ev(3)]),
                           g.nd("Add", [g.nd("Mul", [pW, ev(2)]),
                                        g.nd("Mul", [line, ev(8)])])])
    csum = g.nd("Add", [shifted, g.nd("Add", [pW, line])])
    bg = g.nd("Sub", [realmask, csum])
    return g.nd("Add", [colored, g.nd("Mul", [bg, ev(0)])])


def _t131_build():
    g = _G()
    ROW = g.f([1, 1, G, 1], list(range(G)))
    COL = g.f([1, 1, 1, G], list(range(G)))
    half = g.f([1, 1, 1, 1], [0.5])
    big = g.f([1, 1, 1, 1], [BIG])
    col_out = _t131_solve_col(g, "input", ROW, COL, half, big)
    tin = g.nd("Transpose", ["input"], perm=[0, 1, 3, 2])
    rt = _t131_solve_col(g, tin, ROW, COL, half, big)
    row_out = g.nd("Transpose", [rt], perm=[0, 1, 3, 2])
    # vertical-wall flag: wall (2) occupies a single column
    pW = g.nd("Slice", ["input", g.i64([2]), g.i64([3]), g.i64([1])])
    colW = g.nd("ReduceMax", [pW], axes=[2], keepdims=1)
    cmax = g.nd("ReduceMax", [g.nd("Mul", [colW, COL])], axes=[3], keepdims=1)
    cmin = g.nd("Sub", [big, g.nd("ReduceMax",
               [g.nd("Mul", [colW, g.nd("Sub", [big, COL])])], axes=[3], keepdims=1)])
    cv = g.nd("Cast", [g.nd("Less", [g.nd("Sub", [cmax, cmin]), half])], to=F)
    one = g.f([1, 1, 1, 1], [1.0])
    g.nd("Add", [g.nd("Mul", [col_out, cv]),
                 g.nd("Mul", [row_out, g.nd("Sub", [one, cv])])], "output")
    return _model(g)


# =========================================================================== #
# dispatch                                                                     #
# =========================================================================== #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    A = None
    try:
        A = _t30_detect(prs)
    except Exception:
        A = None
    if A is not None:
        try:
            out.append((f"t30_align_A{A}", _check(_t30_build(A))))
        except Exception:
            pass

    try:
        if _t226_detect(prs):
            out.append(("t226_bandfill", _check(_t226_build())))
    except Exception:
        pass

    try:
        if _t284_detect(prs):
            out.append(("t284_dotbox", _check(_t284_build())))
    except Exception:
        pass

    try:
        if _t131_detect(prs):
            out.append(("t131_slidewall", _check(_t131_build())))
    except Exception:
        pass

    return out
