"""family_crk3_3 -- crack module for slice IDX=3 of the unsolved NeuroGolf tasks.

Each detected task gets its own structural detector (validated EXACTLY against the
provided train/test pairs in numpy) plus a static opset-10 ONNX builder.  All
intermediates are static-shape; data-dependent geometry uses computed index grids
+ ReduceMax/Min/Mod/Abs/Less, never dynamic Resize/Pad.
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


def _pairs(ex):
    out = []
    for k in ("train", "test"):
        for p in ex.get(k, []):
            out.append((np.array(p["input"]), np.array(p["output"])))
    return out


def _slc(g, src, lo, hi, axis):
    s = g.i64([lo]); e = g.i64([hi]); a = g.i64([axis])
    return g.nd("Slice", [src, s, e, a])


# =========================================================================== #
# TASK 389 -- keep the color-5 shape, recolor it to the OTHER color, erase rest
# =========================================================================== #
def _t389_rule(i):
    u = set(np.unique(i).tolist())
    others = [c for c in u if c not in (0, 5)]
    if 5 not in u or len(others) != 1:
        return None
    X = others[0]
    o = np.zeros_like(i)
    o[i == 5] = X
    return o


def _t389_detect(prs):
    okany = False
    for i, o in prs:
        r = _t389_rule(i)
        if r is None or not np.array_equal(r, o):
            return False
        okany = True
    return okany


def _t389_build():
    g = _G()
    mask5 = _slc(g, "input", 5, 6, 1)                      # [1,1,30,30]
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)  # [1,1,30,30]
    pres = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)   # [1,10,1,1]
    gate = g.nd("Clip", [pres], min=0.0, max=1.0)                  # 1 where color present
    keepX = g.f([1, 10, 1, 1], [0, 1, 1, 1, 1, 0, 1, 1, 1, 1])     # drop ch0 & ch5
    xgate = g.nd("Mul", [gate, keepX])                            # 1 only at X channel
    colorpart = g.nd("Mul", [mask5, xgate])                       # [1,10,30,30]
    bg0 = g.nd("Sub", [realmask, mask5])                          # real non-5 cells
    e0 = g.f([1, 10, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    bgpart = g.nd("Mul", [bg0, e0])
    g.nd("Add", [colorpart, bgpart], "output")
    return _model(g)


# =========================================================================== #
# TASK 180 -- 8x8 split into four 4x4 quadrants; overlay with priority         #
#             TR > BL > BR > TL (first colored quadrant wins).  out = 4x4       #
# =========================================================================== #
def _t180_rule(a):
    if a.shape != (8, 8):
        return None
    TL = a[0:4, 0:4]; TR = a[0:4, 4:8]; BL = a[4:8, 0:4]; BR = a[4:8, 4:8]
    o = np.zeros((4, 4), int)
    for i in range(4):
        for j in range(4):
            if TR[i, j]:
                o[i, j] = TR[i, j]
            elif BL[i, j]:
                o[i, j] = BL[i, j]
            elif BR[i, j]:
                o[i, j] = BR[i, j]
            else:
                o[i, j] = TL[i, j]
    return o


def _t180_detect(prs):
    okany = False
    for i, o in prs:
        r = _t180_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
        okany = True
    return okany


def _quad(g, r0, r1, c0, c1):
    t = _slc(g, "input", r0, r1, 2)
    return _slc(g, t, c0, c1, 3)


def _t180_build():
    g = _G()
    TL = _quad(g, 0, 4, 0, 4)
    TR = _quad(g, 0, 4, 4, 8)
    BL = _quad(g, 4, 8, 0, 4)
    BR = _quad(g, 4, 8, 4, 8)
    one = g.f([1, 1, 1, 1], [1.0])

    def nz(Q):
        tot = g.nd("ReduceSum", [Q], axes=[1], keepdims=1)   # =1 on every real cell
        bg = _slc(g, Q, 0, 1, 1)                              # channel0
        return g.nd("Sub", [tot, bg])                        # 1 if colored
    nzTR = nz(TR); nzBL = nz(BL); nzBR = nz(BR)
    iTR = g.nd("Sub", [one, nzTR])
    iBL = g.nd("Sub", [one, nzBL])
    iBR = g.nd("Sub", [one, nzBR])
    selTR = nzTR
    selBL = g.nd("Mul", [iTR, nzBL])
    t1 = g.nd("Mul", [iTR, iBL])
    selBR = g.nd("Mul", [t1, nzBR])
    selTL = g.nd("Mul", [t1, iBR])
    p1 = g.nd("Mul", [TR, selTR])
    p2 = g.nd("Mul", [BL, selBL])
    p3 = g.nd("Mul", [BR, selBR])
    p4 = g.nd("Mul", [TL, selTL])
    s = g.nd("Add", [p1, p2])
    s = g.nd("Add", [s, p3])
    res = g.nd("Add", [s, p4])                                # [1,10,4,4]
    g.nd("Pad", [res], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, G - 4, G - 4])
    return _model(g)


# =========================================================================== #
# TASK 334 -- single non-bg color in {1,2,3}; emit a fixed 3x3 glyph (color 5) #
# =========================================================================== #
_T334_GLYPHS = {
    2: [[5, 5, 5], [0, 5, 0], [0, 5, 0]],
    1: [[0, 5, 0], [5, 5, 5], [0, 5, 0]],
    3: [[0, 0, 5], [0, 0, 5], [5, 5, 5]],
}


def _t334_rule(a):
    cols = [c for c in np.unique(a) if c != 0]
    if len(cols) != 1 or cols[0] not in _T334_GLYPHS:
        return None
    return np.array(_T334_GLYPHS[cols[0]], int)


def _t334_detect(prs):
    okany = False
    for i, o in prs:
        r = _t334_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
        okany = True
    return okany


def _glyph_onehot(grid):
    """grid 3x3 of {0,5} -> [1,10,3,3] one-hot const values."""
    arr = np.zeros((1, 10, 3, 3), np.float32)
    g = np.array(grid)
    arr[0, 0] = (g == 0).astype(np.float32)
    arr[0, 5] = (g == 5).astype(np.float32)
    return arr


def _t334_build():
    g = _G()
    pres = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)  # [1,10,1,1]
    gate = g.nd("Clip", [pres], min=0.0, max=1.0)
    acc = None
    for color, grid in _T334_GLYPHS.items():
        gl = g.f([1, 10, 3, 3], _glyph_onehot(grid).ravel())
        gc = _slc(g, gate, color, color + 1, 1)                  # [1,1,1,1]
        part = g.nd("Mul", [gl, gc])
        acc = part if acc is None else g.nd("Add", [acc, part])
    g.nd("Pad", [acc], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, G - 3, G - 3])
    return _model(g)


# =========================================================================== #
# TASK 11 -- 11x11 of nine 3x3 blocks (sep rows/cols=5).  Every block holds a  #
#            full color set except ONE which is missing the marker color; emit #
#            that special block upscaled (3x3 per cell) with the 5 separators. #
# =========================================================================== #
def _t11_blocks(a):
    return {(ci, cj): a[ci * 4:ci * 4 + 3, cj * 4:cj * 4 + 3] for ci in range(3) for cj in range(3)}


def _t11_rule(a):
    if a.shape != (11, 11):
        return None
    bc = {}
    pres = {}
    for c in range(1, 10):
        if c == 5:
            continue
        cnt = 0
        for ci in range(3):
            for cj in range(3):
                blk = a[ci * 4:ci * 4 + 3, cj * 4:cj * 4 + 3]
                p = int((blk == c).any())
                pres[(c, ci, cj)] = p
                cnt += p
        bc[c] = cnt
    markers = [c for c in bc if bc[c] == 8]
    if len(markers) != 1:
        return None
    m = markers[0]
    sp = [(ci, cj) for ci in range(3) for cj in range(3)
          if pres[(m, ci, cj)] == 0 and a[ci * 4:ci * 4 + 3, cj * 4:cj * 4 + 3].any()]
    if len(sp) != 1:
        return None
    ci, cj = sp[0]
    blk = a[ci * 4:ci * 4 + 3, cj * 4:cj * 4 + 3]
    o = np.zeros((11, 11), int)
    o[3, :] = 5; o[7, :] = 5; o[:, 3] = 5; o[:, 7] = 5
    for i in range(3):
        for j in range(3):
            o[i * 4:i * 4 + 3, j * 4:j * 4 + 3] = blk[i, j]
    return o


def _t11_detect(prs):
    okany = False
    for i, o in prs:
        r = _t11_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
        okany = True
    return okany


def _t11_build():
    g = _G()
    eight = g.f([1, 10, 1, 1], [8.0] * 10)
    half = g.f([1, 10, 1, 1], [0.5] * 10)
    keepmask = g.f([1, 10, 1, 1], [0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
    one = g.f([1, 1, 1, 1], [1.0])

    blks = {}
    presv = {}
    for ci in range(3):
        for cj in range(3):
            t = _slc(g, "input", ci * 4, ci * 4 + 3, 2)
            blk = _slc(g, t, cj * 4, cj * 4 + 3, 3)              # [1,10,3,3]
            blks[(ci, cj)] = blk
            cnt = g.nd("ReduceSum", [blk], axes=[2, 3], keepdims=1)  # [1,10,1,1]
            presv[(ci, cj)] = g.nd("Clip", [cnt], min=0.0, max=1.0)
    # block-count per color
    bc = None
    for k in presv:
        bc = presv[k] if bc is None else g.nd("Add", [bc, presv[k]])
    d = g.nd("Abs", [g.nd("Sub", [bc, eight])])
    lt = g.nd("Less", [d, half])
    markerchan = g.nd("Cast", [lt], to=int(F))
    markerchan = g.nd("Mul", [markerchan, keepmask])             # [1,10,1,1]
    # selected special block
    sel = None
    for k in presv:
        mp = g.nd("Mul", [presv[k], markerchan])
        mp = g.nd("ReduceSum", [mp], axes=[1], keepdims=1)       # [1,1,1,1]
        w = g.nd("Sub", [one, mp])                              # 1 if special
        part = g.nd("Mul", [blks[k], w])                       # [1,10,3,3]
        sel = part if sel is None else g.nd("Add", [sel, part])
    # upscale 3x3 -> 11x11 blocks via R @ sel @ R^T
    R = np.zeros((11, 3), np.float32)
    for j in range(3):
        for dj in range(3):
            R[j * 4 + dj, j] = 1.0
    Rc = g.f([1, 1, 11, 3], R.ravel())
    Rt = g.f([1, 1, 3, 11], R.T.ravel())
    m1 = g.nd("MatMul", [Rc, sel])                              # [1,10,11,3]
    up = g.nd("MatMul", [m1, Rt])                               # [1,10,11,11]
    # separators on channel 5
    sep = np.zeros((11, 11), np.float32)
    sep[3, :] = 1; sep[7, :] = 1; sep[:, 3] = 1; sep[:, 7] = 1
    sepc = g.f([1, 1, 11, 11], sep.ravel())
    e5 = g.f([1, 10, 1, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    seponehot = g.nd("Mul", [sepc, e5])                        # [1,10,11,11]
    comb = g.nd("Add", [up, seponehot])
    g.nd("Pad", [comb], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, G - 11, G - 11])
    return _model(g)


# =========================================================================== #
# TASKS 400 & 351 -- a 5x5 solid block of a marker color occludes a region of  #
#   a 180-symmetric grid.  Emit the 180-rotation partner of that region (=the   #
#   restored content), placed top-left.                                         #
# =========================================================================== #
def _sym_winmax(mask):
    H, W = mask.shape
    best = 0
    for i in range(H - 4):
        for j in range(W - 4):
            best = max(best, int(mask[i:i + 5, j:j + 5].sum()))
    return best


def _sym_rule(a):
    H, W = a.shape
    if H < 5 or W < 5:
        return None
    mk = []
    for c in range(1, 10):
        m = (a == c)
        if int(m.sum()) == 25 and _sym_winmax(m) == 25:
            mk.append(c)
    if len(mk) != 1:
        return None
    c = mk[0]
    ys, xs = np.where(a == c)
    r0, c0 = ys.min(), xs.min()
    out = np.zeros((5, 5), int)
    for i in range(5):
        for j in range(5):
            out[i, j] = a[(H - 1 - r0) - i, (W - 1 - c0) - j]
    return out


def _sym_detect(prs):
    okany = False
    for i, o in prs:
        r = _sym_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
        okany = True
    return okany


def _sym_build():
    g = _G()
    mask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)         # [1,1,30,30]
    count = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)     # [1,10,1,1]
    # depthwise 5x5 ones conv -> per-channel window sums
    Wk = g.f([10, 1, 5, 5], [1.0] * (10 * 25))
    conv = g.nd("Conv", ["input", Wk], kernel_shape=[5, 5], group=10,
                strides=[1, 1], pads=[0, 0, 0, 0])                    # [1,10,26,26]
    winmax = g.nd("ReduceMax", [conv], axes=[2, 3], keepdims=1)       # [1,10,1,1]
    c25 = g.f([1, 10, 1, 1], [25.0] * 10)
    halfv = g.f([1, 10, 1, 1], [0.5] * 10)
    mc = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [count, c25])]), halfv])], to=int(F))
    mw = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [winmax, c25])]), halfv])], to=int(F))
    markerchan = g.nd("Mul", [mc, mw])                               # [1,10,1,1]
    markermask = g.nd("ReduceSum", [g.nd("Mul", ["input", markerchan])], axes=[1], keepdims=1)

    rIdx = g.f([1, 1, G, 1], list(range(G)))
    cIdx = g.f([1, 1, 1, G], list(range(G)))
    one = g.f([1, 1, 1, 1], [1.0])
    big = g.f([1, 1, 1, 1], [1000.0])
    five = g.f([1, 1, 1, 1], [5.0])
    h1 = g.f([1, 1, 1, 1], [0.5])

    anyrow = g.nd("Clip", [g.nd("ReduceSum", [mask], axes=[3], keepdims=1)], min=0.0, max=1.0)
    Hm1 = g.nd("ReduceMax", [g.nd("Mul", [rIdx, anyrow])], axes=[2], keepdims=1)   # [1,1,1,1]
    anycol = g.nd("Clip", [g.nd("ReduceSum", [mask], axes=[2], keepdims=1)], min=0.0, max=1.0)
    Wm1 = g.nd("ReduceMax", [g.nd("Mul", [cIdx, anycol])], axes=[3], keepdims=1)   # [1,1,1,1]

    anyrm = g.nd("Clip", [g.nd("ReduceSum", [markermask], axes=[3], keepdims=1)], min=0.0, max=1.0)
    invrm = g.nd("Sub", [one, anyrm])
    rmasked = g.nd("Add", [g.nd("Mul", [rIdx, anyrm]), g.nd("Mul", [big, invrm])])
    r0 = g.nd("ReduceMin", [rmasked], axes=[2], keepdims=1)          # [1,1,1,1]
    anycm = g.nd("Clip", [g.nd("ReduceSum", [markermask], axes=[2], keepdims=1)], min=0.0, max=1.0)
    invcm = g.nd("Sub", [one, anycm])
    cmasked = g.nd("Add", [g.nd("Mul", [cIdx, anycm]), g.nd("Mul", [big, invcm])])
    c0 = g.nd("ReduceMin", [cmasked], axes=[3], keepdims=1)          # [1,1,1,1]

    A = g.nd("Sub", [Hm1, r0])
    B = g.nd("Sub", [Wm1, c0])
    # Mrow[i,k] = (i<5) & (i+k==A)
    sR = g.nd("Add", [rIdx, cIdx])                                   # [1,1,30,30]
    c2R = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [sR, A])]), h1])], to=int(F))
    c1R = g.nd("Cast", [g.nd("Less", [rIdx, five])], to=int(F))      # [1,1,30,1]
    Mrow = g.nd("Mul", [c1R, c2R])
    # N[l,j] = (j<5) & (l+j==B)
    c2N = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [sR, B])]), h1])], to=int(F))
    c1N = g.nd("Cast", [g.nd("Less", [cIdx, five])], to=int(F))      # [1,1,1,30]
    N = g.nd("Mul", [c1N, c2N])
    m1 = g.nd("MatMul", [Mrow, "input"])                            # [1,10,30,30]
    g.nd("MatMul", [m1, N], "output")
    return _model(g)


# =========================================================================== #
# TASK 320 -- each column has a contiguous run of color-2 ending at the grid    #
#   bottom; recolor the bottom floor(h/2) cells of each run to 8.               #
# =========================================================================== #
def _t320_rule(a):
    ic = set(np.unique(a).tolist())
    if ic - {0, 2}:
        return None
    H, W = a.shape
    m = (a == 2).astype(int)
    o = a.copy()
    for c in range(W):
        col = m[:, c]
        fh = int(col.sum()) // 2
        below = np.cumsum(col[::-1])[::-1] - col
        rec = (col == 1) & (below < fh)
        o[rec, c] = 8
    return o


def _t320_detect(prs):
    okany = False
    for i, o in prs:
        r = _t320_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
        okany = True
    return okany


def _t320_build():
    g = _G()
    mask2 = _slc(g, "input", 2, 3, 1)                                # [1,1,30,30]
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    h = g.nd("ReduceSum", [mask2], axes=[2], keepdims=1)             # [1,1,1,30]
    halfc = g.f([1, 1, 1, 1], [0.5])
    fh = g.nd("Floor", [g.nd("Mul", [h, halfc])])                   # floor(h/2)
    L = np.zeros((30, 30), np.float32)
    for r in range(30):
        for rp in range(30):
            if rp > r:
                L[r, rp] = 1.0
    Lc = g.f([1, 1, 30, 30], L.ravel())
    below = g.nd("MatMul", [Lc, mask2])                            # strictly-below count
    rec = g.nd("Mul", [mask2, g.nd("Cast", [g.nd("Less", [below, fh])], to=int(F))])
    ch0 = g.nd("Sub", [realmask, mask2])
    ch2 = g.nd("Sub", [mask2, rec])
    z = g.nd("Sub", [mask2, mask2])                               # zero [1,1,30,30]
    g.nd("Concat", [ch0, z, ch2, z, z, z, z, z, rec, z], "output", axis=1)
    return _model(g)


# =========================================================================== #
# TASK 288 -- diagonal rays (inner color) emanate up-left & up-right from the   #
#   endpoints of the structure's TOP row, into the empty rows above.            #
# =========================================================================== #
def _t288_rule(a):
    H, W = a.shape
    nz = np.argwhere(a != 0)
    if len(nz) == 0:
        return None
    botrow = nz[:, 0].max()
    cols = np.where(a[botrow] != 0)[0]
    center = (cols.min() + cols.max()) // 2
    inner = a[botrow, center]
    toprow = nz[:, 0].min()
    tcols = np.where(a[toprow] != 0)[0]
    CL, CR = tcols.min(), tcols.max()
    o = a.copy()
    k = 1
    while toprow - k >= 0:
        if CL - k >= 0 and o[toprow - k, CL - k] == 0:
            o[toprow - k, CL - k] = inner
        if CR + k < W and o[toprow - k, CR + k] == 0:
            o[toprow - k, CR + k] = inner
        k += 1
    return o


def _t288_detect(prs):
    okany = False
    for i, o in prs:
        r = _t288_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
        okany = True
    return okany


def _castlt(g, x, y):
    return g.nd("Cast", [g.nd("Less", [x, y])], to=int(F))


def _t288_build():
    g = _G()
    rIdx = g.f([1, 1, G, 1], list(range(G)))
    cIdx = g.f([1, 1, 1, G], list(range(G)))
    one = g.f([1, 1, 1, 1], [1.0])
    big = g.f([1, 1, 1, 1], [1000.0])
    half = g.f([1, 1, 1, 1], [0.5])
    e0 = g.f([1, 10, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    colored = g.nd("ReduceSum", [_slc(g, "input", 1, 10, 1)], axes=[1], keepdims=1)  # [1,1,30,30]
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)

    anyrow = g.nd("Clip", [g.nd("ReduceSum", [colored], axes=[3], keepdims=1)], min=0.0, max=1.0)
    invrow = g.nd("Sub", [one, anyrow])
    R = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [rIdx, anyrow]), g.nd("Mul", [big, invrow])])],
             axes=[2], keepdims=1)                                # [1,1,1,1]
    bot = g.nd("ReduceMax", [g.nd("Mul", [rIdx, anyrow])], axes=[2], keepdims=1)

    # width of grid
    anycolR = g.nd("Clip", [g.nd("ReduceSum", [realmask], axes=[2], keepdims=1)], min=0.0, max=1.0)
    Wm1 = g.nd("ReduceMax", [g.nd("Mul", [cIdx, anycolR])], axes=[3], keepdims=1)
    Wp = g.nd("Add", [Wm1, one])

    # row R endpoints
    rowRm = _castlt(g, g.nd("Abs", [g.nd("Sub", [rIdx, R])]), half)   # [1,1,30,1]
    rowRvec = g.nd("ReduceSum", [g.nd("Mul", [colored, rowRm])], axes=[2], keepdims=1)  # [1,1,1,30]
    anycR = g.nd("Clip", [rowRvec], min=0.0, max=1.0)
    invcR = g.nd("Sub", [one, anycR])
    CL = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [cIdx, anycR]), g.nd("Mul", [big, invcR])])],
              axes=[3], keepdims=1)
    CR = g.nd("ReduceMax", [g.nd("Mul", [cIdx, anycR])], axes=[3], keepdims=1)

    # bottom-row center -> inner color
    botm = _castlt(g, g.nd("Abs", [g.nd("Sub", [rIdx, bot])]), half)
    botvec = g.nd("ReduceSum", [g.nd("Mul", [colored, botm])], axes=[2], keepdims=1)
    anycB = g.nd("Clip", [botvec], min=0.0, max=1.0)
    invcB = g.nd("Sub", [one, anycB])
    CLb = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [cIdx, anycB]), g.nd("Mul", [big, invcB])])],
               axes=[3], keepdims=1)
    CRb = g.nd("ReduceMax", [g.nd("Mul", [cIdx, anycB])], axes=[3], keepdims=1)
    center = g.nd("Floor", [g.nd("Mul", [g.nd("Add", [CLb, CRb]), half])])
    centermask = g.nd("Mul", [_castlt(g, g.nd("Abs", [g.nd("Sub", [cIdx, center])]), half), botm])  # [1,1,30,30]
    innervec = g.nd("ReduceSum", [g.nd("Mul", ["input", centermask])], axes=[2, 3], keepdims=1)  # [1,10,1,1]

    # rays
    diff = g.nd("Sub", [rIdx, cIdx])
    sumg = g.nd("Add", [rIdx, cIdx])
    tgtL = g.nd("Sub", [R, CL])
    tgtR = g.nd("Add", [R, CR])
    leftd = _castlt(g, g.nd("Abs", [g.nd("Sub", [diff, tgtL])]), half)
    rightd = _castlt(g, g.nd("Abs", [g.nd("Sub", [sumg, tgtR])]), half)
    belowtop = _castlt(g, rIdx, R)                                  # [1,1,30,1]
    withinw = _castlt(g, cIdx, Wp)                                  # [1,1,1,30]
    raysum = g.nd("Clip", [g.nd("Add", [leftd, rightd])], min=0.0, max=1.0)
    raymask = g.nd("Mul", [g.nd("Mul", [raysum, belowtop]), withinw])  # [1,1,30,30]

    rm_bg = g.nd("Mul", [raymask, e0])
    rm_in = g.nd("Mul", [raymask, innervec])
    g.nd("Add", [g.nd("Sub", ["input", rm_bg]), rm_in], "output")
    return _model(g)


# =========================================================================== #
# TASK 379 -- 2-markers shoot perpendicular beams (color 2) to the nearest full #
#   8-line(s); a 3x3 ring of 8 is stamped at each crossing (center stays 2).    #
# =========================================================================== #
def _t379_rule(a):
    ic = set(np.unique(a).tolist())
    if ic - {0, 2, 8}:
        return None
    H, W = a.shape
    o = a.copy()
    hlines = [r for r in range(H) if (a[r] == 8).all()]
    vlines = [c for c in range(W) if (a[:, c] == 8).all()]
    markers = list(zip(*np.where(a == 2)))
    if hlines and not vlines:
        for (r0, c0) in markers:
            above = [L for L in hlines if L < r0]
            below = [L for L in hlines if L > r0]
            tg = ([max(above)] if above else []) + ([min(below)] if below else [])
            for L in tg:
                lo, hi = min(r0, L), max(r0, L)
                for r in range(lo, hi + 1):
                    o[r, c0] = 2
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = L + dr, c0 + dc
                        if 0 <= rr < H and 0 <= cc < W:
                            o[rr, cc] = 2 if (dr == 0 and dc == 0) else 8
                o[L, c0] = 2
    elif vlines and not hlines:
        for (r0, c0) in markers:
            left = [L for L in vlines if L < c0]
            right = [L for L in vlines if L > c0]
            tg = ([max(left)] if left else []) + ([min(right)] if right else [])
            for L in tg:
                lo, hi = min(c0, L), max(c0, L)
                for c in range(lo, hi + 1):
                    o[r0, c] = 2
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r0 + dr, L + dc
                        if 0 <= rr < H and 0 <= cc < W:
                            o[rr, cc] = 2 if (dr == 0 and dc == 0) else 8
                o[r0, L] = 2
    else:
        return None
    return o


def _t379_detect(prs):
    okany = False
    for i, o in prs:
        r = _t379_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
        okany = True
    return okany


def _t379_build():
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0])
    half = g.f([1, 1, 1, 1], [0.5])
    ch8 = _slc(g, "input", 8, 9, 1)
    M = _slc(g, "input", 2, 3, 1)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)

    # horizontal line mask
    Wc = g.nd("ReduceSum", [realmask], axes=[3], keepdims=1)          # [1,1,30,1]
    row8 = g.nd("ReduceSum", [ch8], axes=[3], keepdims=1)
    eqh = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [row8, Wc])]), half])], to=int(F))
    poshr = g.nd("Cast", [g.nd("Less", [half, Wc])], to=int(F))
    hlinerow = g.nd("Mul", [eqh, poshr])
    hline8 = g.nd("Mul", [hlinerow, realmask])                       # [1,1,30,30]
    # vertical line mask
    Hc = g.nd("ReduceSum", [realmask], axes=[2], keepdims=1)          # [1,1,1,30]
    col8 = g.nd("ReduceSum", [ch8], axes=[2], keepdims=1)
    eqv = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [col8, Hc])]), half])], to=int(F))
    posv = g.nd("Cast", [g.nd("Less", [half, Hc])], to=int(F))
    vlinecol = g.nd("Mul", [eqv, posv])
    vline8 = g.nd("Mul", [vlinecol, realmask])

    notlh = g.nd("Sub", [one, hline8])
    notlv = g.nd("Sub", [one, vline8])

    # shift matrices
    def diagmat(off):  # M[r,r']=1 iff r'==r+off
        A = np.zeros((30, 30), np.float32)
        for r in range(30):
            rp = r + off
            if 0 <= rp < 30:
                A[r, rp] = 1.0
        return A
    Sdown = g.f([1, 1, 30, 30], diagmat(-1).ravel())   # (Sdown@X)[r]=X[r-1]
    Sup = g.f([1, 1, 30, 30], diagmat(+1).ravel())
    Eright = g.f([1, 1, 30, 30], diagmat(+1).ravel())  # (X@Eright)[c]=X[c-1]? see below
    Eleft = g.f([1, 1, 30, 30], diagmat(-1).ravel())

    def beam(Smat, notline, vertical):
        cur = M
        for _ in range(12):
            masked = g.nd("Mul", [cur, notline])
            if vertical:
                shifted = g.nd("MatMul", [Smat, masked])
            else:
                shifted = g.nd("MatMul", [masked, Smat])
            cur = g.nd("Max", [M, shifted])
        return cur
    # horizontal lines -> vertical beams
    down = beam(Sdown, notlh, True)
    up = beam(Sup, notlh, True)
    # vertical lines -> horizontal beams; shift_right: X@E with E[c',c]=1 iff c==c'+1
    right = beam(Eright, notlv, False)   # (X@Eright)[r,c]=sum X[r,c']E[c',c]; E[c',c]=1 iff c==c'+1 -> X[r,c-1]
    left = beam(Eleft, notlv, False)

    dh = g.nd("Mul", [g.nd("ReduceMax", [g.nd("Mul", [down, hline8])], axes=[2], keepdims=1), one])
    down_k = g.nd("Mul", [down, dh])
    uh = g.nd("ReduceMax", [g.nd("Mul", [up, hline8])], axes=[2], keepdims=1)
    up_k = g.nd("Mul", [up, uh])
    trail_h = g.nd("Max", [down_k, up_k])
    rh = g.nd("ReduceMax", [g.nd("Mul", [right, vline8])], axes=[3], keepdims=1)
    right_k = g.nd("Mul", [right, rh])
    lh = g.nd("ReduceMax", [g.nd("Mul", [left, vline8])], axes=[3], keepdims=1)
    left_k = g.nd("Mul", [left, lh])
    trail_v = g.nd("Max", [left_k, right_k])
    trail = g.nd("Max", [trail_h, trail_v])

    crossing = g.nd("Max", [g.nd("Mul", [trail_h, hline8]), g.nd("Mul", [trail_v, vline8])])
    ones3 = g.f([1, 1, 3, 3], [1.0] * 9)
    dil = g.nd("Conv", [crossing, ones3], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    boxdil = g.nd("Clip", [dil], min=0.0, max=1.0)
    boxring = g.nd("Mul", [g.nd("Clip", [g.nd("Sub", [boxdil, crossing])], min=0.0, max=1.0), realmask])

    allline = g.nd("Max", [hline8, vline8])
    final8 = g.nd("Clip", [g.nd("Sub", [g.nd("Max", [allline, boxring]), crossing])], min=0.0, max=1.0)
    final2 = g.nd("Clip", [g.nd("Sub", [trail, boxring])], min=0.0, max=1.0)
    final0 = g.nd("Sub", [g.nd("Sub", [realmask, final8]), final2])
    z = g.nd("Sub", [M, M])
    g.nd("Concat", [final0, z, final2, z, z, z, z, z, final8, z], "output", axis=1)
    return _model(g)


# =========================================================================== #
# TASK 189 -- 9x9: a full 8-row and full 8-col split off a 2x2 colour KEY and a #
#   6x6 TEMPLATE (color-3 shapes) in opposite corners.  Output (6x6) = template #
#   with each 3x3 quadrant's 3s recolored to the key colour of that quadrant.   #
# =========================================================================== #
def _t189_rule(a):
    ic = set(np.unique(a).tolist())
    if ic - {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}:
        return None
    H, W = a.shape
    if H != 9 or W != 9:
        return None
    hl = [r for r in range(H) if (a[r] == 8).all()]
    vl = [c for c in range(W) if (a[:, c] == 8).all()]
    if len(hl) != 1 or len(vl) != 1:
        return None
    R8, C8 = hl[0], vl[0]
    if R8 not in (2, 6) or C8 not in (2, 6):
        return None
    toff_r = 3 if R8 == 2 else 0
    toff_c = 3 if C8 == 2 else 0
    koff_r = 0 if R8 == 2 else 7
    koff_c = 0 if C8 == 2 else 7
    key = a[koff_r:koff_r + 2, koff_c:koff_c + 2]
    tmpl3 = (a[toff_r:toff_r + 6, toff_c:toff_c + 6] == 3)
    o = np.zeros((6, 6), int)
    for r in range(6):
        for c in range(6):
            if tmpl3[r, c]:
                o[r, c] = key[r // 3, c // 3]
    return o


def _t189_detect(prs):
    okany = False
    for i, o in prs:
        r = _t189_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
        okany = True
    return okany


def _t189_build():
    g = _G()
    rIdx = g.f([1, 1, G, 1], list(range(G)))
    cIdx = g.f([1, 1, 1, G], list(range(G)))
    sP = g.f([1, 1, G, 1], [p // 3 for p in range(G)])
    sQ = g.f([1, 1, 1, G], [q // 3 for q in range(G)])
    one = g.f([1, 1, 1, 1], [1.0])
    half = g.f([1, 1, 1, 1], [0.5])
    two = g.f([1, 1, 1, 1], [2.0])
    six = g.f([1, 1, 1, 1], [6.0])
    three = g.f([1, 1, 1, 1], [3.0])
    seven = g.f([1, 1, 1, 1], [7.0])
    e0 = g.f([1, 10, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    reg = np.zeros((30, 30), np.float32); reg[:6, :6] = 1.0
    region6 = g.f([1, 1, 30, 30], reg.ravel())

    ch8 = _slc(g, "input", 8, 9, 1)
    input3 = _slc(g, "input", 3, 4, 1)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)

    Wc = g.nd("ReduceSum", [realmask], axes=[3], keepdims=1)         # [1,1,30,1]
    row8 = g.nd("ReduceSum", [ch8], axes=[3], keepdims=1)
    hlinerow = g.nd("Mul", [_castlt(g, g.nd("Abs", [g.nd("Sub", [row8, Wc])]), half),
                            _castlt(g, half, Wc)])                  # [1,1,30,1]
    Hc = g.nd("ReduceSum", [realmask], axes=[2], keepdims=1)
    col8 = g.nd("ReduceSum", [ch8], axes=[2], keepdims=1)
    vlinecol = g.nd("Mul", [_castlt(g, g.nd("Abs", [g.nd("Sub", [col8, Hc])]), half),
                            _castlt(g, half, Hc)])                  # [1,1,1,30]

    R8 = g.nd("ReduceSum", [g.nd("Mul", [hlinerow, rIdx])], axes=[2], keepdims=1)   # [1,1,1,1]
    C8 = g.nd("ReduceSum", [g.nd("Mul", [vlinecol, cIdx])], axes=[3], keepdims=1)
    isR2 = _castlt(g, g.nd("Abs", [g.nd("Sub", [R8, two])]), half)
    isC2 = _castlt(g, g.nd("Abs", [g.nd("Sub", [C8, two])]), half)
    toff_r = g.nd("Mul", [three, isR2])
    toff_c = g.nd("Mul", [three, isC2])
    koff_r = g.nd("Mul", [seven, g.nd("Sub", [one, isR2])])
    koff_c = g.nd("Mul", [seven, g.nd("Sub", [one, isC2])])

    plt6 = _castlt(g, rIdx, six)        # p<6 [1,1,30,1]
    qlt6 = _castlt(g, cIdx, six)        # q<6 [1,1,1,30]
    # Tr[p,q]=(p<6)&(q==toff_r+p) ; Tc[p,q]=(q<6)&(p==toff_c+q)
    Tr = g.nd("Mul", [plt6, _castlt(g, g.nd("Abs", [g.nd("Sub", [g.nd("Sub", [cIdx, rIdx]), toff_r])]), half)])
    Tc = g.nd("Mul", [qlt6, _castlt(g, g.nd("Abs", [g.nd("Sub", [g.nd("Sub", [rIdx, cIdx]), toff_c])]), half)])
    # Pr[p,q]=(p<6)&(q==koff_r+floor(p/3)) ; Pc[p,q]=(q<6)&(p==koff_c+floor(q/3))
    Pr = g.nd("Mul", [plt6, _castlt(g, g.nd("Abs", [g.nd("Sub", [g.nd("Sub", [cIdx, sP]), koff_r])]), half)])
    Pc = g.nd("Mul", [qlt6, _castlt(g, g.nd("Abs", [g.nd("Sub", [g.nd("Sub", [rIdx, sQ]), koff_c])]), half)])

    template3 = g.nd("MatMul", [g.nd("MatMul", [Tr, input3]), Tc])   # [1,1,30,30]
    keyfield = g.nd("MatMul", [g.nd("MatMul", [Pr, "input"]), Pc])   # [1,10,30,30]

    colorpart = g.nd("Mul", [keyfield, template3])
    bgpart = g.nd("Mul", [e0, g.nd("Sub", [region6, template3])])
    g.nd("Add", [colorpart, bgpart], "output")
    return _model(g)


# =========================================================================== #
# dispatch                                                                    #
# =========================================================================== #
_SOLVERS = [
    ("t389", _t389_detect, _t389_build),
    ("t180", _t180_detect, _t180_build),
    ("t334", _t334_detect, _t334_build),
    ("t11", _t11_detect, _t11_build),
    ("tsym", _sym_detect, _sym_build),
    ("t320", _t320_detect, _t320_build),
    ("t288", _t288_detect, _t288_build),
    ("t379", _t379_detect, _t379_build),
    ("t189", _t189_detect, _t189_build),
]


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []
    for name, detect, build in _SOLVERS:
        try:
            if detect(prs):
                m = _check(build())
                out.append((name, m))
        except Exception:
            pass
    return out
