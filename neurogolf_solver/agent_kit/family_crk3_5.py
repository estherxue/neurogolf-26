"""family_crk3_5 -- crack module for slice IDX=5 of the unsolved NeuroGolf tasks.

Each detected task gets its own structural detector (validated EXACTLY against the
provided train/test/arc-gen pairs in numpy) plus a static opset-10 ONNX builder.
All intermediates are static-shape; data-dependent geometry uses computed index
grids + ReduceMax/ReduceSum/Mod/Abs/Less, never dynamic Resize/Pad.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL
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


def _pairs(ex, secs=("train", "test")):
    out = []
    for k in secs:
        for p in ex.get(k, []):
            out.append((np.array(p["input"]), np.array(p["output"])))
    return out


def _allpairs(ex):
    return _pairs(ex, ("train", "test", "arc-gen"))


def _slc(g, src, lo, hi, axis):
    s = g.i64([lo]); e = g.i64([hi]); a = g.i64([axis])
    return g.nd("Slice", [src, s, e, a])


def _shift(g, src, dr, dc):
    """Translate content down by dr, right by dc (negative = up/left), zero-fill.
    Keeps [.,.,30,30] via static Pad(attr)+Slice."""
    t = src
    if dr > 0:
        t = g.nd("Pad", [t], pads=[0, 0, dr, 0, 0, 0, 0, 0], mode="constant", value=0.0)
        t = _slc(g, t, 0, G, 2)
    elif dr < 0:
        k = -dr
        t = g.nd("Pad", [t], pads=[0, 0, 0, 0, 0, 0, k, 0], mode="constant", value=0.0)
        t = _slc(g, t, k, k + G, 2)
    if dc > 0:
        t = g.nd("Pad", [t], pads=[0, 0, 0, dc, 0, 0, 0, 0], mode="constant", value=0.0)
        t = _slc(g, t, 0, G, 3)
    elif dc < 0:
        k = -dc
        t = g.nd("Pad", [t], pads=[0, 0, 0, 0, 0, 0, 0, k], mode="constant", value=0.0)
        t = _slc(g, t, k, k + G, 3)
    return t


# =========================================================================== #
# TASK 357 -- bouncing diagonal path from the bottom-left marker              #
#   grid HxW, single marker '1' at (H-1, 0); fill the whole grid with 8       #
#   except a triangle-wave path of 1s bouncing between cols 0 and W-1.        #
# =========================================================================== #
def _t357_rule(a):
    H, W = a.shape
    nz = np.argwhere(a != 0)
    if len(nz) != 1:
        return None
    r0, c0 = nz[0]
    if r0 != H - 1 or c0 != 0 or a[r0, c0] != 1 or W < 2:
        return None
    o = np.full((H, W), 8, dtype=int)
    c, dc = 0, 1
    for d in range(H):
        r = r0 - d
        if r < 0:
            break
        o[r, c] = 1
        nc = c + dc
        if nc < 0 or nc >= W:
            dc = -dc
            nc = c + dc
        c = nc
    return o


def _t357_detect(prs):
    for i, o in prs:
        r = _t357_rule(i)
        if r is None or not np.array_equal(r, o):
            return False
    return True


def _t357_build():
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0])
    two = g.f([1, 1, 1, 1], [2.0])
    half = g.f([1, 1, 1, 1], [0.5])
    Rg = g.f([1, 1, G, 1], list(range(G)))
    Cg = g.f([1, 1, 1, G], list(range(G)))

    M = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)        # [1,1,30,30]
    colmask = g.nd("ReduceMax", [M], axes=[2], keepdims=1)        # [1,1,1,30]
    W = g.nd("ReduceSum", [colmask], axes=[3], keepdims=1)        # [1,1,1,1]
    rowmask = g.nd("ReduceMax", [M], axes=[3], keepdims=1)        # [1,1,30,1]
    H = g.nd("ReduceSum", [rowmask], axes=[2], keepdims=1)        # [1,1,1,1]

    d = g.nd("Sub", [g.nd("Sub", [H, one]), Rg])                  # H-1-r  [1,1,30,1]
    Wm1 = g.nd("Sub", [W, one])                                   # [1,1,1,1]
    p = g.nd("Mul", [Wm1, two])                                   # period
    m = g.nd("Mod", [d, p], fmod=1)                               # [1,1,30,1]
    tri = g.nd("Sub", [Wm1, g.nd("Abs", [g.nd("Sub", [m, Wm1])])])  # [1,1,30,1]
    adiff = g.nd("Abs", [g.nd("Sub", [Cg, tri])])                 # [1,1,30,30]
    path = g.nd("Cast", [g.nd("Less", [adiff, half])], to=F)      # [1,1,30,30]

    ch1 = g.nd("Mul", [M, path])
    ch8 = g.nd("Mul", [M, g.nd("Sub", [one, path])])
    z = g.nd("Mul", [M, g.f([1, 1, 1, 1], [0.0])])               # zeros [1,1,30,30]
    g.nd("Concat", [z, ch1, z, z, z, z, z, z, ch8, z], "output", axis=1)
    return _model(g)


# =========================================================================== #
# TASK 28 -- fixed 10x10 two-color frame template                             #
#   2 single markers (one in top half, one in bottom half); output is a       #
#   constant region template recolored: top-marker color -> region1,          #
#   bottom-marker color -> region2.                                           #
# =========================================================================== #
_T28_REG = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
], dtype=int)


def _t28_rule(a):
    if a.shape != (10, 10):
        return None
    nz = np.argwhere(a != 0)
    if len(nz) != 2:
        return None
    ms = sorted((int(r), int(c), int(a[r, c])) for r, c in nz)
    if not (ms[0][0] < 5 and ms[-1][0] >= 5):
        return None
    topcol, botcol = ms[0][2], ms[-1][2]
    if topcol == botcol:
        return None
    o = np.zeros((10, 10), dtype=int)
    o[_T28_REG == 1] = topcol
    o[_T28_REG == 2] = botcol
    return o


def _t28_detect(prs):
    for i, o in prs:
        r = _t28_rule(i)
        if r is None or not np.array_equal(r, o):
            return False
    return True


def _t28_build():
    g = _G()
    # constant [1,1,30,30] region masks (10x10 content, top-left anchored)
    m1 = np.zeros((1, 1, G, G), np.float32)
    m2 = np.zeros((1, 1, G, G), np.float32)
    bg = np.zeros((1, 1, G, G), np.float32)
    m1[0, 0, :10, :10] = (_T28_REG == 1)
    m2[0, 0, :10, :10] = (_T28_REG == 2)
    bg[0, 0, :10, :10] = (_T28_REG == 0)
    M1 = g.f([1, 1, G, G], m1)
    M2 = g.f([1, 1, G, G], m2)
    BG = g.f([1, 1, G, G], bg)
    chmask = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * 9)

    top_slice = _slc(g, "input", 0, 5, 2)                         # [1,10,5,30]
    bot_slice = _slc(g, "input", 5, 10, 2)                        # [1,10,5,30]
    top_has = g.nd("ReduceMax", [top_slice], axes=[2, 3], keepdims=1)  # [1,10,1,1]
    bot_has = g.nd("ReduceMax", [bot_slice], axes=[2, 3], keepdims=1)
    top_has = g.nd("Mul", [top_has, chmask])
    bot_has = g.nd("Mul", [bot_has, chmask])

    colored = g.nd("Add", [g.nd("Mul", [M1, top_has]),
                           g.nd("Mul", [M2, bot_has])])           # [1,10,30,30] ch0=0
    rest = _slc(g, colored, 1, 10, 1)                             # [1,9,30,30]
    g.nd("Concat", [BG, rest], "output", axis=1)
    return _model(g)


# =========================================================================== #
# TASK 225 -- diagonal 2x2-block corner projection                            #
#   one solid 2x2 block of 4 colors somewhere; each block cell is projected   #
#   (point-reflected) to a 2x2 solid block in the diagonally-opposite corner, #
#   offset 2 cells away (clipped to grid).                                     #
# =========================================================================== #
def _np_shift(Y, dr, dc):
    out = np.zeros_like(Y); H, W = Y.shape[-2], Y.shape[-1]
    r0 = max(0, dr); r1 = min(H, H + dr); c0 = max(0, dc); c1 = min(W, W + dc)
    sr = max(0, -dr); sc = max(0, -dc)
    out[..., r0:r1, c0:c1] = Y[..., sr:sr + (r1 - r0), sc:sc + (c1 - c0)]
    return out


def _t225_rule(a):
    a = np.array(a, int)
    nz = np.argwhere(a != 0)
    if len(nz) != 4:
        return None
    r0, c0 = nz.min(0); r1, c1 = nz.max(0)
    if r1 - r0 != 1 or c1 - c0 != 1:          # must be a 2x2 footprint
        return None
    P = (a != 0).astype(int)
    aP = _np_shift(P, 1, 0); lP = _np_shift(P, 0, 1)
    bP = _np_shift(P, -1, 0); rP = _np_shift(P, 0, -1)
    mTL = P * (1 - aP) * (1 - lP); mTR = P * (1 - aP) * (1 - rP)
    mBL = P * (1 - bP) * (1 - lP); mBR = P * (1 - bP) * (1 - rP)
    cTL = a * mTL; cTR = a * mTR; cBL = a * mBL; cBR = a * mBR

    def dil(Y, ddr, ddc):
        return Y + _np_shift(Y, ddr, 0) + _np_shift(Y, 0, ddc) + _np_shift(Y, ddr, ddc)
    BR = dil(_np_shift(cTL, 2, 2), 1, 1)
    BL = dil(_np_shift(cTR, 2, -2), 1, -1)
    TR = dil(_np_shift(cBL, -2, 2), -1, 1)
    TL = dil(_np_shift(cBR, -2, -2), -1, -1)
    corners = BR + BL + TR + TL
    out = np.where(corners != 0, corners, a)
    return out


def _t225_detect(prs):
    seen = False
    for i, o in prs:
        r = _t225_rule(i)
        if r is None:
            return False
        if not np.array_equal(r, np.array(o, int)):
            return False
        if not np.array_equal(r, np.array(i, int)):
            seen = True
    return seen


def _t225_build():
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0])
    ch19 = _slc(g, "input", 1, 10, 1)
    P = g.nd("ReduceSum", [ch19], axes=[1], keepdims=1)
    R = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    aP = _shift(g, P, 1, 0); lP = _shift(g, P, 0, 1)
    bP = _shift(g, P, -1, 0); rP = _shift(g, P, 0, -1)

    def nott(x):
        return g.nd("Sub", [one, x])
    mTL = g.nd("Mul", [g.nd("Mul", [P, nott(aP)]), nott(lP)])
    mTR = g.nd("Mul", [g.nd("Mul", [P, nott(aP)]), nott(rP)])
    mBL = g.nd("Mul", [g.nd("Mul", [P, nott(bP)]), nott(lP)])
    mBR = g.nd("Mul", [g.nd("Mul", [P, nott(bP)]), nott(rP)])
    cTL = g.nd("Mul", ["input", mTL]); cTR = g.nd("Mul", ["input", mTR])
    cBL = g.nd("Mul", ["input", mBL]); cBR = g.nd("Mul", ["input", mBR])

    def kern(positions):
        w = np.zeros((10, 1, 4, 4), np.float32)
        for ki, kj in positions:
            w[:, 0, ki, kj] = 1.0
        return g.f([10, 1, 4, 4], w)

    def conv(src, positions, pads):
        return g.nd("Conv", [src, kern(positions)], group=10, kernel_shape=[4, 4],
                    pads=pads, strides=[1, 1], dilations=[1, 1])
    BR = conv(cTL, [(0, 0), (0, 1), (1, 0), (1, 1)], [3, 3, 0, 0])
    BL = conv(cTR, [(0, 2), (0, 3), (1, 2), (1, 3)], [3, 0, 0, 3])
    TR = conv(cBL, [(2, 0), (2, 1), (3, 0), (3, 1)], [0, 3, 3, 0])
    TL = conv(cBR, [(2, 2), (2, 3), (3, 2), (3, 3)], [0, 0, 3, 3])
    corners = g.nd("Add", [g.nd("Add", [BR, BL]), g.nd("Add", [TR, TL])])
    corners = g.nd("Mul", [corners, R])
    cmask = g.nd("ReduceSum", [corners], axes=[1], keepdims=1)
    in0 = _slc(g, "input", 0, 1, 1)
    out0 = g.nd("Mul", [in0, nott(cmask)])
    inrest = _slc(g, "input", 1, 10, 1)
    crest = _slc(g, corners, 1, 10, 1)
    outrest = g.nd("Add", [inrest, crest])
    g.nd("Concat", [out0, outrest], "output", axis=1)
    return _model(g)


# =========================================================================== #
# TASK 128 -- bottom-anchored solid blocks jump up by their own height        #
#   each color is a solid rectangle resting on the bottom row; output moves   #
#   it up by exactly its height (so it sits directly above its old spot).     #
# =========================================================================== #
def _t128_rule(a):
    a = np.array(a, int)
    H, W = a.shape
    out = np.zeros_like(a)
    changed = False
    for c in range(W):
        col = a[:, c]
        for cc in set(col[col > 0]):
            hc = int((col == cc).sum())
            # require bottom-anchored solid run
            rows = np.where(col == cc)[0]
            if not (rows.max() == H - 1 and len(rows) == rows.max() - rows.min() + 1):
                return None
            for r in range(H):
                k = H - 1 - r
                if hc <= k <= 2 * hc - 1:
                    out[r, c] = cc
                    if out[r, c] != a[r, c]:
                        changed = True
    return out if changed else None


def _t128_detect(prs):
    good = False
    for i, o in prs:
        r = _t128_rule(i)
        if r is None or not np.array_equal(r, np.array(o, int)):
            return False
        good = True
    return good


def _t128_build():
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0])
    two = g.f([1, 1, 1, 1], [2.0])
    half = g.f([1, 1, 1, 1], [0.5])
    rowg = g.f([1, 1, G, 1], list(range(G)))
    R = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)        # real cells
    rowHas = g.nd("ReduceMax", [R], axes=[3], keepdims=1)         # [1,1,30,1]
    H = g.nd("ReduceSum", [rowHas], axes=[2], keepdims=1)         # [1,1,1,1]
    k = g.nd("Sub", [g.nd("Sub", [H, one]), rowg])               # [1,1,30,1]
    ch19 = _slc(g, "input", 1, 10, 1)                             # [1,9,30,30]
    hcol = g.nd("ReduceSum", [ch19], axes=[2], keepdims=1)        # [1,9,1,30]
    condA = g.nd("Cast", [g.nd("Less", [g.nd("Sub", [hcol, k]), half])], to=F)
    twoh = g.nd("Mul", [hcol, two])
    condB = g.nd("Cast", [g.nd("Greater", [g.nd("Sub", [twoh, k]), half])], to=F)
    condH = g.nd("Cast", [g.nd("Greater", [hcol, half])], to=F)
    mask = g.nd("Mul", [g.nd("Mul", [condA, condB]), condH])      # [1,9,30,30]
    colored = g.nd("ReduceSum", [mask], axes=[1], keepdims=1)     # [1,1,30,30]
    out0 = g.nd("Mul", [R, g.nd("Sub", [one, colored])])
    g.nd("Concat", [out0, mask], "output", axis=1)
    return _model(g)


# =========================================================================== #
# TASK 340 -- markers slide to the wall whose colour they match               #
#   rectangular frame with 4 differently-coloured walls; each interior marker #
#   slides to be adjacent to the wall of its own colour, others are deleted.  #
# =========================================================================== #
def _t340_rule(a):
    a = np.array(a, int)
    H, W = a.shape
    if H < 3 or W < 3:
        return None
    topc = a[0, 1]; botc = a[H - 1, 1]; leftc = a[1, 0]; rightc = a[H - 2, W - 1]
    if len({topc, botc, leftc, rightc}) != 4 or 0 in {topc, botc, leftc, rightc}:
        return None
    out = np.zeros_like(a)
    out[0, :] = a[0, :]; out[H - 1, :] = a[H - 1, :]
    out[:, 0] = a[:, 0]; out[:, W - 1] = a[:, W - 1]
    changed = False
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            v = a[r, c]
            if v == 0:
                continue
            changed = True
            if v == topc:
                out[1, c] = topc
            elif v == botc:
                out[H - 2, c] = botc
            elif v == leftc:
                out[r, 1] = leftc
            elif v == rightc:
                out[r, W - 2] = rightc
    return out if changed else None


def _t340_detect(prs):
    good = False
    for i, o in prs:
        r = _t340_rule(i)
        if r is None or not np.array_equal(r, np.array(o, int)):
            return False
        good = True
    return good


def _t340_build():
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0]); two = g.f([1, 1, 1, 1], [2.0])
    half = g.f([1, 1, 1, 1], [0.5])
    rowg = g.f([1, 1, G, 1], list(range(G)))
    colg = g.f([1, 1, 1, G], list(range(G)))

    def eqr(grid, val):  # cast(|grid-val|<0.5)
        return g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [grid, val])]), half])], to=F)

    R = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    rowHas = g.nd("ReduceMax", [R], axes=[3], keepdims=1)
    colHas = g.nd("ReduceMax", [R], axes=[2], keepdims=1)
    H = g.nd("ReduceSum", [rowHas], axes=[2], keepdims=1)
    W = g.nd("ReduceSum", [colHas], axes=[3], keepdims=1)
    Hm1 = g.nd("Sub", [H, one]); Hm2 = g.nd("Sub", [H, two])
    Wm1 = g.nd("Sub", [W, one]); Wm2 = g.nd("Sub", [W, two])

    isR0 = eqr(rowg, g.f([1, 1, 1, 1], [0.0])); isRH1 = eqr(rowg, Hm1)
    isC0 = eqr(colg, g.f([1, 1, 1, 1], [0.0])); isCW1 = eqr(colg, Wm1)
    isR1 = eqr(rowg, one); isRHm2 = eqr(rowg, Hm2)
    isC1 = eqr(colg, one); isCWm2 = eqr(colg, Wm2)

    irow = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [rowg, half])], to=F),
                        g.nd("Cast", [g.nd("Less", [rowg, g.nd("Sub", [Hm1, half])])], to=F)])
    icol = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [colg, half])], to=F),
                        g.nd("Cast", [g.nd("Less", [colg, g.nd("Sub", [Wm1, half])])], to=F)])
    intr = g.nd("Mul", [g.nd("Mul", [irow, icol]), R])

    inp19 = _slc(g, "input", 1, 10, 1)
    topc = _slc(g, _slc(g, "input", 0, 1, 2), 1, 2, 3)            # input[:,:,0:1,1:2]
    leftc = _slc(g, _slc(g, "input", 1, 2, 2), 0, 1, 3)           # input[:,:,1:2,0:1]
    botrow = g.nd("ReduceSum", [g.nd("Mul", ["input", isRH1])], axes=[2], keepdims=1)
    botc = g.nd("ReduceMax", [botrow], axes=[3], keepdims=1)      # [1,10,1,1]
    rightcol = g.nd("ReduceSum", [g.nd("Mul", ["input", isCW1])], axes=[3], keepdims=1)
    rightc = g.nd("ReduceMax", [rightcol], axes=[2], keepdims=1)  # [1,10,1,1]

    def match(c11):
        c19 = _slc(g, c11, 1, 10, 1)
        return g.nd("Mul", [g.nd("ReduceSum", [g.nd("Mul", [inp19, c19])], axes=[1], keepdims=1), intr])
    mTop = match(topc); mBot = match(botc); mLeft = match(leftc); mRight = match(rightc)

    colTop = g.nd("ReduceMax", [mTop], axes=[2], keepdims=1)      # [1,1,1,30]
    colBot = g.nd("ReduceMax", [mBot], axes=[2], keepdims=1)
    rowLeft = g.nd("ReduceMax", [mLeft], axes=[3], keepdims=1)    # [1,1,30,1]
    rowRight = g.nd("ReduceMax", [mRight], axes=[3], keepdims=1)

    pTop = g.nd("Mul", [colTop, isR1]); pBot = g.nd("Mul", [colBot, isRHm2])
    pLeft = g.nd("Mul", [rowLeft, isC1]); pRight = g.nd("Mul", [rowRight, isCWm2])
    mk = g.nd("Add", [g.nd("Add", [g.nd("Mul", [pTop, topc]), g.nd("Mul", [pBot, botc])]),
                      g.nd("Add", [g.nd("Mul", [pLeft, leftc]), g.nd("Mul", [pRight, rightc])])])

    border = g.nd("Cast", [g.nd("Greater", [g.nd("Add", [g.nd("Add", [isR0, isRH1]),
                                                          g.nd("Add", [isC0, isCW1])]), half])], to=F)
    wall19 = _slc(g, g.nd("Mul", ["input", border]), 1, 10, 1)
    mk19 = _slc(g, mk, 1, 10, 1)
    outrest = g.nd("Add", [wall19, mk19])
    colored = g.nd("ReduceSum", [outrest], axes=[1], keepdims=1)
    out0 = g.nd("Mul", [R, g.nd("Sub", [one, colored])])
    g.nd("Concat", [out0, outrest], "output", axis=1)
    return _model(g)


# =========================================================================== #
# TASK 109 -- remove central +-divider, mirror the content quadrant 4-fold     #
#   a full row & full col of colour L cross at the centre; one quadrant holds  #
#   a shape; output is (H-1)x(W-1) dihedral-symmetric, recoloured to L.        #
# =========================================================================== #
def _t109_rule(a):
    a = np.array(a, int); H, W = a.shape
    if H % 2 == 0 or W % 2 == 0 or H < 3 or W < 3:
        return None
    r0 = (H - 1) // 2; c0 = (W - 1) // 2
    L = a[r0, c0]
    if L == 0 or not (np.all(a[r0, :] == L) and np.all(a[:, c0] == L)):
        return None
    oH, oW = H - 1, W - 1
    out = np.zeros((oH, oW), int)
    for i in range(oH):
        fi = min(i, oH - 1 - i)
        for j in range(oW):
            fj = min(j, oW - 1 - j)
            if a[fi, fj] != 0:
                out[i, j] = L
    return out


def _t109_detect(prs):
    good = False
    for i, o in prs:
        r = _t109_rule(i)
        if r is None or r.shape != np.array(o).shape or not np.array_equal(r, np.array(o, int)):
            return False
        good = True
    return good


def _t109_build():
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0]); two = g.f([1, 1, 1, 1], [2.0])
    half = g.f([1, 1, 1, 1], [0.5])
    rowg = g.f([1, 1, G, 1], list(range(G)))
    colg = g.f([1, 1, 1, G], list(range(G)))
    R = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    rowHas = g.nd("ReduceMax", [R], axes=[3], keepdims=1)
    colHas = g.nd("ReduceMax", [R], axes=[2], keepdims=1)
    H = g.nd("ReduceSum", [rowHas], axes=[2], keepdims=1)
    W = g.nd("ReduceSum", [colHas], axes=[3], keepdims=1)
    Hm1 = g.nd("Sub", [H, one]); Wm1 = g.nd("Sub", [W, one])
    Hm2 = g.nd("Sub", [H, two]); Wm2 = g.nd("Sub", [W, two])
    r0 = g.nd("Mul", [Hm1, half]); c0 = g.nd("Mul", [Wm1, half])

    def cast_lt(x, y):
        return g.nd("Cast", [g.nd("Less", [x, y])], to=F)
    # fold targets
    tr = g.nd("Mul", [half, g.nd("Sub", [Hm2, g.nd("Abs", [g.nd("Sub", [g.nd("Mul", [rowg, two]), Hm2])])])])
    tc = g.nd("Mul", [half, g.nd("Sub", [Wm2, g.nd("Abs", [g.nd("Sub", [g.nd("Mul", [colg, two]), Wm2])])])])
    Frow = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colg, tr])]), half])], to=F)
    Fcol = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowg, tc])]), half])], to=F)
    folded = g.nd("MatMul", [g.nd("MatMul", [Frow, "input"]), Fcol])
    shapeP = g.nd("ReduceSum", [_slc(g, folded, 1, 10, 1)], axes=[1], keepdims=1)

    rowSel = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowg, r0])]), half])], to=F)
    colSel = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colg, c0])]), half])], to=F)
    center = g.nd("Mul", [g.nd("Mul", ["input", rowSel]), colSel])
    L_oh = g.nd("ReduceSum", [center], axes=[2, 3], keepdims=1)         # [1,10,1,1]
    outColored = g.nd("Mul", [shapeP, L_oh])
    outReal = g.nd("Mul", [cast_lt(rowg, Hm1), cast_lt(colg, Wm1)])
    out0 = g.nd("Mul", [outReal, g.nd("Sub", [one, shapeP])])
    g.nd("Concat", [out0, _slc(g, outColored, 1, 10, 1)], "output", axis=1)
    return _model(g)


# =========================================================================== #
# dispatch                                                                    #
# =========================================================================== #
_SOLVERS = [
    ("t357", _t357_detect, _t357_build),
    ("t28", _t28_detect, _t28_build),
    ("t225", _t225_detect, _t225_build),
    ("t128", _t128_detect, _t128_build),
    ("t340", _t340_detect, _t340_build),
    ("t109", _t109_detect, _t109_build),
]


def candidates(ex):
    prs = _allpairs(ex)
    if not prs:
        return []
    out = []
    for name, detect, build in _SOLVERS:
        try:
            if detect(prs):
                out.append((name, _check(build())))
        except Exception:
            pass
    return out
