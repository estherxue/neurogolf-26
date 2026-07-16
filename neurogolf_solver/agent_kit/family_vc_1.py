"""family_vc_1 — verifier-translated static ONNX solvers for three re-arc rules.

task005 / verify_045e512c  (21x21, bg=0): the largest object (3x3 bbox, 4..9 cells)
    has small (<=3 cell) colour markers at offset +-4 in some of the 8 directions.
    Paint pattern-shaped copies along each marked ray at steps of 4, coloured by the
    marker colour.  ONNX: 5x5 score conv (+1 inner 3x3 / -3 ring) -> unique argmax
    locates the main bbox; per-direction shifted boxes read marker colours
    (ReduceMax); a grouped dilated comb conv replicates the coloured k=1 stamps
    outward; Max with the value image; one-hot + 21x21 clip.

task101 / verify_447fd412  (bg=0, template = the only object containing colour 1,
    bbox<=4x4; other objects are colour-2 upscales k=1..5 of its colour-2 part with
    the colour-1 positions empty): complete each occurrence with the upscaled
    colour-1 part.  ONNX: flood colour-1 through nonbg (15 bounded dilate steps) ->
    template mask; anchor via MatMul shift; extract 4x4 red/blue patches as RUNTIME
    Conv weights; per scale k: constant-matrix upscale, three correlation convs
    (red exact match / blue positions empty / 8-neighbourhood isolation == the
    verifier's object filter) + 3x-canvas bound masks; ConvTranspose stamps the
    blue cells back; output = input + stamp * ([-1@ch0, +1@ch1]).

task209 / verify_8a004b2b  (bg=0, colour 4 = exactly 4 rectangle-corner cells):
    key pattern outside the corner bbox, partial k-times-upscaled drawing strictly
    inside.  reliable colours = those whose floor(inside_count/key_count) equals
    the MOSTCOMMON such ratio (verifier x26=mostcommon); k_h,k_w = bbox ratios of
    reliable cells; paint the
    upscaled key aligned by reliable-ulcorner; crop the corner bbox.  ONNX: min/max
    index reductions, Floor(Div) ratios, MatMul translations, constant-free runtime
    upscale matrices U[r,i]=[i==floor(r/k)], MatMul crop, one-hot.

All detection gates are numpy mirrors of the ONNX numerics, validated 266/266
against train+test+arc-gen expected outputs (arc-gen itself is untouched held-out:
gates only consult train+test).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
INT64 = onnx.TensorProto.INT64
H30 = 30


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
        self.inits.append(oh.make_tensor(
            n, F, list(dims), [float(v) for v in np.asarray(vals, np.float64).ravel()]))
        return n

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _model(g, name):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    used = {i for n in g.nodes for i in n.input}
    inits = [t for t in g.inits if t.name in used]
    m = oh.make_model(oh.make_graph(g.nodes, name, [x], [y], inits),
                      ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    onnx.checker.check_model(m, full_check=True)
    return m


def _consts(g):
    g.rowidx = g.f([1, 1, H30, 1], list(range(H30)))
    g.colidx = g.f([1, 1, 1, H30], list(range(H30)))
    g.half = g.f([1, 1, 1, 1], [0.5])
    g.nhalf = g.f([1, 1, 1, 1], [-0.5])
    g.one = g.f([1, 1, 1, 1], [1.0])
    g.cbig = g.f([1, 1, 1, 1], [1000.0])


def _gt(g, a, b):
    return g.nd("Cast", [g.nd("Greater", [a, b])], to=F)


def _lt(g, a, b):
    return g.nd("Cast", [g.nd("Less", [a, b])], to=F)


def _eqm(g, a, b):
    """|a-b| < 0.5  (exact integer equality) as float 0/1."""
    return _lt(g, g.nd("Abs", [g.nd("Sub", [a, b])]), g.half)


def _slice_ch(g, c):
    return g.nd("Slice", ["input", g.i64([c]), g.i64([c + 1]), g.i64([1])])


def _value_img(g):
    """V[.] = colour value 0..9 (0 also outside the grid)."""
    w = g.f([1, 10, 1, 1], list(range(10)))
    return g.nd("Conv", ["input", w], kernel_shape=[1, 1])


def _minmax(g, has, idx, axis):
    """(min, max) of index over cells where has>0; has/idx shaped for `axis`."""
    mx = g.nd("ReduceMax", [g.nd("Mul", [has, idx])], axes=[axis], keepdims=1)
    inv = g.nd("Mul", [has, g.nd("Sub", [g.cbig, idx])])
    mn = g.nd("Sub", [g.cbig, g.nd("ReduceMax", [inv], axes=[axis], keepdims=1)])
    return mn, mx


def _rowspan(g, mask):
    has = g.nd("ReduceMax", [mask], axes=[3], keepdims=1)          # [1,1,30,1]
    return _minmax(g, has, g.rowidx, 2)


def _colspan(g, mask):
    has = g.nd("ReduceMax", [mask], axes=[2], keepdims=1)          # [1,1,1,30]
    return _minmax(g, has, g.colidx, 3)


# =========================================================================== #
# task005 — 045e512c                                                          #
# =========================================================================== #
def _np_005(a):
    a = np.asarray(a, int)
    Hd, Wd = a.shape
    if Hd > 30 or Wd > 30 or Hd < 4 or Wd < 4:
        return None
    N = (a > 0).astype(int)
    # score conv: +1 on the 3x3 window at (i..i+2), -3 on the surrounding ring
    Np = np.pad(N, ((1, 3), (1, 3)))
    S = np.zeros((Hd, Wd), int)
    for u in range(5):
        for v in range(5):
            wgt = 1 if (1 <= u <= 3 and 1 <= v <= 3) else -3
            S += wgt * Np[u:u + Hd, v:v + Wd]
    mx = S.max()
    pos = np.argwhere(S == mx)
    if len(pos) != 1 or mx < 4:
        return None
    r0, c0 = map(int, pos[0])
    if r0 + 2 >= Hd or c0 + 2 >= Wd:
        return None
    B = np.zeros((Hd, Wd), int)
    B[r0:r0 + 3, c0:c0 + 3] = 1
    M = N * B
    R = np.zeros((Hd, Wd), int)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            colors = []
            for u in range(3):
                for v in range(3):
                    y, x = r0 + u + 4 * di, c0 + v + 4 * dj
                    if 0 <= y < Hd and 0 <= x < Wd and a[y, x] > 0:
                        colors.append(int(a[y, x]))
            cd = max(colors) if colors else 0
            if cd == 0:
                continue
            for k in range(1, 8):
                for u in range(3):
                    for v in range(3):
                        if M[r0 + u, c0 + v]:
                            y, x = r0 + u + 4 * k * di, c0 + v + 4 * k * dj
                            if 0 <= y < Hd and 0 <= x < Wd:
                                R[y, x] = cd
    return np.maximum(a, R)


_DIRS8 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


def _build_005():
    g = _G()
    _consts(g)
    V = _value_img(g)
    allsum = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    ch0 = _slice_ch(g, 0)
    N = g.nd("Sub", [allsum, ch0])                                  # nonbg mask

    k5 = np.full((1, 1, 5, 5), -3.0)
    k5[0, 0, 1:4, 1:4] = 1.0
    S = g.nd("Conv", [N, g.f([1, 1, 5, 5], k5)], kernel_shape=[5, 5],
             pads=[1, 1, 3, 3])
    m = g.nd("ReduceMax", [S], axes=[2, 3], keepdims=1)
    P = _gt(g, S, g.nd("Sub", [m, g.half]))                         # one-hot bbox corner
    B = g.nd("Conv", [P, g.f([1, 1, 3, 3], np.ones((1, 1, 3, 3)))],
             kernel_shape=[3, 3], pads=[2, 2, 0, 0])                # 3x3 box mask
    M = g.nd("Mul", [N, B])                                        # pattern mask

    w8 = np.zeros((8, 1, 9, 9))
    for d, (di, dj) in enumerate(_DIRS8):
        w8[d, 0, 4 - 4 * di, 4 - 4 * dj] = 1.0
    W8 = g.f([8, 1, 9, 9], w8)
    BD = g.nd("Conv", [B, W8], kernel_shape=[9, 9], pads=[4, 4, 4, 4])   # shifted boxes
    C8 = g.nd("ReduceMax", [g.nd("Mul", [V, BD])], axes=[2, 3], keepdims=1)  # colours
    MD = g.nd("Conv", [M, W8], kernel_shape=[9, 9], pads=[4, 4, 4, 4])   # shifted patterns
    A = g.nd("Mul", [MD, C8])                                       # coloured k=1 stamps

    comb = np.zeros((8, 1, 13, 13))
    for d, (di, dj) in enumerate(_DIRS8):
        for kk in range(7):                                         # shifts 0..6 (k=1..7)
            comb[d, 0, 6 - kk * di, 6 - kk * dj] = 1.0
    R8 = g.nd("Conv", [A, g.f([8, 1, 13, 13], comb)], kernel_shape=[13, 13],
              dilations=[4, 4], group=8, pads=[24, 24, 24, 24])
    R = g.nd("ReduceSum", [R8], axes=[1], keepdims=1)

    OUTv = g.nd("Max", [V, g.nd("Mul", [R, allsum])])   # allsum = in-grid mask
    OH = _eqm(g, OUTv, g.f([1, 10, 1, 1], list(range(10))))
    g.nd("Mul", [OH, allsum], "output")
    return _model(g, "vc1_005")


# =========================================================================== #
# task101 — 447fd412                                                          #
# =========================================================================== #
def _np_101(a):
    a = np.asarray(a, int)
    Hd, Wd = a.shape
    if Hd > 30 or Wd > 30:
        return None
    A = np.zeros((30, 30), int)
    A[:Hd, :Wd] = a
    NB = (A > 0).astype(int)
    X1 = (A == 1).astype(int)
    X2 = (A == 2).astype(int)
    if X1.sum() == 0 or X2.sum() == 0:
        return None
    # flood colour-1 through nonbg, 8-adjacent, 15 bounded steps (mirrors ONNX)
    T = X1.copy()
    for _ in range(15):
        P = np.pad(T, 1)
        dil = np.zeros_like(T)
        for dr in (0, 1, 2):
            for dc in (0, 1, 2):
                dil = np.maximum(dil, P[dr:dr + 30, dc:dc + 30])
        T = dil * NB
    ys, xs = np.where(T > 0)
    r0, c0 = int(ys.min()), int(xs.min())
    h_t, w_t = int(ys.max()) - r0 + 1, int(xs.max()) - c0 + 1
    if h_t > 4 or w_t > 4:
        return None
    Kr = np.zeros((4, 4), int)
    Kb = np.zeros((4, 4), int)
    Kr[:h_t, :w_t] = (X2 * T)[r0:r0 + h_t, c0:c0 + w_t]
    Kb[:h_t, :w_t] = X1[r0:r0 + h_t, c0:c0 + w_t]
    if Kr.sum() == 0 or Kb.sum() == 0:
        return None
    stamps = np.zeros((30, 30), bool)
    for k in range(1, 6):
        Krk = np.kron(Kr, np.ones((k, k), int))
        Kbk = np.kron(Kb, np.ones((k, k), int))
        n = int(Krk.sum())
        Sz = 29 + 4 * k
        # correlation maps; out index o <-> displacement delta = o-(4k-1)
        RC = np.zeros((Sz, Sz), int)
        BC = np.zeros((Sz, Sz), int)
        DC = np.zeros((Sz, Sz), int)
        X2p = np.pad(X2, 4 * k - 1)
        NBp = np.pad(NB, 4 * k - 1)
        for (u, v) in zip(*np.where(Krk)):
            RC += X2p[u:u + Sz, v:v + Sz]
        for (u, v) in zip(*np.where(Kbk)):
            BC += NBp[u:u + Sz, v:v + Sz]
        Dk = np.zeros((4 * k + 2, 4 * k + 2), int)                  # dilate8(Krk)
        for dr in (0, 1, 2):
            for dc in (0, 1, 2):
                Dk[dr:dr + 4 * k, dc:dc + 4 * k] |= Krk.astype(bool)
        NBp2 = np.pad(NB, 4 * k)
        for (u, v) in zip(*np.where(Dk)):
            DC += NBp2[u:u + Sz, v:v + Sz]
        match = (RC == n) & (BC == 0) & (DC == n)
        # 3x-canvas bounds of the verifier's `occurrences`
        didx = np.arange(Sz) - (4 * k - 1)
        rok = (didx >= -Hd) & (didx <= 2 * Hd - k * h_t)
        cok = (didx >= -Wd) & (didx <= 2 * Wd - k * w_t)
        match &= rok[:, None] & cok[None, :]
        for (o_r, o_c) in zip(*np.where(match)):
            dr_, dc_ = int(o_r) - (4 * k - 1), int(o_c) - (4 * k - 1)
            for (u, v) in zip(*np.where(Kbk)):
                y, x = dr_ + int(u), dc_ + int(v)
                if 0 <= y < 30 and 0 <= x < 30:
                    stamps[y, x] = True
    ing = np.zeros((30, 30), bool)
    ing[:Hd, :Wd] = True
    stamps &= ing
    out = A.copy()
    out[stamps] = 1
    return out[:Hd, :Wd]


def _build_101():
    g = _G()
    _consts(g)
    allsum = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)     # in-grid mask
    ch0 = _slice_ch(g, 0)
    NB = g.nd("Sub", [allsum, ch0])
    X1 = _slice_ch(g, 1)
    X2 = _slice_ch(g, 2)

    # flood colour-1 through nonbg (8-adjacency), 15 steps
    T = X1
    for _ in range(15):
        d = g.nd("MaxPool", [T], kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                 strides=[1, 1])
        T = g.nd("Mul", [d, NB])

    minrT, maxrT = _rowspan(g, T)
    mincT, maxcT = _colspan(g, T)
    h_t = g.nd("Add", [g.nd("Sub", [maxrT, minrT]), g.one])
    w_t = g.nd("Add", [g.nd("Sub", [maxcT, mincT]), g.one])

    _, maxrG = _rowspan(g, allsum)
    _, maxcG = _colspan(g, allsum)
    Hg = g.nd("Add", [maxrG, g.one])
    Wg = g.nd("Add", [maxcG, g.one])
    twoH = g.nd("Add", [Hg, Hg])
    twoW = g.nd("Add", [Wg, Wg])
    negH = g.nd("Sub", [g.nd("Sub", [Hg, Hg]), Hg])                 # -H
    negW = g.nd("Sub", [g.nd("Sub", [Wg, Wg]), Wg])

    # anchor template patch to the origin:  out[r,c] = in[r+minr, c+minc]
    Srow = _eqm(g, g.colidx, g.nd("Add", [g.rowidx, minrT]))
    Scol = _eqm(g, g.rowidx, g.nd("Add", [g.colidx, mincT]))
    Kr30 = g.nd("MatMul", [Srow, g.nd("MatMul", [g.nd("Mul", [X2, T]), Scol])])
    Kb30 = g.nd("MatMul", [Srow, g.nd("MatMul", [X1, Scol])])
    z2 = g.i64([0, 0])
    e2 = g.i64([4, 4])
    ax23 = g.i64([2, 3])
    Kr4 = g.nd("Slice", [Kr30, z2, e2, ax23])                       # [1,1,4,4]
    Kb4 = g.nd("Slice", [Kb30, z2, e2, ax23])

    stamps = []
    for k in range(1, 6):
        if k == 1:
            Krk, Kbk = Kr4, Kb4
        else:
            U = np.zeros((4 * k, 4))
            for i in range(4 * k):
                U[i, i // k] = 1.0
            Uk = g.f([4 * k, 4], U)
            UkT = g.f([4, 4 * k], U.T)
            Krk = g.nd("MatMul", [Uk, g.nd("MatMul", [Kr4, UkT])])  # [1,1,4k,4k]
            Kbk = g.nd("MatMul", [Uk, g.nd("MatMul", [Kb4, UkT])])
        nk = g.nd("ReduceSum", [Krk], axes=[2, 3], keepdims=1)
        Dk = g.nd("MaxPool", [Krk], kernel_shape=[3, 3], pads=[2, 2, 2, 2],
                  strides=[1, 1])                                   # [1,1,4k+2,4k+2]
        p1 = 4 * k - 1
        RC = g.nd("Conv", [X2, Krk], kernel_shape=[4 * k, 4 * k],
                  pads=[p1, p1, p1, p1])
        BC = g.nd("Conv", [NB, Kbk], kernel_shape=[4 * k, 4 * k],
                  pads=[p1, p1, p1, p1])
        DC = g.nd("Conv", [NB, Dk], kernel_shape=[4 * k + 2, 4 * k + 2],
                  pads=[4 * k, 4 * k, 4 * k, 4 * k])
        c1 = g.nd("Greater", [RC, g.nd("Sub", [nk, g.half])])
        c2 = g.nd("Less", [BC, g.half])
        c3 = g.nd("Less", [DC, g.nd("Add", [nk, g.half])])
        match = g.nd("Cast", [g.nd("And", [g.nd("And", [c1, c2]), c3])], to=F)
        # 3x-canvas bounds:  -H <= delta_r <= 2H - k*h_t   (delta = o - (4k-1))
        Sz = 29 + 4 * k
        oid_r = g.f([1, 1, Sz, 1], list(range(Sz)))
        oid_c = g.f([1, 1, 1, Sz], list(range(Sz)))
        p1c = g.f([1, 1, 1, 1], [float(p1)])
        dr = g.nd("Sub", [oid_r, p1c])
        dc = g.nd("Sub", [oid_c, p1c])
        kht = g.nd("Mul", [h_t, g.f([1, 1, 1, 1], [float(k)])])
        kwt = g.nd("Mul", [w_t, g.f([1, 1, 1, 1], [float(k)])])
        rm = g.nd("Mul", [
            _gt(g, dr, g.nd("Sub", [negH, g.half])),
            _lt(g, dr, g.nd("Add", [g.nd("Sub", [twoH, kht]), g.half]))])
        cm = g.nd("Mul", [
            _gt(g, dc, g.nd("Sub", [negW, g.half])),
            _lt(g, dc, g.nd("Add", [g.nd("Sub", [twoW, kwt]), g.half]))])
        match = g.nd("Mul", [g.nd("Mul", [match, rm]), cm])
        st = g.nd("ConvTranspose", [match, Kbk], kernel_shape=[4 * k, 4 * k],
                  pads=[p1, p1, p1, p1])                            # [1,1,30,30]
        stamps.append(st)

    ST = g.nd("Sum", stamps)
    SM = g.nd("Mul", [_gt(g, ST, g.half), allsum])
    E = g.f([1, 10, 1, 1], [-1.0, 1.0] + [0.0] * 8)
    g.nd("Add", ["input", g.nd("Mul", [SM, E])], "output")
    return _model(g, "vc1_101")


# =========================================================================== #
# task209 — 8a004b2b                                                          #
# =========================================================================== #
def _bb(m):
    ys, xs = np.where(m)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _np_209(a):
    a = np.asarray(a, int)
    Hd, Wd = a.shape
    if Hd > 30 or Wd > 30:
        return None
    A = np.zeros((30, 30), int)
    A[:Hd, :Wd] = a
    ys, xs = np.where(A == 4)
    if ys.size != 4:
        return None
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    if set(zip(ys.tolist(), xs.tolist())) != {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None
    NB = A > 0
    B = np.zeros((30, 30), bool)
    B[r0:r1 + 1, c0:c1 + 1] = True
    B2 = np.zeros((30, 30), bool)
    B2[r0 + 1:r1, c0 + 1:c1] = True
    Km = NB & ~B
    Im = NB & B2
    if not Km.any() or not Im.any():
        return None
    ic = np.array([int((Im & (A == c)).sum()) for c in range(10)])
    kc = np.array([int((Km & (A == c)).sum()) for c in range(10)])
    if ((ic > 0) & (kc == 0)).any():
        return None
    present = ic > 0
    kpos = kc > 0
    ratio = ic // np.where(kpos, kc, 1)
    # verifier uses mostcommon(ratios over the inner palette), NOT max:
    # x26 = mostcommon(x25) with tie-break = smallest ratio (set-of-ints order).
    rlist = [int(ratio[c]) for c in range(10) if present[c]]
    mode = max(set(rlist), key=rlist.count)
    relv = kpos & (ratio == mode)
    RI = Im & relv[A]
    RK = Km & relv[A]
    if not RI.any() or not RK.any():
        return None
    ir0, ir1, jc0, jc1 = _bb(RI)
    kr0, kr1, kk0, kk1 = _bb(RK)
    kh = (ir1 - ir0 + 1) // (kr1 - kr0 + 1)
    kw = (jc1 - jc0 + 1) // (kk1 - kk0 + 1)
    if kh < 1 or kw < 1:
        return None
    Kr0, Kr1, Kc0, Kc1 = _bb(Km)
    Kh, Kw = Kr1 - Kr0 + 1, Kc1 - Kc0 + 1
    # anchored + extent-masked key grid
    Kg = np.zeros((30, 30), int)
    src = A[Kr0:Kr0 + Kh, Kc0:Kc0 + Kw]
    Kg[:Kh, :Kw] = src
    ridx = np.arange(30)
    UP = Kg[np.minimum(ridx // kh, 29)][:, np.minimum(ridx // kw, 29)]
    UPnb = UP > 0
    uprel = UPnb & relv[UP]
    if not uprel.any():
        return None
    ur0, _, uc0, _ = _bb(uprel)
    ai0, _, aj0, _ = _bb(Im)                    # verifier: x49 = ulcorner(x15)
    sr, sc = ai0 - ur0, aj0 - uc0
    Tup = np.zeros((30, 30), int)
    lo_r, hi_r = max(0, sr), min(30, 30 + sr)
    lo_c, hi_c = max(0, sc), min(30, 30 + sc)
    Tup[lo_r:hi_r, lo_c:hi_c] = UP[lo_r - sr:hi_r - sr, lo_c - sc:hi_c - sc]
    painted = np.where(Tup > 0, Tup, A)
    return painted[r0:r1 + 1, c0:c1 + 1]


def _build_209():
    g = _G()
    _consts(g)
    V = _value_img(g)
    allsum = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    ch0 = _slice_ch(g, 0)
    NB = g.nd("Sub", [allsum, ch0])
    C4 = _slice_ch(g, 4)

    r0, r1 = _rowspan(g, C4)
    c0, c1 = _colspan(g, C4)
    rowin = g.nd("Mul", [_gt(g, g.rowidx, g.nd("Sub", [r0, g.half])),
                         _lt(g, g.rowidx, g.nd("Add", [r1, g.half]))])
    colin = g.nd("Mul", [_gt(g, g.colidx, g.nd("Sub", [c0, g.half])),
                         _lt(g, g.colidx, g.nd("Add", [c1, g.half]))])
    B = g.nd("Mul", [rowin, colin])
    rowin2 = g.nd("Mul", [_gt(g, g.rowidx, g.nd("Add", [r0, g.half])),
                          _lt(g, g.rowidx, g.nd("Sub", [r1, g.half]))])
    colin2 = g.nd("Mul", [_gt(g, g.colidx, g.nd("Add", [c0, g.half])),
                          _lt(g, g.colidx, g.nd("Sub", [c1, g.half]))])
    B2 = g.nd("Mul", [rowin2, colin2])
    Km = g.nd("Mul", [NB, g.nd("Sub", [g.one, B])])
    Im = g.nd("Mul", [NB, B2])

    ic = g.nd("ReduceSum", [g.nd("Mul", ["input", Im])], axes=[2, 3], keepdims=1)
    kc = g.nd("ReduceSum", [g.nd("Mul", ["input", Km])], axes=[2, 3], keepdims=1)
    presentf = _gt(g, ic, g.half)
    kposf = _gt(g, kc, g.half)
    safe = g.nd("Add", [kc, g.nd("Sub", [g.one, kposf])])
    ratio = g.nd("Floor", [g.nd("Div", [ic, safe])])                # [1,10,1,1]
    # mostcommon(ratio over the inner palette): score[c] = #present colors sharing
    # ratio[c]; among present colours on the max score, pick the winning ratio the
    # way CPython's max(set(.),key=count) breaks ties — set-of-small-ints bucket
    # order == priority (ratio&7 asc, then larger value).  Verified == mostcommon
    # on all 266 train+test+arc-gen.
    ratioT = g.nd("Transpose", [ratio], perm=[0, 2, 3, 1])           # [1,1,1,10]
    presentT = g.nd("Transpose", [presentf], perm=[0, 2, 3, 1])      # [1,1,1,10]
    eqpair = _eqm(g, ratio, ratioT)                                 # [1,10,1,10]
    score = g.nd("ReduceSum", [g.nd("Mul", [eqpair, presentT])], axes=[3], keepdims=1)
    maxsc = g.nd("ReduceMax", [g.nd("Mul", [score, presentf])], axes=[1], keepdims=1)
    onmax = g.nd("Mul", [presentf, _eqm(g, score, maxsc)])          # present & top score
    c8 = g.f([1, 1, 1, 1], [8.0])
    c100 = g.f([1, 1, 1, 1], [100.0])
    rmod8 = g.nd("Sub", [ratio, g.nd("Mul", [c8, g.nd("Floor", [g.nd("Div", [ratio, c8])])])])
    prio = g.nd("Sub", [g.nd("Mul", [rmod8, c100]), ratio])         # (ratio&7)*100 - ratio
    candP = g.nd("Add", [g.nd("Mul", [prio, onmax]),
                         g.nd("Mul", [g.cbig, g.nd("Sub", [g.one, onmax])])])
    minP = g.nd("ReduceMin", [candP], axes=[1], keepdims=1)
    winsel = g.nd("Mul", [onmax, _eqm(g, prio, minP)])              # winning-ratio colours
    mode = g.nd("ReduceMax", [g.nd("Mul", [ratio, winsel])], axes=[1], keepdims=1)
    rel = g.nd("Mul", [kposf, _eqm(g, ratio, mode)])                # [1,10,1,1]

    relcell = g.nd("ReduceSum", [g.nd("Mul", ["input", rel])], axes=[1], keepdims=1)
    RI = g.nd("Mul", [relcell, Im])
    RK = g.nd("Mul", [relcell, Km])
    ir0, ir1 = _rowspan(g, RI)
    jc0, jc1 = _colspan(g, RI)
    kr0, kr1 = _rowspan(g, RK)
    kk0, kk1 = _colspan(g, RK)
    hI = g.nd("Add", [g.nd("Sub", [ir1, ir0]), g.one])
    wI = g.nd("Add", [g.nd("Sub", [jc1, jc0]), g.one])
    hK = g.nd("Add", [g.nd("Sub", [kr1, kr0]), g.one])
    wK = g.nd("Add", [g.nd("Sub", [kk1, kk0]), g.one])
    kh = g.nd("Floor", [g.nd("Div", [hI, hK])])
    kw = g.nd("Floor", [g.nd("Div", [wI, wK])])

    Kr0, Kr1 = _rowspan(g, Km)
    Kc0, Kc1 = _colspan(g, Km)
    Kh = g.nd("Add", [g.nd("Sub", [Kr1, Kr0]), g.one])
    Kw = g.nd("Add", [g.nd("Sub", [Kc1, Kc0]), g.one])
    SrowK = _eqm(g, g.colidx, g.nd("Add", [g.rowidx, Kr0]))
    ScolK = _eqm(g, g.rowidx, g.nd("Add", [g.colidx, Kc0]))
    Kg0 = g.nd("MatMul", [SrowK, g.nd("MatMul", [V, ScolK])])
    ext = g.nd("Mul", [_lt(g, g.rowidx, g.nd("Sub", [Kh, g.half])),
                       _lt(g, g.colidx, g.nd("Sub", [Kw, g.half]))])
    Kg = g.nd("Mul", [Kg0, ext])

    # runtime upscale matrices:  U[r,i] = [i == floor(r/kh)]
    D1 = g.nd("Sub", [g.rowidx, g.nd("Mul", [g.colidx, kh])])
    Ukh = g.nd("Mul", [_gt(g, D1, g.nhalf), _lt(g, D1, g.nd("Sub", [kh, g.half]))])
    D2 = g.nd("Sub", [g.colidx, g.nd("Mul", [g.rowidx, kw])])
    UkwT = g.nd("Mul", [_gt(g, D2, g.nhalf), _lt(g, D2, g.nd("Sub", [kw, g.half]))])
    UP = g.nd("MatMul", [Ukh, g.nd("MatMul", [Kg, UkwT])])

    UPoh = _eqm(g, UP, g.f([1, 10, 1, 1], list(range(10))))         # [1,10,30,30]
    uprel = g.nd("ReduceSum", [g.nd("Mul", [UPoh, rel])], axes=[1], keepdims=1)
    ur0, _ = _rowspan(g, uprel)
    uc0, _ = _colspan(g, uprel)
    ai0, _ = _rowspan(g, Im)                    # verifier: x49 = ulcorner(x15)
    aj0, _ = _colspan(g, Im)
    sr = g.nd("Sub", [ai0, ur0])
    sc = g.nd("Sub", [aj0, uc0])
    Trow = _eqm(g, g.colidx, g.nd("Sub", [g.rowidx, sr]))           # [s == r - sr]
    Tcol = _eqm(g, g.rowidx, g.nd("Sub", [g.colidx, sc]))           # [s == c - sc]
    Tup = g.nd("MatMul", [Trow, g.nd("MatMul", [UP, Tcol])])
    Tnb = _gt(g, Tup, g.half)
    painted = g.nd("Add", [g.nd("Mul", [V, g.nd("Sub", [g.one, Tnb])]), Tup])

    bh = g.nd("Add", [g.nd("Sub", [r1, r0]), g.one])
    bw = g.nd("Add", [g.nd("Sub", [c1, c0]), g.one])
    Crow = g.nd("Mul", [_eqm(g, g.colidx, g.nd("Add", [g.rowidx, r0])),
                        _lt(g, g.rowidx, g.nd("Sub", [bh, g.half]))])
    Ccol = g.nd("Mul", [_eqm(g, g.rowidx, g.nd("Add", [g.colidx, c0])),
                        _lt(g, g.colidx, g.nd("Sub", [bw, g.half]))])
    OUTv = g.nd("MatMul", [Crow, g.nd("MatMul", [painted, Ccol])])
    OH = _eqm(g, OUTv, g.f([1, 10, 1, 1], list(range(10))))
    ingrid = g.nd("Mul", [_lt(g, g.rowidx, g.nd("Sub", [bh, g.half])),
                          _lt(g, g.colidx, g.nd("Sub", [bw, g.half]))])
    g.nd("Mul", [OH, ingrid], "output")
    return _model(g, "vc1_209")


# =========================================================================== #
# entry point                                                                 #
# =========================================================================== #
def _pairs(examples):
    out = []
    for s in ("train", "test"):
        for e in examples.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                return []
            out.append((a, b))
    return out


def _matches(prs, fn):
    for a, b in prs:
        try:
            o = fn(a)
        except Exception:
            return False
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(examples):
    prs = _pairs(examples)
    if not prs:
        return
    for name, ref, builder in (
        ("vc1_rays045e512c", _np_005, _build_005),
        ("vc1_stamp447fd412", _np_101, _build_101),
        ("vc1_upkey8a004b2b", _np_209, _build_209),
    ):
        if _matches(prs, ref):
            try:
                m = builder()
            except Exception:
                continue
            yield (name, m)
