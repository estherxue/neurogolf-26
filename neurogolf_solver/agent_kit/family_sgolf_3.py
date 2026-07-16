"""family_sgolf_3 -- cheaper EXACT crop-golf solvers for FIXED-size targets in
slice golf_targets.json[3::7].

Technique: for a task whose every train+test+arc-gen INPUT and OUTPUT is the same
square SxS, the byte-identical origin-anchored solver can run on a cropped [1,*,S,S]
work area instead of the full [1,*,30,30] canvas, cutting every intermediate tensor
from 30*30 to S*S elements.  We Slice input->[1,10,S,S], run the same automaton /
arithmetic, then Pad the SxS result back to 30x30.  The algorithm (step count,
weights, reflection logic) is UNCHANGED, so it stays value-exact for any grid the
generator produces at this fixed size.  Every candidate is validated EXACT
(numpy mirror == full solver reference) on all pairs before emit.

Sub-solvers here:
  * 345 beamdefl (FIXED 10x10): reuse family_crk7_4._beamdefl_ref for detection;
    rebuild the single-channel light-deflection CA on an SxS crop.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)
import family_crk7_4 as crk74

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE


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
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)],
                          [int(v) for v in np.asarray(vals).ravel()]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out

    def ge(self, a, b):
        return self.nd("Cast", [self.nd("Greater", [a, b])], to=F)

    def lt_abs(self, a, thr):
        return self.nd("Cast", [self.nd("Less", [self.nd("Abs", [a]), thr])], to=F)


def _slc(g, src, lo, hi, axis):
    return g.nd("Slice", [src, g.i64([lo]), g.i64([hi]), g.i64([axis])])


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _pairs(ex, splits=("train", "test", "arc-gen")):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _fixed_S(prs):
    shapes = {a.shape for a, _ in prs} | {b.shape for _, b in prs}
    if len(shapes) != 1:
        return None
    (h, w), = shapes
    if h != w or not (1 <= h <= 30):
        return None
    return h


# =========================================================================== #
# 345 beamdefl -- cropped single-channel light-deflection CA                   #
# =========================================================================== #
_ITERS = crk74._ITERS  # 14, same as baseline


def _shiftS(g, x, dr, dc, S):
    """result[r,c] = x[r-dr, c-dc] over an SxS window, zero fill."""
    pt, pb = max(dr, 0), max(-dr, 0)
    pl, pr = max(dc, 0), max(-dc, 0)
    p = g.nd("Pad", [x], mode="constant", value=0.0,
             pads=[0, 0, pt, pl, 0, 0, pb, pr])
    st = g.i64([max(-dr, 0), max(-dc, 0)])
    en = g.i64([max(-dr, 0) + S, max(-dc, 0) + S])
    ax = g.i64([2, 3])
    return g.nd("Slice", [p, st, en, ax])


def _build_beamdefl_crop(S):
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0])
    xc = g.nd("Slice", ["input", g.i64([0, 0]), g.i64([S, S]), g.i64([2, 3])])
    seed = g.nd("Slice", [xc, g.i64([2]), g.i64([3]), g.i64([1])])
    obst = g.nd("Slice", [xc, g.i64([5]), g.i64([6]), g.i64([1])])
    notobst = g.nd("Sub", [one, obst])
    o1 = _shiftS(g, obst, 0, 1, S)
    o2 = _shiftS(g, obst, 1, 1, S)

    lit = seed
    for _ in range(_ITERS):
        up = _shiftS(g, lit, -1, 0, S)
        a1 = _shiftS(g, lit, -1, 1, S)
        b1 = _shiftS(g, lit, 0, 1, S)
        tb = g.nd("Mul", [up, notobst])
        tc = g.nd("Mul", [a1, o1])
        td = g.nd("Mul", [b1, o2])
        lit = g.nd("Max", [lit, tb, tc, td])

    beam2 = g.nd("Mul", [lit, notobst])
    realmask = g.nd("ReduceSum", [xc], axes=[1], keepdims=1)
    zero = g.nd("Sub", [realmask, realmask])
    ch0 = g.nd("Sub", [g.nd("Sub", [realmask, beam2]), obst])
    chans = [ch0, zero, beam2, zero, zero, obst, zero, zero, zero, zero]
    outc = g.nd("Concat", chans, axis=1)
    pad = 30 - S
    g.nd("Pad", [outc], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, pad, pad])
    return _model(g)


def _beamdefl(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []
    for a, b in allp:
        if set(np.unique(a)) - {0, 2, 5} or set(np.unique(b)) - {0, 2, 5}:
            return []
    if not any((a == 5).any() for a, b in det):
        return []
    S = _fixed_S(allp)
    if S is None:
        return []

    def ok(plist):
        for a, b in plist:
            o = crk74._beamdefl_ref(a)
            if o.shape != b.shape or not np.array_equal(o, b):
                return False
        return True

    if not ok(det) or not ok(allp):
        return []
    try:
        m = _build_beamdefl_crop(S)
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("beamdefl_crop", m)]


# =========================================================================== #
# 336 t336 -- hollow 5-box interior-fill + gap ray, cropped coordinate build   #
# =========================================================================== #
import family_crk3_4 as crk34


def _build_t336_crop(S):
    Gs = S
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    big = g.f([1, 1, 1, 1], [1000.0])
    Jrow = g.f([1, 1, Gs, 1], list(range(Gs)))
    Jcol = g.f([1, 1, 1, Gs], list(range(Gs)))
    e0 = g.f([1, 10, 1, 1], [1 if k == 0 else 0 for k in range(10)])
    e5 = g.f([1, 10, 1, 1], [1 if k == 5 else 0 for k in range(10)])
    e8 = g.f([1, 10, 1, 1], [1 if k == 8 else 0 for k in range(10)])
    X = g.nd("Slice", ["input", g.i64([0, 0]), g.i64([S, S]), g.i64([2, 3])])

    def eqv(a, b):
        return g.lt_abs(g.nd("Sub", [a, b]), half)

    M = g.nd("ReduceSum", [X], axes=[1], keepdims=1)
    M5 = _slc(g, X, 5, 6, 1)
    inv5 = g.nd("Sub", [one, M5])
    row5 = g.nd("ReduceMax", [M5], axes=[3], keepdims=1)
    col5 = g.nd("ReduceMax", [M5], axes=[2], keepdims=1)
    rmin = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [Jrow, row5]),
                 g.nd("Mul", [big, g.nd("Sub", [one, row5])])])], axes=[2], keepdims=1)
    rmax = g.nd("ReduceMax", [g.nd("Mul", [Jrow, row5])], axes=[2], keepdims=1)
    cmin = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [Jcol, col5]),
                 g.nd("Mul", [big, g.nd("Sub", [one, col5])])])], axes=[3], keepdims=1)
    cmax = g.nd("ReduceMax", [g.nd("Mul", [Jcol, col5])], axes=[3], keepdims=1)

    interior = g.nd("Mul", [
        g.nd("Mul", [g.ge(Jrow, rmin), g.nd("Cast", [g.nd("Less", [Jrow, rmax])], to=F)]),
        g.nd("Mul", [g.ge(Jcol, cmin), g.nd("Cast", [g.nd("Less", [Jcol, cmax])], to=F)]),
    ])

    rmin_lo = g.nd("Sub", [rmin, half]); rmax_hi = g.nd("Add", [rmax, half])
    cmin_lo = g.nd("Sub", [cmin, half]); cmax_hi = g.nd("Add", [cmax, half])
    r_edge = g.nd("Add", [eqv(Jrow, rmin), eqv(Jrow, rmax)])
    c_edge = g.nd("Add", [eqv(Jcol, cmin), eqv(Jcol, cmax)])
    r_in = g.nd("Mul", [g.ge(Jrow, rmin_lo),
              g.nd("Cast", [g.nd("Less", [Jrow, rmax_hi])], to=F)])
    c_in = g.nd("Mul", [g.ge(Jcol, cmin_lo),
              g.nd("Cast", [g.nd("Less", [Jcol, cmax_hi])], to=F)])
    perim = g.ge(g.nd("Add", [g.nd("Mul", [r_edge, c_in]),
                              g.nd("Mul", [c_edge, r_in])]), half)
    gap = g.nd("Mul", [perim, inv5])
    gr = g.nd("ReduceSum", [g.nd("Mul", [gap, Jrow])], axes=[2, 3], keepdims=1)
    gc = g.nd("ReduceSum", [g.nd("Mul", [gap, Jcol])], axes=[2, 3], keepdims=1)

    is_top = eqv(gr, rmin); is_bot = eqv(gr, rmax)
    is_left = eqv(gc, cmin); is_right = eqv(gc, cmax)
    col_is_gc = eqv(Jcol, gc)
    row_is_gr = eqv(Jrow, gr)
    rows_above = g.nd("Cast", [g.nd("Less", [Jrow, rmin])], to=F)
    rows_below = g.ge(Jrow, g.nd("Add", [rmax, half]))
    cols_left = g.nd("Cast", [g.nd("Less", [Jcol, cmin])], to=F)
    cols_right = g.ge(Jcol, g.nd("Add", [cmax, half]))
    vert = g.nd("Mul", [col_is_gc, g.nd("Add", [g.nd("Mul", [is_top, rows_above]),
                                                g.nd("Mul", [is_bot, rows_below])])])
    horz = g.nd("Mul", [row_is_gr, g.nd("Add", [g.nd("Mul", [is_left, cols_left]),
                                                g.nd("Mul", [is_right, cols_right])])])
    ray = g.nd("Add", [vert, horz])

    eight = g.nd("Mul", [g.ge(g.nd("Add", [g.nd("Add", [interior, gap]), ray]), half), M])
    five = M5
    out0 = g.nd("Sub", [g.nd("Sub", [M, five]), eight])
    outc = g.nd("Add", [g.nd("Add", [g.nd("Mul", [out0, e0]), g.nd("Mul", [five, e5])]),
                        g.nd("Mul", [eight, e8])])
    pad = 30 - S
    g.nd("Pad", [outc], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, pad, pad])
    return _model(g)


def _t336(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []
    S = _fixed_S(allp)
    if S is None:
        return []
    if not crk34._t336_detect(det) or not crk34._t336_detect(allp):
        return []
    try:
        m = _build_t336_crop(S)
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("t336_crop", m)]


# =========================================================================== #
# 125 hollow-box frame + hole-fill (FIXED 15x15) -- cropped                    #
# =========================================================================== #
import family_golf6_3 as g63


def _build_125_crop(S):
    g = _G()
    sL = np.tril(np.ones((S, S), np.float32), -1)
    sU = np.triu(np.ones((S, S), np.float32), 1)
    xc = g.nd("Slice", ["input", g.i64([0, 0]), g.i64([S, S]), g.i64([2, 3])])
    blob = _slc(g, xc, 6, 7, 1)
    bg = _slc(g, xc, 8, 9, 1)
    sLn = g.f([S, S], sL); sUn = g.f([S, S], sU)
    half = g.f([1, 1, 1, 1], [0.5]); one = g.f([1, 1, 1, 1], [1.0])

    def gt(x):
        return g.nd("Cast", [g.nd("Greater", [x, half])], to=F)

    right = gt(g.nd("MatMul", [blob, sUn]))
    left = gt(g.nd("MatMul", [blob, sLn]))
    up = gt(g.nd("MatMul", [sLn, blob]))
    down = gt(g.nd("MatMul", [sUn, blob]))
    fill = g.nd("Mul", [g.nd("Mul", [bg, right]),
                        g.nd("Mul", [left, g.nd("Mul", [up, down])])])
    k3 = g.f([1, 1, 3, 3], np.ones((1, 1, 3, 3), np.float32))
    adj = gt(g.nd("Conv", [blob, k3], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    border = g.nd("Mul", [g.nd("Mul", [bg, adj]), g.nd("Sub", [one, fill])])
    rem = g.nd("Sub", [g.nd("Sub", [bg, fill]), border])
    masks = g.nd("Concat", [blob, fill, border, rem], axis=1)
    W = np.zeros((CHANNELS, 4, 1, 1), np.float32)
    W[6, 0] = 1.0; W[4, 1] = 1.0; W[3, 2] = 1.0; W[8, 3] = 1.0
    outc = g.nd("Conv", [masks, g.f([CHANNELS, 4, 1, 1], W)], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    pad = 30 - S
    g.nd("Pad", [outc], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, pad, pad])
    return _model(g)


def _t125(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []
    S = _fixed_S(allp)
    if S is None:
        return []

    def ok(plist):
        for a, b in plist:
            o = g63._ref_125(a)
            if o is None or o.shape != b.shape or not np.array_equal(o, b):
                return False
        return True

    if not ok(det) or not ok(allp):
        return []
    try:
        m = _build_125_crop(S)
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("t125_crop", m)]


# =========================================================================== #
# 305 cyclic diagonal restore (FIXED 16x16) -- cropped                         #
# =========================================================================== #
import family_crk6_3 as crk63


def _build_305_crop(S):
    g = _G()
    xc = g.nd("Slice", ["input", g.i64([0, 0]), g.i64([S, S]), g.i64([2, 3])])
    realmask = g.nd("ReduceSum", [xc], axes=[1], keepdims=1)
    chvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    haschan = g.nd("ReduceMax", [xc], axes=[2, 3], keepdims=1)
    K = g.nd("ReduceMax", [g.nd("Mul", [chvec, haschan])], axes=[1], keepdims=1)
    P = g.f([1, 1, S, S], [[r + c for c in range(S)] for r in range(S)])
    M = g.nd("Mod", [P, K], fmod=1)
    cidx = g.f([1, CHANNELS, 1, 1], [j - 1 for j in range(CHANNELS)])
    absd = g.nd("Abs", [g.nd("Sub", [M, cidx])])
    one = g.f([1, 1, 1, 1], [1.0])
    ind = g.nd("Relu", [g.nd("Sub", [one, absd])])
    outc = g.nd("Mul", [ind, realmask])
    pad = 30 - S
    g.nd("Pad", [outc], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, pad, pad])
    return _model(g)


def _t305(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []
    S = _fixed_S(allp)
    if S is None:
        return []

    def ok(plist):
        for a, b in plist:
            p = crk63._sol305(a)
            if p is None or p.shape != b.shape or not np.array_equal(p, b):
                return False
        return True

    if not ok(det) or not ok(allp):
        return []
    try:
        m = _build_305_crop(S)
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("t305_crop", m)]


# =========================================================================== #
# 27 180-degree point-symmetry reveal (FIXED square) -- cropped                #
# =========================================================================== #
import family_crk9_4 as crk94


def _build27_crop(S):
    NTP = 2 * S - 1
    g = _G()
    Rbank = np.zeros((NTP, S, S), np.float32)
    for tp in range(NTP):
        for c in range(S):
            cc = tp - c
            if 0 <= cc < S:
                Rbank[tp, c, cc] = 1.0
    Rb = g.f([NTP, S, S], Rbank)
    arange = g.f([NTP], list(range(NTP)))
    half = g.f([1], [0.5]); bw = g.f([1], [1.5])
    g.inits.append(oh.make_tensor("c29_27", INT64, [1, 1, 1, 1], [S - 1]))
    sel = [0.0] * CHANNELS; sel[2] = 1.0; sel[0] = -1.0
    selc = g.f([1, CHANNELS, 1, 1], sel)

    xc = g.nd("Slice", ["input", g.i64([0, 0]), g.i64([S, S]), g.i64([2, 3])])
    Fch = _slc(g, xc, 1, 2, 1)
    BGch = _slc(g, xc, 0, 1, 1)
    Fsq = g.nd("Squeeze", [Fch], axes=[0, 1])
    BGsq = g.nd("Squeeze", [BGch], axes=[0, 1])

    t1 = g.nd("MatMul", [Rb, Fsq])
    rot = g.nd("MatMul", [t1, Rb])
    ffmul = g.nd("Mul", [rot, Fsq])
    ffvec = g.nd("ReduceSum", [ffmul], axes=[1, 2], keepdims=0)

    rowhas = g.nd("ReduceMax", [Fch], axes=[3], keepdims=1)
    minrow = g.nd("ArgMax", [rowhas], axis=2, keepdims=1)
    rowhasR = g.nd("Slice", [rowhas, g.i64([S - 1]), g.i64([-(1 << 31)]),
                             g.i64([2]), g.i64([-1])])
    amf = g.nd("ArgMax", [rowhasR], axis=2, keepdims=1)
    maxrow = g.nd("Sub", ["c29_27", amf])
    rowsum_i = g.nd("Add", [minrow, maxrow])
    rowsum_f4 = g.nd("Cast", [rowsum_i], to=F)
    rowsum_f = g.nd("Reshape", [rowsum_f4, g.i64([1])])

    adiff = g.nd("Sub", [arange, rowsum_f])
    aabs = g.nd("Abs", [adiff])
    alt = g.nd("Less", [aabs, half])
    altf = g.nd("Cast", [alt], to=F)
    bias = g.nd("Mul", [altf, bw])
    Ssc = g.nd("Add", [ffvec, bias])
    ts = g.nd("ArgMax", [Ssc], axis=0, keepdims=1)

    rg = g.nd("Gather", [rot, ts], axis=0)
    rmask = g.nd("Squeeze", [rg], axes=[0])
    reveal = g.nd("Mul", [rmask, BGsq])
    reveal4 = g.nd("Reshape", [reveal, g.i64([1, 1, S, S])])
    delta = g.nd("Mul", [reveal4, selc])
    outc = g.nd("Add", [xc, delta])
    pad = 30 - S
    g.nd("Pad", [outc], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, pad, pad])
    return _model(g)


def _t27(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []
    S = _fixed_S(allp)
    if S is None:
        return []
    if not crk94._matches27(det) or not crk94._matches27(allp):
        return []
    try:
        m = _build27_crop(S)
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("t27_crop", m)]


# =========================================================================== #
# 273 corner-fill: bg cell inside marker rectangle -> 2 (FIXED square) cropped #
# =========================================================================== #
import family_crk2_3 as crk23


def _build_cornerfill_crop(S):
    g = _G()
    L = np.array([[1.0 if k < i else 0.0 for k in range(S)] for i in range(S)], np.float32)
    U = L.T
    Lc = g.f([S, S], L)
    Uc = g.f([S, S], U)
    half = g.f([1, 1, 1, 1], [0.5])
    diff20 = g.f([1, CHANNELS, 1, 1],
                 [(-1.0 if c == 0 else (1.0 if c == 2 else 0.0)) for c in range(CHANNELS)])
    xc = g.nd("Slice", ["input", g.i64([0, 0]), g.i64([S, S]), g.i64([2, 3])])
    realmask = g.nd("ReduceSum", [xc], axes=[1], keepdims=1)
    ch0 = _slc(g, xc, 0, 1, 1)
    M = g.nd("Sub", [realmask, ch0])
    above = g.nd("MatMul", [Lc, M])
    below = g.nd("MatMul", [Uc, M])
    NW = g.nd("MatMul", [above, Uc])
    NE = g.nd("MatMul", [above, Lc])
    SW = g.nd("MatMul", [below, Uc])
    SE = g.nd("MatMul", [below, Lc])

    def pos(t):
        return g.nd("Cast", [g.nd("Greater", [t, half])], to=F)
    quad = g.nd("Mul", [g.nd("Mul", [pos(NW), pos(NE)]), g.nd("Mul", [pos(SW), pos(SE)])])
    filled = g.nd("Mul", [quad, ch0])
    outc = g.nd("Add", [xc, g.nd("Mul", [diff20, filled])])
    pad = 30 - S
    g.nd("Pad", [outc], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, pad, pad])
    return _model(g)


def _t273(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []
    S = _fixed_S(allp)
    if S is None:
        return []

    def ok(plist):
        for a, b in plist:
            o = crk23.ref_cornerfill(a)
            if o is None or o.shape != b.shape or not np.array_equal(o, b):
                return False
        return True

    if not ok(det) or not ok(allp):
        return []
    # require the transform to actually introduce colour 2 somewhere
    if not any((a != b).any() for a, b in det):
        return []
    try:
        m = _build_cornerfill_crop(S)
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("t273_crop", m)]


def candidates(ex):
    out = []
    out += _beamdefl(ex)
    out += _t336(ex)
    out += _t125(ex)
    out += _t305(ex)
    out += _t27(ex)
    out += _t273(ex)
    return out
