"""family_sgolf_1 -- cheaper EXACT golf solvers via fixed-size CROP.

Technique (ANTI-OVERFIT SAFE): for tasks whose every train+test+arc-gen input AND
output is the same square SxS, the accepted full-30x30 graph is byte-identical when
run on the real SxS region.  We Slice the input to [1,10,S,S], run the same node
sequence with all shifts kept at SxS (so every intermediate is S*S* instead of
900* elements), then Pad the SxS result back to [1,10,30,30].  The algorithm and
step counts are UNCHANGED, so correctness is identical for any grid the generator
can produce at this fixed size; only the working resolution (hence intermediate
memory, hence cost) shrinks.

Each candidate is gated on the base family's numpy reference reproducing every
pair exactly AND all inputs/outputs being one fixed square, then validated EXACT by
the shared harness (train+test+arc-gen) before it can score.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT,
)

import family_crk4_5 as crk4_5
import family_crk4_4 as crk4_4
import family_crk4_1 as crk4_1
import family_crk2_3 as crk2_3
import family_crk3_4 as crk3_4

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE
G = HEIGHT  # 30


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

    def i64d(self, vals, dims):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, list(dims), [int(v) for v in vals]))
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


def _shift(g, t, dy, dx, S):
    """Shift content by (dy,dx): out[r,c]=t[r-dy,c-dx], zero filled, kept SxS."""
    h0, w0 = max(dy, 0), max(dx, 0)
    h1, w1 = max(-dy, 0), max(-dx, 0)
    pad = g.nd("Pad", [t], mode="constant", value=0.0,
               pads=[0, 0, h0, w0, 0, 0, h1, w1])
    return g.nd("Slice", [pad, g.i64([h1, w1]), g.i64([h1 + S, w1 + S]), g.i64([2, 3])])


def _crop_in(g, S):
    return g.nd("Slice", ["input", g.i64([0, 0]), g.i64([S, S]), g.i64([2, 3])])


def _pad_out(g, t, S, name="output"):
    p = G - S
    return g.nd("Pad", [t], name, mode="constant", value=0.0,
                pads=[0, 0, 0, 0, 0, 0, p, p])


# ----------------------------------------------------------------------------- #
# task397 -- 3-tails below 2x2 blocks (cropped)                                  #
# ----------------------------------------------------------------------------- #
def _build_397_crop(S):
    g = _G()
    X = _crop_in(g, S)
    e_nz = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    e3 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 3 else 0.0 for c in range(CHANNELS)])
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])

    appears = g.nd("Max", [X, _shift(g, X, 0, -1, S),
                           _shift(g, X, -1, 0, S), _shift(g, X, -1, -1, S)])
    k = g.nd("ReduceSum", [g.nd("Mul", [appears, e_nz])], axes=[1], keepdims=1)

    colored = g.nd("ReduceSum", [g.nd("Mul", [X, e_nz])], axes=[1], keepdims=1)
    top = _shift(g, colored, 1, 0, S)
    left = _shift(g, colored, 0, 1, S)
    TL = g.nd("Mul", [g.nd("Mul", [colored, g.nd("Sub", [one, top])]),
                      g.nd("Sub", [one, left])])
    kplane = g.nd("Mul", [TL, k])
    seed = g.nd("Add", [_shift(g, kplane, 2, 0, S), _shift(g, kplane, 2, 1, S)])

    fuel = seed
    for _ in range(5):
        fuel = g.nd("Max", [seed, g.nd("Sub", [_shift(g, fuel, 1, 0, S), one])])
    realmask = g.nd("ReduceSum", [X], axes=[1], keepdims=1)
    tail = g.nd("Mul", [g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [fuel, half])], to=F),
                                     realmask]),
                        g.nd("Sub", [one, colored])])

    out = g.nd("Add", [g.nd("Mul", [X, g.nd("Sub", [one, tail])]),
                       g.nd("Mul", [tail, e3])])
    _pad_out(g, out, S)
    return _model(g)


# ----------------------------------------------------------------------------- #
# task270 -- plus-markers (cropped MatMul-shift graph)                           #
# ----------------------------------------------------------------------------- #
def _plane_crop(g, X, ch):
    return g.nd("Slice", [X, g.i64([ch]), g.i64([ch + 1]), g.i64([1])])


def _build_270_crop(S):
    idx = np.arange(S)
    DIFF = idx[:, None] - idx[None, :]
    A = (DIFF >= 0).astype(np.float32)
    B = (DIFF <= 0).astype(np.float32)
    SP = (DIFF == 1).astype(np.float32)
    SM = (DIFF == -1).astype(np.float32)

    g = _G()
    X = _crop_in(g, S)
    c1 = _plane_crop(g, X, 1)
    c2 = _plane_crop(g, X, 2)
    c3 = _plane_crop(g, X, 3)
    c7 = _plane_crop(g, X, 7)
    Cc = g.nd("Concat", [c1, c2], axis=1)
    Mk = g.nd("Concat", [c7, c3], axis=1)

    Af = g.f([1, 1, S, S], A)
    Bf = g.f([1, 1, S, S], B)
    SPf = g.f([1, 1, S, S], SP)
    SMf = g.f([1, 1, S, S], SM)

    Cup = g.nd("MatMul", [SMf, Cc])
    Cdown = g.nd("MatMul", [SPf, Cc])
    Cleft = g.nd("MatMul", [Cc, SPf])
    Cright = g.nd("MatMul", [Cc, SMf])

    north = g.nd("Clip", [g.nd("MatMul", [Af, Mk])], min=0.0, max=1.0)
    south = g.nd("Clip", [g.nd("MatMul", [Bf, Mk])], min=0.0, max=1.0)
    west = g.nd("Clip", [g.nd("MatMul", [Mk, Bf])], min=0.0, max=1.0)
    east = g.nd("Clip", [g.nd("MatMul", [Mk, Af])], min=0.0, max=1.0)

    armN = g.nd("Mul", [Cup, north])
    armS = g.nd("Mul", [Cdown, south])
    armW = g.nd("Mul", [Cleft, west])
    armE = g.nd("Mul", [Cright, east])
    arm = g.nd("Add", [g.nd("Add", [armN, armS]), g.nd("Add", [armW, armE])])
    arm = g.nd("Clip", [arm], min=0.0, max=1.0)

    arm7 = g.nd("Slice", [arm, g.i64([0]), g.i64([1]), g.i64([1])])
    arm3 = g.nd("Slice", [arm, g.i64([1]), g.i64([2]), g.i64([1])])

    real = g.nd("ReduceSum", [X], axes=[1], keepdims=1)
    out0 = g.nd("Sub", [g.nd("Sub", [g.nd("Sub", [g.nd("Sub", [real, c1]), c2]), arm7]), arm3])
    z = g.nd("Sub", [c1, c1])

    res = g.nd("Concat", [out0, c1, c2, arm3, z, z, z, arm7, z, z], axis=1)
    _pad_out(g, res, S)
    return _model(g)


# ----------------------------------------------------------------------------- #
# task240 -- frame D4 symmetrisation (cropped)                                    #
# ----------------------------------------------------------------------------- #
def _build_240_crop(S):
    g = _G()
    X = _crop_in(g, S)
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    rowidx = g.f([1, 1, S, 1], list(range(S)))
    colidx = g.f([1, 1, 1, S], list(range(S)))
    P2vals = [[1.0 if (a - b) % 2 == 0 else 0.0 for b in range(S)] for a in range(S)]
    P2 = g.f([S, S], np.array(P2vals).ravel())

    inputC = g.nd("Slice", [X, g.i64([1]), g.i64([CHANNELS]), g.i64([1])])
    M = g.nd("ReduceMax", [X], axes=[1], keepdims=1)
    realrow = g.nd("ReduceMax", [M], axes=[3], keepdims=1)
    Ssum = g.nd("ReduceSum", [realrow], axes=[2], keepdims=1)
    N = g.nd("Sub", [Ssum, one])

    mr = g.nd("Min", [rowidx, g.nd("Sub", [N, rowidx])])
    mc = g.nd("Min", [colidx, g.nd("Sub", [N, colidx])])
    Bh = g.nd("Cast", [g.nd("Less", [mr, g.nd("Add", [mc, half])])], to=F)
    Bv = g.nd("Cast", [g.nd("Less", [mc, g.nd("Add", [mr, half])])], to=F)
    Hstrict = g.nd("Cast", [g.nd("Less", [mr, mc])], to=F)
    Vstrict = g.nd("Cast", [g.nd("Less", [mc, mr])], to=F)
    d1 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, colidx])]), half])], to=F)
    d2 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.nd("Add", [rowidx, colidx]), N])]), half])], to=F)
    diagmask = g.nd("Max", [d1, d2])
    notdiag = g.nd("Sub", [one, diagmask])

    edgecells = g.nd("Mul", [inputC, notdiag])
    corners = g.nd("Mul", [inputC, diagmask])
    Hedge = g.nd("Mul", [edgecells, Hstrict])
    Vedge = g.nd("Mul", [edgecells, Vstrict])
    Hdash = g.nd("Mul", [g.nd("Mul", [g.nd("MatMul", [Hedge, P2]), Bh]), notdiag])
    Vdash = g.nd("Mul", [g.nd("Mul", [g.nd("MatMul", [P2, Vedge]), Bv]), notdiag])
    canvas = g.nd("Max", [g.nd("Max", [corners, Hdash]), Vdash])

    R = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.nd("Add", [rowidx, colidx]), N])]), half])], to=F)
    lr = g.nd("MatMul", [canvas, R])
    ud = g.nd("MatMul", [R, canvas])
    rot = g.nd("MatMul", [R, lr])
    Tc = g.nd("Transpose", [canvas], perm=[0, 1, 3, 2])
    Tlr = g.nd("Transpose", [lr], perm=[0, 1, 3, 2])
    Tud = g.nd("Transpose", [ud], perm=[0, 1, 3, 2])
    Trot = g.nd("Transpose", [rot], perm=[0, 1, 3, 2])
    sym = canvas
    for o in [lr, ud, rot, Tc, Tlr, Tud, Trot]:
        sym = g.nd("Max", [sym, o])
    ink9 = g.nd("Mul", [sym, M])
    ink_any = g.nd("ReduceMax", [ink9], axes=[1], keepdims=1)
    ch0 = g.nd("Mul", [M, g.nd("Sub", [one, ink_any])])
    res = g.nd("Concat", [ch0, ink9], axis=1)
    _pad_out(g, res, S)
    return _model(g)


# ----------------------------------------------------------------------------- #
# task93 -- barstack fill (cropped)                                               #
# ----------------------------------------------------------------------------- #
_CBIG = 1000.0


def _nbg_mask_crop(g, X):
    realmask = g.nd("ReduceSum", [X], axes=[1], keepdims=1)
    ch0 = g.nd("Slice", [X, g.i64([0]), g.i64([1]), g.i64([1])])
    return g.nd("Sub", [realmask, ch0])


def _build_93_crop(S):
    g = _G()
    X = _crop_in(g, S)
    rowidx = g.f([1, 1, S, 1], list(range(S)))
    colidx = g.f([1, 1, 1, S], list(range(S)))
    half = g.f([1, 1, 1, 1], [0.5])
    cbig = g.f([1, 1, 1, 1], [_CBIG])
    oh0 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 0 else 0.0 for c in range(CHANNELS)])

    counts = g.nd("ReduceSum", [X], axes=[2, 3], keepdims=1)
    bgneg = g.f([1, CHANNELS, 1, 1], [-_CBIG * 1000] + [0.0] * (CHANNELS - 1))
    amax = g.nd("ArgMax", [g.nd("Add", [counts, bgneg])], axis=1, keepdims=1)
    idx = g.i64d(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    gate = g.nd("Cast", [g.nd("Equal", [amax, idx])], to=F)
    barmask = g.nd("ReduceSum", [g.nd("Mul", [X, gate])], axes=[1], keepdims=1)
    Mnb = _nbg_mask_crop(g, X)
    marker = g.nd("Sub", [Mnb, barmask])

    rowhas = g.nd("ReduceMax", [barmask], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [barmask], axes=[2], keepdims=1)
    rb = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    rt = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    cb = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    ct = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])

    def C(t):
        return g.nd("Cast", [t], to=F)
    gt = lambda a, b: C(g.nd("Greater", [a, g.nd("Add", [b, half])]))
    lt = lambda a, b: C(g.nd("Less", [a, g.nd("Sub", [b, half])]))
    inrows = g.nd("Mul", [C(g.nd("Greater", [rowidx, g.nd("Sub", [rt, half])])),
                          C(g.nd("Less", [rowidx, g.nd("Add", [rb, half])]))])
    incols = g.nd("Mul", [C(g.nd("Greater", [colidx, g.nd("Sub", [ct, half])])),
                          C(g.nd("Less", [colidx, g.nd("Add", [cb, half])]))])
    c_gt_cb = gt(colidx, cb); c_lt_ct = lt(colidx, ct)
    r_gt_rb = gt(rowidx, rb); r_lt_rt = lt(rowidx, rt)

    rightcount = g.nd("ReduceSum", [g.nd("Mul", [marker, c_gt_cb])], axes=[3], keepdims=1)
    leftcount = g.nd("ReduceSum", [g.nd("Mul", [marker, c_lt_ct])], axes=[3], keepdims=1)
    abovecount = g.nd("ReduceSum", [g.nd("Mul", [marker, r_lt_rt])], axes=[2], keepdims=1)
    belowcount = g.nd("ReduceSum", [g.nd("Mul", [marker, r_gt_rb])], axes=[2], keepdims=1)

    rightfill = g.nd("Mul", [g.nd("Mul", [c_gt_cb, inrows]),
                  C(g.nd("Less", [g.nd("Sub", [colidx, cb]), g.nd("Add", [rightcount, half])]))])
    leftfill = g.nd("Mul", [g.nd("Mul", [c_lt_ct, inrows]),
                  C(g.nd("Less", [g.nd("Sub", [ct, colidx]), g.nd("Add", [leftcount, half])]))])
    abovefill = g.nd("Mul", [g.nd("Mul", [r_lt_rt, incols]),
                  C(g.nd("Less", [g.nd("Sub", [rt, rowidx]), g.nd("Add", [abovecount, half])]))])
    belowfill = g.nd("Mul", [g.nd("Mul", [r_gt_rb, incols]),
                  C(g.nd("Less", [g.nd("Sub", [rowidx, rb]), g.nd("Add", [belowcount, half])]))])
    fill = g.nd("Max", [g.nd("Max", [rightfill, leftfill]), g.nd("Max", [abovefill, belowfill])])
    Bcells = g.nd("Clip", [g.nd("Add", [barmask, fill])], min=0.0, max=1.0)
    content = g.nd("Mul", [gate, Bcells])
    realmask = g.nd("ReduceSum", [X], axes=[1], keepdims=1)
    bg = g.nd("Mul", [oh0, g.nd("Sub", [realmask, Bcells])])
    res = g.nd("Add", [content, bg])
    _pad_out(g, res, S)
    return _model(g)


# ----------------------------------------------------------------------------- #
# task154 -- reflect 5s across nearest bar (cropped)                              #
# ----------------------------------------------------------------------------- #
def _mat_colL(n, S):
    M = np.zeros((S, S), np.float32)
    for c in range(S):
        if c + n < S:
            M[c + n, c] = 1
    return M


def _mat_rowU(n, S):
    B = np.zeros((S, S), np.float32)
    for r in range(S):
        if r + n < S:
            B[r, r + n] = 1
    return B


def _build_154_crop(S):
    g = _G()
    X = _crop_in(g, S)
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    two = g.f([1, 1, 1, 1], [2.0])
    big = g.f([1, 1, 1, 1], [1000.0])
    Jrow = g.f([1, 1, S, 1], list(range(S)))
    Jcol = g.f([1, 1, 1, S], list(range(S)))
    A1 = g.f([S, S], _mat_colL(1, S)); A2 = g.f([S, S], _mat_colL(2, S))
    B1 = g.f([S, S], _mat_rowU(1, S)); B2 = g.f([S, S], _mat_rowU(2, S))
    e0 = g.f([1, 10, 1, 1], [1 if k == 0 else 0 for k in range(10)])
    e2 = g.f([1, 10, 1, 1], [1 if k == 2 else 0 for k in range(10)])
    e5 = g.f([1, 10, 1, 1], [1 if k == 5 else 0 for k in range(10)])

    def ge(a, b):
        return g.nd("Cast", [g.nd("Greater", [a, b])], to=F)

    def lt_abs(a, thr):
        return g.nd("Cast", [g.nd("Less", [g.nd("Abs", [a]), thr])], to=F)

    def slc(src, lo, hi, axis):
        return g.nd("Slice", [src, g.i64([lo]), g.i64([hi]), g.i64([axis])])

    M = g.nd("ReduceSum", [X], axes=[1], keepdims=1)
    T2 = slc(X, 2, 3, 1)
    F5 = slc(X, 5, 6, 1)
    cL = lambda t, n: g.nd("MatMul", [t, A1 if n == 1 else A2])
    rU = lambda t, n: g.nd("MatMul", [B1 if n == 1 else B2, t])

    h3 = g.nd("Mul", [g.nd("Mul", [T2, cL(T2, 1)]), cL(T2, 2)])
    v3 = g.nd("Mul", [g.nd("Mul", [T2, rU(T2, 1)]), rU(T2, 2)])
    rowbar = g.nd("ReduceMax", [h3], axes=[3], keepdims=1)
    colbar = g.nd("ReduceMax", [v3], axes=[2], keepdims=1)
    sumrow = g.nd("ReduceSum", [rowbar], axes=[2], keepdims=1)
    sumcol = g.nd("ReduceSum", [colbar], axes=[3], keepdims=1)
    horiz = ge(g.nd("Sub", [sumrow, sumcol]), half)

    def bars(bar, J, ax):
        inv = g.nd("Sub", [one, bar])
        p1 = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [J, bar]),
                  g.nd("Mul", [big, inv])])], axes=[ax], keepdims=1)
        p2 = g.nd("ReduceMax", [g.nd("Mul", [J, bar])], axes=[ax], keepdims=1)
        return p1, p2

    def target(J, p1, p2):
        d1 = g.nd("Abs", [g.nd("Sub", [J, p1])])
        d2 = g.nd("Abs", [g.nd("Sub", [J, p2])])
        sideA = g.nd("Cast", [g.nd("Less", [d1, g.nd("Add", [d2, half])])], to=F)
        t1 = g.nd("Sub", [g.nd("Mul", [two, p1]), J])
        t2 = g.nd("Sub", [g.nd("Mul", [two, p2]), J])
        return g.nd("Add", [g.nd("Mul", [sideA, t1]),
                            g.nd("Mul", [g.nd("Sub", [one, sideA]), t2])])

    p1, p2 = bars(rowbar, Jrow, 2)
    tr = target(Jrow, p1, p2)
    trT = g.nd("Transpose", [tr], perm=[0, 1, 3, 2])
    Tr = lt_abs(g.nd("Sub", [Jrow, trT]), half)
    reflH = g.nd("MatMul", [Tr, F5])
    p1c, p2c = bars(colbar, Jcol, 3)
    tc = target(Jcol, p1c, p2c)
    tcT = g.nd("Transpose", [tc], perm=[0, 1, 3, 2])
    Tc = lt_abs(g.nd("Sub", [tcT, Jcol]), half)
    reflV = g.nd("MatMul", [F5, Tc])

    refl = g.nd("Add", [g.nd("Mul", [horiz, reflH]),
                        g.nd("Mul", [g.nd("Sub", [one, horiz]), reflV])])
    out5 = g.nd("Mul", [refl, M])
    out0 = g.nd("Sub", [g.nd("Sub", [M, T2]), out5])
    res = g.nd("Add", [g.nd("Add", [g.nd("Mul", [out0, e0]), g.nd("Mul", [T2, e2])]),
                       g.nd("Mul", [out5, e5])])
    _pad_out(g, res, S)
    return _model(g)


# ----------------------------------------------------------------------------- #
# generic detection helpers                                                      #
# ----------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int); b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            out.append((a, b))
    return out


def _fixed_square(prs):
    """Return S if every input and output is the same SxS square, else None."""
    shapes = {a.shape for a, _ in prs} | {b.shape for _, b in prs}
    if len(shapes) != 1:
        return None
    (h, w), = shapes
    if h != w or not (1 <= h <= G):
        return None
    return h


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    # task397: 3-tails below 2x2 blocks
    S = _fixed_square(prs)
    if S is not None and all(a.shape == b.shape and
                             np.array_equal(crk4_5._solve_397(a), b) for a, b in prs):
        try:
            m = _build_397_crop(S)
            onnx.checker.check_model(m, full_check=True)
            out.append(("sgolf_397_crop", m))
        except Exception:
            pass

    # task270: plus-markers, centres {1,2} markers {3,7}
    if S is not None:
        incol = set(); outcol = set()
        for a, b in prs:
            incol |= set(np.unique(a).tolist()); outcol |= set(np.unique(b).tolist())
        if incol <= {0, 1, 2, 3, 7} and outcol <= {0, 1, 2, 3, 7} and \
                all(a.shape == b.shape and np.array_equal(crk4_4._ref_270(a), b) for a, b in prs):
            try:
                m = _build_270_crop(S)
                onnx.checker.check_model(m, full_check=True)
                out.append(("sgolf_270_crop", m))
            except Exception:
                pass

    # task240: frame D4 symmetrisation
    if S is not None and S >= 3 and all(
            a.shape == b.shape and crk4_1._ref_frame(a) is not None
            and np.array_equal(crk4_1._ref_frame(a), b) for a, b in prs):
        try:
            m = _build_240_crop(S)
            onnx.checker.check_model(m, full_check=True)
            out.append(("sgolf_240_crop", m))
        except Exception:
            pass

    # task93: barstack fill
    if S is not None and all(
            a.shape == b.shape and crk2_3.ref_barstack(a) is not None
            and np.array_equal(crk2_3.ref_barstack(a), b) for a, b in prs):
        try:
            m = _build_93_crop(S)
            onnx.checker.check_model(m, full_check=True)
            out.append(("sgolf_93_crop", m))
        except Exception:
            pass

    # task154: reflect 5s across nearest bar
    if S is not None and crk3_4._refl_detect(prs):
        try:
            m = _build_154_crop(S)
            onnx.checker.check_model(m, full_check=True)
            out.append(("sgolf_154_crop", m))
        except Exception:
            pass

    return out
