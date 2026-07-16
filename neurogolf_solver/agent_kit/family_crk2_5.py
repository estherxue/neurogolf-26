"""family_crk2_5 -- a grab-bag of exact, structurally-detected ARC->ONNX solvers
for previously-unsolved NeuroGolf 2026 tasks (slice IDX=5).

Each sub-family detects its rule by mirroring the transform in numpy and requiring
an EXACT reproduction of EVERY provided (train+test+arc-gen) pair before emitting an
opset-10 ONNX model.  Several rules need DATA-DEPENDENT positioning (the grid size /
object position varies per example); those are realised with the standard
"computed index-grid mask (+ MatMul shift)" trick so the graph keeps static shapes.

Tasks covered
-------------
  halve   (task 188) : input is exactly two identical halves (left|right OR top|bottom);
                       output = one half.  The doubled axis is detected per-input and
                       the truncation is a data-dependent column/row mask (W/2 or H/2).
  altfill (task 232) : each isolated coloured dot extends rightward along its row with
                       an alternating [dotcolour, 5(gray)] stripe to the grid edge.
  quad    (task 342) : a 2x2 block (the most-frequent colour) is recoloured cell-by-cell
                       with the four single dots placed in the four diagonal quadrants.
  border  (task  40) : two parallel full border lines (left|right cols OR top|bottom
                       rows); every interior marker takes the colour of the NEARER line.
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


def _common(g):
    g.rowidx = g.f([1, 1, H, 1], list(range(H)))     # [1,1,30,1]
    g.colidx = g.f([1, 1, 1, W], list(range(W)))     # [1,1,1,30]
    g.half = g.f([1, 1, 1, 1], [0.5])
    g.one = g.f([1, 1, 1, 1], [1.0])
    g.colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    g.e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))


def _realmask(g):
    return g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)   # [1,1,30,30]


def _eq(g, a, b):
    return g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [a, b])]), g.half])], to=F)


def _lt(g, a, b):
    return g.nd("Cast", [g.nd("Less", [a, b])], to=F)


def _gt(g, a, b):
    return g.nd("Cast", [g.nd("Greater", [a, b])], to=F)


# =========================================================================== #
# task 188 -- halve the doubled axis                                           #
# =========================================================================== #
def _build_halve():
    g = _G(); _common(g)
    rm = _realmask(g)
    colhas = g.nd("ReduceMax", [rm], axes=[2], keepdims=1)             # [1,1,1,30]
    rowhas = g.nd("ReduceMax", [rm], axes=[3], keepdims=1)             # [1,1,30,1]
    Wt = g.nd("ReduceSum", [colhas], axes=[3], keepdims=1)            # [1,1,1,1]
    Ht = g.nd("ReduceSum", [rowhas], axes=[2], keepdims=1)            # [1,1,1,1]
    Whalf = g.nd("Mul", [Wt, g.half])
    Hhalf = g.nd("Mul", [Ht, g.half])

    color = g.nd("ReduceSum", [g.nd("Mul", ["input", g.colvec])], axes=[1], keepdims=1)
    # shift colour plane left by Whalf via MatMul selection
    t1 = g.nd("Sub", [g.colidx, g.rowidx])                            # [1,1,30,30] = j-i
    s_neg = g.nd("Cast", [g.nd("Less",
              [g.nd("Abs", [g.nd("Add", [t1, Whalf])]), g.half])], to=F)   # j==i-Whalf
    cleft = g.nd("MatMul", [color, s_neg])                            # color[r,c+Whalf]
    leftcol = _lt(g, g.colidx, Whalf)                                 # [1,1,1,30]
    mism = g.nd("Cast", [g.nd("Greater",
              [g.nd("Abs", [g.nd("Sub", [color, cleft])]), g.half])], to=F)
    mc = g.nd("ReduceSum", [g.nd("Mul", [g.nd("Mul", [mism, leftcol]), rm])],
              axes=[2, 3], keepdims=1)                                # [1,1,1,1]
    # Whalf must be integer (W even)
    wfloor = g.nd("Cast", [g.nd("Cast", [Whalf], to=INT64)], to=F)
    iseven = g.nd("Cast", [g.nd("Less",
              [g.nd("Abs", [g.nd("Sub", [Whalf, wfloor])]), g.f([1, 1, 1, 1], [0.25])])], to=F)
    g_h = g.nd("Mul", [iseven, g.nd("Cast", [g.nd("Less", [mc, g.half])], to=F)])

    Hout = g.nd("Mul", ["input", leftcol])
    Vout = g.nd("Mul", ["input", _lt(g, g.rowidx, Hhalf)])
    g.nd("Add", [g.nd("Mul", [Hout, g_h]),
                 g.nd("Mul", [Vout, g.nd("Sub", [g.one, g_h])])], "output")
    return _model(g)


def _ref_halve(a):
    Hh, Ww = a.shape
    if Ww % 2 == 0 and np.array_equal(a[:, :Ww // 2], a[:, Ww // 2:]):
        return a[:, :Ww // 2]
    if Hh % 2 == 0 and np.array_equal(a[:Hh // 2, :], a[Hh // 2:, :]):
        return a[:Hh // 2, :]
    return None


# =========================================================================== #
# task 232 -- alternating rightward stripe from each dot (alt colour = 5)      #
# =========================================================================== #
_ALT = 5


def _build_altfill():
    g = _G(); _common(g)
    rm = _realmask(g)
    masknb = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    Xnb = g.nd("Mul", ["input", masknb])                             # bg channel zeroed
    nbmask = g.nd("ReduceSum", [Xnb], axes=[1], keepdims=1)          # [1,1,30,30]
    hasdot = g.nd("ReduceMax", [nbmask], axes=[3], keepdims=1)       # [1,1,30,1]
    dotcol = g.nd("ReduceSum", [g.nd("Mul", [nbmask, g.colidx])], axes=[3], keepdims=1)
    dotOH = g.nd("ReduceMax", [Xnb], axes=[3], keepdims=1)           # [1,10,30,1]

    ge = g.nd("Cast", [g.nd("Less", [g.nd("Sub", [dotcol, g.colidx]), g.half])], to=F)
    active = g.nd("Mul", [g.nd("Mul", [ge, hasdot]), rm])            # [1,1,30,30]

    colpar = g.f([1, 1, 1, W], [c % 2 for c in range(W)])
    dchalf = g.nd("Mul", [dotcol, g.half])
    dcfloor = g.nd("Cast", [g.nd("Cast", [dchalf], to=INT64)], to=F)
    dotpar = g.nd("Sub", [dotcol, g.nd("Add", [dcfloor, dcfloor])])  # dotcol mod 2
    even = _eq(g, colpar, dotpar)                                    # [1,1,30,30]
    evenM = g.nd("Mul", [active, even])
    oddM = g.nd("Mul", [active, g.nd("Sub", [g.one, even])])

    outE = g.nd("Mul", [dotOH, evenM])                              # [1,10,30,30]
    e5 = g.f([1, CHANNELS, 1, 1], [1.0 if c == _ALT else 0.0 for c in range(CHANNELS)])
    outO = g.nd("Mul", [e5, oddM])
    bg = g.nd("Mul", [rm, g.nd("Sub", [g.one, active])])
    g.nd("Add", [g.nd("Add", [outE, outO]), g.nd("Mul", [g.e0, bg])], "output")
    return _model(g)


def _ref_altfill(a):
    Hh, Ww = a.shape
    out = np.zeros_like(a)
    for r in range(Hh):
        nz = [c for c in range(Ww) if a[r, c] != 0]
        if not nz:
            continue
        if len(nz) != 1:
            return None
        c0 = nz[0]; v = a[r, c0]
        for c in range(c0, Ww):
            out[r, c] = v if (c - c0) % 2 == 0 else _ALT
    return out


# =========================================================================== #
# task 342 -- recolour 2x2 block by quadrant dots                              #
# =========================================================================== #
def _build_quad():
    g = _G(); _common(g)
    rm = _realmask(g)
    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)   # [1,10,1,1]
    bgneg = g.f([1, CHANNELS, 1, 1], [-1e9] + [0.0] * (CHANNELS - 1))
    amax = g.nd("ArgMax", [g.nd("Add", [counts, bgneg])], axis=1, keepdims=1)
    chidx = g.i64(list(range(CHANNELS)))
    chidx2 = g.nd("Reshape", [g.nd("Cast", [chidx], to=F), g.i64([1, CHANNELS, 1, 1])])
    blockgate = _eq(g, g.nd("Cast", [amax], to=F), chidx2)           # [1,10,1,1]
    blockcells = g.nd("Mul", ["input", blockgate])                  # [1,10,30,30]
    blockmask = g.nd("ReduceSum", [blockcells], axes=[1], keepdims=1)

    rowhas = g.nd("ReduceMax", [blockmask], axes=[3], keepdims=1)    # [1,1,30,1]
    colhas = g.nd("ReduceMax", [blockmask], axes=[2], keepdims=1)    # [1,1,1,30]
    cbig = g.f([1, 1, 1, 1], [1000.0])
    r0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
          [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, g.rowidx])])], axes=[2], keepdims=1)])
    c0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
          [g.nd("Mul", [colhas, g.nd("Sub", [cbig, g.colidx])])], axes=[3], keepdims=1)])
    r1 = g.nd("Add", [r0, g.one]); c1 = g.nd("Add", [c0, g.one])

    masknb = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    dots = g.nd("Sub", [g.nd("Mul", ["input", masknb]), blockcells])  # [1,10,30,30]

    mtop = _lt(g, g.rowidx, r0)
    mbot = _gt(g, g.rowidx, r1)
    mlft = _lt(g, g.colidx, c0)
    mrgt = _gt(g, g.colidx, c1)

    def qcol(mr, mc):
        return g.nd("ReduceSum", [g.nd("Mul", [dots, g.nd("Mul", [mr, mc])])],
                    axes=[2, 3], keepdims=1)                          # [1,10,1,1]
    cTL = qcol(mtop, mlft); cTR = qcol(mtop, mrgt)
    cBL = qcol(mbot, mlft); cBR = qcol(mbot, mrgt)

    def pos(rr, cc):
        return g.nd("Mul", [_eq(g, g.rowidx, rr), _eq(g, g.colidx, cc)])  # [1,1,30,30]
    pTL = pos(r0, c0); pTR = pos(r0, c1); pBL = pos(r1, c0); pBR = pos(r1, c1)

    placed = g.nd("Add", [g.nd("Add", [g.nd("Mul", [cTL, pTL]), g.nd("Mul", [cTR, pTR])]),
                          g.nd("Add", [g.nd("Mul", [cBL, pBL]), g.nd("Mul", [cBR, pBR])])])
    cov = g.nd("Add", [g.nd("Add", [pTL, pTR]), g.nd("Add", [pBL, pBR])])
    bg = g.nd("Mul", [rm, g.nd("Sub", [g.one, cov])])
    g.nd("Add", [placed, g.nd("Mul", [g.e0, bg])], "output")
    return _model(g)


def _ref_quad(a):
    from collections import Counter
    cnt = Counter(a[a != 0].tolist())
    blk = [c for c, n in cnt.items() if n >= 4]
    if len(blk) != 1:
        return None
    bc = blk[0]
    ys, xs = np.where(a == bc)
    if len(ys) != 4:
        return None
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    out = np.zeros_like(a)
    dots = [(r, c, a[r, c]) for r in range(a.shape[0]) for c in range(a.shape[1])
            if a[r, c] != 0 and a[r, c] != bc]
    if len(dots) != 4:
        return None
    for (r, c, v) in dots:
        top, bot, left, right = r < r0, r > r1, c < c0, c > c1
        if top and left:
            out[r0, c0] = v
        elif top and right:
            out[r0, c1] = v
        elif bot and left:
            out[r1, c0] = v
        elif bot and right:
            out[r1, c1] = v
        else:
            return None
    return out


# =========================================================================== #
# task 40 -- recolour markers by nearer of two parallel border lines           #
# =========================================================================== #
def _build_border():
    g = _G(); _common(g)
    rm = _realmask(g)
    colhas = g.nd("ReduceMax", [rm], axes=[2], keepdims=1)           # [1,1,1,30]
    rowhas = g.nd("ReduceMax", [rm], axes=[3], keepdims=1)           # [1,1,30,1]
    Wt = g.nd("ReduceSum", [colhas], axes=[3], keepdims=1)
    Ht = g.nd("ReduceSum", [rowhas], axes=[2], keepdims=1)
    Wm1 = g.nd("Sub", [Wt, g.one]); Hm1 = g.nd("Sub", [Ht, g.one])

    masknb = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    Xnb = g.nd("Mul", ["input", masknb])
    nbmask = g.nd("ReduceSum", [Xnb], axes=[1], keepdims=1)          # [1,1,30,30]

    # vertical-border colours
    col0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([3])])  # [1,10,30,1]
    L_oh = g.nd("ReduceMax", [col0], axes=[2], keepdims=1)           # [1,10,1,1]
    rcmask = _eq(g, g.colidx, Wm1)                                   # [1,1,1,30]
    R_oh = g.nd("ReduceMax", [g.nd("Mul", ["input", rcmask])], axes=[2, 3], keepdims=1)
    mV = g.nd("Mul", [Wm1, g.half])
    leftsel = _lt(g, g.colidx, mV)
    Vres = g.nd("Add", [g.nd("Add",
              [g.nd("Mul", [L_oh, g.nd("Mul", [nbmask, leftsel])]),
               g.nd("Mul", [R_oh, g.nd("Mul", [nbmask, g.nd("Sub", [g.one, leftsel])])])]),
              g.nd("Mul", [g.e0, g.nd("Mul", [rm, g.nd("Sub", [g.one, nbmask])])])])

    # horizontal-border colours
    row0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([2])])  # [1,10,1,30]
    T_oh = g.nd("ReduceMax", [row0], axes=[3], keepdims=1)           # [1,10,1,1]
    brmask = _eq(g, g.rowidx, Hm1)                                   # [1,1,30,1]
    B_oh = g.nd("ReduceMax", [g.nd("Mul", ["input", brmask])], axes=[2, 3], keepdims=1)
    mHm = g.nd("Mul", [Hm1, g.half])
    topsel = _lt(g, g.rowidx, mHm)
    Hres = g.nd("Add", [g.nd("Add",
              [g.nd("Mul", [T_oh, g.nd("Mul", [nbmask, topsel])]),
               g.nd("Mul", [B_oh, g.nd("Mul", [nbmask, g.nd("Sub", [g.one, topsel])])])]),
              g.nd("Mul", [g.e0, g.nd("Mul", [rm, g.nd("Sub", [g.one, nbmask])])])])

    # orientation gate: row0 fully non-bg -> horizontal
    row0nb = g.nd("ReduceSum", [g.nd("Mul", [nbmask, _eq(g, g.rowidx, g.f([1, 1, 1, 1], [0.0]))])],
                  axes=[2, 3], keepdims=1)
    gate_h = g.nd("Cast", [g.nd("Greater", [row0nb, g.nd("Sub", [Wt, g.half])])], to=F)
    g.nd("Add", [g.nd("Mul", [Hres, gate_h]),
                 g.nd("Mul", [Vres, g.nd("Sub", [g.one, gate_h])])], "output")
    return _model(g)


def _ref_border(a):
    Hh, Ww = a.shape

    def full_col(c):
        col = a[:, c]
        return col[0] != 0 and np.all(col == col[0])

    def full_row(r):
        row = a[r, :]
        return row[0] != 0 and np.all(row == row[0])

    out = a.copy()
    if full_col(0) and full_col(Ww - 1):
        L, R = a[0, 0], a[0, Ww - 1]
        m = (Ww - 1) / 2.0
        for r in range(Hh):
            for c in range(Ww):
                if a[r, c] != 0:
                    out[r, c] = L if c < m else R
        return out
    if full_row(0) and full_row(Hh - 1):
        T, B = a[0, 0], a[Hh - 1, 0]
        m = (Hh - 1) / 2.0
        for r in range(Hh):
            for c in range(Ww):
                if a[r, c] != 0:
                    out[r, c] = T if r < m else B
        return out
    return None


# =========================================================================== #
# task 293 -- swap z-order of two crossing bars at their intersection           #
# =========================================================================== #
def _build_swap():
    g = _G(); _common(g)
    rm = _realmask(g)
    masknb = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    nbmask = g.nd("ReduceSum", [g.nd("Mul", ["input", masknb])], axes=[1], keepdims=1)

    colnb = g.nd("ReduceSum", [nbmask], axes=[2], keepdims=1)         # [1,1,1,30]
    colreal = g.nd("ReduceSum", [rm], axes=[2], keepdims=1)
    vcol = g.nd("Mul", [_eq(g, colnb, colreal), _gt(g, colreal, g.half)])  # [1,1,1,30]
    rownb = g.nd("ReduceSum", [nbmask], axes=[3], keepdims=1)         # [1,1,30,1]
    rowreal = g.nd("ReduceSum", [rm], axes=[3], keepdims=1)
    hrow = g.nd("Mul", [_eq(g, rownb, rowreal), _gt(g, rowreal, g.half)])  # [1,1,30,1]

    IM = g.nd("Mul", [g.nd("Mul", [vcol, hrow]), rm])                # [1,1,30,30]
    vbar_only = g.nd("Mul", [g.nd("Mul", [vcol, rm]), g.nd("Sub", [g.one, hrow])])
    hbar_only = g.nd("Mul", [g.nd("Mul", [hrow, rm]), g.nd("Sub", [g.one, vcol])])

    color = g.nd("ReduceSum", [g.nd("Mul", ["input", g.colvec])], axes=[1], keepdims=1)
    V = g.nd("ReduceMax", [g.nd("Mul", [color, vbar_only])], axes=[2, 3], keepdims=1)
    Hc = g.nd("ReduceMax", [g.nd("Mul", [color, hbar_only])], axes=[2, 3], keepdims=1)
    newc = g.nd("Sub", [g.nd("Add", [V, Hc]), color])
    tcolor = g.nd("Add", [g.nd("Mul", [color, g.nd("Sub", [g.one, IM])]),
                          g.nd("Mul", [newc, IM])])                  # [1,1,30,30]
    oh10 = g.nd("Cast", [g.nd("Less",
              [g.nd("Abs", [g.nd("Sub", [tcolor, g.colvec])]), g.half])], to=F)
    g.nd("Mul", [oh10, rm], "output")
    return _model(g)


def _ref_swap(a):
    Hh, Ww = a.shape
    vcol = [c for c in range(Ww) if np.all(a[:, c] != 0)]
    hrow = [r for r in range(Hh) if np.all(a[r, :] != 0)]
    if not vcol or not hrow:
        return None
    Vs, Hs = set(), set()
    for c in vcol:
        for r in range(Hh):
            if r not in hrow:
                Vs.add(int(a[r, c]))
    for r in hrow:
        for c in range(Ww):
            if c not in vcol:
                Hs.add(int(a[r, c]))
    if len(Vs) != 1 or len(Hs) != 1:
        return None
    V = next(iter(Vs)); Hh2 = next(iter(Hs))
    out = a.copy()
    for r in hrow:
        for c in vcol:
            out[r, c] = V + Hh2 - a[r, c]
    return out


# =========================================================================== #
# task 197 -- expand each seed row to the template pattern, recoloured          #
# =========================================================================== #
def _build_template():
    g = _G(); _common(g)
    rm = _realmask(g)
    masknb = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    nbmask = g.nd("ReduceSum", [g.nd("Mul", ["input", masknb])], axes=[1], keepdims=1)
    rownb = g.nd("ReduceSum", [nbmask], axes=[3], keepdims=1)         # [1,1,30,1]
    rowreal = g.nd("ReduceSum", [rm], axes=[3], keepdims=1)
    rowfull = g.nd("Mul", [_eq(g, rownb, rowreal), _gt(g, rowreal, g.half)])  # [1,1,30,1]

    tplOH = g.nd("ReduceSum", [g.nd("Mul", ["input", rowfull])], axes=[2], keepdims=1)
    tplcolor = g.nd("ReduceSum", [g.nd("Mul", [tplOH, g.colvec])], axes=[1], keepdims=1)  # [1,1,1,30]
    color = g.nd("ReduceSum", [g.nd("Mul", ["input", g.colvec])], axes=[1], keepdims=1)   # [1,1,30,30]

    Cp2 = g.nd("Reshape", [color, g.i64([H, W])])                    # [30,30] (r,c)
    tpl_c = g.nd("Reshape", [tplcolor, g.i64([W, 1])])               # [30,1]
    tpl_j = g.nd("Reshape", [tplcolor, g.i64([1, W])])               # [1,30]
    half2 = g.f([1, 1], [0.5])
    Tmatch = g.nd("Cast", [g.nd("Less",
              [g.nd("Abs", [g.nd("Sub", [tpl_c, tpl_j])]), half2])], to=F)   # [30,30]
    A = g.nd("Reshape", [Cp2, g.i64([H, W, 1])])                     # [30,30,1]
    B = g.nd("Reshape", [Tmatch, g.i64([1, W, W])])                  # [1,30,30]
    P = g.nd("Mul", [A, B])                                          # [30,30,30] (r,c,j)
    mx = g.nd("ReduceMax", [P], axes=[1], keepdims=0)                # [30,30] (r,j)
    outcolor = g.nd("Reshape", [mx, g.i64([1, 1, H, W])])
    oh10 = g.nd("Cast", [g.nd("Less",
              [g.nd("Abs", [g.nd("Sub", [outcolor, g.colvec])]), g.half])], to=F)
    g.nd("Mul", [oh10, rm], "output")
    return _model(g)


def _ref_template(a):
    Hh, Ww = a.shape
    full = [r for r in range(Hh) if np.all(a[r, :] != 0)]
    if len(full) != 1:
        return None
    R = full[0]; tpl = a[R, :]
    out = a.copy()
    for r in range(Hh):
        if r == R or np.all(a[r, :] == 0):
            continue
        seed = a[r, :]; mp = {}
        for c in range(Ww):
            if seed[c] != 0:
                tc = int(tpl[c])
                if tc in mp and mp[tc] != seed[c]:
                    return None
                mp[tc] = int(seed[c])
        for c in range(Ww):
            if int(tpl[c]) not in mp:
                return None
        out[r, :] = [mp[int(tpl[c])] for c in range(Ww)]
    return out


# =========================================================================== #
# task 239 -- colour-frequency histogram (columns sorted by count desc)         #
# =========================================================================== #
def _build_histogram():
    g = _G(); _common(g)
    mask0 = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    cnt = g.nd("Mul", [g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1), mask0])  # [1,10,1,1]
    present = g.nd("Cast", [g.nd("Greater", [cnt, g.half])], to=F)    # [1,10,1,1]

    cnt_c = g.nd("Reshape", [cnt, g.i64([CHANNELS, 1])])              # [10,1]
    cnt_cp = g.nd("Reshape", [cnt, g.i64([1, CHANNELS])])            # [1,10]
    cidx_c = g.f([CHANNELS, 1], list(range(CHANNELS)))
    cidx_cp = g.f([1, CHANNELS], list(range(CHANNELS)))
    half2 = g.f([1, 1], [0.5])
    greater = g.nd("Cast", [g.nd("Greater", [cnt_cp, cnt_c])], to=F)  # [10,10] cnt[c']>cnt[c]
    equal = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [cnt_cp, cnt_c])]), half2])], to=F)
    less = g.nd("Cast", [g.nd("Less", [cidx_cp, cidx_c])], to=F)      # c' < c
    outrank = g.nd("Add", [greater, g.nd("Mul", [equal, less])])      # [10,10]
    present_cp = g.nd("Reshape", [present, g.i64([1, CHANNELS])])
    M = g.nd("Mul", [outrank, present_cp])
    col_c = g.nd("ReduceSum", [M], axes=[1], keepdims=1)              # [10,1]
    colpos = g.nd("Reshape", [col_c, g.i64([1, CHANNELS, 1, 1])])     # [1,10,1,1]

    colmask = _eq(g, g.colidx, colpos)                               # [1,10,1,30]
    rowmask = g.nd("Cast", [g.nd("Less", [g.rowidx, cnt])], to=F)    # [1,10,30,1]
    placed = g.nd("Mul", [g.nd("Mul", [colmask, rowmask]), present])  # [1,10,30,30]

    covered = g.nd("ReduceSum", [placed], axes=[1], keepdims=1)
    maxcount = g.nd("ReduceMax", [cnt], axes=[1], keepdims=1)
    ndist = g.nd("ReduceSum", [present], axes=[1], keepdims=1)
    Rect = g.nd("Mul", [_lt(g, g.rowidx, maxcount), _lt(g, g.colidx, ndist)])
    bg = g.nd("Mul", [Rect, g.nd("Sub", [g.one, covered])])
    g.nd("Add", [placed, g.nd("Mul", [g.e0, bg])], "output")
    return _model(g)


def _ref_histogram(a):
    from collections import Counter
    cnt = Counter(a.ravel().tolist())
    cnt.pop(0, None)
    if not cnt:
        return None
    items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    maxc = max(c for _, c in items)
    if maxc > 30 or len(items) > 30:
        return None
    out = np.zeros((maxc, len(items)), int)
    for j, (col, c) in enumerate(items):
        out[:c, j] = col
    return out


# =========================================================================== #
# task 247 -- max-frequency colours as full columns, ordered by leftmost col    #
# =========================================================================== #
def _build_maxcols():
    g = _G(); _common(g)
    mask0 = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    cnt = g.nd("Mul", [g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1), mask0])
    present = g.nd("Cast", [g.nd("Greater", [cnt, g.half])], to=F)
    maxcount = g.nd("ReduceMax", [cnt], axes=[1], keepdims=1)         # [1,1,1,1]
    selected = g.nd("Mul", [present, _eq(g, cnt, maxcount)])          # [1,10,1,1]

    cbig = g.f([1, 1, 1, 1], [1000.0])
    masked = g.nd("Mul", ["input", g.nd("Sub", [cbig, g.colidx])])
    mxv = g.nd("ReduceMax", [masked], axes=[2, 3], keepdims=1)
    mincol = g.nd("Sub", [cbig, mxv])                                 # [1,10,1,1]

    mc_c = g.nd("Reshape", [mincol, g.i64([CHANNELS, 1])])
    mc_cp = g.nd("Reshape", [mincol, g.i64([1, CHANNELS])])
    cidx_c = g.f([CHANNELS, 1], list(range(CHANNELS)))
    cidx_cp = g.f([1, CHANNELS], list(range(CHANNELS)))
    half2 = g.f([1, 1], [0.5])
    lessmc = g.nd("Cast", [g.nd("Less", [mc_cp, mc_c])], to=F)
    eqmc = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [mc_cp, mc_c])]), half2])], to=F)
    lesscol = g.nd("Cast", [g.nd("Less", [cidx_cp, cidx_c])], to=F)
    outrank = g.nd("Add", [lessmc, g.nd("Mul", [eqmc, lesscol])])
    sel_cp = g.nd("Reshape", [selected, g.i64([1, CHANNELS])])
    col_c = g.nd("ReduceSum", [g.nd("Mul", [outrank, sel_cp])], axes=[1], keepdims=1)
    colpos = g.nd("Reshape", [col_c, g.i64([1, CHANNELS, 1, 1])])

    colmask = _eq(g, g.colidx, colpos)                               # [1,10,1,30]
    rowmask = _lt(g, g.rowidx, maxcount)                             # [1,1,30,1]
    placed = g.nd("Mul", [g.nd("Mul", [colmask, rowmask]), selected])  # [1,10,30,30]
    covered = g.nd("ReduceSum", [placed], axes=[1], keepdims=1)
    ndist = g.nd("ReduceSum", [selected], axes=[1], keepdims=1)
    Rect = g.nd("Mul", [_lt(g, g.rowidx, maxcount), _lt(g, g.colidx, ndist)])
    bg = g.nd("Mul", [Rect, g.nd("Sub", [g.one, covered])])
    g.nd("Add", [placed, g.nd("Mul", [g.e0, bg])], "output")
    return _model(g)


def _ref_maxcols(a):
    from collections import Counter
    cnt = Counter(a.ravel().tolist())
    cnt.pop(0, None)
    if not cnt:
        return None
    mx = max(cnt.values())
    sel = [c for c in cnt if cnt[c] == mx]
    sel.sort(key=lambda c: (int(np.where(a == c)[1].min()), c))
    if mx > 30 or len(sel) > 30:
        return None
    out = np.zeros((mx, len(sel)), int)
    for j, c in enumerate(sel):
        out[:, j] = c
    return out


# =========================================================================== #
# task 88 -- crop the marker-box INTERIOR, recolour shape -> marker colour      #
# =========================================================================== #
def _build_innercrop():
    g = _G(); _common(g)
    cbig = g.f([1, 1, 1, 1], [1000.0])
    mask0 = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    cnt = g.nd("Mul", [g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1), mask0])
    present = g.nd("Cast", [g.nd("Greater", [cnt, g.half])], to=F)
    big = g.f([1, CHANNELS, 1, 1], [1e9] * CHANNELS)
    cntP = g.nd("Add", [cnt, g.nd("Mul", [g.nd("Sub", [g.one, present]), big])])
    mincnt = g.nd("ReduceMin", [cntP], axes=[1], keepdims=1)
    maxcnt = g.nd("ReduceMax", [cnt], axes=[1], keepdims=1)
    Mgate = g.nd("Mul", [present, _eq(g, cnt, mincnt)])              # marker colour [1,10,1,1]
    Sgate = g.nd("Mul", [present, _eq(g, cnt, maxcnt)])              # shape colour

    sCells = g.nd("ReduceSum", [g.nd("Mul", ["input", Sgate])], axes=[1], keepdims=1)
    Xr = g.nd("Add", [g.nd("Mul", ["input", g.nd("Sub", [g.one, Sgate])]),
                      g.nd("Mul", [Mgate, sCells])])                # shape recoloured to marker

    Mmask = g.nd("ReduceSum", [g.nd("Mul", ["input", Mgate])], axes=[1], keepdims=1)
    rowhas = g.nd("ReduceMax", [Mmask], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [Mmask], axes=[2], keepdims=1)
    maxrow = g.nd("ReduceMax", [g.nd("Mul", [rowhas, g.rowidx])], axes=[2], keepdims=1)
    minrow = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, g.rowidx])])], axes=[2], keepdims=1)])
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, g.colidx])], axes=[3], keepdims=1)
    mincol = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [colhas, g.nd("Sub", [cbig, g.colidx])])], axes=[3], keepdims=1)])

    mr = g.nd("Add", [minrow, g.one]); mc = g.nd("Add", [mincol, g.one])
    bh = g.nd("Sub", [g.nd("Sub", [maxrow, minrow]), g.one])         # maxrow-minrow-1
    bw = g.nd("Sub", [g.nd("Sub", [maxcol, mincol]), g.one])

    diff_c = g.nd("Sub", [g.nd("Add", [g.colidx, mc]), g.rowidx])
    match_c = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_c]), g.half])], to=F)
    Scol = g.nd("Mul", [match_c, _lt(g, g.colidx, bw)])
    diff_r = g.nd("Sub", [g.colidx, g.nd("Add", [g.rowidx, mr])])
    match_r = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_r]), g.half])], to=F)
    Srow = g.nd("Mul", [match_r, _lt(g, g.rowidx, bh)])
    g.nd("MatMul", [Srow, g.nd("MatMul", [Xr, Scol])], "output")
    return _model(g)


def _ref_innercrop(a):
    from collections import Counter
    cnt = Counter(a.ravel().tolist())
    cnt.pop(0, None)
    if len(cnt) != 2:
        return None
    items = sorted(cnt.items(), key=lambda kv: kv[1])
    if items[0][1] == items[1][1]:
        return None
    Mm = items[0][0]; S = items[1][0]
    ys, xs = np.where(a == Mm)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    sub = a[r0 + 1:r1, c0 + 1:c1].copy()
    return np.where(sub == S, Mm, sub)


# =========================================================================== #
# task 117 -- 4-fold reflect content about the symmetric marker's centre        #
# =========================================================================== #
def _build_reflect4():
    g = _G(); _common(g)
    cbig = g.f([1, 1, 1, 1], [1000.0])
    mask0 = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    cnt = g.nd("Mul", [g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1), mask0])
    present = g.nd("Cast", [g.nd("Greater", [cnt, g.half])], to=F)    # [1,10,1,1]

    # per-channel bbox centre (x2)
    rowhasC = g.nd("ReduceMax", ["input"], axes=[3], keepdims=1)       # [1,10,30,1]
    colhasC = g.nd("ReduceMax", ["input"], axes=[2], keepdims=1)       # [1,10,1,30]
    maxrowC = g.nd("ReduceMax", [g.nd("Mul", [rowhasC, g.rowidx])], axes=[2], keepdims=1)
    minrowC = g.nd("Sub", [cbig, g.nd("ReduceMax",
               [g.nd("Mul", [rowhasC, g.nd("Sub", [cbig, g.rowidx])])], axes=[2], keepdims=1)])
    maxcolC = g.nd("ReduceMax", [g.nd("Mul", [colhasC, g.colidx])], axes=[3], keepdims=1)
    mincolC = g.nd("Sub", [cbig, g.nd("ReduceMax",
               [g.nd("Mul", [colhasC, g.nd("Sub", [cbig, g.colidx])])], axes=[3], keepdims=1)])
    RRc = g.nd("Add", [minrowC, maxrowC])                            # [1,10,1,1]
    CCc = g.nd("Add", [mincolC, maxcolC])

    # per-channel self-symmetry via batched reflection
    inputT = g.nd("Reshape", ["input", g.i64([CHANNELS, H, W])])      # [10,30,30]
    ii = g.f([1, H, 1], list(range(H)))
    jj = g.f([1, 1, W], list(range(W)))
    ipj = g.nd("Add", [ii, jj])                                      # [1,30,30]
    RRc3 = g.nd("Reshape", [RRc, g.i64([CHANNELS, 1, 1])])
    CCc3 = g.nd("Reshape", [CCc, g.i64([CHANNELS, 1, 1])])
    half3 = g.f([1, 1, 1], [0.5])
    Rcol_c = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [ipj, CCc3])]), half3])], to=F)
    Rrow_c = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [ipj, RRc3])]), half3])], to=F)
    refH = g.nd("MatMul", [inputT, Rcol_c])                          # reflect cols about CC_c
    refV = g.nd("MatMul", [Rrow_c, inputT])                          # reflect rows about RR_c
    diffH = g.nd("ReduceSum", [g.nd("Abs", [g.nd("Sub", [refH, inputT])])], axes=[1, 2], keepdims=0)
    diffV = g.nd("ReduceSum", [g.nd("Abs", [g.nd("Sub", [refV, inputT])])], axes=[1, 2], keepdims=0)
    h05 = g.f([1], [0.5])
    selfHV = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [diffH, h05])], to=F),
                          g.nd("Cast", [g.nd("Less", [diffV, h05])], to=F)])  # [10]
    selfsym = g.nd("Reshape", [selfHV, g.i64([1, CHANNELS, 1, 1])])
    markerg = g.nd("Mul", [present, selfsym])                        # [1,10,1,1]
    RR = g.nd("ReduceSum", [g.nd("Mul", [markerg, RRc])], axes=[1], keepdims=1)  # [1,1,1,1]
    CC = g.nd("ReduceSum", [g.nd("Mul", [markerg, CCc])], axes=[1], keepdims=1)

    ipj4 = g.nd("Add", [g.rowidx, g.colidx])                         # [1,1,30,30]
    Rcol = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [ipj4, CC])]), g.half])], to=F)
    Rrow = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [ipj4, RR])]), g.half])], to=F)
    Xnb = g.nd("Mul", ["input", mask0])                              # background channel zeroed
    Href = g.nd("MatMul", [Xnb, Rcol])
    Vref = g.nd("MatMul", [Rrow, Xnb])
    HVref = g.nd("MatMul", [Rrow, Href])
    union = g.nd("Max", [Xnb, Href, Vref, HVref])                    # [1,10,30,30] (ch0 = 0)
    rm = _realmask(g)
    nbany = g.nd("ReduceSum", [union], axes=[1], keepdims=1)
    bg = g.nd("Mul", [rm, g.nd("Sub", [g.one, nbany])])
    g.nd("Add", [union, g.nd("Mul", [g.e0, bg])], "output")
    return _model(g)


def _selfsym(a, m):
    ys, xs = np.where(a == m)
    rc = ys.min() + ys.max(); cc = xs.min() + xs.max()
    cells = set(zip(ys.tolist(), xs.tolist()))
    for (r, c) in cells:
        if (rc - r, c) not in cells or (r, cc - c) not in cells:
            return False
    return True


def _ref_reflect4(a):
    from collections import Counter
    cnt = Counter(a.ravel().tolist())
    cnt.pop(0, None)
    if len(cnt) != 2:
        return None
    cols = list(cnt)
    syms = [m for m in cols if _selfsym(a, m)]
    if len(syms) != 1:
        return None
    mk = syms[0]
    H0, W0 = a.shape
    ys, xs = np.where(a == mk)
    rc = ys.min() + ys.max(); cc = xs.min() + xs.max()
    out = np.zeros_like(a)
    for r in range(H0):
        for c in range(W0):
            if a[r, c] != 0:
                for (rr, ccc) in [(r, c), (rc - r, c), (r, cc - c), (rc - r, cc - c)]:
                    if 0 <= rr < H0 and 0 <= ccc < W0:
                        if out[rr, ccc] not in (0, a[r, c]):
                            return None
                        out[rr, ccc] = a[r, c]
    return out


# =========================================================================== #
# task 256 -- left-anchored triangle: grow (colour 3) above, shrink (1) below    #
# =========================================================================== #
def _build_triangle():
    g = _G(); _common(g)
    rm = _realmask(g)
    mask0 = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    Xnb = g.nd("Mul", ["input", mask0])
    nbmask = g.nd("ReduceSum", [Xnb], axes=[1], keepdims=1)          # [1,1,30,30]
    rowhas = g.nd("ReduceMax", [nbmask], axes=[3], keepdims=1)       # [1,1,30,1]
    R = g.nd("ReduceSum", [g.nd("Mul", [rowhas, g.rowidx])], axes=[2], keepdims=1)  # [1,1,1,1]
    L = g.nd("ReduceSum", [nbmask], axes=[2, 3], keepdims=1)         # [1,1,1,1]
    cnt = g.nd("Mul", [g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1), mask0])
    Cgate = g.nd("Cast", [g.nd("Greater", [cnt, g.half])], to=F)     # [1,10,1,1]

    lenr = g.nd("Sub", [g.nd("Add", [L, R]), g.rowidx])             # [1,1,30,1] = L+R-r
    fillmask = _lt(g, g.colidx, lenr)                               # [1,1,30,30]
    covered = g.nd("Mul", [fillmask, rm])

    upm = _lt(g, g.rowidx, R)
    midm = _eq(g, g.rowidx, R)
    dnm = _gt(g, g.rowidx, R)
    e3 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 3 else 0.0 for c in range(CHANNELS)])
    e1 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 1 else 0.0 for c in range(CHANNELS)])
    color_oh = g.nd("Add", [g.nd("Add", [g.nd("Mul", [e3, upm]), g.nd("Mul", [Cgate, midm])]),
                            g.nd("Mul", [e1, dnm])])                 # [1,10,30,1]
    content = g.nd("Mul", [color_oh, covered])
    bg = g.nd("Mul", [rm, g.nd("Sub", [g.one, covered])])
    g.nd("Add", [content, g.nd("Mul", [g.e0, bg])], "output")
    return _model(g)


def _ref_triangle(a):
    H0, W0 = a.shape
    rows = [r for r in range(H0) if np.any(a[r] != 0)]
    if len(rows) != 1:
        return None
    R = rows[0]; row = a[R]
    nz = np.where(row != 0)[0]
    if nz.size == 0 or nz[0] != 0 or not np.array_equal(nz, np.arange(nz.size)):
        return None
    L = nz.size; C = int(row[0])
    if np.any(row[:L] != C):
        return None
    out = np.zeros_like(a)
    for r in range(H0):
        ln = L + R - r
        col = 3 if r < R else (C if r == R else 1)
        if ln > 0:
            out[r, :min(ln, W0)] = col
    return out


# =========================================================================== #
# task 108 -- 2x2 block-reduce (per-block colour) then 4x nearest upscale        #
# =========================================================================== #
def _build_blockup():
    g = _G(); _common(g)
    rm = _realmask(g)
    Cp = g.nd("ReduceSum", [g.nd("Mul", ["input", g.colvec])], axes=[1], keepdims=1)

    def stride2(sr, sc):
        return g.nd("Slice", [Cp, g.i64([sr, sc]), g.i64([H, W]), g.i64([2, 3]), g.i64([2, 2])])
    block = g.nd("Max", [stride2(0, 0), stride2(0, 1), stride2(1, 0), stride2(1, 1)])  # [1,1,15,15]
    scales = g.f([4], [1.0, 1.0, 4.0, 4.0])
    up = g.nd("Resize", [block, scales], mode="nearest")             # [1,1,60,60]
    crop = g.nd("Slice", [up, g.i64([0, 0]), g.i64([H, W]), g.i64([2, 3])])  # [1,1,30,30]

    colhas = g.nd("ReduceMax", [rm], axes=[2], keepdims=1)
    rowhas = g.nd("ReduceMax", [rm], axes=[3], keepdims=1)
    W2 = g.nd("ReduceSum", [colhas], axes=[3], keepdims=1)
    W2 = g.nd("Add", [W2, W2])
    H2 = g.nd("ReduceSum", [rowhas], axes=[2], keepdims=1)
    H2 = g.nd("Add", [H2, H2])
    outmask = g.nd("Mul", [_lt(g, g.colidx, W2), _lt(g, g.rowidx, H2)])
    oh10 = g.nd("Cast", [g.nd("Less",
              [g.nd("Abs", [g.nd("Sub", [crop, g.colvec])]), g.half])], to=F)
    g.nd("Mul", [oh10, outmask], "output")
    return _model(g)


def _ref_blockup(a):
    H0, W0 = a.shape
    bh, bw = (H0 + 1) // 2, (W0 + 1) // 2
    block = np.zeros((bh, bw), int)
    for i in range(bh):
        for j in range(bw):
            blk = a[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            nz = blk[blk != 0]
            if nz.size and len(set(nz.tolist())) > 1:
                return None
            block[i, j] = nz[0] if nz.size else 0
    return np.repeat(np.repeat(block, 4, axis=0), 4, axis=1)


# =========================================================================== #
# task 12 -- stamp a fixed 5x5 (X-diagonal=centre colour, +cross=arm colour)     #
# =========================================================================== #
_C_OFF = [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (-2, -2), (-2, 2), (2, -2), (2, 2)]
_O_OFF = [(-2, 0), (-1, 0), (1, 0), (2, 0), (0, -2), (0, -1), (0, 1), (0, 2)]


def _kernel(offs):
    k = np.zeros((5, 5), np.float32)
    for dy, dx in offs:
        k[2 + dy, 2 + dx] = 1.0
    return k


def _build_stamp12():
    g = _G(); _common(g)
    rm = _realmask(g)
    mask0 = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    cnt = g.nd("Mul", [g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1), mask0])
    present = g.nd("Cast", [g.nd("Greater", [cnt, g.half])], to=F)
    maxcnt = g.nd("ReduceMax", [cnt], axes=[1], keepdims=1)
    big = g.f([1, CHANNELS, 1, 1], [1e9] * CHANNELS)
    cntP = g.nd("Add", [cnt, g.nd("Mul", [g.nd("Sub", [g.one, present]), big])])
    mincnt = g.nd("ReduceMin", [cntP], axes=[1], keepdims=1)
    Ogate = g.nd("Mul", [present, _eq(g, cnt, maxcnt)])              # arm colour
    Cgate = g.nd("Mul", [present, _eq(g, cnt, mincnt)])              # centre colour

    centermask = g.nd("ReduceSum", [g.nd("Mul", ["input", Cgate])], axes=[1], keepdims=1)
    WO = g.f([1, 1, 5, 5], _kernel(_O_OFF))
    WC = g.f([1, 1, 5, 5], _kernel(_C_OFF))
    Oplace = g.nd("Conv", [centermask, WO], kernel_shape=[5, 5], pads=[2, 2, 2, 2])
    Cplace = g.nd("Conv", [centermask, WC], kernel_shape=[5, 5], pads=[2, 2, 2, 2])
    Op = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [Oplace, g.half])], to=F), rm])
    Cp = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [Cplace, g.half])], to=F), rm])
    covered = g.nd("Cast", [g.nd("Greater", [g.nd("Add", [Op, Cp]), g.half])], to=F)
    bg = g.nd("Mul", [rm, g.nd("Sub", [g.one, covered])])
    g.nd("Add", [g.nd("Add", [g.nd("Mul", [Ogate, Op]), g.nd("Mul", [Cgate, Cp])]),
                 g.nd("Mul", [g.e0, bg])], "output")
    return _model(g)


def _ref_stamp12(a):
    from collections import Counter
    cnt = Counter(a.ravel().tolist())
    cnt.pop(0, None)
    if len(cnt) != 2:
        return None
    items = sorted(cnt.items(), key=lambda kv: kv[1])
    if items[0][1] == items[1][1]:
        return None
    C = items[0][0]; O = items[1][0]
    H0, W0 = a.shape
    out = np.zeros_like(a)
    cys, cxs = np.where(a == C)
    for cy, cx in zip(cys.tolist(), cxs.tolist()):
        for dy, dx in _O_OFF:
            y, x = cy + dy, cx + dx
            if 0 <= y < H0 and 0 <= x < W0:
                out[y, x] = O
        for dy, dx in _C_OFF:
            y, x = cy + dy, cx + dx
            if 0 <= y < H0 and 0 <= x < W0:
                out[y, x] = C
    return out


# =========================================================================== #
# task 277 -- recolour smallest 8-connected object -> 2, all others -> 1         #
# =========================================================================== #
INT32 = onnx.TensorProto.INT32


def _shift8(g, x, dr, dc):
    pt, pb = max(dr, 0), max(-dr, 0)
    pl, pr = max(dc, 0), max(-dc, 0)
    p = g.nd("Pad", [x], mode="constant", value=0.0, pads=[0, 0, pt, pl, 0, 0, pb, pr])
    st = g.i64([max(-dr, 0), max(-dc, 0)])
    en = g.i64([max(-dr, 0) + H, max(-dc, 0) + W])
    return g.nd("Slice", [p, st, en, g.i64([2, 3])])


def _build_smallobj(T=30):
    g = _G(); _common(g)
    rm = _realmask(g)
    mask0 = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    M = g.nd("ReduceSum", [g.nd("Mul", ["input", mask0])], axes=[1], keepdims=1)  # [1,1,30,30]

    P = g.f([1, 1, H, W], np.arange(1, H * W + 1))
    L = g.nd("Mul", [M, P])
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for _ in range(T):
        nbrs = [_shift8(g, L, dr, dc) for dr, dc in dirs]
        L = g.nd("Mul", [g.nd("Max", [L] + nbrs), M])

    Lcol = g.nd("Reshape", [L, g.i64([H * W, 1])])
    Lrow = g.nd("Reshape", [L, g.i64([1, H * W])])
    Ef = g.nd("Cast", [g.nd("Equal", [g.nd("Cast", [Lcol], to=INT32),
                                      g.nd("Cast", [Lrow], to=INT32)])], to=F)
    size_col = g.nd("ReduceSum", [Ef], axes=[1], keepdims=1)          # [900,1]
    size2d = g.nd("Mul", [g.nd("Reshape", [size_col, g.i64([1, 1, H, W])]), M])
    big = g.f([1, 1, 1, 1], [1e6])
    sizeM = g.nd("Add", [size2d, g.nd("Mul", [g.nd("Sub", [g.one, M]), big])])
    minsize = g.nd("ReduceMin", [sizeM], axes=[2, 3], keepdims=1)
    smallmask = g.nd("Mul", [M, _eq(g, size2d, minsize)])
    restmask = g.nd("Mul", [M, g.nd("Sub", [g.one, smallmask])])

    e1 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 1 else 0.0 for c in range(CHANNELS)])
    e2 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 2 else 0.0 for c in range(CHANNELS)])
    bg = g.nd("Mul", [rm, g.nd("Sub", [g.one, M])])
    g.nd("Add", [g.nd("Add", [g.nd("Mul", [e2, smallmask]), g.nd("Mul", [e1, restmask])]),
                 g.nd("Mul", [g.e0, bg])], "output")
    return _model(g)


def _comps8(a):
    from collections import deque
    H0, W0 = a.shape
    seen = np.zeros((H0, W0), bool); out = []
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for i in range(H0):
        for j in range(W0):
            if a[i, j] != 0 and not seen[i, j]:
                q = deque([(i, j)]); seen[i, j] = True; cells = [(i, j)]
                while q:
                    r, c = q.popleft()
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H0 and 0 <= nc < W0 and a[nr, nc] != 0 and not seen[nr, nc]:
                            seen[nr, nc] = True; q.append((nr, nc)); cells.append((nr, nc))
                out.append(cells)
    return out


def _ref_smallobj(a):
    cs = _comps8(a)
    if len(cs) < 2:
        return None
    sizes = [len(c) for c in cs]
    mn = min(sizes)
    out = np.zeros_like(a)
    for c in cs:
        col = 2 if len(c) == mn else 1
        for (r, cc) in c:
            out[r, cc] = col
    return out


# =========================================================================== #
# task 204 -- fill enclosed regions: odd cell-count -> 7, even -> 2              #
# =========================================================================== #
_D4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _build_roomfill(Tflood=60, Tlabel=30):
    g = _G(); _common(g)
    rm = _realmask(g)
    mask0 = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    Xnb = g.nd("Mul", ["input", mask0])
    nbmask = g.nd("ReduceSum", [Xnb], axes=[1], keepdims=1)          # [1,1,30,30]
    passable = g.nd("Sub", [g.one, nbmask])                         # bg or padding

    bc = np.zeros((H, W), np.float32); bc[0, :] = 1; bc[-1, :] = 1; bc[:, 0] = 1; bc[:, -1] = 1
    borderc = g.f([1, 1, H, W], bc)
    reach = g.nd("Mul", [passable, borderc])
    for _ in range(Tflood):
        nb = [_shift8(g, reach, dr, dc) for dr, dc in _D4]
        reach = g.nd("Mul", [passable, g.nd("Max", [reach] + nb)])
    realbg = g.nd("Mul", [rm, g.nd("Sub", [g.one, nbmask])])
    interior = g.nd("Mul", [realbg, g.nd("Sub", [g.one, reach])])

    P = g.f([1, 1, H, W], np.arange(1, H * W + 1))
    L = g.nd("Mul", [interior, P])
    for _ in range(Tlabel):
        nb = [_shift8(g, L, dr, dc) for dr, dc in _D4]
        L = g.nd("Mul", [g.nd("Max", [L] + nb), interior])
    Lcol = g.nd("Reshape", [L, g.i64([H * W, 1])])
    Lrow = g.nd("Reshape", [L, g.i64([1, H * W])])
    Ef = g.nd("Cast", [g.nd("Equal", [g.nd("Cast", [Lcol], to=INT32),
                                      g.nd("Cast", [Lrow], to=INT32)])], to=F)
    size_col = g.nd("ReduceSum", [Ef], axes=[1], keepdims=1)
    size2d = g.nd("Mul", [g.nd("Reshape", [size_col, g.i64([1, 1, H, W])]), interior])
    sfloor = g.nd("Cast", [g.nd("Cast", [g.nd("Mul", [size2d, g.half])], to=INT64)], to=F)
    par = g.nd("Sub", [size2d, g.nd("Add", [sfloor, sfloor])])       # 1 odd, 0 even
    odd = g.nd("Mul", [interior, par])
    even = g.nd("Mul", [interior, g.nd("Sub", [g.one, par])])

    e7 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 7 else 0.0 for c in range(CHANNELS)])
    e2 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 2 else 0.0 for c in range(CHANNELS)])
    bg_out = g.nd("Mul", [realbg, reach])
    g.nd("Add", [g.nd("Add", [Xnb, g.nd("Add", [g.nd("Mul", [e7, odd]), g.nd("Mul", [e2, even])])]),
                 g.nd("Mul", [g.e0, bg_out])], "output")
    return _model(g)


def _ref_roomfill(a):
    from collections import deque
    H0, W0 = a.shape
    reach = np.zeros((H0, W0), bool); q = deque()
    for i in range(H0):
        for j in (0, W0 - 1):
            if a[i, j] == 0 and not reach[i, j]:
                reach[i, j] = True; q.append((i, j))
    for j in range(W0):
        for i in (0, H0 - 1):
            if a[i, j] == 0 and not reach[i, j]:
                reach[i, j] = True; q.append((i, j))
    while q:
        r, c = q.popleft()
        for dr, dc in _D4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H0 and 0 <= nc < W0 and a[nr, nc] == 0 and not reach[nr, nc]:
                reach[nr, nc] = True; q.append((nr, nc))
    out = a.copy(); seen = np.zeros((H0, W0), bool); found = False
    for i in range(H0):
        for j in range(W0):
            if a[i, j] == 0 and not reach[i, j] and not seen[i, j]:
                q = deque([(i, j)]); seen[i, j] = True; cells = [(i, j)]
                while q:
                    r, c = q.popleft()
                    for dr, dc in _D4:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H0 and 0 <= nc < W0 and a[nr, nc] == 0 and not reach[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True; q.append((nr, nc)); cells.append((nr, nc))
                col = 7 if len(cells) % 2 == 1 else 2
                for (r, c) in cells:
                    out[r, c] = col
                found = True
    return out if found else None


# =========================================================================== #
# dispatch                                                                      #
# =========================================================================== #
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


def _matches(prs, fn):
    for a, b in prs:
        o = fn(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


_RULES = [
    ("halve", _ref_halve, _build_halve),
    ("altfill", _ref_altfill, _build_altfill),
    ("quad", _ref_quad, _build_quad),
    ("border", _ref_border, _build_border),
    ("swap", _ref_swap, _build_swap),
    ("template", _ref_template, _build_template),
    ("histogram", _ref_histogram, _build_histogram),
    ("maxcols", _ref_maxcols, _build_maxcols),
    ("innercrop", _ref_innercrop, _build_innercrop),
    ("reflect4", _ref_reflect4, _build_reflect4),
    ("triangle", _ref_triangle, _build_triangle),
    ("blockup", _ref_blockup, _build_blockup),
    ("stamp12", _ref_stamp12, _build_stamp12),
    ("smallobj", _ref_smallobj, _build_smallobj),
    ("roomfill", _ref_roomfill, _build_roomfill),
]


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []
    out = []
    for name, ref, build in _RULES:
        try:
            if not _matches(prs, ref):
                continue
            m = build()
            onnx.checker.check_model(m, full_check=True)
        except Exception:
            continue
        out.append((f"crk2_5_{name}", m))
    return out
