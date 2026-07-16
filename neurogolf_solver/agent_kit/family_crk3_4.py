"""family_crk3_4 -- crack module for slice IDX=4 of the unsolved NeuroGolf tasks.

Each detected task gets its own structural detector (validated EXACTLY against the
provided train/test pairs in numpy) plus a static opset-10 ONNX builder.  All
intermediates are static-shape; data-dependent geometry uses computed index grids
+ ReduceMax/Mod/Abs/Less, never dynamic Resize/Pad.
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

    # convenience comparison helpers returning FLOAT 0/1
    def ge(self, a, b):
        return self.nd("Cast", [self.nd("Greater", [a, b])], to=F)

    def lt_abs(self, a, thr):
        return self.nd("Cast", [self.nd("Less", [self.nd("Abs", [a]), thr])], to=F)


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
# TASK 298 -- concentric-ring colours cycled one step outward (pointwise)     #
#   distinct colours from outer ring inward = [a, b, c]; recolor a->c, b->a,   #
#   c->b  (each colour becomes the colour of the ring just outside it).        #
# =========================================================================== #
def _t298_order(g):
    """list of distinct colours outer-ring -> inner, by walking corners."""
    H, W = g.shape
    order, seen = [], set()
    R = (min(H, W) + 1) // 2
    for d in range(R):
        col = int(g[d, d])
        if col not in seen:
            seen.add(col); order.append(col)
    return order


def _t298_rule(g):
    order = _t298_order(g)
    if len(order) < 2:
        return None
    perm = {order[i]: order[i - 1] for i in range(len(order))}
    o = g.copy()
    for k, v in perm.items():
        o[g == k] = v
    return o


def _t298_arith(g):
    """numpy mirror of the in-graph arithmetic remap (3 distinct ring colours
    read at corners (0,0),(1,1),(2,2); cycle a->c, b->a, c->b)."""
    H, W = g.shape
    if min(H, W) < 3:
        return None
    a, b, c = float(g[0, 0]), float(g[1, 1]), float(g[2, 2])
    S = g.astype(float)
    C = (S + (c - a) * (np.abs(S - a) < 0.5)
           + (a - b) * (np.abs(S - b) < 0.5)
           + (b - c) * (np.abs(S - c) < 0.5))
    return C.astype(int)


def _t298_detect(prs):
    for i, o in prs:
        if i.shape != o.shape:
            return False
        r = _t298_arith(i)
        if r is None or not np.array_equal(r, o):
            return False
    return True


def _slc2(g, src, r0, r1, c0, c1):
    s = g.i64([r0, c0]); e = g.i64([r1, c1]); a = g.i64([2, 3])
    return g.nd("Slice", [src, s, e, a])


def _t298_build():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    chvec = g.f([1, 10, 1, 1], list(range(10)))
    X = "input"
    M = g.nd("ReduceSum", [X], axes=[1], keepdims=1)            # [1,1,30,30]
    S = g.nd("ReduceSum", [g.nd("Mul", [X, chvec])], axes=[1], keepdims=1)
    a = _slc2(g, S, 0, 1, 0, 1)                                 # [1,1,1,1]
    b = _slc2(g, S, 1, 2, 1, 2)
    c = _slc2(g, S, 2, 3, 2, 3)
    eqa = g.lt_abs(g.nd("Sub", [S, a]), half)
    eqb = g.lt_abs(g.nd("Sub", [S, b]), half)
    eqc = g.lt_abs(g.nd("Sub", [S, c]), half)
    C = g.nd("Add", [S, g.nd("Mul", [g.nd("Sub", [c, a]), eqa])])
    C = g.nd("Add", [C, g.nd("Mul", [g.nd("Sub", [a, b]), eqb])])
    C = g.nd("Add", [C, g.nd("Mul", [g.nd("Sub", [b, c]), eqc])])
    onehot = g.lt_abs(g.nd("Sub", [C, chvec]), half)           # [1,10,30,30]
    g.nd("Mul", [onehot, M], "output")
    return _model(g)


# =========================================================================== #
# TASK 13 -- two single-cell markers -> periodic alternating stripes filling   #
#   the whole grid.  Stripes run along the LONGER grid axis; period = the       #
#   marker separation along that axis; the marker with the smaller coordinate   #
#   provides the first colour, alternating with the second.                     #
# =========================================================================== #
def _t13_rule(g):
    H, W = g.shape
    cells = np.argwhere(g != 0)
    if len(cells) != 2:
        return None
    (r0, c0), (r1, c1) = cells
    v0, v1 = g[r0, c0], g[r1, c1]
    o = np.zeros_like(g)
    vertical = (W >= H)
    if vertical:
        d = abs(c1 - c0)
        if d == 0:
            return None
        first = v0 if c0 < c1 else v1
        second = v1 if c0 < c1 else v0
        cmin = min(c0, c1)
        k = 0; c = cmin
        while c < W:
            o[:, c] = first if k % 2 == 0 else second
            c += d; k += 1
    else:
        d = abs(r1 - r0)
        if d == 0:
            return None
        first = v0 if r0 < r1 else v1
        second = v1 if r0 < r1 else v0
        rmin = min(r0, r1)
        k = 0; r = rmin
        while r < H:
            o[r, :] = first if k % 2 == 0 else second
            r += d; k += 1
    return o


def _t13_detect(prs):
    for i, o in prs:
        if i.shape != o.shape:
            return False
        cells = np.argwhere(i != 0)
        if len(cells) != 2:
            return False
        r = _t13_rule(i)
        if r is None or not np.array_equal(r, o):
            return False
    return True


def _t13_build():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    nhalf = g.f([1, 1, 1, 1], [-0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    two = g.f([1, 1, 1, 1], [2.0])
    big = g.f([1, 1, 1, 1], [1000.0])
    chvec = g.f([1, 10, 1, 1], list(range(10)))
    Jcol = g.f([1, 1, 1, G], list(range(G)))
    Jrow = g.f([1, 1, G, 1], list(range(G)))

    X = "input"
    # real-cell mask (incl. colour-0 cells, excl. padding)
    M = g.nd("ReduceSum", [X], axes=[1], keepdims=1)            # [1,1,30,30]
    # colour-value field
    Smul = g.nd("Mul", [X, chvec])
    S = g.nd("ReduceSum", [Smul], axes=[1], keepdims=1)         # [1,1,30,30]

    def stripe_color(axis_red, J, ax):
        # axis_red: reduce S along this axis to project markers onto the line
        val = g.nd("ReduceSum", [S], axes=[axis_red], keepdims=1)   # [.. line ..]
        marker = g.ge(val, half)                                    # 1 where marker
        inv = g.nd("Sub", [one, marker])
        Jm = g.nd("Mul", [J, marker])
        cand = g.nd("Add", [Jm, g.nd("Mul", [big, inv])])
        amin = g.nd("ReduceMin", [cand], axes=[ax], keepdims=1)     # smaller coord
        amax = g.nd("ReduceMax", [Jm], axes=[ax], keepdims=1)       # larger coord
        d = g.nd("Sub", [amax, amin])
        dsafe = g.nd("Max", [d, one])
        twod = g.nd("Mul", [dsafe, two])
        eqmin = g.lt_abs(g.nd("Sub", [J, amin]), half)
        eqmax = g.lt_abs(g.nd("Sub", [J, amax]), half)
        first = g.nd("ReduceSum", [g.nd("Mul", [val, eqmin])], axes=[ax], keepdims=1)
        second = g.nd("ReduceSum", [g.nd("Mul", [val, eqmax])], axes=[ax], keepdims=1)
        t = g.nd("Sub", [J, amin])
        ge0 = g.ge(t, nhalf)
        modd = g.nd("Mod", [t, dsafe], fmod=1)
        isstep = g.lt_abs(modd, half)
        stripe = g.nd("Mul", [ge0, isstep])
        par_mod = g.nd("Mod", [t, twod], fmod=1)
        parity = g.ge(par_mod, g.nd("Sub", [dsafe, half]))         # 1 if >= d
        diff = g.nd("Sub", [second, first])
        colr = g.nd("Add", [first, g.nd("Mul", [diff, parity])])
        line = g.nd("Mul", [stripe, colr])                          # colour along line
        return line

    # vertical: project markers onto columns (reduce rows axis=2), line over cols
    Vline = stripe_color(2, Jcol, 3)                               # [1,1,1,30]
    onehotV = g.lt_abs(g.nd("Sub", [Vline, chvec]), half)          # [1,10,1,30]
    outV = g.nd("Mul", [onehotV, M])                               # [1,10,30,30]

    # horizontal: project markers onto rows (reduce cols axis=3), line over rows
    Uline = stripe_color(3, Jrow, 2)                               # [1,1,30,1]
    onehotH = g.lt_abs(g.nd("Sub", [Uline, chvec]), half)          # [1,10,30,1]
    outH = g.nd("Mul", [onehotH, M])                               # [1,10,30,30]

    # orientation: vertical iff W >= H
    rowany = g.nd("ReduceMax", [M], axes=[3], keepdims=1)          # [1,1,30,1]
    maxrow = g.nd("ReduceMax", [g.nd("Mul", [Jrow, rowany])], axes=[2], keepdims=1)
    colany = g.nd("ReduceMax", [M], axes=[2], keepdims=1)          # [1,1,1,30]
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [Jcol, colany])], axes=[3], keepdims=1)
    vert = g.ge(g.nd("Sub", [maxcol, maxrow]), nhalf)             # [1,1,1,1]
    invv = g.nd("Sub", [one, vert])
    g.nd("Add", [g.nd("Mul", [vert, outV]), g.nd("Mul", [invv, outH])], "output")
    return _model(g)


# =========================================================================== #
# TASK 336 -- hollow 5-box: fill interior with 8, fill the single border gap   #
#   with 8 and shoot an 8 ray straight outward through the gap to the edge.     #
# =========================================================================== #
def _t336_rule(g):
    H, W = g.shape
    ys, xs = np.where(g == 5)
    if len(ys) == 0:
        return None
    rmin, rmax, cmin, cmax = ys.min(), ys.max(), xs.min(), xs.max()
    o = g.copy()
    o[rmin + 1:rmax, cmin + 1:cmax] = 8
    gap = None
    for c in range(cmin, cmax + 1):
        if g[rmin, c] != 5: gap = (rmin, c, 'up')
        if g[rmax, c] != 5: gap = (rmax, c, 'down')
    for r in range(rmin, rmax + 1):
        if g[r, cmin] != 5: gap = (r, cmin, 'left')
        if g[r, cmax] != 5: gap = (r, cmax, 'right')
    if gap is None:
        return o
    gr, gc, d = gap
    o[gr, gc] = 8
    if d == 'up':    o[0:rmin, gc] = 8
    elif d == 'down': o[rmax + 1:H, gc] = 8
    elif d == 'left': o[gr, 0:cmin] = 8
    else:            o[gr, cmax + 1:W] = 8
    return o


def _t336_detect(prs):
    seen = False
    for i, o in prs:
        if i.shape != o.shape:
            return False
        if set(np.unique(i).tolist()) - {0, 5}:
            return False
        r = _t336_rule(i)
        if r is None or not np.array_equal(r, o):
            return False
        seen = True
    return seen


def _t336_build():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    big = g.f([1, 1, 1, 1], [1000.0])
    Jrow = g.f([1, 1, G, 1], list(range(G)))
    Jcol = g.f([1, 1, 1, G], list(range(G)))
    e0 = g.f([1, 10, 1, 1], [1 if k == 0 else 0 for k in range(10)])
    e5 = g.f([1, 10, 1, 1], [1 if k == 5 else 0 for k in range(10)])
    e8 = g.f([1, 10, 1, 1], [1 if k == 8 else 0 for k in range(10)])
    X = "input"

    def eqv(a, b):
        return g.lt_abs(g.nd("Sub", [a, b]), half)

    M = g.nd("ReduceSum", [X], axes=[1], keepdims=1)
    M5 = _slc(g, X, 5, 6, 1)                                   # [1,1,30,30]
    inv5 = g.nd("Sub", [one, M5])

    row5 = g.nd("ReduceMax", [M5], axes=[3], keepdims=1)       # [1,1,30,1]
    col5 = g.nd("ReduceMax", [M5], axes=[2], keepdims=1)       # [1,1,1,30]
    rmin = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [Jrow, row5]),
                 g.nd("Mul", [big, g.nd("Sub", [one, row5])])])], axes=[2], keepdims=1)
    rmax = g.nd("ReduceMax", [g.nd("Mul", [Jrow, row5])], axes=[2], keepdims=1)
    cmin = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [Jcol, col5]),
                 g.nd("Mul", [big, g.nd("Sub", [one, col5])])])], axes=[3], keepdims=1)
    cmax = g.nd("ReduceMax", [g.nd("Mul", [Jcol, col5])], axes=[3], keepdims=1)

    # interior (strict)
    interior = g.nd("Mul", [
        g.nd("Mul", [g.ge(Jrow, rmin), g.nd("Cast", [g.nd("Less", [Jrow, rmax])], to=F)]),
        g.nd("Mul", [g.ge(Jcol, cmin), g.nd("Cast", [g.nd("Less", [Jcol, cmax])], to=F)]),
    ])

    # bounding-box edges & ranges
    rmin_lo = g.nd("Sub", [rmin, half]); rmax_hi = g.nd("Add", [rmax, half])
    cmin_lo = g.nd("Sub", [cmin, half]); cmax_hi = g.nd("Add", [cmax, half])
    r_edge = g.nd("Add", [eqv(Jrow, rmin), eqv(Jrow, rmax)])   # [1,1,30,1]
    c_edge = g.nd("Add", [eqv(Jcol, cmin), eqv(Jcol, cmax)])   # [1,1,1,30]
    r_in = g.nd("Mul", [g.ge(Jrow, rmin_lo),
              g.nd("Cast", [g.nd("Less", [Jrow, rmax_hi])], to=F)])  # [1,1,30,1]
    c_in = g.nd("Mul", [g.ge(Jcol, cmin_lo),
              g.nd("Cast", [g.nd("Less", [Jcol, cmax_hi])], to=F)])  # [1,1,1,30]
    perim = g.ge(g.nd("Add", [g.nd("Mul", [r_edge, c_in]),
                              g.nd("Mul", [c_edge, r_in])]), half)    # [1,1,30,30]
    gap = g.nd("Mul", [perim, inv5])                          # one cell
    gr = g.nd("ReduceSum", [g.nd("Mul", [gap, Jrow])], axes=[2, 3], keepdims=1)
    gc = g.nd("ReduceSum", [g.nd("Mul", [gap, Jcol])], axes=[2, 3], keepdims=1)

    is_top = eqv(gr, rmin); is_bot = eqv(gr, rmax)
    is_left = eqv(gc, cmin); is_right = eqv(gc, cmax)
    col_is_gc = eqv(Jcol, gc)                                 # [1,1,1,30]
    row_is_gr = eqv(Jrow, gr)                                 # [1,1,30,1]
    rows_above = g.nd("Cast", [g.nd("Less", [Jrow, rmin])], to=F)
    rows_below = g.ge(Jrow, g.nd("Add", [rmax, half]))
    cols_left = g.nd("Cast", [g.nd("Less", [Jcol, cmin])], to=F)
    cols_right = g.ge(Jcol, g.nd("Add", [cmax, half]))
    vert = g.nd("Mul", [col_is_gc, g.nd("Add", [g.nd("Mul", [is_top, rows_above]),
                                                g.nd("Mul", [is_bot, rows_below])])])
    horz = g.nd("Mul", [row_is_gr, g.nd("Add", [g.nd("Mul", [is_left, cols_left]),
                                                g.nd("Mul", [is_right, cols_right])])])
    ray = g.nd("Add", [vert, horz])                          # [1,1,30,30]

    eight = g.nd("Mul", [g.ge(g.nd("Add", [g.nd("Add", [interior, gap]), ray]), half), M])
    five = M5
    out0 = g.nd("Sub", [g.nd("Sub", [M, five]), eight])
    g.nd("Add", [g.nd("Add", [g.nd("Mul", [out0, e0]), g.nd("Mul", [five, e5])]),
                 g.nd("Mul", [eight, e8])], "output")
    return _model(g)


# =========================================================================== #
# TASK 235 -- three 4x4 shapes (cols 0-3 / 5-8 / 10-13) classified against a    #
#   fixed 4-template dictionary -> a 3x3 grid whose row k is the colour of      #
#   block k repeated.                                                           #
# =========================================================================== #
_T235 = {
    2: np.ones((4, 4), int),
    3: np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 1, 1, 1]]),
    4: np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1]]),
    8: np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]]),
}


def _t235_rule(g):
    if g.shape != (4, 14):
        return None
    B = (g > 0).astype(int)
    blocks = [B[:, 0:4], B[:, 5:9], B[:, 10:14]]
    o = np.zeros((3, 3), int)
    for k, blk in enumerate(blocks):
        col = 0
        for c, t in _T235.items():
            if blk.sum() + t.sum() - 2 * (blk * t).sum() == 0:
                col = c
        if col == 0:
            return None
        o[k, :] = col
    return o


def _t235_detect(prs):
    seen = False
    for i, o in prs:
        r = _t235_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
        seen = True
    return seen


def _t235_build():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    twof = g.f([1, 1, 1, 1], [2.0])
    chvec = g.f([1, 10, 1, 1], list(range(10)))
    Jrow = g.f([1, 1, G, 1], list(range(G)))
    Jcol = g.f([1, 1, 1, G], list(range(G)))
    tmpl = {c: g.f([1, 1, 4, 4], _T235[c].ravel()) for c in _T235}
    sumt = {c: g.f([1, 1, 1, 1], [float(_T235[c].sum())]) for c in _T235}
    colc = {c: g.f([1, 1, 1, 1], [float(c)]) for c in _T235}
    B = _slc(g, "input", 5, 6, 1)                              # [1,1,30,30]

    def block_color(c0):
        blk = g.nd("Slice", [B, g.i64([0, c0]), g.i64([4, c0 + 4]), g.i64([2, 3])])
        sb = g.nd("ReduceSum", [blk], axes=[2, 3], keepdims=1)
        acc = None
        for c in _T235:
            dot = g.nd("ReduceSum", [g.nd("Mul", [blk, tmpl[c]])], axes=[2, 3], keepdims=1)
            dist = g.nd("Sub", [g.nd("Add", [sb, sumt[c]]), g.nd("Mul", [twof, dot])])
            match = g.nd("Cast", [g.nd("Less", [dist, half])], to=F)
            term = g.nd("Mul", [match, colc[c]])
            acc = term if acc is None else g.nd("Add", [acc, term])
        return acc                                            # [1,1,1,1]

    col0 = block_color(0); col1 = block_color(5); col2 = block_color(10)
    U = g.nd("Add", [g.nd("Add", [
        g.nd("Mul", [col0, g.lt_abs(Jrow, half)]),
        g.nd("Mul", [col1, g.lt_abs(g.nd("Sub", [Jrow, g.f([1, 1, 1, 1], [1.0])]), half)])]),
        g.nd("Mul", [col2, g.lt_abs(g.nd("Sub", [Jrow, twof]), half)])])  # [1,1,30,1]
    lt3r = g.nd("Cast", [g.nd("Less", [Jrow, g.f([1, 1, 1, 1], [2.5])])], to=F)
    lt3c = g.nd("Cast", [g.nd("Less", [Jcol, g.f([1, 1, 1, 1], [2.5])])], to=F)
    C = g.nd("Mul", [U, lt3c])                                # [1,1,30,30]
    grid = g.nd("Mul", [lt3r, lt3c])
    onehot = g.lt_abs(g.nd("Sub", [C, chvec]), half)
    g.nd("Mul", [onehot, grid], "output")
    return _model(g)


# =========================================================================== #
# TASK 222 -- keep only the solid monochrome rectangle (>=2x3 or 3x2), blank    #
#   out all the surrounding single-colour noise.                                #
# =========================================================================== #
def _t222_rule(g):
    H, W = g.shape

    def uni(wh, ww):
        keep = np.zeros((H, W), bool)
        for r in range(H - wh + 1):
            for c in range(W - ww + 1):
                v = g[r, c]
                if v != 0 and (g[r:r + wh, c:c + ww] == v).all():
                    keep[r:r + wh, c:c + ww] = True
        return keep
    keep = uni(2, 3) | uni(3, 2)
    return np.where(keep, g, 0)


def _t222_detect(prs):
    seen = False
    for i, o in prs:
        if i.shape != o.shape:
            return False
        r = _t222_rule(i)
        if not np.array_equal(r, o):
            return False
        # require the kept region to be a genuine block (non-trivial) and the
        # input to be otherwise noisy -> avoid matching trivial tasks
        if (o != 0).sum() < 6 or (i != 0).sum() < 30:
            return False
        seen = True
    return seen


def _mat_colL(n):
    M = np.zeros((G, G), np.float32)
    for c in range(G):
        if c + n < G:
            M[c + n, c] = 1
    return M


def _mat_colR(n):
    M = np.zeros((G, G), np.float32)
    for c in range(G):
        if c - n >= 0:
            M[c - n, c] = 1
    return M


def _mat_rowU(n):
    B = np.zeros((G, G), np.float32)
    for r in range(G):
        if r + n < G:
            B[r, r + n] = 1
    return B


def _mat_rowD(n):
    B = np.zeros((G, G), np.float32)
    for r in range(G):
        if r - n >= 0:
            B[r, r - n] = 1
    return B


def _t222_build():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    chvec = g.f([1, 10, 1, 1], list(range(10)))
    A1 = g.f([G, G], _mat_colL(1)); A2 = g.f([G, G], _mat_colL(2))
    AR1 = g.f([G, G], _mat_colR(1)); AR2 = g.f([G, G], _mat_colR(2))
    B1 = g.f([G, G], _mat_rowU(1)); B2 = g.f([G, G], _mat_rowU(2))
    BD1 = g.f([G, G], _mat_rowD(1)); BD2 = g.f([G, G], _mat_rowD(2))
    X = "input"
    M = g.nd("ReduceSum", [X], axes=[1], keepdims=1)
    S = g.nd("ReduceSum", [g.nd("Mul", [X, chvec])], axes=[1], keepdims=1)

    cL = lambda t, n: g.nd("MatMul", [t, A1 if n == 1 else A2])
    cR = lambda t, n: g.nd("MatMul", [t, AR1 if n == 1 else AR2])
    rU = lambda t, n: g.nd("MatMul", [B1 if n == 1 else B2, t])
    rD = lambda t, n: g.nd("MatMul", [BD1 if n == 1 else BD2, t])
    eq = lambda a, b: g.lt_abs(g.nd("Sub", [a, b]), half)
    nz = lambda a: g.ge(a, half)

    def prod(xs):
        acc = xs[0]
        for x in xs[1:]:
            acc = g.nd("Mul", [acc, x])
        return acc

    def add(xs):
        acc = xs[0]
        for x in xs[1:]:
            acc = g.nd("Add", [acc, x])
        return acc

    # anchor 2x3
    s1 = cL(S, 1); s2 = cL(S, 2); d = rU(S, 1); d1 = cL(d, 1); d2 = cL(d, 2)
    a23 = prod([nz(S), eq(S, s1), eq(S, s2), eq(S, d), eq(S, d1), eq(S, d2)])
    cov23 = g.ge(add([a23, cR(a23, 1), cR(a23, 2),
                       rD(a23, 1), rD(cR(a23, 1), 1), rD(cR(a23, 2), 1)]), half)
    # anchor 3x2
    u1 = rU(S, 1); u1c = cL(u1, 1); u2 = rU(S, 2); u2c = cL(u2, 1)
    a32 = prod([nz(S), eq(S, s1), eq(S, u1), eq(S, u1c), eq(S, u2), eq(S, u2c)])
    cov32 = g.ge(add([a32, cR(a32, 1), rD(a32, 1), rD(cR(a32, 1), 1),
                       rD(a32, 2), rD(cR(a32, 1), 2)]), half)

    keep = g.ge(g.nd("Add", [cov23, cov32]), half)
    C = g.nd("Mul", [keep, S])
    onehot = g.lt_abs(g.nd("Sub", [C, chvec]), half)
    g.nd("Mul", [onehot, M], "output")
    return _model(g)


# =========================================================================== #
# TASK 154 / 390 -- two open 2-brackets define two mirror lines (their solid    #
#   bars).  Each 5 reflects across the NEAREST bar to the outside; originals     #
#   removed, brackets kept.                                                      #
# =========================================================================== #
def _maxrun(a):
    best = cur = 0
    for v in a:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def _refl_rule(g):
    H, W = g.shape
    two = (g == 2); five = (g == 5)
    barrows = [r for r in range(H) if _maxrun(two[r]) >= 3]
    barcols = [c for c in range(W) if _maxrun(two[:, c]) >= 3]
    o = g.copy(); o[five] = 0
    if len(barrows) == 2 and len(barcols) == 0:
        p = sorted(barrows)
        for (r, c) in np.argwhere(five):
            pr = p[0] if abs(r - p[0]) <= abs(r - p[1]) else p[1]
            nr = 2 * pr - r
            if 0 <= nr < H:
                o[nr, c] = 5
    elif len(barcols) == 2 and len(barrows) == 0:
        p = sorted(barcols)
        for (r, c) in np.argwhere(five):
            pc = p[0] if abs(c - p[0]) <= abs(c - p[1]) else p[1]
            nc = 2 * pc - c
            if 0 <= nc < W:
                o[r, nc] = 5
    else:
        return None
    return o


def _refl_detect(prs):
    seen = False
    for i, o in prs:
        if i.shape != o.shape:
            return False
        if set(np.unique(i).tolist()) - {0, 2, 5}:
            return False
        r = _refl_rule(i)
        if r is None or not np.array_equal(r, o):
            return False
        seen = True
    return seen


def _refl_build():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    two = g.f([1, 1, 1, 1], [2.0])
    big = g.f([1, 1, 1, 1], [1000.0])
    Jrow = g.f([1, 1, G, 1], list(range(G)))
    Jcol = g.f([1, 1, 1, G], list(range(G)))
    A1 = g.f([G, G], _mat_colL(1)); A2 = g.f([G, G], _mat_colL(2))
    B1 = g.f([G, G], _mat_rowU(1)); B2 = g.f([G, G], _mat_rowU(2))
    e0 = g.f([1, 10, 1, 1], [1 if k == 0 else 0 for k in range(10)])
    e2 = g.f([1, 10, 1, 1], [1 if k == 2 else 0 for k in range(10)])
    e5 = g.f([1, 10, 1, 1], [1 if k == 5 else 0 for k in range(10)])
    X = "input"
    M = g.nd("ReduceSum", [X], axes=[1], keepdims=1)
    T2 = _slc(g, X, 2, 3, 1)
    F5 = _slc(g, X, 5, 6, 1)
    cL = lambda t, n: g.nd("MatMul", [t, A1 if n == 1 else A2])
    rU = lambda t, n: g.nd("MatMul", [B1 if n == 1 else B2, t])

    h3 = g.nd("Mul", [g.nd("Mul", [T2, cL(T2, 1)]), cL(T2, 2)])
    v3 = g.nd("Mul", [g.nd("Mul", [T2, rU(T2, 1)]), rU(T2, 2)])
    rowbar = g.nd("ReduceMax", [h3], axes=[3], keepdims=1)      # [1,1,30,1]
    colbar = g.nd("ReduceMax", [v3], axes=[2], keepdims=1)      # [1,1,1,30]
    sumrow = g.nd("ReduceSum", [rowbar], axes=[2], keepdims=1)
    sumcol = g.nd("ReduceSum", [colbar], axes=[3], keepdims=1)
    horiz = g.ge(g.nd("Sub", [sumrow, sumcol]), half)          # [1,1,1,1]

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

    # horizontal (reflect rows)
    p1, p2 = bars(rowbar, Jrow, 2)
    tr = target(Jrow, p1, p2)                                  # [1,1,30,1]
    trT = g.nd("Transpose", [tr], perm=[0, 1, 3, 2])           # [1,1,1,30]
    Tr = g.lt_abs(g.nd("Sub", [Jrow, trT]), half)              # [1,1,30,30] new,x
    reflH = g.nd("MatMul", [Tr, F5])
    # vertical (reflect cols)
    p1c, p2c = bars(colbar, Jcol, 3)
    tc = target(Jcol, p1c, p2c)                                # [1,1,1,30]
    tcT = g.nd("Transpose", [tc], perm=[0, 1, 3, 2])           # [1,1,30,1]
    Tc = g.lt_abs(g.nd("Sub", [tcT, Jcol]), half)              # [1,1,30,30] x,new
    reflV = g.nd("MatMul", [F5, Tc])

    refl = g.nd("Add", [g.nd("Mul", [horiz, reflH]),
                        g.nd("Mul", [g.nd("Sub", [one, horiz]), reflV])])
    out5 = g.nd("Mul", [refl, M])
    out0 = g.nd("Sub", [g.nd("Sub", [M, T2]), out5])
    g.nd("Add", [g.nd("Add", [g.nd("Mul", [out0, e0]), g.nd("Mul", [T2, e2])]),
                 g.nd("Mul", [out5, e5])], "output")
    return _model(g)


# =========================================================================== #
# TASK 181 -- mirror the 8-shape horizontally to the side indicated by the      #
#   4-arrow's tip (the 4-shape's heavier outer column).                          #
# =========================================================================== #
def _t181_rule(g):
    H, W = g.shape
    er, ec = np.where(g == 8); fr, fc = np.where(g == 4)
    if len(er) == 0 or len(fr) == 0:
        return None
    c8min, c8max = ec.min(), ec.max()
    fcol = np.array([(g[:, c] == 4).sum() for c in range(W)])
    fcols = np.where(fcol > 0)[0]
    fcmin, fcmax = fcols.min(), fcols.max()
    left = fcol[fcmin] > fcol[fcmax]
    o = g.copy()
    for r, c in zip(er, ec):
        nc = 2 * c8min - 1 - c if left else 2 * c8max + 1 - c
        if 0 <= nc < W:
            o[r, nc] = 8
    return o


def _t181_detect(prs):
    seen = False
    for i, o in prs:
        if i.shape != o.shape:
            return False
        if set(np.unique(i).tolist()) - {0, 4, 8}:
            return False
        r = _t181_rule(i)
        if r is None or not np.array_equal(r, o):
            return False
        seen = True
    return seen


def _t181_build():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    two = g.f([1, 1, 1, 1], [2.0])
    big = g.f([1, 1, 1, 1], [1000.0])
    Jcol = g.f([1, 1, 1, G], list(range(G)))
    e0 = g.f([1, 10, 1, 1], [1 if k == 0 else 0 for k in range(10)])
    e4 = g.f([1, 10, 1, 1], [1 if k == 4 else 0 for k in range(10)])
    e8 = g.f([1, 10, 1, 1], [1 if k == 8 else 0 for k in range(10)])
    X = "input"
    M = g.nd("ReduceSum", [X], axes=[1], keepdims=1)
    E = _slc(g, X, 8, 9, 1)
    Fo = _slc(g, X, 4, 5, 1)

    def eqv(a, b):
        return g.lt_abs(g.nd("Sub", [a, b]), half)

    col8 = g.nd("ReduceMax", [E], axes=[2], keepdims=1)        # [1,1,1,30]
    inv8 = g.nd("Sub", [one, col8])
    c8min = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [Jcol, col8]),
                  g.nd("Mul", [big, inv8])])], axes=[3], keepdims=1)
    c8max = g.nd("ReduceMax", [g.nd("Mul", [Jcol, col8])], axes=[3], keepdims=1)

    fcol = g.nd("ReduceSum", [Fo], axes=[2], keepdims=1)       # [1,1,1,30] counts
    fany = g.ge(fcol, half)
    finv = g.nd("Sub", [one, fany])
    fcmin = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [Jcol, fany]),
                   g.nd("Mul", [big, finv])])], axes=[3], keepdims=1)
    fcmax = g.nd("ReduceMax", [g.nd("Mul", [Jcol, fany])], axes=[3], keepdims=1)
    cntmin = g.nd("ReduceSum", [g.nd("Mul", [fcol, eqv(Jcol, fcmin)])], axes=[3], keepdims=1)
    cntmax = g.nd("ReduceSum", [g.nd("Mul", [fcol, eqv(Jcol, fcmax)])], axes=[3], keepdims=1)
    left = g.ge(g.nd("Sub", [cntmin, cntmax]), half)          # [1,1,1,1]

    # target column for each source col c
    tL = g.nd("Sub", [g.nd("Sub", [g.nd("Mul", [two, c8min]), one]), Jcol])  # 2c8min-1-c
    tR = g.nd("Sub", [g.nd("Add", [g.nd("Mul", [two, c8max]), one]), Jcol])  # 2c8max+1-c
    tgt = g.nd("Add", [g.nd("Mul", [left, tL]),
                       g.nd("Mul", [g.nd("Sub", [one, left]), tR])])  # [1,1,1,30]
    tgtT = g.nd("Transpose", [tgt], perm=[0, 1, 3, 2])        # [1,1,30,1] over c
    Tc = g.lt_abs(g.nd("Sub", [tgtT, Jcol]), half)            # [1,1,30,30] c,nc
    reflE = g.nd("MatMul", [E, Tc])

    out8 = g.nd("Mul", [g.ge(g.nd("Add", [E, reflE]), half), M])
    out0 = g.nd("Sub", [g.nd("Sub", [M, out8]), Fo])
    g.nd("Add", [g.nd("Add", [g.nd("Mul", [out0, e0]), g.nd("Mul", [Fo, e4])]),
                 g.nd("Mul", [out8, e8])], "output")
    return _model(g)


# =========================================================================== #
# TASK 190 -- a solid block with diagonal "stub" cells; extend each stub as a    #
#   diagonal ray (away from the block) to the grid edge.                         #
# =========================================================================== #
def _t190_rule(g):
    H, W = g.shape
    col = (g != 0)
    if col.sum() == 0:
        return None
    colorval = g.max()
    orth = np.zeros((H, W), int)
    orth[1:, :] += col[:-1, :]; orth[:-1, :] += col[1:, :]
    orth[:, 1:] += col[:, :-1]; orth[:, :-1] += col[:, 1:]
    stub = col & (orth == 0)
    if stub.sum() == 0:
        return None
    out = col.copy()
    for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        for (r, c) in np.argwhere(stub):
            pr, pc = r - dr, c - dc
            if 0 <= pr < H and 0 <= pc < W and col[pr, pc]:
                rr, cc = r, c
                while 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = True; rr += dr; cc += dc
    return np.where(out, colorval, 0)


def _t190_detect(prs):
    seen = False
    for i, o in prs:
        if i.shape != o.shape or len(set(np.unique(i).tolist()) - {0}) != 1:
            return False
        r = _t190_rule(i)
        if r is None or not np.array_equal(r, o):
            return False
        seen = True
    return seen


_POW = [1, 2, 4, 8, 16]


def _t190_build():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    chvec = g.f([1, 10, 1, 1], list(range(10)))
    AL = {s: g.f([G, G], _mat_colL(s)) for s in _POW}
    AR = {s: g.f([G, G], _mat_colR(s)) for s in _POW}
    BU = {s: g.f([G, G], _mat_rowU(s)) for s in _POW}
    BD = {s: g.f([G, G], _mat_rowD(s)) for s in _POW}
    X = "input"
    M = g.nd("ReduceSum", [X], axes=[1], keepdims=1)
    S = g.nd("ReduceSum", [g.nd("Mul", [X, chvec])], axes=[1], keepdims=1)
    col = g.ge(S, half)
    colorval = g.nd("ReduceMax", [S], axes=[2, 3], keepdims=1)

    sL = lambda t, s: g.nd("MatMul", [t, AL[s]])   # t[r,c+s]
    sR = lambda t, s: g.nd("MatMul", [t, AR[s]])   # t[r,c-s]
    sU = lambda t, s: g.nd("MatMul", [BU[s], t])   # t[r+s,c]
    sD = lambda t, s: g.nd("MatMul", [BD[s], t])   # t[r-s,c]
    orr = lambda a, b: g.ge(g.nd("Add", [a, b]), half)

    above = sD(col, 1); below = sU(col, 1); lft = sR(col, 1); rgt = sL(col, 1)
    orth = g.nd("Add", [g.nd("Add", [above, below]), g.nd("Add", [lft, rgt])])
    stub = g.nd("Mul", [col, g.nd("Cast", [g.nd("Less", [orth, half])], to=F)])

    # per direction shift fns: shift ray in +dir by s
    shifts = {
        "DR": lambda t, s: sR(sD(t, s), s),   # (r-s,c-s)
        "DL": lambda t, s: sL(sD(t, s), s),   # (r-s,c+s)
        "UR": lambda t, s: sR(sU(t, s), s),   # (r+s,c-s)
        "UL": lambda t, s: sL(sU(t, s), s),   # (r+s,c+s)
    }
    out = col
    for name, sh in shifts.items():
        seed = g.nd("Mul", [stub, sh(col, 1)])   # anti-dir neighbour colored
        ray = seed
        for s in _POW:
            ray = orr(ray, sh(ray, s))
        out = orr(out, ray)

    C = g.nd("Mul", [out, colorval])
    onehot = g.lt_abs(g.nd("Sub", [C, chvec]), half)
    g.nd("Mul", [onehot, M], "output")
    return _model(g)


# =========================================================================== #
# TASK 64 -- a solid rectangle + isolated marker pixels of a 3rd colour.  Each   #
#   marker aligned with the rectangle (row within its rows, or col within its    #
#   cols) shoots a line in its own colour filling the gap to the rectangle.      #
# =========================================================================== #
def _t64_rule(g):
    from collections import Counter
    H, W = g.shape
    real = (g != 0)
    if real.sum() == 0:
        return None
    bg = Counter(g[real].tolist()).most_common(1)[0][0]
    others = [c for c in np.unique(g) if c != 0 and c != bg]
    rectc = None
    for c in others:
        ys, xs = np.where(g == c)
        area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        if len(ys) == area and len(ys) >= 4:
            rectc = c; r1, r2, c1, c2 = ys.min(), ys.max(), xs.min(), xs.max()
    if rectc is None:
        return None
    o = g.copy()
    for mc in others:
        if mc == rectc:
            continue
        for (r, c) in np.argwhere(g == mc):
            if r1 <= r <= r2:
                if c < c1: o[r, c + 1:c1] = mc
                elif c > c2: o[r, c2 + 1:c] = mc
            elif c1 <= c <= c2:
                if r < r1: o[r + 1:r1, c] = mc
                elif r > r2: o[r2 + 1:r, c] = mc
    return o


def _t64_detect(prs):
    seen = False
    for i, o in prs:
        if i.shape != o.shape:
            return False
        if 0 in np.unique(i) and (i == 0).sum() and (i != 0).sum() < 20:
            return False
        r = _t64_rule(i)
        if r is None or not np.array_equal(r, o):
            return False
        seen = True
    return seen


def _tri_gt(G_):    # M[a,b]=1 if a>b
    return np.array([[1.0 if a > b else 0.0 for b in range(G_)] for a in range(G_)], np.float32)


def _tri_lt(G_):
    return np.array([[1.0 if a < b else 0.0 for b in range(G_)] for a in range(G_)], np.float32)


def _t64_build():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    big = g.f([1, 1, 1, 1], [1000.0])
    chvec = g.f([1, 10, 1, 1], list(range(10)))
    Jrow = g.f([1, 1, G, 1], list(range(G)))
    Jcol = g.f([1, 1, 1, G], list(range(G)))
    A1 = g.f([G, G], _mat_colL(1)); AR1 = g.f([G, G], _mat_colR(1))
    B1 = g.f([G, G], _mat_rowU(1)); BD1 = g.f([G, G], _mat_rowD(1))
    Ugt = g.f([G, G], _tri_gt(G)); Ult = g.f([G, G], _tri_lt(G))
    Lgt = g.f([G, G], _tri_gt(G)); Llt = g.f([G, G], _tri_lt(G))
    X = "input"
    M = g.nd("ReduceSum", [X], axes=[1], keepdims=1)
    S = g.nd("ReduceSum", [g.nd("Mul", [X, chvec])], axes=[1], keepdims=1)
    cL = lambda t: g.nd("MatMul", [t, A1])
    cR = lambda t: g.nd("MatMul", [t, AR1])
    rU = lambda t: g.nd("MatMul", [B1, t])
    rD = lambda t: g.nd("MatMul", [BD1, t])
    eq = lambda a, b: g.lt_abs(g.nd("Sub", [a, b]), half)
    nz = lambda a: g.ge(a, half)
    ci = lambda op, a, b: g.nd("Cast", [g.nd(op, [a, b])], to=F)

    cnt = g.nd("ReduceSum", [X], axes=[2, 3], keepdims=1)       # [1,10,1,1]
    maxcnt = g.nd("ReduceMax", [cnt], axes=[1], keepdims=1)
    bgoh = ci("Greater", cnt, g.nd("Sub", [maxcnt, half]))      # one-hot bg ch
    bgmask = g.nd("ReduceSum", [g.nd("Mul", [X, bgoh])], axes=[1], keepdims=1)
    nonbg = g.nd("Sub", [M, bgmask])

    ru1 = rU(S)
    anchor = g.nd("Mul", [g.nd("Mul", [nz(S), eq(S, cL(S))]),
                          g.nd("Mul", [eq(S, ru1), eq(S, cL(ru1))])])
    keep = g.ge(g.nd("Add", [g.nd("Add", [anchor, cR(anchor)]),
                             g.nd("Add", [rD(anchor), rD(cR(anchor))])]), half)
    rectM = g.nd("Mul", [keep, nonbg])
    markM = g.nd("Mul", [nonbg, g.nd("Sub", [one, keep])])

    rowR = g.nd("ReduceMax", [rectM], axes=[3], keepdims=1)
    colR = g.nd("ReduceMax", [rectM], axes=[2], keepdims=1)
    r1 = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [Jrow, rowR]),
              g.nd("Mul", [big, g.nd("Sub", [one, rowR])])])], axes=[2], keepdims=1)
    r2 = g.nd("ReduceMax", [g.nd("Mul", [Jrow, rowR])], axes=[2], keepdims=1)
    c1 = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [Jcol, colR]),
              g.nd("Mul", [big, g.nd("Sub", [one, colR])])])], axes=[3], keepdims=1)
    c2 = g.nd("ReduceMax", [g.nd("Mul", [Jcol, colR])], axes=[3], keepdims=1)
    m = g.nd("ReduceMax", [g.nd("Mul", [markM, S])], axes=[2, 3], keepdims=1)

    sufR = g.ge(g.nd("MatMul", [markM, Ugt]), half)
    preL = g.ge(g.nd("MatMul", [markM, Ult]), half)
    sufD = g.ge(g.nd("MatMul", [Llt, markM]), half)   # markers strictly below
    preU = g.ge(g.nd("MatMul", [Lgt, markM]), half)   # markers strictly above

    rowIn = g.nd("Mul", [g.ge(Jrow, g.nd("Sub", [r1, half])),
                         ci("Less", Jrow, g.nd("Add", [r2, half]))])
    colIn = g.nd("Mul", [g.ge(Jcol, g.nd("Sub", [c1, half])),
                         ci("Less", Jcol, g.nd("Add", [c2, half]))])
    rfill = g.nd("Mul", [g.nd("Mul", [rowIn, ci("Greater", Jcol, c2)]), sufR])
    lfill = g.nd("Mul", [g.nd("Mul", [rowIn, ci("Less", Jcol, c1)]), preL])
    dfill = g.nd("Mul", [g.nd("Mul", [colIn, ci("Greater", Jrow, r2)]), sufD])
    ufill = g.nd("Mul", [g.nd("Mul", [colIn, ci("Less", Jrow, r1)]), preU])
    fill = g.ge(g.nd("Add", [g.nd("Add", [rfill, lfill]),
                             g.nd("Add", [dfill, ufill])]), half)
    C = g.nd("Add", [g.nd("Mul", [S, g.nd("Sub", [one, fill])]), g.nd("Mul", [m, fill])])
    onehot = g.lt_abs(g.nd("Sub", [C, chvec]), half)
    g.nd("Mul", [onehot, M], "output")
    return _model(g)


# =========================================================================== #
# TASK 275 -- fractal: input is two adjacent n x n squares; the monochrome one   #
#   is a SHAPE stamp, the other a colour SELECTOR; output (n^2 x n^2) places the  #
#   shape coloured by selector[i,j] at block (i,j)  (a Kronecker product).        #
# =========================================================================== #
def _t275_rule(g):
    H, W = g.shape
    if W == 2 * H:
        n = H; A = g[:, :n]; B = g[:, n:]
    elif H == 2 * W:
        n = W; A = g[:n, :]; B = g[n:, :]
    else:
        return None
    mono = lambda blk: len(np.unique(blk[blk != 0])) <= 1
    if mono(B):
        Sb = (B != 0).astype(int); Gs = A
    elif mono(A):
        Sb = (A != 0).astype(int); Gs = B
    else:
        return None
    out = np.zeros((n * n, n * n), int)
    for i in range(n):
        for j in range(n):
            out[i * n:(i + 1) * n, j * n:(j + 1) * n] = Sb * Gs[i, j]
    return out


def _t275_detect(prs):
    seen = False
    for i, o in prs:
        r = _t275_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
        seen = True
    return seen


def _t275_build():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    two = g.f([1, 1, 1, 1], [2.0])
    big = g.f([1, 1, 1, 1], [1000.0])
    chvec = g.f([1, 10, 1, 1], list(range(10)))
    Jrow = g.f([1, 1, G, 1], list(range(G)))
    Jcol = g.f([1, 1, 1, G], list(range(G)))
    X = "input"
    lt = lambda a, b: g.nd("Cast", [g.nd("Less", [a, b])], to=F)
    gt = lambda a, b: g.nd("Cast", [g.nd("Greater", [a, b])], to=F)
    eqa = lambda a, b: g.lt_abs(g.nd("Sub", [a, b]), half)
    M = g.nd("ReduceSum", [X], axes=[1], keepdims=1)
    S0 = g.nd("ReduceSum", [g.nd("Mul", [X, chvec])], axes=[1], keepdims=1)

    rowany = g.nd("ReduceMax", [M], axes=[3], keepdims=1)
    colany = g.nd("ReduceMax", [M], axes=[2], keepdims=1)
    Hin = g.nd("Add", [g.nd("ReduceMax", [g.nd("Mul", [Jrow, rowany])], axes=[2], keepdims=1), one])
    Win = g.nd("Add", [g.nd("ReduceMax", [g.nd("Mul", [Jcol, colany])], axes=[3], keepdims=1), one])
    horiz = g.ge(g.nd("Sub", [Win, Hin]), half)
    n = g.nd("Add", [g.nd("Mul", [horiz, Hin]),
                     g.nd("Mul", [g.nd("Sub", [one, horiz]), Win])])
    twoN = g.nd("Mul", [n, two]); nm = g.nd("Sub", [n, half])

    rlt = lt(Jrow, n); clt = lt(Jcol, n)
    maskA = g.nd("Mul", [rlt, clt])
    Afield = g.nd("Mul", [S0, maskA])
    # horizontal: B = cols [n,2n) shifted left by n
    maskBh = g.nd("Mul", [rlt, g.nd("Mul", [gt(Jcol, nm), lt(Jcol, twoN)])])
    Brawh = g.nd("Mul", [S0, maskBh])
    Sh = g.lt_abs(g.nd("Sub", [Jrow, g.nd("Add", [Jcol, n])]), half)   # [k,c] k==c+n
    Bfh = g.nd("MatMul", [Brawh, Sh])
    # vertical: B = rows [n,2n) shifted up by n
    maskBv = g.nd("Mul", [g.nd("Mul", [gt(Jrow, nm), lt(Jrow, twoN)]), clt])
    Brawv = g.nd("Mul", [S0, maskBv])
    Up = g.lt_abs(g.nd("Sub", [Jcol, g.nd("Add", [Jrow, n])]), half)   # [r,k] k==r+n
    Bfv = g.nd("MatMul", [Up, Brawv])
    Bfield = g.nd("Add", [g.nd("Mul", [horiz, Bfh]),
                          g.nd("Mul", [g.nd("Sub", [one, horiz]), Bfv])])

    # mono check on B
    Bnz = g.ge(Bfield, half)
    maxB = g.nd("ReduceMax", [Bfield], axes=[2, 3], keepdims=1)
    minB = g.nd("ReduceMin", [g.nd("Add", [Bfield, g.nd("Mul", [big, g.nd("Sub", [one, Bnz])])])],
                axes=[2, 3], keepdims=1)
    monoB = g.lt_abs(g.nd("Sub", [maxB, minB]), half)
    notB = g.nd("Sub", [one, monoB])
    Sbin = g.nd("Add", [g.nd("Mul", [monoB, Bnz]),
                        g.nd("Mul", [notB, g.ge(Afield, half)])])
    Gsel = g.nd("Add", [g.nd("Mul", [monoB, Afield]), g.nd("Mul", [notB, Bfield])])

    E = eqa(g.nd("Floor", [g.nd("Div", [Jrow, n])]), Jcol)   # [p,i]
    T = eqa(g.nd("Mod", [Jrow, n], fmod=1), Jcol)            # [p,a]
    ET = g.nd("Transpose", [E], perm=[0, 1, 3, 2])
    TT = g.nd("Transpose", [T], perm=[0, 1, 3, 2])
    outG = g.nd("MatMul", [g.nd("MatMul", [E, Gsel]), ET])
    outS = g.nd("MatMul", [g.nd("MatMul", [T, Sbin]), TT])
    C = g.nd("Mul", [outG, outS])
    n2 = g.nd("Mul", [n, n])
    outmask = g.nd("Mul", [lt(Jrow, n2), lt(Jcol, n2)])
    onehot = g.lt_abs(g.nd("Sub", [C, chvec]), half)
    g.nd("Mul", [onehot, outmask], "output")
    return _model(g)


# =========================================================================== #
# TASK 125 -- frame every 6-block with a 3 border (8-neighbourhood) and fill its #
#   enclosed hole cells with 4.                                                  #
# =========================================================================== #
def _t125_rule(g):
    H, W = g.shape
    six = (g == 6)
    hasL = np.zeros((H, W), bool); hasR = np.zeros((H, W), bool)
    hasU = np.zeros((H, W), bool); hasD = np.zeros((H, W), bool)
    for r in range(H):
        acc = False
        for c in range(W): hasL[r, c] = acc; acc = acc or six[r, c]
        acc = False
        for c in range(W - 1, -1, -1): hasR[r, c] = acc; acc = acc or six[r, c]
    for c in range(W):
        acc = False
        for r in range(H): hasU[r, c] = acc; acc = acc or six[r, c]
        acc = False
        for r in range(H - 1, -1, -1): hasD[r, c] = acc; acc = acc or six[r, c]
    enclosed = (g == 8) & hasL & hasR & hasU & hasD
    adj6 = np.zeros((H, W), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ys = slice(max(0, dy), H + min(0, dy)); xs = slice(max(0, dx), W + min(0, dx))
            ysr = slice(max(0, -dy), H + min(0, -dy)); xsr = slice(max(0, -dx), W + min(0, -dx))
            tmp = np.zeros((H, W), bool); tmp[ys, xs] = six[ysr, xsr]; adj6 |= tmp
    o = g.copy()
    o[enclosed] = 4
    o[(g == 8) & adj6 & (~enclosed)] = 3
    return o


def _t125_detect(prs):
    seen = False
    for i, o in prs:
        if i.shape != o.shape or set(np.unique(i).tolist()) - {0, 6, 8}:
            return False
        if not np.array_equal(_t125_rule(i), o):
            return False
        seen = True
    return seen


def _t125_build():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    A1 = g.f([G, G], _mat_colL(1)); AR1 = g.f([G, G], _mat_colR(1))
    B1 = g.f([G, G], _mat_rowU(1)); BD1 = g.f([G, G], _mat_rowD(1))
    TG = g.f([G, G], _tri_gt(G)); TL = g.f([G, G], _tri_lt(G))
    e3 = g.f([1, 10, 1, 1], [1 if k == 3 else 0 for k in range(10)])
    e4 = g.f([1, 10, 1, 1], [1 if k == 4 else 0 for k in range(10)])
    e6 = g.f([1, 10, 1, 1], [1 if k == 6 else 0 for k in range(10)])
    e8 = g.f([1, 10, 1, 1], [1 if k == 8 else 0 for k in range(10)])
    X = "input"
    M6 = _slc(g, X, 6, 7, 1)
    M8 = _slc(g, X, 8, 9, 1)
    cL = lambda t: g.nd("MatMul", [t, A1])
    cR = lambda t: g.nd("MatMul", [t, AR1])
    rU = lambda t: g.nd("MatMul", [B1, t])
    rD = lambda t: g.nd("MatMul", [BD1, t])

    hasL = g.ge(g.nd("MatMul", [M6, TL]), half)
    hasR = g.ge(g.nd("MatMul", [M6, TG]), half)
    hasU = g.ge(g.nd("MatMul", [TG, M6]), half)
    hasD = g.ge(g.nd("MatMul", [TL, M6]), half)
    enclosed = g.nd("Mul", [g.nd("Mul", [M8, hasL]),
                           g.nd("Mul", [hasR, g.nd("Mul", [hasU, hasD])])])
    d = rD(M6); u = rU(M6)
    nbrs = [d, u, cR(M6), cL(M6), cR(d), cL(d), cR(u), cL(u)]
    acc = nbrs[0]
    for x in nbrs[1:]:
        acc = g.nd("Add", [acc, x])
    adj6 = g.ge(acc, half)
    frame = g.nd("Mul", [g.nd("Mul", [M8, adj6]), g.nd("Sub", [one, enclosed])])
    out8 = g.nd("Sub", [g.nd("Sub", [M8, enclosed]), frame])
    g.nd("Add", [g.nd("Add", [g.nd("Mul", [M6, e6]), g.nd("Mul", [enclosed, e4])]),
                 g.nd("Add", [g.nd("Mul", [frame, e3]), g.nd("Mul", [out8, e8])])], "output")
    return _model(g)


# =========================================================================== #
# dispatch                                                                    #
# =========================================================================== #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []
    # task 298
    try:
        if _t298_detect(prs):
            out.append(("t298", _check(_t298_build())))
    except Exception:
        pass
    # task 13
    try:
        if _t13_detect(prs):
            out.append(("t13", _check(_t13_build())))
    except Exception:
        pass
    # task 336
    try:
        if _t336_detect(prs):
            out.append(("t336", _check(_t336_build())))
    except Exception:
        pass
    # task 235
    try:
        if _t235_detect(prs):
            out.append(("t235", _check(_t235_build())))
    except Exception:
        pass
    # task 222
    try:
        if _t222_detect(prs):
            out.append(("t222", _check(_t222_build())))
    except Exception:
        pass
    # tasks 154 / 390 (reflect 5s across nearest bracket bar)
    try:
        if _refl_detect(prs):
            out.append(("refl", _check(_refl_build())))
    except Exception:
        pass
    # task 181 (mirror 8-shape by 4-arrow direction)
    try:
        if _t181_detect(prs):
            out.append(("t181", _check(_t181_build())))
    except Exception:
        pass
    # task 190 (diagonal stub rays)
    try:
        if _t190_detect(prs):
            out.append(("t190", _check(_t190_build())))
    except Exception:
        pass
    # task 64 (markers shoot lines to aligned rectangle)
    try:
        if _t64_detect(prs):
            out.append(("t64", _check(_t64_build())))
    except Exception:
        pass
    # task 275 (fractal Kronecker stamp)
    try:
        if _t275_detect(prs):
            out.append(("t275", _check(_t275_build())))
    except Exception:
        pass
    # task 125 (frame 6-blocks with 3, fill holes with 4)
    try:
        if _t125_detect(prs):
            out.append(("t125", _check(_t125_build())))
    except Exception:
        pass
    return out
