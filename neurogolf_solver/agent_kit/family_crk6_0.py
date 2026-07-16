"""crk6_0 -- two hard "physics-of-the-grid" families, both exact static graphs.

(A) CONTAINER FILL + OVERFLOW  (task 099)
    Rectangular colour-1 containers with a single-cell gap in their TOP wall and a
    seed pixel inside.  Output fills the interior with the seed colour, fills the
    gap, and adds one OVERFLOW row directly above the rim spanning the box width.
    Implemented as a fixed morphological pipeline (ray tests via triangular MatMul,
    rim via shift, seed flood via masked max-dilation, re-onehot via relu(1-|c-k|)).

(B) BOUNCING DIAGONAL RAY  (task 119)
    A "ball" drawn as a short diagonal segment (or chevron) launches a diagonal ray
    that reflects ONCE off a straight colour-2 wall band and runs to the grid edge,
    leaving a colour-3 trace.  Implemented as a 4-direction beam cellular automaton
    with wall reflection (the wall orientation flips the dr or dc component); beams
    are seeded only from ball cells on a populated diagonal, so straight segments and
    chevrons are both handled.

Both numpy references mirror the ONNX graphs EXACTLY and are validated on every
train+test+arc-gen pair before emitting, so wrong hypotheses are never scored.
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
N99 = 20    # container flood iterations
N119 = 28   # billiard CA iterations


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


def _shift(g, t, dr, dc):
    """Shift content of a [1,K,30,30] tensor by (dr,dc): out[r,c]=t[r-dr,c-dc]."""
    pads = [0, 0, max(dr, 0), max(dc, 0), 0, 0, max(-dr, 0), max(-dc, 0)]
    p = g.nd("Pad", [t], mode="constant", value=0.0, pads=pads)
    rs, cs = max(-dr, 0), max(-dc, 0)
    return g.nd("Slice", [p, g.i64([rs, cs]), g.i64([rs + G, cs + G]), g.i64([2, 3])])


# =========================================================================== #
# (A) container fill                                                           #
# =========================================================================== #
def _sd(x):
    r = np.zeros_like(x); r[1:] = x[:-1]; return r


def _su(x):
    r = np.zeros_like(x); r[:-1] = x[1:]; return r


def _sr(x):
    r = np.zeros_like(x); r[:, 1:] = x[:, :-1]; return r


def _sl(x):
    r = np.zeros_like(x); r[:, :-1] = x[:, 1:]; return r


def solve99(a, wc=1):
    a = np.asarray(a, int)
    H, W = a.shape
    wall = (a == wc).astype(float)
    L = np.triu(np.ones((H, H)), 1)
    dn = np.minimum(L @ wall, 1.0)
    Lc = np.tril(np.ones((W, W)), -1)
    Rc = np.triu(np.ones((W, W)), 1)
    lf = np.minimum(wall @ Lc, 1.0)
    rt = np.minimum(wall @ Rc, 1.0)
    reg = (1 - wall) * dn * lf * rt
    box = np.minimum(wall + reg, 1.0)
    rim = box * (1 - _sd(box))
    bg = (a == 0).astype(float)
    over = _su(rim) * bg
    filled = np.minimum(reg + over, 1.0)
    color = np.zeros((H, W))
    for k in range(2, 10):
        color = np.where(a == k, float(k), color)
    color = color * filled
    for _ in range(N99):
        d = np.maximum.reduce([color, _su(color), _sd(color), _sl(color), _sr(color)])
        color = d * filled
    return np.where((filled > 0) & (color > 0), color.astype(int), a)


def build99():
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0])
    colvec = g.f([1, CHANNELS, 1, 1], [0, 0, 2, 3, 4, 5, 6, 7, 8, 9])
    kvec = g.f([1, 8, 1, 1], [2, 3, 4, 5, 6, 7, 8, 9])
    rng = list(range(G))
    Lb = g.f([G, G], [[1.0 if k > i else 0.0 for k in rng] for i in rng])
    Cl = g.f([G, G], [[1.0 if k < j else 0.0 for j in rng] for k in rng])
    Cr = g.f([G, G], [[1.0 if k > j else 0.0 for j in rng] for k in rng])
    Sdn = g.f([G, G], [[1.0 if k == i - 1 else 0.0 for k in rng] for i in rng])
    Sup = g.f([G, G], [[1.0 if k == i + 1 else 0.0 for k in rng] for i in rng])
    SCL = g.f([G, G], [[1.0 if k == j + 1 else 0.0 for j in rng] for k in rng])
    SCR = g.f([G, G], [[1.0 if k == j - 1 else 0.0 for j in rng] for k in rng])

    wall = g.nd("Slice", ["input", g.i64([1]), g.i64([2]), g.i64([1])])
    bg = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    colorseed = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)

    dn = g.nd("Min", [g.nd("MatMul", [Lb, wall]), one])
    lf = g.nd("Min", [g.nd("MatMul", [wall, Cl]), one])
    rt = g.nd("Min", [g.nd("MatMul", [wall, Cr]), one])
    notwall = g.nd("Sub", [one, wall])
    reg = g.nd("Mul", [g.nd("Mul", [g.nd("Mul", [notwall, dn]), lf]), rt])
    box = g.nd("Min", [g.nd("Add", [wall, reg]), one])
    box_above = g.nd("MatMul", [Sdn, box])
    rim = g.nd("Mul", [box, g.nd("Sub", [one, box_above])])
    over = g.nd("Mul", [g.nd("MatMul", [Sup, rim]), bg])
    filled = g.nd("Min", [g.nd("Add", [reg, over]), one])

    c = g.nd("Mul", [colorseed, filled])
    for _ in range(N99):
        up = g.nd("MatMul", [Sup, c])
        dd = g.nd("MatMul", [Sdn, c])
        le = g.nd("MatMul", [c, SCL])
        ri = g.nd("MatMul", [c, SCR])
        m = g.nd("Max", [c, up])
        m = g.nd("Max", [m, dd])
        m = g.nd("Max", [m, le])
        m = g.nd("Max", [m, ri])
        c = g.nd("Mul", [m, filled])

    diff = g.nd("Sub", [c, kvec])
    ind = g.nd("Relu", [g.nd("Sub", [one, g.nd("Abs", [diff])])])
    inksum = g.nd("ReduceSum", [ind], axes=[1], keepdims=1)
    ink = g.nd("Min", [g.nd("Add", [inksum, wall]), one])
    ch0 = g.nd("Mul", [realmask, g.nd("Sub", [one, ink])])
    g.nd("Concat", [ch0, wall, ind], "output", axis=1)
    return _model(g)


# =========================================================================== #
# (B) bouncing diagonal ray                                                    #
# =========================================================================== #
_DIRS = {"SE": (1, 1), "NW": (-1, -1), "SW": (1, -1), "NE": (-1, 1)}
_RH = {"SE": "NE", "NE": "SE", "NW": "SW", "SW": "NW"}   # horizontal wall flips dr
_RV = {"SE": "SW", "SW": "SE", "NW": "NE", "NE": "NW"}   # vertical wall flips dc


def _np_shift(x, d):
    dr, dc = d
    r = np.zeros_like(x)
    rs0, re0 = max(dr, 0), G + min(dr, 0)
    cs0, ce0 = max(dc, 0), G + min(dc, 0)
    rs1, re1 = max(-dr, 0), G + min(-dr, 0)
    cs1, ce1 = max(-dc, 0), G + min(-dc, 0)
    r[rs0:re0, cs0:ce0] = x[rs1:re1, cs1:ce1]
    return r


def solve119(a, N=N119):
    a = np.asarray(a, int)
    H, W = a.shape
    A = np.zeros((G, G), int); A[:H, :W] = a
    wall = (A == 2).astype(float); ball = (A == 8).astype(float)
    real = np.zeros((G, G), float); real[:H, :W] = 1.0
    rowwall = wall.sum(1); roww = real.sum(1)
    colwall = wall.sum(0); colh = real.sum(0)
    Hw = float(((rowwall > 0) & (rowwall >= roww - 0.5) & (roww > 0)).any())
    Vw = float(((colwall > 0) & (colwall >= colh - 0.5) & (colh > 0)).any())
    se = ball * np.clip(_np_shift(ball, (1, 1)) + _np_shift(ball, (-1, -1)), 0, 1)
    sw = ball * np.clip(_np_shift(ball, (1, -1)) + _np_shift(ball, (-1, 1)), 0, 1)
    B = {"SE": se.copy(), "NW": se.copy(), "SW": sw.copy(), "NE": sw.copy()}
    blocked = {}; fw = {}
    for name, d in _DIRS.items():
        nd = (-d[0], -d[1])
        swl = _np_shift(wall, nd); srl = _np_shift(real, nd)
        blocked[name] = np.clip(swl + (1 - srl), 0, 1); fw[name] = swl
    visited = np.zeros((G, G))
    for _ in range(N):
        free = {n: B[n] * (1 - blocked[n]) for n in _DIRS}
        hit = {n: B[n] * fw[n] for n in _DIRS}
        newB = {}
        for name, d in _DIRS.items():
            s = _np_shift(free[name], d)
            s = s + Hw * _np_shift(hit[_RH[name]], d) + Vw * _np_shift(hit[_RV[name]], d)
            newB[name] = np.clip(s, 0, 1) * real
        B = newB
        cur = np.clip(B["SE"] + B["NW"] + B["SW"] + B["NE"], 0, 1)
        visited = np.clip(visited + cur, 0, 1)
    out = A.copy()
    mark = (visited > 0) & (A != 8) & (A != 2)
    out[mark] = 3
    return out[:H, :W]


def build119():
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0])
    half = g.f([1, 1, 1, 1], [0.5])
    coeff = g.f([1, CHANNELS, 1, 1], [-1, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    wall = g.nd("Slice", ["input", g.i64([2]), g.i64([3]), g.i64([1])])
    ball = g.nd("Slice", ["input", g.i64([8]), g.i64([9]), g.i64([1])])
    real = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)

    rowwall = g.nd("ReduceSum", [wall], axes=[3], keepdims=1)
    roww = g.nd("ReduceSum", [real], axes=[3], keepdims=1)
    eqrow = g.nd("Cast", [g.nd("Greater", [rowwall, g.nd("Sub", [roww, half])])], to=F)
    posrow = g.nd("Cast", [g.nd("Greater", [roww, half])], to=F)
    Hw = g.nd("Min", [g.nd("ReduceSum", [g.nd("Mul", [eqrow, posrow])], axes=[2, 3], keepdims=1), one])
    colwall = g.nd("ReduceSum", [wall], axes=[2], keepdims=1)
    colh = g.nd("ReduceSum", [real], axes=[2], keepdims=1)
    eqcol = g.nd("Cast", [g.nd("Greater", [colwall, g.nd("Sub", [colh, half])])], to=F)
    poscol = g.nd("Cast", [g.nd("Greater", [colh, half])], to=F)
    Vw = g.nd("Min", [g.nd("ReduceSum", [g.nd("Mul", [eqcol, poscol])], axes=[2, 3], keepdims=1), one])

    blocked = {}; fw = {}
    for name, d in _DIRS.items():
        nd0, nd1 = -d[0], -d[1]
        sw = _shift(g, wall, nd0, nd1)
        sr = _shift(g, real, nd0, nd1)
        blocked[name] = g.nd("Min", [g.nd("Add", [sw, g.nd("Sub", [one, sr])]), one])
        fw[name] = sw

    se = g.nd("Mul", [ball, g.nd("Min", [g.nd("Add", [_shift(g, ball, 1, 1), _shift(g, ball, -1, -1)]), one])])
    sw_ = g.nd("Mul", [ball, g.nd("Min", [g.nd("Add", [_shift(g, ball, 1, -1), _shift(g, ball, -1, 1)]), one])])
    B = {"SE": se, "NW": se, "SW": sw_, "NE": sw_}
    visited = None
    for _ in range(N119):
        free = {n: g.nd("Mul", [B[n], g.nd("Sub", [one, blocked[n]])]) for n in _DIRS}
        hit = {n: g.nd("Mul", [B[n], fw[n]]) for n in _DIRS}
        newB = {}
        for name, d in _DIRS.items():
            cont = _shift(g, free[name], d[0], d[1])
            ht = g.nd("Mul", [Hw, _shift(g, hit[_RH[name]], d[0], d[1])])
            vt = g.nd("Mul", [Vw, _shift(g, hit[_RV[name]], d[0], d[1])])
            s = g.nd("Add", [g.nd("Add", [cont, ht]), vt])
            newB[name] = g.nd("Mul", [g.nd("Min", [s, one]), real])
        B = newB
        cur = g.nd("Min", [g.nd("Add", [g.nd("Add", [B["SE"], B["NW"]]),
                                        g.nd("Add", [B["SW"], B["NE"]])]), one])
        visited = cur if visited is None else g.nd("Min", [g.nd("Add", [visited, cur]), one])

    mark = g.nd("Mul", [g.nd("Mul", [visited, g.nd("Sub", [one, ball])]), g.nd("Sub", [one, wall])])
    delta = g.nd("Mul", [mark, coeff])
    g.nd("Add", ["input", delta], "output")
    return _model(g)


# =========================================================================== #
# (C) corner diagonal rays                                                     #
# =========================================================================== #
def solve378(a):
    a = np.asarray(a, int)
    H, W = a.shape
    A = np.zeros((G, G), int); A[:H, :W] = a
    block = wallc = None; ns = nw = 0
    for k in range(1, 10):
        m = (A == k); cnt = int(m.sum())
        if cnt == 0:
            continue
        ys, xs = np.where(m)
        area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        if cnt == area:
            block = k; ns += 1
        else:
            wallc = k; nw += 1
    if block is None or wallc is None or ns != 1 or nw != 1:
        return a.copy()
    wall = (A == wallc).astype(float)
    wR = _np_shift(wall, (0, -1)); wL = _np_shift(wall, (0, 1))
    wD = _np_shift(wall, (-1, 0)); wU = _np_shift(wall, (1, 0))
    cNW = wall * wR * wD * (1 - wL) * (1 - wU)
    cSW = wall * wR * wU * (1 - wL) * (1 - wD)
    cNE = wall * wL * wD * (1 - wR) * (1 - wU)
    cSE = wall * wL * wU * (1 - wR) * (1 - wD)

    def beam(c, dr, dc):
        b = c.copy()
        for d in (1, 2, 4, 8, 16):
            b = np.clip(b + _np_shift(b, (dr * d, dc * d)), 0, 1)
        return b
    tr = np.clip(beam(cNW, -1, -1) + beam(cSW, 1, -1) + beam(cNE, -1, 1) + beam(cSE, 1, 1), 0, 1)
    mark = (tr * (A == 0).astype(float)) > 0
    out = A.copy(); out[mark] = block
    return out[:H, :W]


def build378():
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0])
    half = g.f([1, 1, 1, 1], [0.5])
    big = g.f([1, 1, 1, 1], [100.0])
    rvec = g.f([1, 1, G, 1], list(range(G)))
    cvec = g.f([1, 1, 1, G], list(range(G)))
    chanmask = g.f([1, CHANNELS, 1, 1], [0] + [1] * 9)
    bgcoeff = g.f([1, CHANNELS, 1, 1], [-1] + [0] * 9)

    count = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    rowpres = g.nd("ReduceMax", ["input"], axes=[3], keepdims=1)
    colpres = g.nd("ReduceMax", ["input"], axes=[2], keepdims=1)
    maxr = g.nd("ReduceMax", [g.nd("Mul", [rowpres, rvec])], axes=[2], keepdims=1)
    minr = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [rowpres, rvec]),
                                           g.nd("Mul", [g.nd("Sub", [one, rowpres]), big])])],
                axes=[2], keepdims=1)
    maxc = g.nd("ReduceMax", [g.nd("Mul", [colpres, cvec])], axes=[3], keepdims=1)
    minc = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [colpres, cvec]),
                                           g.nd("Mul", [g.nd("Sub", [one, colpres]), big])])],
                axes=[3], keepdims=1)
    hrows = g.nd("Add", [g.nd("Sub", [maxr, minr]), one])
    wcols = g.nd("Add", [g.nd("Sub", [maxc, minc]), one])
    area = g.nd("Mul", [hrows, wcols])
    solid = g.nd("Relu", [g.nd("Sub", [one, g.nd("Abs", [g.nd("Sub", [count, area])])])])
    present = g.nd("Cast", [g.nd("Greater", [count, half])], to=F)
    blockflag = g.nd("Mul", [g.nd("Mul", [solid, present]), chanmask])
    wallflag = g.nd("Mul", [g.nd("Mul", [g.nd("Sub", [one, solid]), present]), chanmask])

    wall = g.nd("ReduceSum", [g.nd("Mul", ["input", wallflag])], axes=[1], keepdims=1)
    wR = _shift(g, wall, 0, -1); wL = _shift(g, wall, 0, 1)
    wD = _shift(g, wall, -1, 0); wU = _shift(g, wall, 1, 0)
    nwL = g.nd("Sub", [one, wL]); nwR = g.nd("Sub", [one, wR])
    nwU = g.nd("Sub", [one, wU]); nwD = g.nd("Sub", [one, wD])

    def corner(h, v, nh, nv):
        return g.nd("Mul", [g.nd("Mul", [g.nd("Mul", [wall, h]), v]), g.nd("Mul", [nh, nv])])
    cNW = corner(wR, wD, nwL, nwU)
    cSW = corner(wR, wU, nwL, nwD)
    cNE = corner(wL, wD, nwR, nwU)
    cSE = corner(wL, wU, nwR, nwD)

    def beam(c, dr, dc):
        b = c
        for d in (1, 2, 4, 8, 16):
            b = g.nd("Min", [g.nd("Add", [b, _shift(g, b, dr * d, dc * d)]), one])
        return b
    tr = g.nd("Min", [g.nd("Add", [g.nd("Add", [beam(cNW, -1, -1), beam(cSW, 1, -1)]),
                                   g.nd("Add", [beam(cNE, -1, 1), beam(cSE, 1, 1)])]), one])
    ch0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])
    mark = g.nd("Mul", [tr, ch0])
    delta = g.nd("Add", [g.nd("Mul", [mark, blockflag]), g.nd("Mul", [mark, bgcoeff])])
    g.nd("Add", ["input", delta], "output")
    return _model(g)


# =========================================================================== #
# detection                                                                   #
# =========================================================================== #
def _pairs(examples):
    out = []
    for sec in ("train", "test", "arc-gen"):
        for e in examples.get(sec, []):
            out.append((np.asarray(e["input"], int), np.asarray(e["output"], int)))
    return out


def _match(pairs, fn):
    changed = False
    for a, b in pairs:
        if a.shape != b.shape or not np.array_equal(fn(a), b):
            return False
        if not np.array_equal(a, b):
            changed = True
    return changed


def candidates(examples):
    pairs = _pairs(examples)
    if not pairs:
        return []
    out = []
    if _match(pairs, solve99):
        out.append(("container_fill", build99()))
    if _match(pairs, solve119):
        out.append(("bounce_ray", build119()))
    if _match(pairs, solve378):
        out.append(("corner_rays", build378()))
    return out
