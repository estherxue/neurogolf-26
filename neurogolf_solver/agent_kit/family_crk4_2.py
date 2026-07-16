"""family_crk4_2 -- hard-tail ARC->ONNX solvers (opset-10, origin-anchored).

Three data-dependent rules, each validated to EXACT one-hot equality on every
train+test+arc-gen pair before emission.  All use the verified
"computed shift / select matrix + MatMul" recipe so shapes stay STATIC while the
geometry (grid height/width, shape position) is read off the one-hot tensor.

  reflecttile  (task 376)  Vertically (or horizontally) reflect-tile the grid:
                output row i = input row tri(i) with tri the triangle wave of
                period 2*(H-1).  Output height = nseg*(H-1)+1 for an integer
                segment count nseg detected from the pairs.  One MatMul on the
                H axis with a data-dependent [30,30] selection matrix M[i,k] =
                (k==tri(i)) * (i < nseg*(H-1)+1).

  coltile8     (task 388)  Every column that contains a non-background cell has
                its background cells painted colour 8; the resulting tile is then
                replicated 2x2 (output 2H x 2W).  The 2x2 replication is two
                data-dependent shift MatMuls (right by W, down by H).

  centerbox    (task 245)  A moving shape (colour S) is translated so its
                bounding box is centred inside the box spanned by marker colour M
                (4 corner dots); markers stay put.  Displacement (dy,dx) =
                ((Mr0+Mr1-Sr0-Sr1)/2, (Mc0+Mc1-Sc0-Sc1)/2) applied with the
                two-MatMul shift-plane idiom.

A numpy mirror reproduces the float-then-threshold semantics exactly; a candidate
is yielded only when it reproduces EVERY available pair.
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
_CBIG = 1000.0


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0
        self._cache = {}

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

    def _shared(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def half(self):
        return self._shared("half", lambda: self.f([1, 1, 1, 1], [0.5]))

    def one(self):
        return self._shared("one", lambda: self.f([1, 1, 1, 1], [1.0]))

    def cbig(self):
        return self._shared("cbig", lambda: self.f([1, 1, 1, 1], [_CBIG]))

    def rowidx(self):
        return self._shared("rowidx", lambda: self.f([1, 1, H, 1], list(range(H))))

    def colidx(self):
        return self._shared("colidx", lambda: self.f([1, 1, 1, W], list(range(W))))


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# graph helpers                                                               #
# --------------------------------------------------------------------------- #
def _plane(g, ch):
    return g.nd("Slice", ["input", g.i64([ch]), g.i64([ch + 1]), g.i64([1])])


def _real(g):
    """[1,1,30,30] presence of real cells (any channel set)."""
    return g.nd("ReduceMax", ["input"], axes=[1], keepdims=1)


def _height(g):
    real = _real(g)
    rrow = g.nd("ReduceMax", [real], axes=[3], keepdims=1)         # [1,1,30,1]
    return g.nd("ReduceSum", [rrow], axes=[2], keepdims=1)         # [1,1,1,1]


def _width(g):
    real = _real(g)
    rcol = g.nd("ReduceMax", [real], axes=[2], keepdims=1)         # [1,1,1,30]
    return g.nd("ReduceSum", [rcol], axes=[3], keepdims=1)         # [1,1,1,1]


def _bbox(g, plane):
    rowidx, colidx, cbig = g.rowidx(), g.colidx(), g.cbig()
    rowhas = g.nd("ReduceMax", [plane], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [plane], axes=[2], keepdims=1)
    r1 = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    r0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    c1 = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    c0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])
    return r0, r1, c0, c1


def _srow(g, dy):
    """[1,1,30,30] matrix that, used as MatMul(S, X), shifts rows down by dy."""
    diff = g.nd("Sub", [g.nd("Sub", [g.rowidx(), g.colidx()]), dy])
    return g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff]), g.half()])], to=F)


def _scol(g, dx):
    """[1,1,30,30] matrix that, used as MatMul(X, S), shifts cols right by dx."""
    diff = g.nd("Sub", [g.nd("Sub", [g.colidx(), g.rowidx()]), dx])
    return g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff]), g.half()])], to=F)


def _shift_plane(g, plane, dy, dx):
    rowshift = g.nd("MatMul", [_srow(g, dy), plane])
    return g.nd("MatMul", [rowshift, _scol(g, dx)])


# --------------------------------------------------------------------------- #
# 376  reflect-tile                                                           #
# --------------------------------------------------------------------------- #
def build_reflecttile(nseg, axis):
    """axis=2 vertical reflection-tiling, axis=3 horizontal."""
    g = _G()
    half = g.half()
    if axis == 2:
        size = _height(g)
        outi = g.rowidx()          # [1,1,30,1] output index along axis
        ini = g.colidx()           # [1,1,1,30] input index along axis
    else:
        size = _width(g)
        outi = g.colidx()          # [1,1,1,30] output col index
        ini = g.rowidx()           # [1,1,30,1] input col index
    Hm1 = g.nd("Sub", [size, g.one()])
    p = g.nd("Add", [Hm1, Hm1])                                    # 2*(H-1)
    # m = outi mod p  (integer mod)
    oi_i = g.nd("Cast", [outi], to=INT64)
    p_i = g.nd("Cast", [p], to=INT64)
    m = g.nd("Cast", [g.nd("Mod", [oi_i, p_i])], to=F)
    # tri = Hm1 - |m - Hm1|
    tri = g.nd("Sub", [Hm1, g.nd("Abs", [g.nd("Sub", [m, Hm1])])])
    # sel[out,in] = |ini - tri| < 0.5
    diff = g.nd("Sub", [ini, tri])
    sel = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff]), half])], to=F)
    # valid output positions: outi < nseg*(H-1)+1
    nsegc = g.f([1, 1, 1, 1], [float(nseg)])
    limit = g.nd("Add", [g.nd("Mul", [Hm1, nsegc]), g.one()])
    valid = g.nd("Cast", [g.nd("Less", [outi, limit])], to=F)
    M = g.nd("Mul", [sel, valid])                                  # [1,1,30,30]
    if axis == 2:
        # output[i,j] = sum_k M[i,k] input[k,j]
        g.nd("MatMul", [M, "input"], "output")
    else:
        # M here indexed [.,.,k(in row),j(out col)]; want output[i,j]=sum_k input[i,k] Mc[k,j]
        # rebuild as proper [in,out]: sel currently [.,.,?]. For horizontal use MatMul(input, Mc)
        g.nd("MatMul", ["input", M], "output")
    return _model(g)


def _tri(i, Hh):
    p = 2 * (Hh - 1)
    m = i % p
    return (Hh - 1) - abs(m - (Hh - 1))


def _ref_reflecttile(a, nseg, axis):
    if axis == 2:
        Hh = a.shape[0]
        if Hh < 2:
            return None
        n = nseg * (Hh - 1) + 1
        if n > 30:
            return None
        out = np.zeros((n, a.shape[1]), int)
        for i in range(n):
            out[i] = a[_tri(i, Hh)]
        return out
    else:
        Ww = a.shape[1]
        if Ww < 2:
            return None
        n = nseg * (Ww - 1) + 1
        if n > 30:
            return None
        out = np.zeros((a.shape[0], n), int)
        for j in range(n):
            out[:, j] = a[:, _tri(j, Ww)]
        return out


# --------------------------------------------------------------------------- #
# 388  column-fill(8) + 2x2 tile                                              #
# --------------------------------------------------------------------------- #
def build_coltile8(fill):
    g = _G()
    nonbg = g.nd("ReduceSum", [g.nd("Slice", ["input", g.i64([1]), g.i64([CHANNELS]),
                                              g.i64([1])])], axes=[1], keepdims=1)
    colhas = g.nd("ReduceMax", [nonbg], axes=[2], keepdims=1)       # [1,1,1,30]
    ch0 = _plane(g, 0)                                              # real-bg cells
    new = g.nd("Mul", [colhas, ch0])                               # [1,1,30,30]
    vec = g.f([1, CHANNELS, 1, 1],
              [(1.0 if c == fill else 0.0) - (1.0 if c == 0 else 0.0) for c in range(CHANNELS)])
    tile = g.nd("Add", ["input", g.nd("Mul", [new, vec])])         # [1,10,30,30]
    Hs = _height(g)
    Ws = _width(g)
    horiz = g.nd("Add", [tile, g.nd("MatMul", [tile, _scol(g, Ws)])])
    g.nd("Add", [horiz, g.nd("MatMul", [_srow(g, Hs), horiz])], "output")
    return _model(g)


def _ref_coltile8(a, fill):
    Hh, Ww = a.shape
    if 2 * Hh > 30 or 2 * Ww > 30:
        return None
    t = a.copy()
    colhas = (a != 0).any(axis=0)
    for c in range(Ww):
        if colhas[c]:
            for r in range(Hh):
                if t[r, c] == 0:
                    t[r, c] = fill
    out = np.zeros((2 * Hh, 2 * Ww), int)
    out[:Hh, :Ww] = t
    out[Hh:, :Ww] = t
    out[:Hh, Ww:] = t
    out[Hh:, Ww:] = t
    return out


# --------------------------------------------------------------------------- #
# 245  centre moving shape inside marker box                                  #
# --------------------------------------------------------------------------- #
def build_centerbox(M, S):
    g = _G()
    pM = _plane(g, M)
    pS = _plane(g, S)
    mr0, mr1, mc0, mc1 = _bbox(g, pM)
    sr0, sr1, sc0, sc1 = _bbox(g, pS)
    half = g.half()
    dy = g.nd("Mul", [g.nd("Sub", [g.nd("Add", [mr0, mr1]), g.nd("Add", [sr0, sr1])]), half])
    dx = g.nd("Mul", [g.nd("Sub", [g.nd("Add", [mc0, mc1]), g.nd("Add", [sc0, sc1])]), half])
    shifted = _shift_plane(g, pS, dy, dx)
    delta = g.nd("Sub", [shifted, pS])
    vec = g.f([1, CHANNELS, 1, 1],
              [(1.0 if c == S else 0.0) - (1.0 if c == 0 else 0.0) for c in range(CHANNELS)])
    g.nd("Add", ["input", g.nd("Mul", [delta, vec])], "output")
    return _model(g)


def _bbox_np(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _ref_centerbox(a, M, S):
    bm = _bbox_np(a == M)
    bs = _bbox_np(a == S)
    if bm is None or bs is None:
        return None
    dy2 = (bm[0] + bm[1]) - (bs[0] + bs[1])
    dx2 = (bm[2] + bm[3]) - (bs[2] + bs[3])
    if dy2 % 2 or dx2 % 2:
        return None
    dy, dx = dy2 // 2, dx2 // 2
    Hh, Ww = a.shape
    out = a.copy()
    out[a == S] = 0
    ys, xs = np.where(a == S)
    for y, x in zip(ys, xs):
        ny, nx = y + dy, x + dx
        if not (0 <= ny < Hh and 0 <= nx < Ww):
            return None
        out[ny, nx] = S
    return out


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
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


def _matches(prs, ref, need_change=True):
    changed = False
    for a, b in prs:
        p = ref(a)
        if p is None or p.shape != b.shape or not np.array_equal(p, b):
            return False
        if not np.array_equal(a, b):
            changed = True
    return changed or not need_change


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out, seen = [], set()

    def emit(name, builder):
        if name in seen:
            return
        try:
            m = builder()
            onnx.checker.check_model(m, full_check=True)
        except Exception:
            return
        seen.add(name)
        out.append((name, m))

    # ---- reflect-tile (output grows along one axis) ----
    samewidth = all(a.shape[1] == b.shape[1] for a, b in prs)
    sameheight = all(a.shape[0] == b.shape[0] for a, b in prs)
    if samewidth and not sameheight:
        # vertical: detect nseg
        nsegs = set()
        ok = True
        for a, b in prs:
            if a.shape[0] < 2 or (a.shape[0] - 1) == 0:
                ok = False
                break
            q, r = divmod(b.shape[0] - 1, a.shape[0] - 1)
            if r != 0 or q < 1:
                ok = False
                break
            nsegs.add(q)
        if ok and len(nsegs) == 1:
            nseg = nsegs.pop()
            if _matches(prs, lambda a, n=nseg: _ref_reflecttile(a, n, 2)):
                emit(f"reflecttile_v_n{nseg}", lambda n=nseg: build_reflecttile(n, 2))
    if sameheight and not samewidth:
        nsegs = set()
        ok = True
        for a, b in prs:
            if a.shape[1] < 2:
                ok = False
                break
            q, r = divmod(b.shape[1] - 1, a.shape[1] - 1)
            if r != 0 or q < 1:
                ok = False
                break
            nsegs.add(q)
        if ok and len(nsegs) == 1:
            nseg = nsegs.pop()
            if _matches(prs, lambda a, n=nseg: _ref_reflecttile(a, n, 3)):
                emit(f"reflecttile_h_n{nseg}", lambda n=nseg: build_reflecttile(n, 3))

    # ---- coltile8 (output exactly 2H x 2W) ----
    if all(b.shape == (2 * a.shape[0], 2 * a.shape[1]) for a, b in prs):
        for fill in [8] + list(range(1, CHANNELS)):
            if _matches(prs, lambda a, f=fill: _ref_coltile8(a, f)):
                emit(f"coltile_f{fill}", lambda f=fill: build_coltile8(f))
                break

    # ---- centerbox (same shape) ----
    if all(a.shape == b.shape for a, b in prs):
        cols = set()
        for a, _ in prs:
            cols |= set(np.unique(a[a != 0]).tolist())
        cols = sorted(cols)
        for Mk in cols:
            for Sh in cols:
                if Mk == Sh:
                    continue
                if _matches(prs, lambda a, Mk=Mk, Sh=Sh: _ref_centerbox(a, Mk, Sh)):
                    emit(f"centerbox_M{Mk}_S{Sh}", lambda Mk=Mk, Sh=Sh: build_centerbox(Mk, Sh))

    return out
