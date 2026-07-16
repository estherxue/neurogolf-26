"""family_crk4_1 -- hard-tail ARC->ONNX solvers (opset-10, origin-anchored).

Data-dependent rules, each validated to EXACT one-hot equality on every
train+test+arc-gen pair before emission.  All keep STATIC shapes while reading
geometry (positions / periods / colours) off the one-hot input tensor.

  cross358  (task 358)  Input is a "plus": a short horizontal colour stub in one
            row R and a short vertical colour stub in one column C, crossing at
            (R,C).  Output extends each stub periodically to fill its whole row /
            column (period = stub length, phased to the original cells).  Built
            with two data-dependent [30,30] circulant-selection MatMuls.

  pin200    (task 200)  A single seed dot of colour X at column c0 (bottom row).
            Output draws vertical period-2 pinstripes of colour X at columns
            c>=c0 with (c-c0) even, plus colour-5 highlight dots on the top row
            (cols with (c-c0)%4==1) and the bottom row ((c-c0)%4==3).  All keyed
            off c0 / X read from the one-hot input.

  frame240  (task 240)  A small motif in one corner of a square grid encodes
            nested square "rings": cells on a grid diagonal are ring corners,
            off-diagonal cells are period-2 edge dashes.  Output is the full
            D4 symmetrisation: corners placed by reflection, edge dashes spread
            period-2 along their ring edge (parity MatMul), bounded to the
            top/bottom (resp. left/right) triangle, then mirrored over D4.
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

    def rowidx(self):
        return self._shared("rowidx", lambda: self.f([1, 1, H, 1], list(range(H))))

    def colidx(self):
        return self._shared("colidx", lambda: self.f([1, 1, 1, W], list(range(W))))

    def translate(self, X, dy, dx):
        """output[i,j] = X[i-dy, j-dx], zero-filled, 30x30 window."""
        h0, w0 = max(dy, 0), max(dx, 0)
        h1, w1 = max(-dy, 0), max(-dx, 0)
        padded = self.nd("Pad", [X], pads=[0, 0, h0, w0, 0, 0, h1, w1],
                         mode="constant", value=0.0)
        return self.nd("Slice", [padded, self.i64([h1, w1]),
                                 self.i64([h1 + H, w1 + W]), self.i64([2, 3])])


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# 358  periodic cross extension                                               #
# --------------------------------------------------------------------------- #
def build_cross():
    g = _G()
    half = g.half()
    one = g.one()
    rowidx = g.rowidx()    # [1,1,30,1]
    colidx = g.colidx()    # [1,1,1,30]

    inputC = g.nd("Slice", ["input", g.i64([1]), g.i64([CHANNELS]), g.i64([1])])  # [1,9,30,30]
    ink_in = g.nd("ReduceSum", [inputC], axes=[1], keepdims=1)     # [1,1,30,30]
    M = g.nd("ReduceMax", ["input"], axes=[1], keepdims=1)         # [1,1,30,30] grid mask

    rowcnt = g.nd("ReduceSum", [ink_in], axes=[3], keepdims=1)     # [1,1,30,1]
    ph = g.nd("ReduceMax", [rowcnt], axes=[2], keepdims=1)         # [1,1,1,1]
    colcnt = g.nd("ReduceSum", [ink_in], axes=[2], keepdims=1)     # [1,1,1,30]
    pv = g.nd("ReduceMax", [colcnt], axes=[3], keepdims=1)         # [1,1,1,1]

    rowmaskR = g.nd("Cast", [g.nd("Greater", [rowcnt, g.nd("Sub", [ph, half])])], to=F)  # [1,1,30,1]
    colmaskC = g.nd("Cast", [g.nd("Greater", [colcnt, g.nd("Sub", [pv, half])])], to=F)  # [1,1,1,30]

    HX = g.nd("Mul", [inputC, rowmaskR])   # [1,9,30,30] only row R
    VX = g.nd("Mul", [inputC, colmaskC])   # [1,9,30,30] only col C

    # ---- horizontal circulant selection Th[k(axis2), j(axis3)] -------------
    diff_h = g.nd("Sub", [colidx, rowidx])                          # j - k  [1,1,30,30]
    ph_i = g.nd("Cast", [ph], to=INT64)
    mod_h = g.nd("Mod", [g.nd("Cast", [diff_h], to=INT64), ph_i])   # [1,1,30,30]
    divis_h = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Cast", [mod_h], to=F)]), half])], to=F)
    inkR_cols = g.nd("ReduceSum", [g.nd("Mul", [ink_in, rowmaskR])], axes=[2], keepdims=1)  # [1,1,1,30]
    inkR_k = g.nd("Reshape", [inkR_cols, g.i64([1, 1, H, 1])])      # [1,1,30,1] (k axis)
    Th = g.nd("Mul", [divis_h, inkR_k])                            # [1,1,30,30]
    Hout = g.nd("MatMul", [HX, Th])                                # [1,9,30,30]

    # ---- vertical circulant selection Tv[i(axis2), k(axis3)] ---------------
    diff_v = g.nd("Sub", [rowidx, colidx])                          # i - k  [1,1,30,30]
    pv_i = g.nd("Cast", [pv], to=INT64)
    mod_v = g.nd("Mod", [g.nd("Cast", [diff_v], to=INT64), pv_i])
    divis_v = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Cast", [mod_v], to=F)]), half])], to=F)
    inkC_rows = g.nd("ReduceSum", [g.nd("Mul", [ink_in, colmaskC])], axes=[3], keepdims=1)  # [1,1,30,1]
    inkC_k = g.nd("Reshape", [inkC_rows, g.i64([1, 1, 1, W])])      # [1,1,1,30] (k axis)
    Tv = g.nd("Mul", [divis_v, inkC_k])                           # [1,1,30,30]
    Vout = g.nd("MatMul", [Tv, VX])                               # [1,9,30,30]

    INK9 = g.nd("Mul", [g.nd("Max", [Hout, Vout]), M])            # [1,9,30,30] clipped to grid
    ink_any = g.nd("ReduceMax", [INK9], axes=[1], keepdims=1)     # [1,1,30,30]
    ch0 = g.nd("Mul", [M, g.nd("Sub", [one, ink_any])])          # [1,1,30,30]
    g.nd("Concat", [ch0, INK9], "output", axis=1)
    return _model(g)


def _ref_cross(a):
    Hh, Ww = a.shape
    ink = (a != 0).astype(int)
    rowcnt = ink.sum(axis=1)
    colcnt = ink.sum(axis=0)
    ph = int(rowcnt.max()); pv = int(colcnt.max())
    if ph < 2 or pv < 2:
        return None
    if (rowcnt == ph).sum() != 1 or (colcnt == pv).sum() != 1:
        return None
    R = int(np.argmax(rowcnt)); C = int(np.argmax(colcnt))
    rcols = np.where(ink[R] != 0)[0]
    ccols = np.where(ink[:, C] != 0)[0]
    if rcols.size != ph or ccols.size != pv:
        return None
    if rcols.max() - rcols.min() + 1 != ph:
        return None
    if ccols.max() - ccols.min() + 1 != pv:
        return None
    out = np.zeros_like(a)
    for j in range(Ww):
        for k in rcols:
            if (j - k) % ph == 0:
                out[R, j] = a[R, k]; break
    for i in range(Hh):
        for k in ccols:
            if (i - k) % pv == 0:
                out[i, C] = a[k, C]; break
    return out


# --------------------------------------------------------------------------- #
# 200  pinstripes + corner highlights                                         #
# --------------------------------------------------------------------------- #
def build_pin():
    g = _G()
    half = g.half()
    one = g.one()
    three = g.f([1, 1, 1, 1], [3.0])
    neghalf = g.f([1, 1, 1, 1], [-0.5])
    rowidx = g.rowidx()    # [1,1,30,1]
    colidx = g.colidx()    # [1,1,1,30]
    vec5 = g.f([1, CHANNELS - 1, 1, 1], [1.0 if c == 4 else 0.0 for c in range(CHANNELS - 1)])
    two_i = g.i64([2])
    four_i = g.i64([4])

    inputC = g.nd("Slice", ["input", g.i64([1]), g.i64([CHANNELS]), g.i64([1])])  # [1,9,30,30]
    ink_in = g.nd("ReduceSum", [inputC], axes=[1], keepdims=1)     # [1,1,30,30]
    M = g.nd("ReduceMax", ["input"], axes=[1], keepdims=1)         # [1,1,30,30]

    seedcol = g.nd("ReduceMax", [ink_in], axes=[2], keepdims=1)    # [1,1,1,30]
    c0 = g.nd("ReduceSum", [g.nd("Mul", [seedcol, colidx])], axes=[3], keepdims=1)  # [1,1,1,1]
    colorvec = g.nd("ReduceMax", [inputC], axes=[2, 3], keepdims=1)  # [1,9,1,1]
    realrow = g.nd("ReduceMax", [M], axes=[3], keepdims=1)         # [1,1,30,1]
    Rmax = g.nd("ReduceMax", [g.nd("Mul", [realrow, rowidx])], axes=[2], keepdims=1)  # [1,1,1,1]

    off = g.nd("Sub", [colidx, c0])                               # [1,1,1,30]
    ge = g.nd("Cast", [g.nd("Greater", [off, neghalf])], to=F)    # c>=c0
    off_i = g.nd("Cast", [off], to=INT64)
    mod2 = g.nd("Cast", [g.nd("Mod", [off_i, two_i])], to=F)
    even = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [mod2]), half])], to=F)
    STRrow = g.nd("Mul", [even, ge])                              # [1,1,1,30]
    mod4 = g.nd("Cast", [g.nd("Mod", [off_i, four_i])], to=F)
    is1 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [mod4, one])]), half])], to=F)
    is3 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [mod4, three])]), half])], to=F)
    top5row = g.nd("Mul", [is1, ge])                             # [1,1,1,30]
    bot5row = g.nd("Mul", [is3, ge])                             # [1,1,1,30]

    row0ind = g.nd("Cast", [g.nd("Less", [rowidx, half])], to=F)  # [1,1,30,1]
    lastrowind = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [rowidx, g.nd("Sub", [Rmax, half])])], to=F), realrow])  # [1,1,30,1]

    P5 = g.nd("Add", [g.nd("Mul", [top5row, row0ind]), g.nd("Mul", [bot5row, lastrowind])])  # [1,1,30,30]
    Xink = g.nd("Mul", [g.nd("Mul", [STRrow, colorvec]), M])      # [1,9,30,30]
    five = g.nd("Mul", [g.nd("Mul", [P5, vec5]), M])              # [1,9,30,30]
    ink9 = g.nd("Add", [Xink, five])                            # [1,9,30,30]
    ink_any = g.nd("ReduceMax", [ink9], axes=[1], keepdims=1)     # [1,1,30,30]
    ch0 = g.nd("Mul", [M, g.nd("Sub", [one, ink_any])])          # [1,1,30,30]
    g.nd("Concat", [ch0, ink9], "output", axis=1)
    return _model(g)


def _ref_pin(a):
    Hh, Ww = a.shape
    ys, xs = np.where(a != 0)
    if ys.size != 1:
        return None
    X = int(a[ys[0], xs[0]]); c0 = int(xs[0])
    if X == 5:
        return None
    out = np.zeros_like(a)
    for r in range(Hh):
        for c in range(Ww):
            off = c - c0
            if off >= 0 and off % 2 == 0:
                out[r, c] = X
    for c in range(Ww):
        off = c - c0
        if off >= 0 and off % 4 == 1:
            out[0, c] = 5
        if off >= 0 and off % 4 == 3:
            out[Hh - 1, c] = 5
    return out


# --------------------------------------------------------------------------- #
# 240  D4 frame builder                                                       #
# --------------------------------------------------------------------------- #
def build_frame():
    g = _G()
    half = g.half()
    one = g.one()
    rowidx = g.rowidx()    # [1,1,30,1]
    colidx = g.colidx()    # [1,1,1,30]
    # parity spread matrix P2[a,b] = 1 if (a-b) even
    P2vals = [[1.0 if (a - b) % 2 == 0 else 0.0 for b in range(W)] for a in range(H)]
    P2 = g.f([H, W], np.array(P2vals).ravel())

    inputC = g.nd("Slice", ["input", g.i64([1]), g.i64([CHANNELS]), g.i64([1])])  # [1,9,30,30]
    M = g.nd("ReduceMax", ["input"], axes=[1], keepdims=1)         # [1,1,30,30]
    realrow = g.nd("ReduceMax", [M], axes=[3], keepdims=1)         # [1,1,30,1]
    S = g.nd("ReduceSum", [realrow], axes=[2], keepdims=1)         # [1,1,1,1]
    N = g.nd("Sub", [S, one])                                     # grid index max

    mr = g.nd("Min", [rowidx, g.nd("Sub", [N, rowidx])])          # [1,1,30,1]
    mc = g.nd("Min", [colidx, g.nd("Sub", [N, colidx])])          # [1,1,1,30]
    Bh = g.nd("Cast", [g.nd("Less", [mr, g.nd("Add", [mc, half])])], to=F)   # mr<=mc
    Bv = g.nd("Cast", [g.nd("Less", [mc, g.nd("Add", [mr, half])])], to=F)   # mc<=mr
    Hstrict = g.nd("Cast", [g.nd("Less", [mr, mc])], to=F)
    Vstrict = g.nd("Cast", [g.nd("Less", [mc, mr])], to=F)
    d1 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, colidx])]), half])], to=F)
    d2 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.nd("Add", [rowidx, colidx]), N])]), half])], to=F)
    diagmask = g.nd("Max", [d1, d2])                             # [1,1,30,30]
    notdiag = g.nd("Sub", [one, diagmask])

    edgecells = g.nd("Mul", [inputC, notdiag])                   # [1,9,30,30]
    corners = g.nd("Mul", [inputC, diagmask])
    Hedge = g.nd("Mul", [edgecells, Hstrict])
    Vedge = g.nd("Mul", [edgecells, Vstrict])
    Hdash = g.nd("Mul", [g.nd("Mul", [g.nd("MatMul", [Hedge, P2]), Bh]), notdiag])
    Vdash = g.nd("Mul", [g.nd("Mul", [g.nd("MatMul", [P2, Vedge]), Bv]), notdiag])
    canvas = g.nd("Max", [g.nd("Max", [corners, Hdash]), Vdash])  # [1,9,30,30]

    R = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.nd("Add", [rowidx, colidx]), N])]), half])], to=F)  # [1,1,30,30]
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
    ink9 = g.nd("Mul", [sym, M])                                 # [1,9,30,30]
    ink_any = g.nd("ReduceMax", [ink9], axes=[1], keepdims=1)
    ch0 = g.nd("Mul", [M, g.nd("Sub", [one, ink_any])])
    g.nd("Concat", [ch0, ink9], "output", axis=1)
    return _model(g)


def _d4(canvas):
    outs = [canvas, canvas[:, ::-1], canvas[::-1], canvas[::-1, ::-1]]
    if canvas.shape[0] == canvas.shape[1]:
        outs += [canvas.T, canvas.T[:, ::-1], canvas.T[::-1], canvas.T[::-1, ::-1]]
    res = np.zeros_like(canvas)
    for o in outs:
        res = np.where(o != 0, o, res)
    return res


def _ref_frame(a):
    Hh, Ww = a.shape
    if Hh != Ww or Hh < 3:
        return None
    N = Hh - 1
    canvas = np.zeros_like(a)
    for r, c in zip(*np.where(a != 0)):
        v = a[r, c]
        if r == c or r == N - c:
            canvas[r, c] = v
        else:
            L = min(r, N - r)
            Lc = min(c, N - c)
            if L < Lc:
                for col in range(L + 1, N - L):
                    if (col - c) % 2 == 0:
                        canvas[r, col] = v
            else:
                for row in range(Lc + 1, N - Lc):
                    if (row - r) % 2 == 0:
                        canvas[row, c] = v
    return _d4(canvas)


# --------------------------------------------------------------------------- #
# 168  diagonal rays from L-tromino empty corners                             #
# --------------------------------------------------------------------------- #
def build_rays():
    g = _G()
    one = g.one()
    inputC = g.nd("Slice", ["input", g.i64([1]), g.i64([CHANNELS]), g.i64([1])])  # [1,9,30,30]
    ink = g.nd("ReduceSum", [inputC], axes=[1], keepdims=1)        # [1,1,30,30] presence
    M = g.nd("ReduceMax", ["input"], axes=[1], keepdims=1)         # [1,1,30,30]
    colorvec = g.nd("ReduceMax", [inputC], axes=[2, 3], keepdims=1)  # [1,9,1,1]
    bg = g.nd("Mul", [M, g.nd("Sub", [one, ink])])                # [1,1,30,30]

    rays = None
    for (dr, dc) in [(1, -1), (1, 1), (-1, -1), (-1, 1)]:
        nb_v = g.translate(ink, -dr, 0)        # ink[i+dr, j]
        nb_h = g.translate(ink, 0, -dc)        # ink[i, j+dc]
        nb_d = g.translate(ink, -dr, -dc)      # ink[i+dr, j+dc]
        corner = g.nd("Mul", [g.nd("Mul", [g.nd("Mul", [bg, nb_v]), nb_h]), nb_d])
        ry, rx = -dr, -dc
        R = g.translate(corner, ry, rx)        # first ray cell
        for m in (1, 2, 4, 8, 16):
            R = g.nd("Max", [R, g.translate(R, m * ry, m * rx)])
        rays = R if rays is None else g.nd("Max", [rays, R])

    rays = g.nd("Mul", [rays, M])                                 # clip to grid
    ray9 = g.nd("Mul", [rays, colorvec])                          # [1,9,30,30] in colour C
    ink9 = g.nd("Max", [inputC, ray9])                           # [1,9,30,30]
    ink_any = g.nd("ReduceMax", [ink9], axes=[1], keepdims=1)
    ch0 = g.nd("Mul", [M, g.nd("Sub", [one, ink_any])])
    g.nd("Concat", [ch0, ink9], "output", axis=1)
    return _model(g)


def _ref_rays(a):
    Hh, Ww = a.shape
    cols = np.unique(a[a != 0])
    if cols.size != 1:
        return None
    C = int(cols[0])
    ink = (a == C)
    out = a.copy()

    def filled(i, j):
        return 0 <= i < Hh and 0 <= j < Ww and ink[i, j]

    for i in range(Hh):
        for j in range(Ww):
            if ink[i, j]:
                continue
            for (dr, dc) in [(1, -1), (1, 1), (-1, -1), (-1, 1)]:
                if filled(i + dr, j) and filled(i, j + dc) and filled(i + dr, j + dc):
                    ry, rx = -dr, -dc
                    k = 1
                    while True:
                        ni, nj = i + k * ry, j + k * rx
                        if not (0 <= ni < Hh and 0 <= nj < Ww):
                            break
                        out[ni, nj] = C
                        k += 1
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


def _matches(prs, ref):
    changed = False
    for a, b in prs:
        p = ref(a)
        if p is None or p.shape != b.shape or not np.array_equal(p, b):
            return False
        if not np.array_equal(a, b):
            changed = True
    return changed


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

    # cross358: same shape, periodic cross extension
    if all(a.shape == b.shape for a, b in prs):
        if _matches(prs, _ref_cross):
            emit("cross358", build_cross)

    # pin200: single seed dot -> pinstripes + highlights
    if all(a.shape == b.shape for a, b in prs):
        if all(np.count_nonzero(a) == 1 for a, b in prs) and _matches(prs, _ref_pin):
            emit("pin200", build_pin)

    # frame240: corner motif -> D4 nested-ring frame
    if all(a.shape == b.shape and a.shape[0] == a.shape[1] for a, b in prs):
        if _matches(prs, _ref_frame):
            emit("frame240", build_frame)

    # rays168: L-tromino empty corners shoot diagonal rays
    if all(a.shape == b.shape for a, b in prs):
        if _matches(prs, _ref_rays):
            emit("rays168", build_rays)

    return out
