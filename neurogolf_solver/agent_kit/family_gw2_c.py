"""family_gw2_c — from-scratch cheap-ONNX rebuilds for golf ranks 32..52.

Each solver is a hand-built opset-10 ONNX (onnx.helper) implementing the TRUE
minimal rule, gated by a pure-numpy mirror that is bit-exact on train+test+arc-gen.

task253 (verify_a61ba2ce): the grid contains four L-trominoes (each a 2x2 bbox with
    one corner missing), one per colour.  Output is a fixed 4x4 grid: each tromino is
    placed (orientation preserved) into the 2x2 quadrant given by the corner its
    'elbow' points at (== the full row / full column of its 2x2 bbox).  Background
    (the central 2x2) stays 0.
    ONNX: per-channel 2x2 dyn-crop via selection MatMul (Srow @ X @ Scol), quadrant
    from argmax row/col sum, scatter into 4x4 via two more selection MatMuls, rebuild
    the background channel, Pad to 30x30.  All intermediates tiny -> very cheap.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
INT64 = onnx.TensorProto.INT64


class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def c(self, dtype, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, dtype, list(dims),
                                         np.asarray(vals).ravel().tolist()))
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


def _eqm(g, a, b):
    """float 0/1 mask = (|a-b| < 0.5)."""
    d = g.nd("Abs", [g.nd("Sub", [a, b])])
    return g.nd("Cast", [g.nd("Less", [d, g.c(F, [1, 1, 1, 1], [0.5])])], to=F)


# --------------------------------------------------------------------------- #
# task253 — a61ba2ce                                                          #
# --------------------------------------------------------------------------- #
def build_253():
    g = _G()
    inp = "input"

    # per-channel bbox top-left (min row / min col)
    rowhas = g.nd("ReduceMax", [inp], axes=[3], keepdims=1)      # [1,10,30,1]
    colhas = g.nd("ReduceMax", [inp], axes=[2], keepdims=1)      # [1,10,1,30]
    ridx = g.c(F, [1, 1, 30, 1], list(range(30)))
    cidx = g.c(F, [1, 1, 1, 30], list(range(30)))
    c30 = g.c(F, [1, 1, 1, 1], [30.0])
    minr = g.nd("Sub", [c30, g.nd("ReduceMax",
                [g.nd("Mul", [rowhas, g.nd("Sub", [c30, ridx])])], axes=[2], keepdims=1)])  # [1,10,1,1]
    minc = g.nd("Sub", [c30, g.nd("ReduceMax",
                [g.nd("Mul", [colhas, g.nd("Sub", [c30, cidx])])], axes=[3], keepdims=1)])  # [1,10,1,1]

    # 2x2 dyn-crop:  Mcrop = Srow @ (X @ Scol)
    k2r = g.c(F, [1, 1, 2, 1], [0, 1])
    k2c = g.c(F, [1, 1, 1, 2], [0, 1])
    jrow = g.c(F, [1, 1, 1, 30], list(range(30)))   # column coord for Srow
    irow = g.c(F, [1, 1, 30, 1], list(range(30)))   # row coord for Scol
    Srow = _eqm(g, g.nd("Add", [minr, k2r]), jrow)  # [1,10,2,30]
    Scol = _eqm(g, irow, g.nd("Add", [minc, k2c]))  # [1,10,30,2]
    tmp = g.nd("MatMul", [inp, Scol])               # [1,10,30,2]
    Mcrop = g.nd("MatMul", [Srow, tmp])             # [1,10,2,2]

    # quadrant = full row / full col of the 2x2 (== argmax of row/col sums)
    rowsum = g.nd("ReduceSum", [Mcrop], axes=[3], keepdims=1)    # [1,10,2,1]
    colsum = g.nd("ReduceSum", [Mcrop], axes=[2], keepdims=1)    # [1,10,1,2]
    dr = g.c(F, [1, 1, 2, 1], [-1, 1])
    dc = g.c(F, [1, 1, 1, 2], [-1, 1])
    zero = g.c(F, [1, 1, 1, 1], [0.0])
    rdiff = g.nd("ReduceSum", [g.nd("Mul", [rowsum, dr])], axes=[2], keepdims=1)  # [1,10,1,1]
    cdiff = g.nd("ReduceSum", [g.nd("Mul", [colsum, dc])], axes=[3], keepdims=1)  # [1,10,1,1]
    qr = g.nd("Cast", [g.nd("Greater", [rdiff, zero])], to=F)    # [1,10,1,1]
    qc = g.nd("Cast", [g.nd("Greater", [cdiff, zero])], to=F)    # [1,10,1,1]

    # scatter Mcrop into a 4x4 at quadrant (qr,qc):  out4 = Pr @ Mcrop @ Pc
    two = g.c(F, [1, 1, 1, 1], [2.0])
    a2 = g.c(F, [1, 1, 1, 2], [0, 1])       # crop-row index
    b2 = g.c(F, [1, 1, 2, 1], [0, 1])       # crop-col index
    i4r = g.c(F, [1, 1, 4, 1], [0, 1, 2, 3])
    j4c = g.c(F, [1, 1, 1, 4], [0, 1, 2, 3])
    Tr = g.nd("Add", [g.nd("Mul", [two, qr]), a2])      # [1,10,1,2]
    Pr = _eqm(g, i4r, Tr)                                # [1,10,4,2]
    Tc = g.nd("Add", [g.nd("Mul", [two, qc]), b2])      # [1,10,2,1]
    Pc = _eqm(g, j4c, Tc)                                # [1,10,2,4]
    out4 = g.nd("MatMul", [g.nd("MatMul", [Pr, Mcrop]), Pc])     # [1,10,4,4]

    # keep only foreground channels, rebuild background channel 0
    fgmask = g.c(F, [1, 10, 1, 1], [0] + [1] * 9)
    fg = g.nd("Mul", [out4, fgmask])                     # [1,10,4,4]
    fgsum = g.nd("ReduceSum", [fg], axes=[1], keepdims=1)       # [1,1,4,4]
    one = g.c(F, [1, 1, 1, 1], [1.0])
    e0 = g.c(F, [1, 10, 1, 1], [1] + [0] * 9)
    bg = g.nd("Mul", [g.nd("Sub", [one, fgsum]), e0])   # [1,10,4,4]
    onehot = g.nd("Add", [fg, bg])                       # [1,10,4,4]
    g.nd("Pad", [onehot], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, 26, 26])
    return _model(g, "gw2c_253")


# --------------------------------------------------------------------------- #
# task228 — 952a094c  (eject box inner-corner markers to opposite outer corner) #
# --------------------------------------------------------------------------- #
_N = 10   # all grids are fixed 10x10


def build_228():
    g = _G()
    i64 = lambda v: g.c(INT64, [len(v)], v)
    X = g.nd("Slice", ["input", i64([0, 0]), i64([_N, _N]), i64([2, 3])])   # [1,10,10,10]

    fgmask = g.c(F, [1, 10, 1, 1], [0] + [1] * 9)
    e0 = g.c(F, [1, 10, 1, 1], [1] + [0] * 9)
    counts = g.nd("Mul", [g.nd("ReduceSum", [X], axes=[2, 3], keepdims=1), fgmask])  # [1,10,1,1]
    maxc = g.nd("ReduceMax", [counts], axes=[1], keepdims=1)
    eVecB = g.nd("Mul", [_eqm(g, counts, maxc), fgmask])        # [1,10,1,1] onehot of frame colour

    Bmask = g.nd("ReduceSum", [g.nd("Mul", [X, eVecB])], axes=[1], keepdims=1)  # [1,1,10,10]
    rH = g.nd("ReduceMax", [Bmask], axes=[3], keepdims=1)       # [1,1,10,1]
    cH = g.nd("ReduceMax", [Bmask], axes=[2], keepdims=1)       # [1,1,1,10]
    ri = g.c(F, [1, 1, _N, 1], list(range(_N)))
    ci = g.c(F, [1, 1, 1, _N], list(range(_N)))
    cN = g.c(F, [1, 1, 1, 1], [float(_N)])
    r0 = g.nd("Sub", [cN, g.nd("ReduceMax", [g.nd("Mul", [rH, g.nd("Sub", [cN, ri])])], axes=[2], keepdims=1)])
    r1 = g.nd("ReduceMax", [g.nd("Mul", [rH, ri])], axes=[2], keepdims=1)
    c0 = g.nd("Sub", [cN, g.nd("ReduceMax", [g.nd("Mul", [cH, g.nd("Sub", [cN, ci])])], axes=[3], keepdims=1)])
    c1 = g.nd("ReduceMax", [g.nd("Mul", [cH, ci])], axes=[3], keepdims=1)
    one = g.c(F, [1, 1, 1, 1], [1.0])
    r0m = g.nd("Sub", [r0, one]); r1p = g.nd("Add", [r1, one])
    r0p = g.nd("Add", [r0, one]); r1m = g.nd("Sub", [r1, one])
    c0m = g.nd("Sub", [c0, one]); c1p = g.nd("Add", [c1, one])
    c0p = g.nd("Add", [c0, one]); c1m = g.nd("Sub", [c1, one])

    # marker one-hot (foreground, non-frame)
    Mkeep = g.nd("Mul", [X, g.nd("Sub", [fgmask, eVecB])])      # [1,10,10,10]

    # row/col remap matrices (top-inner->bottom-outer, bottom-inner->top-outer, etc.)
    A = g.c(F, [1, 1, _N, 1], list(range(_N)))   # target rows
    I = g.c(F, [1, 1, 1, _N], list(range(_N)))   # source rows
    RowMap = g.nd("Add", [
        g.nd("Mul", [_eqm(g, A, r1p), _eqm(g, I, r0p)]),
        g.nd("Mul", [_eqm(g, A, r0m), _eqm(g, I, r1m)])])       # [1,1,10,10]
    Js = g.c(F, [1, 1, _N, 1], list(range(_N)))   # source cols
    Jt = g.c(F, [1, 1, 1, _N], list(range(_N)))   # target cols
    ColMap = g.nd("Add", [
        g.nd("Mul", [_eqm(g, Js, c0p), _eqm(g, Jt, c1p)]),
        g.nd("Mul", [_eqm(g, Js, c1m), _eqm(g, Jt, c0m)])])     # [1,1,10,10]
    reloc = g.nd("MatMul", [g.nd("MatMul", [RowMap, Mkeep]), ColMap])   # [1,10,10,10]

    Bborder = g.nd("Mul", [X, eVecB])                          # [1,10,10,10]
    fg = g.nd("Add", [Bborder, reloc])                         # [1,10,10,10]
    sumfg = g.nd("ReduceSum", [fg], axes=[1], keepdims=1)      # [1,1,10,10]
    bg = g.nd("Mul", [g.nd("Sub", [one, sumfg]), e0])          # [1,10,10,10]
    onehot = g.nd("Add", [fg, bg])
    g.nd("Pad", [onehot], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, 30 - _N, 30 - _N])
    return _model(g, "gw2c_228")


def _ref_228(a):
    a = np.asarray(a, int)
    if a.ndim != 2 or a.shape != (_N, _N):
        return None
    cs = [c for c in range(1, 10) if (a == c).any()]
    if not cs:
        return None
    counts = {c: int((a == c).sum()) for c in cs}
    B = max(counts, key=lambda c: counts[c])
    ys, xs = np.where(a == B)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    out = np.zeros_like(a)
    out[a == B] = B
    mr, mc = (r0 + r1) / 2, (c0 + c1) / 2
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            v = a[i, j]
            if v == 0 or v == B:
                continue
            ni = r1 + 1 if i < mr else r0 - 1
            nj = c1 + 1 if j < mc else c0 - 1
            if not (0 <= ni < a.shape[0] and 0 <= nj < a.shape[1]):
                return None
            out[ni, nj] = v
    return out


def _ref_253(a):
    a = np.asarray(a, int)
    if a.ndim != 2 or max(a.shape) > 30:
        return None
    out = np.zeros((4, 4), int)
    cs = [c for c in range(1, 10) if (a == c).any()]
    for c in cs:
        ys, xs = np.where(a == c)
        rmin, rmax, cmin, cmax = ys.min(), ys.max(), xs.min(), xs.max()
        if rmax - rmin != 1 or cmax - cmin != 1:
            return None
        block = np.zeros((2, 2), int)
        for y, x in zip(ys, xs):
            block[y - rmin, x - cmin] = 1
        if block.sum() != 3:
            return None
        qr = 1 if block[1].sum() > block[0].sum() else 0
        qc = 1 if block[:, 1].sum() > block[:, 0].sum() else 0
        out[2 * qr:2 * qr + 2, 2 * qc:2 * qc + 2] = block * c
    return out


# --------------------------------------------------------------------------- #
def _pairs(examples):
    out = []
    for s in ("train", "test"):
        for e in examples.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                return []
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
    if _matches(prs, _ref_253):
        try:
            yield ("gw2c_a61ba2ce", build_253())
        except Exception:
            pass
    if _matches(prs, _ref_228):
        try:
            yield ("gw2c_952a094c", build_228())
        except Exception:
            pass
