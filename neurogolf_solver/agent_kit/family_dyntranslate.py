"""DATA-DEPENDENT TRANSLATE / SNAP family (opset-10, origin-anchored).

Every rule MOVES one coloured object by a DATA-DEPENDENT displacement that is
computed from the content itself (a target marker / another block).  The grids
are zero-padded TOP-LEFT to 30x30, sizes vary per example, so the displacement
cannot be baked into static `Pad`/`Slice` attributes.  Instead we follow the
verified "computed shift-matrix + MatMul" recipe:

    * read positions off the one-hot tensor with position-weighted
      ReduceSum / ReduceMax against constant index grids
      ROW=[1,1,30,1]=arange(30), COL=[1,1,1,30]=arange(30);
    * derive a scalar displacement (dy,dx) (e.g. sign of a target offset, or the
      gap to make two boxes adjacent);
    * build a [1,1,30,30] permutation/shift matrix  S[i,k]=1 iff (i-k)==dy  with
      diff=ROW-COL-dy ; S = Cast(Less(Abs(diff),0.5),FLOAT)  and apply it with two
      MatMuls  ( Srow @ plane @ Scol )  to translate a colour plane by (dy,dx)
      while keeping STATIC shapes.

The moved object is routed back into the one-hot tensor additively
(output = input + (shifted - plane) (x) (e_O - e_bg)) so the vacated cells become
real background and every other colour / the padding is preserved exactly.

Rules (the matching one + colours are inferred structurally from the pairs)
---------------------------------------------------------------------------
  king     A single cell of colour M takes ONE king-move step toward a single
           target cell of colour T:  (dy,dx)=(sign(rT-rM), sign(cT-cM)).
  adjacent A coloured object O slides along the single axis on which it overlaps
           a target block T, toward T, until their bounding boxes are adjacent
           (gap 0).  dy/dx are the signed gaps; the perpendicular axis is left
           untouched automatically because an overlap on it zeroes the indicator.
  slide    A coloured object M is shifted by a fixed period P (down/right) along
           the axis of a static marker line K; the axis is read from the marker's
           bounding box (vertical iff its row-span exceeds its col-span).  Markers
           and every other colour stay put.

A numpy mirror reproduces the EXACT ONNX float-then-threshold semantics and a
candidate is emitted only when it matches EVERY available train+test+arc-gen pair
(the grader's gate), so wrong hypotheses are dropped before scoring.
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
        self._half = None
        self._nhalf = None
        self._one = None
        self._cbig = None
        self._rowidx = None
        self._colidx = None

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

    # shared scalars / index grids ----------------------------------------- #
    def half(self):
        if self._half is None:
            self._half = self.f([1, 1, 1, 1], [0.5])
        return self._half

    def nhalf(self):
        if self._nhalf is None:
            self._nhalf = self.f([1, 1, 1, 1], [-0.5])
        return self._nhalf

    def one(self):
        if self._one is None:
            self._one = self.f([1, 1, 1, 1], [1.0])
        return self._one

    def cbig(self):
        if self._cbig is None:
            self._cbig = self.f([1, 1, 1, 1], [_CBIG])
        return self._cbig

    def rowidx(self):
        if self._rowidx is None:
            self._rowidx = self.f([1, 1, H, 1], list(range(H)))
        return self._rowidx

    def colidx(self):
        if self._colidx is None:
            self._colidx = self.f([1, 1, 1, W], list(range(W)))
        return self._colidx


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# graph helpers                                                               #
# --------------------------------------------------------------------------- #
def _plane(g, ch):
    """Channel `ch` of the input as a [1,1,30,30] tensor."""
    return g.nd("Slice", ["input", g.i64([ch]), g.i64([ch + 1]), g.i64([1])])


def _bbox(g, plane):
    """Return (r0,r1,c0,c1) bounding-box scalars [1,1,1,1] of a 0/1 presence plane."""
    rowidx, colidx, cbig = g.rowidx(), g.colidx(), g.cbig()
    rowhas = g.nd("ReduceMax", [plane], axes=[3], keepdims=1)        # [1,1,30,1]
    colhas = g.nd("ReduceMax", [plane], axes=[2], keepdims=1)        # [1,1,1,30]
    r1 = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    r0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    c1 = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    c0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])
    return r0, r1, c0, c1


def _sign(g, x):
    """Integer sign of scalar x: (x>0.5) - (x<-0.5)  -> {-1,0,1} float."""
    pos = g.nd("Cast", [g.nd("Greater", [x, g.half()])], to=F)
    neg = g.nd("Cast", [g.nd("Less", [x, g.nhalf()])], to=F)
    return g.nd("Sub", [pos, neg])


def _shift_plane(g, plane, dy, dx):
    """Rigidly translate a [1,1,30,30] plane by data-dependent (dy,dx) scalars
    using two MatMuls with computed [1,1,30,30] shift matrices (zero fill)."""
    rowidx, colidx, half = g.rowidx(), g.colidx(), g.half()
    # Srow[i,k]=1 iff (i-k)==dy   (i = output row = axis2, k = input row = axis3)
    diffr = g.nd("Sub", [g.nd("Sub", [rowidx, colidx]), dy])         # [1,1,30,30]
    srow = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diffr]), half])], to=F)
    rowshift = g.nd("MatMul", [srow, plane])                         # shift rows
    # Scol[k,j]=1 iff (j-k)==dx   (k = input col = axis2, j = output col = axis3)
    diffc = g.nd("Sub", [g.nd("Sub", [colidx, rowidx]), dx])
    scol = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diffc]), half])], to=F)
    return g.nd("MatMul", [rowshift, scol])                          # shift cols


def _emit_move(g, planeO, O, shifted):
    """output = input + (shifted - planeO) (x) (e_O - e_bg)."""
    delta = g.nd("Sub", [shifted, planeO])                           # [1,1,30,30]
    vec = g.f([1, CHANNELS, 1, 1],
              [(1.0 if c == O else 0.0) - (1.0 if c == 0 else 0.0) for c in range(CHANNELS)])
    g.nd("Add", ["input", g.nd("Mul", [delta, vec])], "output")


# --------------------------------------------------------------------------- #
# ONNX builders                                                                #
# --------------------------------------------------------------------------- #
def build_king(M, T):
    """Single cell of colour M takes one king step toward single cell of colour T.
    The mover is a single cell, so the shifted plane is the cheap rank-1 outer
    product of a row- and a column-selection vector (no [30,30] MatMul)."""
    g = _G()
    pM = _plane(g, M)
    pT = _plane(g, T)
    rowidx, colidx, half = g.rowidx(), g.colidx(), g.half()
    rM, _, cM, _ = _bbox(g, pM)
    rT, _, cT, _ = _bbox(g, pT)
    dy = _sign(g, g.nd("Sub", [rT, rM]))
    dx = _sign(g, g.nd("Sub", [cT, cM]))
    new_r = g.nd("Add", [rM, dy])                                    # [1,1,1,1]
    new_c = g.nd("Add", [cM, dx])
    rowsel = g.nd("Cast", [g.nd("Less",
              [g.nd("Abs", [g.nd("Sub", [rowidx, new_r])]), half])], to=F)   # [1,1,30,1]
    colsel = g.nd("Cast", [g.nd("Less",
              [g.nd("Abs", [g.nd("Sub", [colidx, new_c])]), half])], to=F)   # [1,1,1,30]
    shifted = g.nd("Mul", [rowsel, colsel])                          # [1,1,30,30]
    _emit_move(g, pM, M, shifted)
    return _model(g)


def build_adjacent(O, T):
    """Object O slides along the overlapping axis toward block T until adjacent."""
    g = _G()
    pO = _plane(g, O)
    pT = _plane(g, T)
    or0, or1, oc0, oc1 = _bbox(g, pO)
    tr0, tr1, tc0, tc1 = _bbox(g, pT)
    half, one = g.half(), g.one()

    below = g.nd("Cast", [g.nd("Greater", [g.nd("Sub", [tr0, or1]), half])], to=F)
    above = g.nd("Cast", [g.nd("Greater", [g.nd("Sub", [or0, tr1]), half])], to=F)
    right = g.nd("Cast", [g.nd("Greater", [g.nd("Sub", [tc0, oc1]), half])], to=F)
    left = g.nd("Cast", [g.nd("Greater", [g.nd("Sub", [oc0, tc1]), half])], to=F)

    dy_b = g.nd("Sub", [g.nd("Sub", [tr0, or1]), one])               # tr0-or1-1
    dy_a = g.nd("Add", [g.nd("Sub", [tr1, or0]), one])               # tr1-or0+1
    dx_r = g.nd("Sub", [g.nd("Sub", [tc0, oc1]), one])               # tc0-oc1-1
    dx_l = g.nd("Add", [g.nd("Sub", [tc1, oc0]), one])               # tc1-oc0+1
    dy = g.nd("Add", [g.nd("Mul", [below, dy_b]), g.nd("Mul", [above, dy_a])])
    dx = g.nd("Add", [g.nd("Mul", [right, dx_r]), g.nd("Mul", [left, dx_l])])

    shifted = _shift_plane(g, pO, dy, dx)
    _emit_move(g, pO, O, shifted)
    return _model(g)


def build_slide(M, K, P):
    """Shift object M by period P (down/right) along marker-line K's long axis."""
    g = _G()
    pM = _plane(g, M)
    pK = _plane(g, K)
    half, one = g.half(), g.one()
    kr0, kr1, kc0, kc1 = _bbox(g, pK)
    rspan = g.nd("Sub", [kr1, kr0])
    cspan = g.nd("Sub", [kc1, kc0])
    vert = g.nd("Cast", [g.nd("Greater", [g.nd("Sub", [rspan, cspan]), half])], to=F)
    horiz = g.nd("Sub", [one, vert])
    Pc = g.f([1, 1, 1, 1], [P])
    dy = g.nd("Mul", [vert, Pc])
    dx = g.nd("Mul", [horiz, Pc])
    shifted = _shift_plane(g, pM, dy, dx)
    _emit_move(g, pM, M, shifted)
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy mirrors (reproduce the ONNX float-then-threshold semantics EXACTLY)    #
# --------------------------------------------------------------------------- #
def _onehot(a):
    o = np.zeros((CHANNELS,) + a.shape, np.float64)
    for c in range(CHANNELS):
        o[c] = (a == c)
    return o


def _bbox_np(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _sgn(x):
    return (x > 0) - (x < 0)


def _predict(a, O, dy, dx):
    """Predicted one-hot bool [10,h,w] of: shift colour-O plane by (dy,dx),
    output = input + (shifted - plane_O)(x)(e_O - e_bg), then threshold > 0."""
    h, w = a.shape
    chan = _onehot(a)
    oldO = chan[O].copy()
    newO = np.zeros((h, w), np.float64)
    ys, xs = np.where(oldO > 0)
    for i, j in zip(ys, xs):
        ni, nj = i + dy, j + dx
        if 0 <= ni < h and 0 <= nj < w:
            newO[ni, nj] += 1.0
    chan[O] = newO
    chan[0] = chan[0] + oldO - newO
    return chan > 0


def _ref_king(a, M, T):
    bm = _bbox_np(a == M)
    bt = _bbox_np(a == T)
    if bm is None or bt is None:
        return None
    if (a == M).sum() != 1 or (a == T).sum() != 1:
        return None
    dy = _sgn(bt[0] - bm[0])
    dx = _sgn(bt[2] - bm[2])
    if dy == 0 and dx == 0:
        return None
    return _predict(a, M, dy, dx)


def _ref_adjacent(a, O, T):
    bo = _bbox_np(a == O)
    bt = _bbox_np(a == T)
    if bo is None or bt is None:
        return None
    or0, or1, oc0, oc1 = bo
    tr0, tr1, tc0, tc1 = bt
    below = 1 if (tr0 - or1) > 0 else 0
    above = 1 if (or0 - tr1) > 0 else 0
    right = 1 if (tc0 - oc1) > 0 else 0
    left = 1 if (oc0 - tc1) > 0 else 0
    dy = below * (tr0 - or1 - 1) + above * (tr1 - or0 + 1)
    dx = right * (tc0 - oc1 - 1) + left * (tc1 - oc0 + 1)
    if dy == 0 and dx == 0:
        return None
    return _predict(a, O, dy, dx)


def _ref_slide(a, M, K, P):
    bm = _bbox_np(a == M)
    bk = _bbox_np(a == K)
    if bm is None or bk is None:
        return None
    rspan = bk[1] - bk[0]
    cspan = bk[3] - bk[2]
    vert = rspan > cspan
    dy = P if vert else 0
    dx = 0 if vert else P
    return _predict(a, M, dy, dx)


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
    seen_move = False
    for a, b in prs:
        pred = ref(a)
        if pred is None or not np.array_equal(pred, _onehot(b).astype(bool)):
            return False
        if not np.array_equal(a, b):
            seen_move = True
    return seen_move


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if any(a.shape != b.shape for a, b in prs):     # every rule preserves shape
        return []
    if all(np.array_equal(a, b) for a, b in prs):   # identity -> not us
        return []

    colors = set()
    for a, _ in prs:
        colors |= set(np.unique(a[a != 0]).tolist())
    colors = sorted(colors)
    if len(colors) < 2:
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

    for M in colors:
        for T in colors:
            if M == T:
                continue
            if _matches(prs, lambda a, M=M, T=T: _ref_king(a, M, T)):
                emit(f"king_M{M}_T{T}", lambda M=M, T=T: build_king(M, T))
            if _matches(prs, lambda a, O=M, T=T: _ref_adjacent(a, O, T)):
                emit(f"adj_O{M}_T{T}", lambda O=M, T=T: build_adjacent(O, T))
            for P in range(1, 7):
                if _matches(prs, lambda a, M=M, K=T, P=P: _ref_slide(a, M, K, P)):
                    emit(f"slide_M{M}_K{T}_P{P}", lambda M=M, K=T, P=P: build_slide(M, K, P))
                    break

    return out
