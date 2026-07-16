"""family_crk8_5 -- doubly-periodic pattern completion (data-dependent period).

Target: tasks whose input is a fully-tiled (doubly periodic) pattern that has been
partially OCCLUDED by background holes, and whose output restores the complete
pattern (no background cells in the output).  The vertical / horizontal periods
vary per example, so they are recovered AT INFERENCE TIME and the fill is applied
with computed [30,30] "fold" matrices via MatMul (static shapes throughout).

Algorithm (mirrored exactly in numpy for detection, see `_sim`)
--------------------------------------------------------------
  Xnz       = input one-hot with channel 0 (background) zeroed.
  occ_nz    = sum_c>=1 Xnz[c]            # 1 on non-background cells
  Gmask     = sum_c   input[c]           # 1 over the whole HxW rectangle

  Period detection (per axis) by exact pattern autocorrelation, cheaply via a
  row/col Gram matrix:
    Gv[a,b]   = #cols j where rows a,b share the SAME non-bg colour
    Goccv[a,b]= #cols j where rows a,b are both non-bg
    mismv[a,b]= Goccv-Gv  (= disagreements)        # 0 if rows a,b are compatible
    mismv_p   = sum_a mismv[a,a+p]                  # vertical autocorr at shift p
    pv        = smallest p in 1..K with mismv_p == 0
  (likewise ph horizontally on a column Gram).

  Fold / unfold by the recovered period with a single congruence matrix:
    Mh[i,k] = 1 iff (i-k) mod pv == 0
    Mw[k,j] = 1 iff (k-j) mod ph == 0
    out = Mh @ Xnz @ Mw                              # sum over the 2D residue class
  Every cell receives the (unique) non-bg colour of its residue class, then the
  result is masked back to the real rectangle.  (output > 0) recovers the grid.

Only emitted when the numpy reference reproduces EVERY train+test pair (the grader
then also re-checks all arc-gen), so wrong hypotheses are never scored.
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
G = HEIGHT          # 30
K = 15              # max candidate period


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _Graph:
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

    def i(self, dims, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, list(dims),
                                         [int(v) for v in np.asarray(vals).ravel()]))
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


# --------------------------------------------------------------------------- #
# ONNX builder                                                                #
# --------------------------------------------------------------------------- #
def _period(g, Mflat, occflat, Dstack, half, ramp):
    """Given the [1,30,N] content rows and [1,30,30] occupancy rows, return the
    smallest-period scalar as INT64 [1,1]."""
    # Gram of content rows: [1,30,30]
    Mt = g.nd("Transpose", [Mflat], perm=[0, 2, 1])              # [1,N,30]
    Gv = g.nd("MatMul", [Mflat, Mt])                             # [1,30,30]
    Ot = g.nd("Transpose", [occflat], perm=[0, 2, 1])           # [1,30,30]
    Go = g.nd("MatMul", [occflat, Ot])                          # [1,30,30]
    mism = g.nd("Sub", [Go, Gv])                                # [1,30,30] (>=0)
    # diagonal sums for each candidate p: mism[1,30,30] * Dstack[K,30,30] -> [K,30,30]
    prod = g.nd("Mul", [mism, Dstack])                          # [K,30,30]
    vec = g.nd("ReduceSum", [prod], axes=[1, 2], keepdims=0)    # [K]
    z = g.nd("Cast", [g.nd("Less", [vec, half])], to=F)        # 1 where period
    w = g.nd("Mul", [z, ramp])                                  # weight, smaller p heavier
    idx = g.nd("ArgMax", [w], axis=0, keepdims=1)              # [1] int64 in 0..K-1
    p1 = g.nd("Add", [idx, g.i([1], [1])])                     # period = idx+1
    return g.nd("Reshape", [p1, g.i([2], [1, 1])])             # [1,1] int64


def build():
    g = _Graph()
    half = g.f([1], [0.5])

    # channel-0 (background) zeroed content
    maskh = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    Xnz = g.nd("Mul", ["input", maskh])                         # [1,10,30,30]
    Gmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)  # [1,1,30,30]
    occ = g.nd("ReduceSum", [Xnz], axes=[1], keepdims=1)        # [1,1,30,30]

    # candidate-diagonal selector stack [K,30,30]: D[p-1,a,b]=1 iff b-a==p
    Dst = np.zeros((K, G, G), np.float32)
    for p in range(1, K + 1):
        for a in range(G - p):
            Dst[p - 1, a, a + p] = 1.0
    Dstack = g.f([K, G, G], Dst)
    ramp = g.f([K], [float(K - q) for q in range(K)])           # descending

    # --- vertical period (rows) ---
    Mv = g.nd("Reshape", [g.nd("Transpose", [Xnz], perm=[0, 2, 1, 3]),
                          g.i([3], [1, G, CHANNELS * G])])       # [1,30,300]
    Ov = g.nd("Reshape", [occ, g.i([3], [1, G, G])])            # [1,30,30]
    pv = _period(g, Mv, Ov, Dstack, half, ramp)

    # --- horizontal period (cols) ---
    Mh = g.nd("Reshape", [g.nd("Transpose", [Xnz], perm=[0, 3, 1, 2]),
                          g.i([3], [1, G, CHANNELS * G])])       # [1,30,300]
    Oh = g.nd("Reshape", [g.nd("Transpose", [occ], perm=[0, 1, 3, 2]),
                          g.i([3], [1, G, G])])                  # [1,30,30]
    ph = _period(g, Mh, Oh, Dstack, half, ramp)

    # --- fold matrices from periods ---
    AB = g.i([G, G], np.subtract.outer(np.arange(G), np.arange(G)))  # a-b
    halfm = g.f([1, 1], [0.5])
    MhF = g.nd("Cast", [g.nd("Less", [g.nd("Cast", [g.nd("Mod", [AB, pv], None, fmod=0)], to=F), halfm])], to=F)
    MwF = g.nd("Cast", [g.nd("Less", [g.nd("Cast", [g.nd("Mod", [AB, ph], None, fmod=0)], to=F), halfm])], to=F)

    foldH = g.nd("MatMul", [MhF, Xnz])                          # rows fold [1,10,30,30]
    full = g.nd("MatMul", [foldH, MwF])                         # cols fold [1,10,30,30]
    g.nd("Mul", [full, Gmask], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy reference (mirrors the ONNX semantics EXACTLY)                          #
# --------------------------------------------------------------------------- #
def _onehot(a):
    H, W = a.shape
    X = np.zeros((CHANNELS, G, G), np.float32)
    for c in range(CHANNELS):
        X[c, :H, :W] = (a == c)
    return X


_AB = np.subtract.outer(np.arange(G), np.arange(G))


def _sim(a):
    X = _onehot(a)
    Xnz = X.copy(); Xnz[0] = 0.0
    occ = Xnz.sum(0)                                            # [30,30]
    Gmask = X.sum(0)                                            # [30,30]

    def period(Mflat, Oflat):
        Gv = Mflat @ Mflat.T
        Go = Oflat @ Oflat.T
        mism = Go - Gv
        for p in range(1, K + 1):
            s = sum(mism[x, x + p] for x in range(G - p))
            if abs(s) < 0.5:
                return p
        return 1

    Mv = np.transpose(Xnz, (1, 0, 2)).reshape(G, CHANNELS * G)
    pv = period(Mv, occ.reshape(G, G))
    Mh = np.transpose(Xnz, (2, 0, 1)).reshape(G, CHANNELS * G)
    ph = period(Mh, occ.T.reshape(G, G))

    MhF = (np.mod(_AB, pv) == 0).astype(np.float32)
    MwF = (np.mod(_AB, ph) == 0).astype(np.float32)
    out = np.einsum('ik,ckj->cij', MhF, Xnz)
    out = np.einsum('cik,kj->cij', out, MwF)
    out = out * Gmask[None]
    return (out > 0).astype(np.float32)


def _grid_from(out):
    sel = out > 0.5
    cnt = sel.sum(0)
    return sel, cnt


# --------------------------------------------------------------------------- #
# detection / entry                                                            #
# --------------------------------------------------------------------------- #
def _pairs(ex, splits):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > G or max(b.shape) > G:
                continue
            out.append((a, b))
    return out


def candidates(ex):
    det = _pairs(ex, ("train", "test"))
    if not det:
        return []
    if any(a.shape != b.shape for a, b in det):
        return []
    # output must be fully non-background (this family fills every cell)
    for a, b in det:
        if (b == 0).any():
            return []
    # verify the numpy reference reproduces every train+test pair exactly
    for a, b in det:
        pred = _sim(a)
        tgt = _onehot(b)
        if pred.shape != tgt.shape or not (pred == tgt).all():
            return []
    try:
        model = build()
        onnx.checker.check_model(model, full_check=True)
    except Exception:
        return []
    return [("crk8_5_period", model)]
