"""crk8_1: cracking a slice of hard unsolved tasks.

Currently implements:

* task 17 family -- DOUBLY-PERIODIC PATTERN REPAIR with a DATA-DEPENDENT period.
  The grid carries a 2-D periodic texture partially erased to colour 0 (holes).
  The output refills every hole with the colour its period demands.  The period
  varies per example so it is recovered AT INFERENCE TIME by autocorrelation
  (smallest shift d in 1..K with zero pattern-mismatch on the visible overlap),
  separately per axis, then the fill is realised as an OR over the residue class
  via two tiling matrices and two MatMuls:

      comp = Tv @ X @ Th ,  Tv[i,k]=1 iff (i-k)%Pv==0 ,  Th[k,j]=1 iff (k-j)%Ph==0

  comp[c,i,j] = number of class siblings of colour c, so thresholding (>0) and
  masking to the grid region reproduces the completed pattern.  Channel 0 (the
  hole colour) is dropped because the repaired pattern contains no colour-0 cell.

Detection mirrors the ONNX semantics EXACTLY in numpy and only emits when the
reconstruction reproduces EVERY train+test+arc-gen pair.
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
KMAX = 14


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


# --------------------------------------------------------------------------- #
# ONNX builder                                                                #
# --------------------------------------------------------------------------- #
def _shift_plane(g, t, d, axis):
    """Shift a [1,1,30,30] tensor LEFT/UP by static d>=1 along `axis`
    (2=rows up, 3=cols left): out[...,j] = t[...,j+d], zero-filled."""
    if axis == 3:
        sl = g.nd("Slice", [t, g.i64([d]), g.i64([G]), g.i64([3])])
        return g.nd("Pad", [sl], mode="constant", value=0.0,
                    pads=[0, 0, 0, 0, 0, 0, 0, d])
    sl = g.nd("Slice", [t, g.i64([d]), g.i64([G]), g.i64([2])])
    return g.nd("Pad", [sl], mode="constant", value=0.0,
                pads=[0, 0, 0, 0, 0, 0, d, 0])


def _period_scalar(g, known, color, axis, half):
    """Smallest static d in 1..KMAX whose pattern autocorrelation is 0, as a
    [1,1] float (0 if none)."""
    qstar = None
    remaining = None
    one = g.f([1, 1], [1.0])
    for d in range(1, KMAX + 1):
        shK = _shift_plane(g, known, d, axis)
        shC = _shift_plane(g, color, d, axis)
        both = g.nd("Mul", [known, shK])
        diff = g.nd("Abs", [g.nd("Sub", [color, shC])])
        mism = g.nd("Cast", [g.nd("Greater", [diff, half])], to=F)
        prod = g.nd("Mul", [both, mism])
        score = g.nd("ReduceSum", [prod], axes=[2, 3], keepdims=0)   # [1,1]
        zero = g.nd("Cast", [g.nd("Less", [score, half])], to=F)     # 1 iff period
        if remaining is None:
            gate = zero
            remaining = g.nd("Sub", [one, zero])
        else:
            gate = g.nd("Mul", [zero, remaining])
            remaining = g.nd("Mul", [remaining, g.nd("Sub", [one, zero])])
        contrib = g.nd("Mul", [gate, g.f([1, 1], [float(d)])])
        qstar = contrib if qstar is None else g.nd("Add", [qstar, contrib])
    return qstar


def _tile_mat(g, Dmat, period, half):
    """[1,1,30,30] tiling matrix: 1 where (i-k) divisible by `period`."""
    m = g.nd("Mod", [Dmat, period], fmod=1)                          # [30,30]
    t = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [m]), half])], to=F)  # [30,30]
    return g.nd("Reshape", [t, g.i64([1, 1, G, G])])                  # [1,1,30,30]


def build():
    g = _Graph()
    half = g.f([1, 1], [0.5])

    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    color = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)     # [1,1,30,30]
    inch0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])
    known = g.nd("Sub", [realmask, inch0])

    # data-dependent periods (as [1,1] float)
    Pv = _period_scalar(g, known, color, 2, half)   # vertical (rows)
    Ph = _period_scalar(g, known, color, 3, half)   # horizontal (cols)

    rowg = np.arange(G).reshape(G, 1).repeat(G, 1)
    colg = np.arange(G).reshape(1, G).repeat(G, 0)
    Dmat = g.f([G, G], (rowg - colg).astype(np.float32))             # D[i,k]=i-k

    Tv = _tile_mat(g, Dmat, Pv, half)               # [1,1,30,30]
    Th = _tile_mat(g, Dmat, Ph, half)

    tmp = g.nd("MatMul", [Tv, "input"])             # broadcast -> [1,10,30,30]
    comp = g.nd("MatMul", [tmp, Th])                # [1,10,30,30]

    # drop channel 0 (hole colour), mask to grid region
    chmask = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    masked = g.nd("Mul", [comp, chmask])
    g.nd("Mul", [masked, realmask], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy mirror (EXACTLY mirrors the ONNX graph)                                #
# --------------------------------------------------------------------------- #
def _onehot(a):
    H, W = a.shape
    X = np.zeros((CHANNELS, G, G), np.float32)
    for c in range(CHANNELS):
        X[c, :H, :W] = (a == c)
    return X


def _shift_np(t, d, axis):
    out = np.zeros_like(t)
    if d >= G:
        return out
    if axis == 1:
        out[..., :G - d] = t[..., d:]
    else:
        out[..., :G - d, :] = t[..., d:, :]
    return out


def _period_np(known, color, axis):
    for d in range(1, KMAX + 1):
        shK = _shift_np(known, d, axis)
        shC = _shift_np(color, d, axis)
        both = known * shK
        mism = (np.abs(color - shC) > 0.5).astype(np.float32)
        if float((both * mism).sum()) < 0.5:
            return float(d)
    return 0.0


def _sim(a):
    X = _onehot(a)
    color = sum(c * X[c] for c in range(CHANNELS))         # [30,30]
    realmask = X.sum(0)                                     # [30,30]
    known = realmask - X[0]
    Pv = _period_np(known, color, 0)
    Ph = _period_np(known, color, 1)
    if Pv == 0 or Ph == 0:
        return None
    rowg = np.arange(G).reshape(G, 1)
    colg = np.arange(G).reshape(1, G)
    D = (rowg - colg).astype(np.float32)
    Tv = (np.abs(np.fmod(D, Pv)) < 0.5).astype(np.float32)
    Th = (np.abs(np.fmod(D, Ph)) < 0.5).astype(np.float32)
    comp = np.einsum("ik,ckj->cij", Tv, X)                 # Tv @ X[c]
    comp = np.einsum("cik,kj->cij", comp, Th)              # @ Th
    comp[0] = 0.0
    out = comp * realmask[None]
    return out


def _sim_grid(a):
    out = _sim(a)
    if out is None:
        return None
    realmask = (_onehot(a).sum(0) > 0.5)
    sel = out > 0.5                                        # [10,30,30] bool
    cnt = sel.sum(0)
    if (cnt[realmask] != 1).any():
        return None
    if (cnt[~realmask] != 0).any():
        return None
    return np.argmax(sel, axis=0)


# --------------------------------------------------------------------------- #
# detection / entry point                                                      #
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
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []
    if any(a.shape != b.shape for a, b in allp):
        return []
    if all(np.array_equal(a, b) for a, b in allp):
        return []

    # holes must be colour 0; outputs must contain no colour-0 cell in grid
    for a, b in det:
        d = (a != b)
        if d.any() and set(int(v) for v in a[d].tolist()) != {0}:
            return []
        if (b == 0).any():
            return []

    def ok(plist):
        for a, b in plist:
            pred = _sim_grid(a)
            if pred is None:
                return False
            H, W = b.shape
            if not np.array_equal(pred[:H, :W], b):
                return False
        return True

    if not ok(det) or not ok(allp):
        return []

    try:
        model = build()
        onnx.checker.check_model(model, full_check=True)
    except Exception:
        return []
    return [("crk8_periodic2d", model)]
