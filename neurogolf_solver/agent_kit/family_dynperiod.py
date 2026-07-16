"""VARIABLE-PERIOD periodic completion (DATA-DEPENDENT period, opset 10).

The input carries a spatially periodic pattern that is partially OCCLUDED: a set
of cells are replaced by a single "hole" colour ``h`` (usually the background 0).
The output is the same-size grid with every hole filled by the value its period
demands.  Unlike ``family_periodic`` the period is NOT fixed across the task -- it
varies per example -- so a static graph cannot bake it in.  Instead the period is
recovered AT INFERENCE TIME by autocorrelation and the fill is applied with a
DATA-DEPENDENT shift realised through computed permutation matrices + MatMul
(static shapes throughout).

Pipeline (origin-anchored, size/period independent)
---------------------------------------------------
  realmask = ReduceSum(input, ch)            # [1,1,30,30] 1 on real cells, 0 pad
  color    = sum_c c * input[:,c]            # [1,1,30,30] colour index (0 on pad)
  known    = realmask - input[:,h]           # 1 where real & not a hole
  X        = input with channel h zeroed     # clean one-hot content

  Period per axis (q horizontal, p vertical), as a data-dependent scalar:
    for each candidate d in 1..Qmax:
        score_d = sum over the real region of [ known(x) & known(x+d) & colour
                  differs ]                   # exact pattern autocorrelation
    the period = the SMALLEST d with score 0 (0 if none -> that axis is treated as
    aperiodic, i.e. an identity fill).  Realised with a cumulative "first zero"
    one-hot over the static candidate set, so the period pops out as a [1,1] float.

  Fill (doubling OR, data-dependent shift):
    a residue class repeats every <period> cells, so OR-ing X with all of its
    shifted copies recovers the hidden value wherever ANY copy is visible.  The
    shift by  d = period * 2^k  is done by F @ S  (horizontal) / S @ F (vertical)
    where S[k,j] = 1 iff j == k +/- d is built on the fly from index grids and the
    computed period (|colidx - rowidx -/+ d| < 0.5).  5 doubling steps cover the
    whole 30-wide grid for any period >= 1; an out-of-range shift yields an all-zero
    matrix (a harmless no-op under Max), and period 0 yields the identity matrix.

  Horizontal fill then vertical fill == OR over the full 2D residue class, so every
  hole with a visible class representative is filled.  The result is masked back to
  the real region (padding stays zero); for an all-background hole colour the
  leftover real cells are routed to channel 0.

Detection mirrors the ONNX semantics in numpy EXACTLY (same autocorrelation /
first-zero / doubling-OR) and only emits when the reconstruction reproduces EVERY
train+test+arc-gen pair, so wrong hypotheses are never scored.
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
QMAX = 15


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


# --------------------------------------------------------------------------- #
# ONNX builder                                                                #
# --------------------------------------------------------------------------- #
def _shift_plane(g, t, d, axis):
    """Shift a [1,1,30,30] (or [1,K,30,30]) tensor LEFT/UP by static d>=1 along
    `axis` (2=rows up, 3=cols left): out[...,j] = t[...,j+d], zero-filled."""
    if axis == 3:
        sl = g.nd("Slice", [t, g.i64([d]), g.i64([G]), g.i64([3])])  # [.,.,30,30-d]
        return g.nd("Pad", [sl], mode="constant", value=0.0,
                    pads=[0, 0, 0, 0, 0, 0, 0, d])
    sl = g.nd("Slice", [t, g.i64([d]), g.i64([G]), g.i64([2])])      # [.,.,30-d,30]
    return g.nd("Pad", [sl], mode="constant", value=0.0,
                pads=[0, 0, 0, 0, 0, 0, d, 0])


def _period_scalar(g, known, color, axis, half):
    """Smallest static d in 1..QMAX whose pattern autocorrelation is 0, as a [1,1]
    float (0 if none)."""
    qstar = None
    remaining = None                       # product of (1 - zero_{d'<d}), [1,1]
    one = g.f([1, 1], [1.0])
    for d in range(1, QMAX + 1):
        shK = _shift_plane(g, known, d, axis)
        shC = _shift_plane(g, color, d, axis)
        both = g.nd("Mul", [known, shK])                              # both visible
        diff = g.nd("Abs", [g.nd("Sub", [color, shC])])
        mism = g.nd("Cast", [g.nd("Greater", [diff, half])], to=F)    # colour differs
        prod = g.nd("Mul", [both, mism])
        score = g.nd("ReduceSum", [prod], axes=[2, 3], keepdims=0)    # [1,1]
        zero = g.nd("Cast", [g.nd("Less", [score, half])], to=F)      # 1 iff period
        if remaining is None:
            gate = zero
            remaining = g.nd("Sub", [one, zero])
        else:
            gate = g.nd("Mul", [zero, remaining])
            remaining = g.nd("Mul", [remaining, g.nd("Sub", [one, zero])])
        contrib = g.nd("Mul", [gate, g.f([1, 1], [float(d)])])
        qstar = contrib if qstar is None else g.nd("Add", [qstar, contrib])
    return qstar                                                     # [1,1] float


def _shift_mats(g, t1, d, half, vertical):
    """Two [30,30] permutation matrices shifting by +/- d (d a [1,1] float)."""
    # t1 = colidx-rowidx (horizontal) or rowidx-colidx (vertical); cond |t1 -/+ d|<.5
    s_a = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [t1, d])]), half])], to=F)
    s_b = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Add", [t1, d])]), half])], to=F)
    return s_a, s_b


def _fill_axis(g, Ft, qstar, axis, rowidx, colidx, half):
    """Doubling-OR fill along `axis` (3 horizontal, 2 vertical) by data-dependent
    period qstar ([1,1])."""
    if axis == 3:
        t1 = g.nd("Sub", [colidx, rowidx])            # j - k  -> [30,30]
    else:
        t1 = g.nd("Sub", [rowidx, colidx])            # i - k  -> [30,30]
    cur = Ft
    for k in range(5):
        d = qstar if k == 0 else g.nd("Mul", [qstar, g.f([1, 1], [float(2 ** k)])])
        s_pos, s_neg = _shift_mats(g, t1, d, half, axis == 2)
        if axis == 3:
            a = g.nd("MatMul", [cur, s_pos])          # shift right (+d)
            b = g.nd("MatMul", [cur, s_neg])          # shift left  (-d)
        else:
            a = g.nd("MatMul", [s_pos, cur])          # shift down
            b = g.nd("MatMul", [s_neg, cur])          # shift up
        cur = g.nd("Max", [cur, a, b])
    return cur


def build(h):
    g = _G()
    half = g.f([1, 1], [0.5])
    halfc = g.f([1, 1, 1, 1], [0.5])

    # colour index plane & masks
    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    color = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)          # [1,1,30,30]
    inch = g.nd("Slice", ["input", g.i64([h]), g.i64([h + 1]), g.i64([1])])  # [1,1,30,30]
    known = g.nd("Sub", [realmask, inch])                                 # real & not hole
    maskh = g.f([1, CHANNELS, 1, 1], [0.0 if c == h else 1.0 for c in range(CHANNELS)])
    X = g.nd("Mul", ["input", maskh])                                     # channel h zeroed

    # data-dependent periods
    q_h = _period_scalar(g, known, color, 3, halfc)                       # horizontal
    q_v = _period_scalar(g, known, color, 2, halfc)                       # vertical

    rowidx = g.f([G, 1], list(range(G)))
    colidx = g.f([1, G], list(range(G)))

    Hf = _fill_axis(g, X, q_h, 3, rowidx, colidx, half)
    Vf = _fill_axis(g, Hf, q_v, 2, rowidx, colidx, half)

    masked = g.nd("Mul", [Vf, realmask])                                  # [1,10,30,30]
    if h == 0:
        cp = g.nd("ReduceSum", [masked], axes=[1], keepdims=1)            # content present
        cppos = g.nd("Cast", [g.nd("Greater", [cp, halfc])], to=F)
        bg = g.nd("Mul", [realmask, g.nd("Sub", [g.f([1, 1, 1, 1], [1.0]), cppos])])
        e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
        g.nd("Add", [masked, g.nd("Mul", [bg, e0])], "output")
    else:
        g.nd("Identity", [masked], "output")
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


def _shift(T, d, axis, direction):
    out = np.zeros_like(T)
    if d == 0:
        return T.copy()
    if d >= G:
        return out
    if axis == 1:                       # last spatial axis (cols)
        if direction > 0:
            out[..., :G - d] = T[..., d:]
        else:
            out[..., d:] = T[..., :G - d]
    else:                               # rows
        if direction > 0:
            out[..., :G - d, :] = T[..., d:, :]
        else:
            out[..., d:, :] = T[..., :G - d, :]
    return out


def _period(known, color, axis):
    for d in range(1, QMAX + 1):
        shK = _shift(known, d, axis, +1)
        shC = _shift(color, d, axis, +1)
        both = known * shK
        mism = (np.abs(color - shC) > 0.5).astype(np.float32)
        if float((both * mism).sum()) < 0.5:
            return d
    return 0


def _fill(Ften, dstar, axis):
    cur = Ften
    for k in range(5):
        d = dstar * (2 ** k)
        cur = np.maximum(np.maximum(cur, _shift(cur, d, axis, +1)),
                         _shift(cur, d, axis, -1))
    return cur


def _sim(a, h):
    X = _onehot(a)
    realmask = X.sum(0)                                  # [30,30]
    color = sum(c * X[c] for c in range(CHANNELS))
    known = realmask * (np.abs(color - h) > 0.5)
    Xc = X.copy(); Xc[h] = 0.0
    q_h = _period(known, color, 1)
    q_v = _period(known, color, 0)
    Hf = _fill(Xc, q_h, 1)
    Vf = _fill(Hf, q_v, 0)
    out = Vf * realmask[None]
    if h == 0:
        cp = out[1:].sum(0)
        out = out.copy()
        out[0] = realmask * (1.0 - (cp > 0.5))
    return out, realmask


def _sim_grid(a, h):
    out, realmask = _sim(a, h)
    sel = out > 0.5
    cnt = sel.sum(0)
    if (cnt[realmask > 0.5] != 1).any():
        return None
    if (cnt[realmask <= 0.5] != 0).any():
        return None
    pred = np.argmax(sel, axis=0)
    return pred


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


def _detect_h(prs):
    hs = set()
    for a, b in prs:
        if a.shape != b.shape:
            return None
        d = (a != b)
        if d.any():
            hs |= set(int(v) for v in a[d].tolist())
    if len(hs) != 1:
        return None
    return next(iter(hs))


def candidates(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []
    if any(a.shape != b.shape for a, b in allp):
        return []
    if all(np.array_equal(a, b) for a, b in allp):
        return []

    h = _detect_h(det)
    if h is None or not (0 <= h < CHANNELS):
        return []
    if any((b == h).any() for _, b in det):     # hole colour must not appear in outputs
        return []

    def ok(plist):
        for a, b in plist:
            pred = _sim_grid(a, h)
            if pred is None:
                return False
            H, W = b.shape
            if not np.array_equal(pred[:H, :W], b):
                return False
        return True

    if not ok(det) or not ok(allp):
        return []

    try:
        model = build(h)
        onnx.checker.check_model(model, full_check=True)
    except Exception:
        return []
    return [(f"dynperiod_h{h}", model)]
