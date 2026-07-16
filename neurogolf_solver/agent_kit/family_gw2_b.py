"""family_gw2_b — from-scratch minimal-rule rebuilds (golf ranks 15..32).

Only task110 (484b58aa) yielded a rule that is both exact/general AND cheaply
expressible as a hand-built opset-10 ONNX; the other 16 targets are global
structural transforms (connected-component room-fill, diagonal ray-casting,
fractal self-tiling, multi-object assembly, generic pattern repair) that cannot
be beaten cheaply with a static graph, so they are skipped (no regression).

task110 (verify_484b58aa): every grid is a 29x29 doubly-periodic colour tiling
with a few background(0) "holes".  The output is the hole-free tiling.  Because
all non-hole cells in a period-coset share one colour, the reconstruction is
    out(i,j) = max over the (Pv,Ph)-lattice coset of the colour grid,
where Pv/Ph are the minimal vertical/horizontal periods.  ONNX:
  * one Conv builds the 14 vertical shifts of the colour grid; comparing to the
    grid gives, per candidate period p, whether any two non-hole cells p apart
    disagree -> the minimal disagreement-free p (else "no period").  Same on W.
  * max-coset fill: doubling max-propagation with runtime shift MatMuls at
    steps Pv*2^t / Ph*2^t (covers any period over a 29-cell axis in 5 steps).
Fully fp16 (all live values are colours<=9, 0/1 masks, counts<100, offsets<2048).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
F16 = onnx.TensorProto.FLOAT16
BOOL = onnx.TensorProto.BOOL

PMAX = 10          # max candidate period (observed real periods <= 9)
NSTEP = 5          # doubling steps: covers offsets up to (2^5-1)*P >= 29


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


def _model(g, name, out_dtype=F):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", out_dtype, GRID_SHAPE)
    used = {i for n in g.nodes for i in n.input}
    inits = [t for t in g.inits if t.name in used]
    m = oh.make_model(oh.make_graph(g.nodes, name, [x], [y], inits),
                      ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    onnx.checker.check_model(m, full_check=True)
    return m


def _detect_period(g, C, axis):
    """Return scalar [1,1] fp16 = minimal disagreement-free period in 1..PMAX,
    or 100 (== "no period") if none. axis 0 = vertical, 1 = horizontal."""
    # Build the PMAX shifted copies of C via a single Conv.
    if axis == 0:
        # shift up: channel oc = C[i+oc+1, j]; pad bottom PMAX, kernel (PMAX+1)x1
        Cp = g.nd("Pad", [C], mode="constant", value=0.0,
                  pads=[0, 0, 0, 0, 0, 0, PMAX, 0])          # [1,1,30+PMAX,30]
        W = np.zeros((PMAX, 1, PMAX + 1, 1), np.float32)
        for p in range(1, PMAX + 1):
            W[p - 1, 0, p, 0] = 1.0
        w = g.c(F16, [PMAX, 1, PMAX + 1, 1], W)
        Csh = g.nd("Conv", [Cp, w], kernel_shape=[PMAX + 1, 1])   # [1,PMAX,30,30]
    else:
        Cp = g.nd("Pad", [C], mode="constant", value=0.0,
                  pads=[0, 0, 0, 0, 0, 0, 0, PMAX])          # [1,1,30,30+PMAX]
        W = np.zeros((PMAX, 1, 1, PMAX + 1), np.float32)
        for p in range(1, PMAX + 1):
            W[p - 1, 0, 0, p] = 1.0
        w = g.c(F16, [PMAX, 1, 1, PMAX + 1], W)
        Csh = g.nd("Conv", [Cp, w], kernel_shape=[1, PMAX + 1])   # [1,PMAX,30,30]

    half = g.c(F16, [1, 1, 1, 1], [0.5])
    cnz = g.nd("Greater", [C, half])                      # [1,1,30,30] bool
    snz = g.nd("Greater", [Csh, half])                    # [1,PMAX,30,30] bool
    both = g.nd("And", [cnz, snz])                        # [1,PMAX,30,30]
    ad = g.nd("Abs", [g.nd("Sub", [C, Csh])])
    neq = g.nd("Greater", [ad, half])                     # colours differ (>=1 apart)
    dis = g.nd("And", [both, neq])                        # disagreement mask
    disf = g.nd("Cast", [dis], to=F16)
    anyd = g.nd("ReduceMax", [disf], axes=[2, 3], keepdims=1)  # [1,PMAX,1,1]
    valid = g.nd("Cast", [g.nd("Less", [anyd, half])], to=F16)  # 1 if period ok
    score_w = g.c(F16, [1, PMAX, 1, 1], [100 - p for p in range(1, PMAX + 1)])
    score = g.nd("Mul", [valid, score_w])                 # [1,PMAX,1,1]
    mx = g.nd("ReduceMax", [score], axes=[1], keepdims=1)  # [1,1,1,1]
    c100 = g.c(F16, [1, 1, 1, 1], [100.0])
    P = g.nd("Sub", [c100, mx])                           # [1,1,1,1] period (or 100)
    return g.nd("Reshape", [P, g.c(onnx.TensorProto.INT64, [2], [1, 1])])  # [1,1]


def _kmi(g):
    """constant [1,1,30,30] with M[i,k] = k - i."""
    M = (np.arange(30)[None, :] - np.arange(30)[:, None]).astype(np.float32)
    return g.c(F16, [1, 1, 30, 30], M.reshape(1, 1, 30, 30))


def _shift_pair(g, kmi, base, t):
    """(pos,neg) [1,1,30,30]: pos[i,k]=1 iff k==i+s ; neg = pos^T (k==i-s)."""
    s = g.nd("Mul", [base, g.c(F16, [1, 1], [float(2 ** t)])])   # [1,1]
    half = g.c(F16, [1, 1, 1, 1], [0.5])
    d = g.nd("Abs", [g.nd("Sub", [kmi, s])])
    pos = g.nd("Cast", [g.nd("Less", [d, half])], to=F16)        # k-i == s
    neg = g.nd("Transpose", [pos], perm=[0, 1, 3, 2])            # k-i == -s
    return pos, neg


def _maxprop(g, cur, kmi, base, axis):
    for t in range(NSTEP):
        pos, neg = _shift_pair(g, kmi, base, t)
        if axis == 0:
            a = g.nd("MatMul", [pos, cur])      # cur[i+s,j]
            b = g.nd("MatMul", [neg, cur])      # cur[i-s,j]
        else:
            a = g.nd("MatMul", [cur, neg])      # cur[i,j+s]
            b = g.nd("MatMul", [cur, pos])      # cur[i,j-s]
        cur = g.nd("Max", [cur, g.nd("Max", [a, b])])
    return cur


def build_110():
    g = _G()
    inp = g.nd("Cast", ["input"], to=F16)
    idx10 = g.c(F16, [1, 10, 1, 1], list(range(10)))
    C = g.nd("ReduceSum", [g.nd("Mul", [inp, idx10])], axes=[1], keepdims=1)  # [1,1,30,30]

    Pv = _detect_period(g, C, 0)
    Ph = _detect_period(g, C, 1)

    kmi = _kmi(g)
    cur = _maxprop(g, C, kmi, Pv, 0)
    cur = _maxprop(g, cur, kmi, Ph, 1)

    # mask to the 29x29 grid region (kill padding row/col 29)
    gm = np.zeros((1, 1, 30, 30), np.float32)
    gm[0, 0, :29, :29] = 1.0
    cur = g.nd("Mul", [cur, g.c(F16, [1, 1, 30, 30], gm)])

    # colour grid -> one-hot, channel 0 forced to zero
    half10 = g.c(F16, [1, 1, 1, 1], [0.5])
    adc = g.nd("Abs", [g.nd("Sub", [cur, idx10])])
    onehot = g.nd("Cast", [g.nd("Less", [adc, half10])], to=F16)   # [1,10,30,30]
    chmask = g.c(F16, [1, 10, 1, 1], [0.0] + [1.0] * 9)
    g.nd("Mul", [onehot, chmask], "output")
    return _model(g, "gw2b_110", out_dtype=F16)


# --------------------------------------------------------------------------- #
# numpy mirror (must match the ONNX exactly)                                   #
# --------------------------------------------------------------------------- #
def _minperiod(C, axis):
    dim = C.shape[axis]
    for p in range(1, PMAX + 1):
        if p >= dim:
            break
        a = (C[:-p, :], C[p:, :]) if axis == 0 else (C[:, :-p], C[:, p:])
        both = (a[0] != 0) & (a[1] != 0)
        if not np.any(both & (a[0] != a[1])):
            return p
    return None


def _maxprop_np(C, P, axis):
    H, W = C.shape
    cur = C.copy()
    s = P
    for _ in range(NSTEP):
        up = np.zeros_like(cur)
        dn = np.zeros_like(cur)
        if axis == 0:
            if s < H:
                up[:H - s, :] = cur[s:, :]
                dn[s:, :] = cur[:H - s, :]
        else:
            if s < W:
                up[:, :W - s] = cur[:, s:]
                dn[:, s:] = cur[:, :W - s]
        cur = np.maximum(cur, np.maximum(up, dn))
        s *= 2
    return cur


def _mirror_110(a):
    a = np.asarray(a, int)
    if a.ndim != 2 or a.shape != (29, 29):
        return None
    if (a == 0).all() or a.max() > 9:
        return None
    Pv = _minperiod(a, 0)
    Ph = _minperiod(a, 1)
    cur = a.copy()
    if Pv:
        cur = _maxprop_np(cur, Pv, 0)
    if Ph:
        cur = _maxprop_np(cur, Ph, 1)
    return cur


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def _pairs(examples):
    out = []
    for s in ("train", "test", "arc-gen"):
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
    if _matches(prs, _mirror_110):
        try:
            yield ("gw2b_484b58aa", build_110())
        except Exception:
            pass
