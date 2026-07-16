"""BANDPUNCH family -- single-channel Hillis-Steele doubling rewrite.

Rule (task 202 "g44_bandpunch"): the grid is partitioned into monochrome STRIPES
(maximal runs of equal-coloured rows *or* columns).  Each stripe is solid colour
except for a few 0-cells ("holes").  Every hole is projected across the full
THICKNESS of its stripe: the whole column-within-band (horizontal stripes) or the
whole row-within-band (vertical stripes) is punched to 0.

Equivalently the 0-holes flood along the stripe-thickness axis, bounded by the
band walls (colour changes).  We implement that flood as a Hillis-Steele DOUBLING
propagation on a *single-channel* [1,1,30,30] hole field (offsets 1,2,4,8,16 cover
any distance <=31), gated by a per-row-boundary "openness" field that is the
product-along-the-span of adjacent-row same-band indicators.  Intermediates are
single-channel (3600 B) instead of the 10-channel (36000 B) unroll.

Orientation (rows-striped vs cols-striped) varies per example, so ONE static graph
computes the vertical punch V(x) and the transposed punch T(V(T(x))) and blends
them with a scalar flag s = 1 iff every row is monochrome (horizontal stripes).
Transpose (perm [0,1,3,2]) preserves the top-left origin, so the pad contract holds.

Weights are analytic; the harness grader is the final judge of exactness over
train+test+arc-gen.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

OFFS = [1, 2, 4, 8, 16]


# --------------------------------------------------------------------------- #
# numpy reference -- MUST mirror the ONNX arithmetic exactly                   #
# --------------------------------------------------------------------------- #
def _onehot(g):
    t = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    h, w = g.shape
    for c in range(CHANNELS):
        t[c, :h, :w] = (g == c)
    return t


def _su(a, d):          # shift up by d along rows (row r <- r+d)
    o = np.zeros_like(a)
    if d < a.shape[0]:
        o[:a.shape[0] - d] = a[d:]
    return o


def _sd(a, d):          # shift down by d along rows (row r <- r-d)
    o = np.zeros_like(a)
    o[d:] = a[:a.shape[0] - d]
    return o


def _c01(a):
    return np.clip(a, 0, 1)


def _vpunch_np(X):
    H = X[0].copy()
    rh = _c01(X[1:].sum(2))                     # [9,30]  rows-have-colour (1..9)
    rh0 = np.zeros((CHANNELS, HEIGHT), np.float32)
    rh0[1:] = rh                                # ch0 zeroed
    rhc = rh0[:, :, None]                       # [10,30,1]
    rh_up = np.zeros_like(rhc)
    rh_up[:, :HEIGHT - 1] = rhc[:, 1:]
    O1 = _c01((rhc * rh_up).sum(0))             # [30,1]
    O1 = np.broadcast_to(O1, (HEIGHT, WIDTH)).copy()
    real = _c01(X.sum(0))                        # [30,30]
    P = O1.copy()
    for d in OFFS:
        Hu, Hd = _su(H, d), _sd(H, d)
        cand_down = P * Hu
        cand_up = _sd(P, d) * Hd
        H = np.maximum(np.maximum(H, cand_down), cand_up)
        P = P * _su(P, d)
    Gout = H
    out = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    out[0] = Gout * real
    colcol = np.broadcast_to(rh0[:, :, None], (CHANNELS, HEIGHT, WIDTH))  # [10,30,30]
    out += colcol * real * (1 - Gout)           # channels 1..9 filled, ch0 stays Gout*real
    out[0] = Gout * real                          # (colcol[0]==0 so ch0 unaffected; explicit)
    return out


def _transp(X):
    return np.transpose(X, (0, 2, 1)).copy()


def _model_np(g):
    X = _onehot(g)
    rh = _c01(X[1:].sum(2))
    maxrow = rh.sum(0).max()
    s = 1 - _c01(maxrow - 1)
    V = _vpunch_np(X)
    Vt = _transp(_vpunch_np(_transp(X)))
    return s * V + (1 - s) * Vt


def _decode(R, h, w):
    b = (R > 0)
    if b[:, h:, :].any() or b[:, :, w:].any():
        return None
    out = np.zeros((h, w), int)
    for i in range(h):
        for j in range(w):
            ch = np.where(b[:, i, j])[0]
            if len(ch) != 1:
                return None
            out[i, j] = ch[0]
    return out


# --------------------------------------------------------------------------- #
# ONNX construction                                                           #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def initf(self, dims, vals):
        n = self.nm("w")
        self.inits.append(oh.make_tensor(n, DATA_TYPE, list(dims),
                                         np.asarray(vals, np.float32).ravel().tolist()))
        return n

    def node(self, op, ins, out=None, **kw):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **kw))
        return out

    def clip(self, s, out=None):
        return self.node("Clip", [s], out, min=0.0, max=1.0)

    def conv(self, src, W, pads, out=None):
        cout, cin, kh, kw = W.shape
        wt = self.initf([cout, cin, kh, kw], W)
        return self.node("Conv", [src, wt], out, kernel_shape=[kh, kw], pads=list(pads))

    def shift_up(self, src, d, cin=1, out=None):     # row r <- r+d
        W = np.zeros((cin, cin, d + 1, 1), np.float32)
        for k in range(cin):
            W[k, k, d, 0] = 1.0
        return self.conv(src, W, [0, 0, d, 0], out)

    def shift_down(self, src, d, cin=1, out=None):    # row r <- r-d
        W = np.zeros((cin, cin, d + 1, 1), np.float32)
        for k in range(cin):
            W[k, k, 0, 0] = 1.0
        return self.conv(src, W, [d, 0, 0, 0], out)


def _vpunch(g, X):
    # real mask [1,1,30,30]
    real = g.node("ReduceSum", [X], axes=[1], keepdims=1)
    # H = channel 0  [1,1,30,30]
    Wh = np.zeros((1, CHANNELS, 1, 1), np.float32); Wh[0, 0, 0, 0] = 1.0
    H = g.conv(X, Wh, [0, 0, 0, 0])
    # rowhas per channel, ch0 zeroed  [1,10,30,1]
    rowsum = g.node("ReduceSum", [X], axes=[3], keepdims=1)
    rh = g.clip(rowsum)
    Wz = np.eye(CHANNELS, dtype=np.float32); Wz[0, 0] = 0.0
    RH0 = g.conv(rh, Wz.reshape(CHANNELS, CHANNELS, 1, 1), [0, 0, 0, 0])  # [1,10,30,1]
    # openness O1 [1,1,30,1]
    RHup = g.shift_up(RH0, 1, cin=CHANNELS)
    prod = g.node("Mul", [RH0, RHup])
    O1s = g.node("ReduceSum", [prod], axes=[1], keepdims=1)
    O1 = g.clip(O1s)                                  # [1,1,30,1]
    P = O1
    for d in OFFS:
        Hu = g.shift_up(H, d)
        Hd = g.shift_down(H, d)
        cand_down = g.node("Mul", [P, Hu])
        Pdn = g.shift_down(P, d)
        cand_up = g.node("Mul", [Pdn, Hd])
        H = g.node("Max", [H, cand_down, cand_up])
        Pup = g.shift_up(P, d)
        P = g.node("Mul", [P, Pup])
    Gout = H
    # recolor
    out0 = g.node("Mul", [Gout, real])                # [1,1,30,30]
    one = g.initf([1], [1.0])
    notG = g.node("Sub", [one, Gout])                 # 1-Gout broadcast [1,1,30,30]
    colored = g.node("Mul", [RH0, real])              # [1,10,30,30] ch0=0
    colored2 = g.node("Mul", [colored, notG])         # [1,10,30,30]
    Wc0 = np.zeros((CHANNELS, 1, 1, 1), np.float32); Wc0[0, 0, 0, 0] = 1.0
    ch0t = g.conv(out0, Wc0, [0, 0, 0, 0])            # [1,10,30,30] ch0=out0
    return g.node("Add", [colored2, ch0t])            # [1,10,30,30]


def build():
    g = _G()
    X = "input"
    # s = 1 iff every row monochrome (<=1 colour) : maxrow = max_r sum_k rowhas
    rowsum = g.node("ReduceSum", [X], axes=[3], keepdims=1)
    rh = g.clip(rowsum)
    Wz = np.eye(CHANNELS, dtype=np.float32); Wz[0, 0] = 0.0
    RH0 = g.conv(rh, Wz.reshape(CHANNELS, CHANNELS, 1, 1), [0, 0, 0, 0])
    perrow = g.node("ReduceSum", [RH0], axes=[1], keepdims=1)   # [1,1,30,1]
    maxrow = g.node("ReduceMax", [perrow], keepdims=1)          # [1,1,1,1]
    one = g.initf([1], [1.0])
    excess = g.clip(g.node("Sub", [maxrow, one]))
    s = g.node("Sub", [one, excess])                            # [1,1,1,1]
    oneS = g.node("Sub", [one, s])
    V = _vpunch(g, X)
    Xt = g.node("Transpose", [X], perm=[0, 1, 3, 2])
    Vt0 = _vpunch(g, Xt)
    Vt = g.node("Transpose", [Vt0], perm=[0, 1, 3, 2])
    Va = g.node("Mul", [V, s])
    Vb = g.node("Mul", [Vt, oneS])
    g.node("Add", [Va, Vb], "output")
    xi = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    yo = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "bandpunch", [xi], [yo], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# detection / entry point                                                     #
# --------------------------------------------------------------------------- #
def _pairs(ex, splits):
    out = []
    for sp in splits:
        for e in ex.get(sp, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def candidates(ex):
    tt = _pairs(ex, ("train", "test"))
    if len(tt) < 2:
        return []
    # same shape in/out
    if not all(a.shape == b.shape for a, b in tt):
        return []
    # the transform must actually punch something (0 introduced somewhere)
    if not any(((a != 0) & (b == 0)).any() for a, b in tt):
        return []
    for a, b in tt:
        dec = _decode(_model_np(a), *b.shape)
        if dec is None or not np.array_equal(dec, b):
            return []
    try:
        return [("bandpunch_double", build())]
    except Exception:
        return []
