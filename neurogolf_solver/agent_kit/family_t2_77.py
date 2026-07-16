"""family_t2_77 — cheaper, WORKING rebuild of task 77 (36fdfd69).

Rule (from the generator): the grid holds scattered `static`-colour noise plus a
few solid rectangles.  Inside each rectangle the empty cells were painted red(2)
and the pre-existing static cells become yellow(4) in the output; every other
static cell is left alone.  So: recolour a static cell to yellow iff it lies
inside a red rectangle.

Detection = geodesic horizontal reconstruction of the red mask, gated by a
vertical-support mask, plus two 1-gap "closings" that recolour fully-static
middle rows / columns of a rectangle (rectangles have t<=3, w<=7):

    R    = red mask (uint8, cropped to the 20x21 max grid)
    V    = MaxPool[5,1](R)                       # vertical support (t<=3)
    V2   = V | (V<<1 & V>>1)                      # close 1-col gaps  (QLinearConv)
    F0   = R
    Fk+1 = Where(V2, MaxPool[1,5](Fk), 0)         # 3 geodesic horizontal steps
    S    = F3 | (F3^up & F3^down)                 # close 1-row gaps  (QLinearConv)
    fill = S & ~R                                 # interior static cells
    out  = Where(fill, yellow, input)

The two closings are each a QLinearConv with weight [1,3,1] and output scale
3.5, so round((l + 3*m + r) / 3.5) == (m | (l & r)) stays a clean {0,1} map.

The incumbent (out_blend12/onnx/task077.onnx) uses uint8 `Min`, which ORT 1.23.2
does NOT implement under ORT_DISABLE_ALL, so it fails to load (0 pts locally).
Here every op runs on a dtype ORT supports: MaxPool/uint8, Where/uint8,
Greater/uint8, Cast, Slice(static), Pad, Where(bool,f,f).  Intersections use
`Where(sup_bool, x, 0)` instead of the unsupported uint8 `Min`.  All flood
tensors stay uint8 (1 byte) and the terminal Where writes straight to `output`.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

from ng_utils_shim import GRID_SHAPE

F = TP.FLOAT
I64 = TP.INT64
U8 = TP.UINT8
BOOL = TP.BOOL

CROP_H, CROP_W = 20, 21


class G:
    def __init__(self):
        self.nodes, self.inits, self._k = [], [], 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def c(self, arr, dt):
        a = np.asarray(arr)
        n = self.nm("c")
        if dt == I64:
            vals = a.astype(np.int64).ravel().tolist()
        elif dt == U8:
            vals = a.astype(np.uint8).ravel().tolist()
        else:
            vals = a.astype(np.float64).ravel().tolist()
        self.inits.append(oh.make_tensor(n, dt, list(a.shape) if a.shape else [], vals))
        return n

    def nd(self, op, ins, out=None, **kw):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **kw))
        return out

    def model(self, name, opset=13):
        x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
        y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
        used = {i for n in self.nodes for i in n.input}
        inits = [t for t in self.inits if t.name in used]
        m = oh.make_model(oh.make_graph(self.nodes, name, [x], [y], inits),
                          ir_version=10, opset_imports=[oh.make_opsetid("", opset)])
        onnx.checker.check_model(m, full_check=True)
        return m


# --------------------------------------------------------------------------- #
# numpy reference (gates emission)                                             #
# --------------------------------------------------------------------------- #
def _maxpool_h(F, rad):
    H, W = F.shape
    out = F.copy()
    for k in range(1, rad + 1):
        out[:, : W - k] |= F[:, k:]
        out[:, k:] |= F[:, : W - k]
    return out


def _maxpool_v(F, rad):
    H, W = F.shape
    out = F.copy()
    for k in range(1, rad + 1):
        out[: H - k, :] |= F[k:, :]
        out[k:, :] |= F[: H - k, :]
    return out


def _close_h(V):
    l = np.zeros_like(V); r = np.zeros_like(V)
    l[:, 1:] = V[:, :-1]; r[:, :-1] = V[:, 1:]
    return V | (l & r)


def _close_v(F):
    u = np.zeros_like(F); d = np.zeros_like(F)
    u[1:, :] = F[:-1, :]; d[:-1, :] = F[1:, :]
    return F | (u & d)


def _ref(a):
    a = np.asarray(a, int)
    H, W = a.shape
    if H > CROP_H or W > CROP_W:
        return None
    red = a == 2
    V2 = _close_h(_maxpool_v(red, 2))
    Fm = red.copy()
    for _ in range(3):
        Fm = _maxpool_h(Fm, 2) & V2
    S = _close_v(Fm)
    fill = S & (~red)
    out = a.copy()
    out[fill] = 4
    return out


# --------------------------------------------------------------------------- #
# onnx builder                                                                 #
# --------------------------------------------------------------------------- #
def _build():
    g = G()
    ss = g.c([2, 0, 0], I64)
    se = g.c([3, CROP_H, CROP_W], I64)
    sa = g.c([1, 2, 3], I64)
    xf = g.nd("Slice", ["input", ss, se, sa])            # [1,1,20,21] f32 (red plane)
    R = g.nd("Cast", [xf], to=U8)                        # red mask 0/1

    # QLinearConv shared quant params: x_scale=w_scale=1, zp=0, y_scale=3.5.
    s1 = g.c(1.0, F)
    s35 = g.c(3.5, F)
    zpu = g.c(0, U8)
    wh = g.c(np.array([1, 3, 1]).reshape(1, 1, 1, 3), U8)   # horizontal [1,3,1]
    wv = g.c(np.array([1, 3, 1]).reshape(1, 1, 3, 1), U8)   # vertical   [1,3,1]

    V = g.nd("MaxPool", [R], kernel_shape=[5, 1], pads=[2, 0, 2, 0])   # vertical support
    V2 = g.nd("QLinearConv", [V, s1, zpu, wh, s1, zpu, s35, zpu], pads=[0, 1, 0, 1])
    V2b = g.nd("Cast", [V2], to=BOOL)
    zero = g.c(0, U8)

    Fm = R
    for _ in range(3):
        P = g.nd("MaxPool", [Fm], kernel_shape=[1, 5], pads=[0, 2, 0, 2])
        Fm = g.nd("Where", [V2b, P, zero])               # intersect with support

    S = g.nd("QLinearConv", [Fm, s1, zpu, wv, s1, zpu, s35, zpu], pads=[1, 0, 1, 0])
    fill = g.nd("Greater", [S, R])                       # S & ~red  -> bool
    pp = g.c([0, 0, 0, 0, 0, 0, 30 - CROP_H, 30 - CROP_W], I64)
    fillfull = g.nd("Pad", [fill, pp], mode="constant")  # bool [1,1,30,30]

    ch4 = g.c(np.eye(10)[4].reshape(1, 10, 1, 1), F)     # yellow one-hot
    g.nd("Where", [fillfull, ch4, "input"], "output")
    return g.model("t2_77")


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


def candidates(examples):
    prs = _pairs(examples)
    if not prs:
        return
    for a, b in prs:
        o = _ref(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return
    try:
        yield ("t2_77", _build())
    except Exception:
        return
