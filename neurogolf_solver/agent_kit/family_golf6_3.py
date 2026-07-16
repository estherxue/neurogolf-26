"""family_golf6_3: cheaper EXACT golf re-implementations for slice [3::5] targets.

The integrator auto-picks the cheapest exact solver per task, so each family
below re-derives a *known* rule with fewer / smaller intermediate tensors than
the incumbent solution (cost = params + intermediate-memory bytes; the [1,10,30,30]
input and output tensors are FREE).

Implemented
-----------
* spray4 (task 199): a lone non-bg marker at (r,c) of colour `col` sprays colour 4
  over every real cell in rows <= r whose column has the same parity as c, and
  re-stamps the marker colour one cell below it, at (r+1, c).  The incumbent
  solution computes the non-bg presence map by slicing the 9 colour channels into
  a [1,9,30,30] tensor (32400 B) and reducing.  Because the one-hot input has
  exactly one channel set per real cell, presence == ReduceSum(allchannels) -
  channel0, which only ever materialises [1,1,30,30] tensors.  That removes the
  single largest intermediate of the graph.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64


# --------------------------------------------------------------------------- #
# tiny graph builder
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def iconst(self, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)], list(vals)))
        return nm

    def fconst(self, vals, shape):
        nm = self.name("f")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(shape),
                                         [float(v) for v in vals]))
        return nm

    def scalar(self, v):
        return self.fconst([v], [1])

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(inits))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _slice(g, src, starts, ends, axes, steps=None):
    ins = [src, g.iconst(starts), g.iconst(ends), g.iconst(axes)]
    if steps is not None:
        ins.append(g.iconst(steps))
    return g.node("Slice", ins)


def _onehot_channel(g, ch):
    vals = [1.0 if i == ch else 0.0 for i in range(CHANNELS)]
    return g.fconst(vals, [1, CHANNELS, 1, 1])


def _chan_mask(g, channels):
    vals = [1.0 if i in channels else 0.0 for i in range(CHANNELS)]
    return g.fconst(vals, [1, CHANNELS, 1, 1])


def _argmax_f(g, x, axis):
    a = g.node("ArgMax", [x], axis=axis, keepdims=1)
    return g.node("Cast", [a], to=DATA_TYPE)


def _pairs(ex):
    out = []
    for s in ("train", "test"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


# --------------------------------------------------------------------------- #
# 199  spray4
# --------------------------------------------------------------------------- #
def _ref_199(a):
    nz = np.argwhere(a > 0)
    if len(nz) != 1:
        return None
    r, c = nz[0]
    col = a[r, c]
    H, W = a.shape
    o = np.zeros_like(a)
    for i in range(0, r + 1):
        for j in range(W):
            if (j % 2) == (c % 2):
                o[i, j] = 4
    if r + 1 < H:
        o[r + 1, c] = col
    return o


def _build_199():
    g = _G()
    # ---- cheap presence map: ReduceSum(all channels) - channel0 --------------
    real = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)      # [1,1,30,30]
    ch0 = _slice(g, "input", [0], [1], [1])                          # [1,1,30,30]
    nz = g.node("Sub", [real, ch0])                                  # presence

    rowhas = g.node("ReduceMax", [nz], axes=[3], keepdims=1)         # [1,1,30,1]
    colhas = g.node("ReduceMax", [nz], axes=[2], keepdims=1)         # [1,1,1,30]
    r = _argmax_f(g, rowhas, 2)                                      # [1,1,1,1]
    c = _argmax_f(g, colhas, 3)                                      # [1,1,1,1]
    ridx = g.fconst(list(range(HEIGHT)), [1, 1, HEIGHT, 1])
    cidx = g.fconst(list(range(WIDTH)), [1, 1, 1, WIDTH])

    # region4: rows<=r, same column parity, real cell
    rowmask = g.node("Cast", [g.node("Less", [ridx, g.node("Add", [r, g.scalar(0.5)])])],
                     to=DATA_TYPE)                                   # i<=r -> [1,1,30,1]
    par = g.node("Mod", [g.node("Add", [cidx, c]), g.scalar(2.0)], fmod=1)  # [1,1,1,30]
    colmask = g.node("Cast", [g.node("Less", [par, g.scalar(0.5)])], to=DATA_TYPE)
    region = g.node("Mul", [g.node("Mul", [rowmask, colmask]), real])
    regionb = g.node("Greater", [region, g.scalar(0.5)])

    # marker position (r+1, c)
    posr = g.node("Cast", [g.node("Less",
                  [g.node("Abs", [g.node("Sub", [ridx, g.node("Add", [r, g.scalar(1.0)])])]),
                   g.scalar(0.5)])], to=DATA_TYPE)                   # [1,1,30,1]
    posc = g.node("Cast", [g.node("Less",
                  [g.node("Abs", [g.node("Sub", [cidx, c])]), g.scalar(0.5)])], to=DATA_TYPE)
    pos = g.node("Mul", [g.node("Mul", [posr, posc]), real])
    posb = g.node("Greater", [pos, g.scalar(0.5)])

    # marker colour one-hot [1,10,1,1]
    pres = g.node("ReduceMax", ["input"], axes=[2, 3], keepdims=1)   # [1,10,1,1]
    mcol = g.node("Mul", [pres, _chan_mask(g, set(range(1, CHANNELS)))])

    inner = g.node("Where", [posb, mcol, "input"])                  # marker placed
    g.node("Where", [regionb, _onehot_channel(g, 4), inner], "output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# 125  hollow-box frame + hole-fill
# --------------------------------------------------------------------------- #
# colours: input is {6 (box), 8 (bg)}; output adds {3 (ring drawn one cell outside
# each box, = 8-neighbour dilation of the box intersect bg) and 4 (interior holes
# of the box, = bg cells with a box pixel in all four axis directions)}.
_sL30 = np.tril(np.ones((30, 30), np.float32), -1)   # [r,i]=1 if i<r  (rows above)
_sU30 = np.triu(np.ones((30, 30), np.float32), 1)    # [r,i]=1 if i>r  (rows below)


def _ref_125(a):
    if not set(np.unique(a).tolist()).issubset({6, 8}):
        return None
    H, W = a.shape
    blob = np.zeros((30, 30), np.float32); blob[:H, :W] = (a == 6)
    bg = np.zeros((30, 30), np.float32);   bg[:H, :W] = (a == 8)
    right = (blob @ _sU30 > 0.5); left = (blob @ _sL30 > 0.5)
    up = (_sL30 @ blob > 0.5);    down = (_sU30 @ blob > 0.5)
    fill = bg * right * left * up * down
    # 8-neighbour dilation via 3x3 box sum
    pad = np.pad(blob, 1)
    nb = sum(pad[i:i + 30, j:j + 30] for i in range(3) for j in range(3))
    adj = (nb > 0.5).astype(np.float32)
    border = bg * adj * (1.0 - fill)
    rem = bg - fill - border
    val = blob * 6 + fill * 4 + border * 3 + rem * 8
    return val[:H, :W].astype(int)


def _mat(g, arr):
    arr = np.asarray(arr, np.float32)
    return g.fconst(arr.ravel().tolist(), list(arr.shape))


def _build_125():
    g = _G()
    blob = _slice(g, "input", [6], [7], [1])      # box channel  (colour 6)
    bg = _slice(g, "input", [8], [9], [1])        # bg channel   (colour 8)
    sL = _mat(g, _sL30); sU = _mat(g, _sU30)
    half = g.scalar(0.5); one = g.scalar(1.0)

    def gt(x):
        return g.node("Cast", [g.node("Greater", [x, half])], to=DATA_TYPE)

    right = gt(g.node("MatMul", [blob, sU]))
    left = gt(g.node("MatMul", [blob, sL]))
    up = gt(g.node("MatMul", [sL, blob]))
    down = gt(g.node("MatMul", [sU, blob]))
    fill = g.node("Mul", [g.node("Mul", [bg, right]),
                          g.node("Mul", [left, g.node("Mul", [up, down])])])
    # 8-neighbour dilation via a single 3x3 all-ones depthwise conv
    k3 = _mat(g, np.ones((1, 1, 3, 3), np.float32))
    adj = gt(g.node("Conv", [blob, k3], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    border = g.node("Mul", [g.node("Mul", [bg, adj]), g.node("Sub", [one, fill])])
    rem = g.node("Sub", [g.node("Sub", [bg, fill]), border])

    masks = g.node("Concat", [blob, fill, border, rem], axis=1)   # [1,4,30,30]
    # scatter mask-channel -> colour-channel with a free Conv (writes "output")
    W = np.zeros((CHANNELS, 4, 1, 1), np.float32)
    W[6, 0] = 1.0; W[4, 1] = 1.0; W[3, 2] = 1.0; W[8, 3] = 1.0
    g.node("Conv", [masks, _mat(g, W)], "output", kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# 181  reflect colour-8 across an axis chosen from the colour-4 distribution
# --------------------------------------------------------------------------- #
def _ref_181(a):
    if not set(np.unique(a).tolist()).issubset({0, 4, 8}):
        return None
    H, W = a.shape
    X = np.zeros((10, 30, 30), np.float32)
    for r in range(H):
        for c in range(W):
            X[a[r, c], r, c] = 1
    c5 = np.arange(30, dtype=np.float32)
    t9 = X.sum(0); t13 = X[8]; t17 = X[4]
    t18 = t13.max(axis=0); t19 = 1 - t18
    t23 = (c5 * t18 + 1000.0 * t19).min()
    t25 = (c5 * t18).max()
    t26 = t17.sum(axis=0); t28 = (t26 > 0.5).astype(np.float32); t29 = 1 - t28
    t33 = (c5 * t28 + 1000.0 * t29).min(); t35 = (c5 * t28).max()
    t41 = (t26 * (np.abs(c5 - t33) < 0.5)).sum()
    t47 = (t26 * (np.abs(c5 - t35) < 0.5)).sum()
    t50 = float((t41 - t47) > 0.5)
    t53 = (2 * t23 - 1) - c5; t56 = (2 * t25 + 1) - c5
    t60 = t50 * t53 + (1 - t50) * t56
    t65 = (np.abs(t60[:, None] - c5[None, :]) < 0.5).astype(np.float32)
    t66 = t13 @ t65; t69 = ((t13 + t66) > 0.5).astype(np.float32)
    t70 = t69 * t9
    val = t17 * 4 + t70 * 8
    return val[:H, :W].astype(int)


def _build_181():
    g = _G()
    c1 = g.scalar(0.5); c2 = g.scalar(1.0); c3 = g.scalar(2.0); c4 = g.scalar(1000.0)
    c5 = g.fconst(list(range(WIDTH)), [1, 1, 1, WIDTH])

    def cast(x):
        return g.node("Cast", [x], to=DATA_TYPE)

    t9 = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)
    t13 = _slice(g, "input", [8], [9], [1])
    t17 = _slice(g, "input", [4], [5], [1])
    t18 = g.node("ReduceMax", [t13], axes=[2], keepdims=1)
    t19 = g.node("Sub", [c2, t18])
    t20 = g.node("Mul", [c5, t18]); t21 = g.node("Mul", [c4, t19])
    t22 = g.node("Add", [t20, t21])
    t23 = g.node("ReduceMin", [t22], axes=[3], keepdims=1)
    t24 = g.node("Mul", [c5, t18])
    t25 = g.node("ReduceMax", [t24], axes=[3], keepdims=1)
    t26 = g.node("ReduceSum", [t17], axes=[2], keepdims=1)
    t28 = cast(g.node("Greater", [t26, c1]))
    t29 = g.node("Sub", [c2, t28])
    t30 = g.node("Mul", [c5, t28]); t31 = g.node("Mul", [c4, t29])
    t32 = g.node("Add", [t30, t31])
    t33 = g.node("ReduceMin", [t32], axes=[3], keepdims=1)
    t34 = g.node("Mul", [c5, t28])
    t35 = g.node("ReduceMax", [t34], axes=[3], keepdims=1)
    t39 = cast(g.node("Less", [g.node("Abs", [g.node("Sub", [c5, t33])]), c1]))
    t41 = g.node("ReduceSum", [g.node("Mul", [t26, t39])], axes=[3], keepdims=1)
    t45 = cast(g.node("Less", [g.node("Abs", [g.node("Sub", [c5, t35])]), c1]))
    t47 = g.node("ReduceSum", [g.node("Mul", [t26, t45])], axes=[3], keepdims=1)
    t50 = cast(g.node("Greater", [g.node("Sub", [t41, t47]), c1]))
    t52 = g.node("Sub", [g.node("Mul", [c3, t23]), c2])
    t53 = g.node("Sub", [t52, c5])
    t55 = g.node("Add", [g.node("Mul", [c3, t25]), c2])
    t56 = g.node("Sub", [t55, c5])
    t57 = g.node("Mul", [t50, t53])
    t59 = g.node("Mul", [g.node("Sub", [c2, t50]), t56])
    t60 = g.node("Add", [t57, t59])
    t61 = g.node("Transpose", [t60], perm=[0, 1, 3, 2])
    t65 = cast(g.node("Less", [g.node("Abs", [g.node("Sub", [t61, c5])]), c1]))
    t66 = g.node("MatMul", [t13, t65])
    t69 = cast(g.node("Greater", [g.node("Add", [t13, t66]), c1]))
    t70 = g.node("Mul", [t69, t9])
    t71 = g.node("Sub", [t9, t70])
    t72 = g.node("Sub", [t71, t17])         # -> colour 0
    masks = g.node("Concat", [t72, t17, t70], axis=1)        # [1,3,30,30]
    W = np.zeros((CHANNELS, 3, 1, 1), np.float32)
    W[0, 0] = 1.0; W[4, 1] = 1.0; W[8, 2] = 1.0
    g.node("Conv", [masks, _mat(g, W)], "output", kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# 188  halve: keep the fundamental tile of a 2x-periodic grid (left or top half)
# --------------------------------------------------------------------------- #
def _ref_188(a):
    H, W = a.shape
    if H > 30 or W > 30:
        return None
    X = np.zeros((10, 30, 30), np.float32)
    for r in range(H):
        for c in range(W):
            X[a[r, c], r, c] = 1
    idx = np.arange(30, dtype=np.float32); cvec = np.arange(10, dtype=np.float32)
    t7 = X.sum(0)
    t12 = t7.max(axis=0).sum() * 0.5
    t13 = t7.max(axis=1).sum() * 0.5
    t15 = (X * cvec[:, None, None]).sum(0)
    t20 = (np.abs((idx[None, :] - idx[:, None]) + t12) < 0.5).astype(np.float32)
    t21 = t15 @ t20
    t23 = (idx < t12).astype(np.float32)
    t29 = (np.abs(t15 - t21) > 0.5).astype(np.float32) * t23[None, :] * t7
    t40 = float(abs(t12 - np.trunc(t12)) < 0.25) * float(t29.sum() < 0.5)
    t43 = (idx < t13).astype(np.float32)
    combined = t40 * t23[None, :] + (1 - t40) * t43[:, None]
    keep = combined[:H, :W] > 0.5
    if keep.sum() == 0:
        return None
    mr = np.where(keep.any(1))[0].max(); mc = np.where(keep.any(0))[0].max()
    return a[:mr + 1, :mc + 1]


def _build_188():
    g = _G()
    c1 = g.fconst(list(range(HEIGHT)), [1, 1, HEIGHT, 1])     # row idx
    c2 = g.fconst(list(range(WIDTH)), [1, 1, 1, WIDTH])       # col idx
    half = g.scalar(0.5); one = g.scalar(1.0); qtr = g.scalar(0.25)
    colorw = g.fconst(list(range(CHANNELS)), [1, CHANNELS, 1, 1])

    def fcast(x):
        return g.node("Cast", [x], to=DATA_TYPE)

    t7 = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)
    t8 = g.node("ReduceMax", [t7], axes=[2], keepdims=1)
    t9 = g.node("ReduceMax", [t7], axes=[3], keepdims=1)
    t10 = g.node("ReduceSum", [t8], axes=[3], keepdims=1)
    t11 = g.node("ReduceSum", [t9], axes=[2], keepdims=1)
    t12 = g.node("Mul", [t10, half])
    t13 = g.node("Mul", [t11, half])
    t15 = g.node("Conv", ["input", colorw], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    t16 = g.node("Sub", [c2, c1])
    t17 = g.node("Add", [t16, t12])
    t20 = fcast(g.node("Less", [g.node("Abs", [t17]), half]))
    t21 = g.node("MatMul", [t15, t20])
    t23 = fcast(g.node("Less", [c2, t12]))
    t27 = fcast(g.node("Greater", [g.node("Abs", [g.node("Sub", [t15, t21])]), half]))
    t29 = g.node("Mul", [g.node("Mul", [t27, t23]), t7])
    t30 = g.node("ReduceSum", [t29], axes=[2, 3], keepdims=1)
    t32 = fcast(g.node("Cast", [t12], to=INT64))
    t34 = g.node("Abs", [g.node("Sub", [t12, t32])])
    t37 = fcast(g.node("Less", [t34, qtr]))
    t39 = fcast(g.node("Less", [t30, half]))
    t40 = g.node("Mul", [t37, t39])
    t43 = fcast(g.node("Less", [c1, t13]))
    m1 = g.node("Mul", [t40, t23])
    m2 = g.node("Mul", [g.node("Sub", [one, t40]), t43])
    combined = g.node("Add", [m1, m2])
    g.node("Mul", ["input", combined], "output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# 213  line-colours -> compact stack (rows or columns of the line colours)
# --------------------------------------------------------------------------- #
_C1_213 = [0, 1, 2, 3, 4, 0, 6, 7, 8, 9]          # colour value map (5 -> ignored)
_C44_213 = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]        # decode key vector


def _ref_213(a):
    H, W = a.shape
    if H > 30 or W > 30:
        return None
    X = np.zeros((10, 30, 30), np.float32)
    for r in range(H):
        for c in range(W):
            X[a[r, c], r, c] = 1
    c1 = np.array(_C1_213, np.float32)
    t3 = (X * c1[:, None, None]).sum(0)
    t4 = t3.max(axis=1); t5 = t3.max(axis=0)
    t8 = t4.reshape(10, 3).max(1); t11 = t5.reshape(10, 3).max(1)
    t16 = (t8 > 0.5).sum(); t19 = (t11 > 0.5).sum()
    t27 = float((t4 > 0.5).sum() < (t5 > 0.5).sum())
    idx = np.arange(30, dtype=np.float32)
    t28 = np.zeros(30, np.float32); t28[:10] = t8
    t31 = np.zeros(30, np.float32); t31[:10] = t11
    t38 = t28[:, None] * (idx < t16)[None, :]
    t39 = (idx < t19)[:, None] * t31[None, :]
    t43 = t27 * t38 + (1 - t27) * t39
    m = np.rint(t43).astype(int)
    grid = np.where(m >= 1, m, -1)
    keep = grid >= 0
    if keep.sum() == 0:
        return None
    mr = np.where(keep.any(1))[0].max(); mc = np.where(keep.any(0))[0].max()
    sub = grid[:mr + 1, :mc + 1]
    if (sub < 0).any():
        return None
    return sub


def _build_213():
    g = _G()
    half = g.scalar(0.5); one = g.scalar(1.0)
    colorw = g.fconst(_C1_213, [1, CHANNELS, 1, 1])
    c32 = g.fconst(list(range(WIDTH)), [1, 1, 1, WIDTH])
    c33 = g.fconst(list(range(HEIGHT)), [1, 1, HEIGHT, 1])

    def fcast(x):
        return g.node("Cast", [x], to=DATA_TYPE)

    t3 = g.node("Conv", ["input", colorw], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    t4 = g.node("ReduceMax", [t3], axes=[3], keepdims=1)
    t5 = g.node("ReduceMax", [t3], axes=[2], keepdims=1)
    t7 = g.node("Reshape", [t4, g.iconst([1, 1, 10, 3])])
    t8 = g.node("ReduceMax", [t7], axes=[3], keepdims=1)
    t10 = g.node("Reshape", [t5, g.iconst([1, 1, 10, 3])])
    t11 = g.node("ReduceMax", [t10], axes=[3], keepdims=1)
    t16 = g.node("ReduceSum", [fcast(g.node("Greater", [t8, half]))], axes=[2, 3], keepdims=1)
    t19 = g.node("ReduceSum", [fcast(g.node("Greater", [t11, half]))], axes=[2, 3], keepdims=1)
    t22 = g.node("ReduceSum", [fcast(g.node("Greater", [t4, half]))], axes=[2, 3], keepdims=1)
    t25 = g.node("ReduceSum", [fcast(g.node("Greater", [t5, half]))], axes=[2, 3], keepdims=1)
    t27 = fcast(g.node("Less", [t22, t25]))
    t28 = g.node("Pad", [t8], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 20, 0])
    t30 = g.node("Reshape", [t11, g.iconst([1, 1, 1, 10])])
    t31 = g.node("Pad", [t30], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 0, 20])
    t35 = fcast(g.node("Less", [c32, t16]))
    t37 = fcast(g.node("Less", [c33, t19]))
    t38 = g.node("Mul", [t28, t35])
    t39 = g.node("Mul", [t37, t31])
    t40 = g.node("Mul", [t27, t38])
    t42 = g.node("Mul", [g.node("Sub", [one, t27]), t39])
    t43 = g.node("Add", [t40, t42])
    rounded = g.node("Cast", [g.node("Add", [t43, half])], to=INT64)
    c44 = g.name("i")
    g.inits.append(oh.make_tensor(c44, INT64, [1, CHANNELS, 1, 1], _C44_213))
    g.node("Cast", [g.node("Equal", [rounded, c44])], "output", to=DATA_TYPE)
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# 198  lattice gap-flood: walls pass through, leaked bg -> 4, enclosed bg -> 3
# --------------------------------------------------------------------------- #
def _conv_plus(m):
    out = m.copy()
    out[1:, :] += m[:-1, :]; out[:-1, :] += m[1:, :]
    out[:, 1:] += m[:, :-1]; out[:, :-1] += m[:, 1:]
    return out


def _ref_198(a):
    H, W = a.shape
    if H > 30 or W > 30:
        return None
    X = np.zeros((10, 30, 30), np.float32)
    for r in range(H):
        for c in range(W):
            X[a[r, c], r, c] = 1
    t4 = X[0]; t5 = X.sum(0); t6 = t5 - t4
    Wd = t5.sum(axis=1).max(); Hd = t5.sum(axis=0).max()
    t16 = (t6.sum(axis=1) > Wd * 0.4).astype(np.float32)
    t18 = (t6.sum(axis=0) > Hd * 0.4).astype(np.float32)
    leak = np.maximum(t16[:, None], t18[None, :]) * t4
    for _ in range(14):
        leak = np.minimum(_conv_plus(leak), t4)
    t50 = leak[:H, :W]; t51 = (t4 - leak)[:H, :W]
    val = a.copy()
    val[t50 > 0.5] = 4; val[t51 > 0.5] = 3
    return val


def _build_198():
    g = _G()
    c11 = g.scalar(0.4)
    plus = g.fconst([0, 1, 0, 1, 1, 1, 0, 1, 0], [1, 1, 3, 3])

    def fcast(x):
        return g.node("Cast", [x], to=DATA_TYPE)

    t4 = _slice(g, "input", [0], [1], [1])
    t5 = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)
    t6 = g.node("Sub", [t5, t4])
    t8 = g.node("ReduceMax", [g.node("ReduceSum", [t5], axes=[3], keepdims=1)], axes=[2], keepdims=1)
    t10 = g.node("ReduceMax", [g.node("ReduceSum", [t5], axes=[2], keepdims=1)], axes=[3], keepdims=1)
    t12 = g.node("ReduceSum", [t6], axes=[3], keepdims=1)
    t13 = g.node("ReduceSum", [t6], axes=[2], keepdims=1)
    t16 = fcast(g.node("Greater", [t12, g.node("Mul", [t8, c11])]))
    t18 = fcast(g.node("Greater", [t13, g.node("Mul", [t10, c11])]))
    leak = g.node("Mul", [g.node("Max", [t16, t18]), t4])
    for _ in range(14):
        conv = g.node("Conv", [leak, plus], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        leak = g.node("Min", [conv, t4])
    t51 = g.node("Sub", [t4, leak])
    masks = g.node("Concat", [t4, leak, t51], axis=1)         # [1,3,30,30]
    Wd = np.zeros((CHANNELS, 3, 1, 1), np.float32)
    Wd[0, 0] = -1.0; Wd[4, 1] = 1.0; Wd[3, 2] = 1.0
    delta = g.node("Conv", [masks, _mat(g, Wd)], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    g.node("Add", ["input", delta], "output")
    return _model(g.nodes, g.inits)


def _exact(ref, prs):
    if not prs:
        return False
    saw = False
    for a, b in prs:
        try:
            r = ref(a)
        except Exception:
            return False
        if r is None or not isinstance(r, np.ndarray):
            return False
        if r.shape != b.shape or not np.array_equal(r, b):
            return False
        saw = True
    return saw


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def candidates(examples):
    prs = _pairs(examples)
    out = []
    if _exact(_ref_199, prs):
        out.append(("spray4_golf", _build_199()))
    if _exact(_ref_125, prs):
        out.append(("box125_golf", _build_125()))
    if _exact(_ref_181, prs):
        out.append(("reflect181_golf", _build_181()))
    if _exact(_ref_188, prs):
        out.append(("halve188_golf", _build_188()))
    if _exact(_ref_213, prs):
        out.append(("linecolors213_golf", _build_213()))
    if _exact(_ref_198, prs):
        out.append(("gapflood198_golf", _build_198()))
    return out
