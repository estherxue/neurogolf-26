"""family_scrk_1 : concentric-square "onion-swap + plus-arm" expansion (task 86 family).

Rule (per solid concentric-square object, colors O=border, In=interior):
  1. swap the two colors inside the object's box (border<->interior);
  2. paint solid arms of the border colour O in the 4 cardinal directions,
     extending floor(size/2) cells (1 for 3x3, 2 for 4x4).

Everything is a LOCAL, origin-anchored stencil (shifts / maxpool / cross-convs)
so it is expressed as a single static opset-10 graph with no size dependence.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64


# --------------------------------------------------------------------------- #
# numpy reference (also used for detection)                                    #
# --------------------------------------------------------------------------- #
def _shift(a, dy, dx):
    r = np.zeros_like(a)
    H, W = a.shape
    ys0, ys1 = max(0, dy), min(H, H + dy)
    xs0, xs1 = max(0, dx), min(W, W + dx)
    sy0, sy1 = max(0, -dy), min(H, H - dy)
    sx0, sx1 = max(0, -dx), min(W, W - dx)
    r[ys0:ys1, xs0:xs1] = a[sy0:sy1, sx0:sx1]
    return r


def _dil(a, r):
    out = a.astype(bool).copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out = out | _shift(a.astype(bool), dy, dx)
    return out


def _solve(I):
    H, W = I.shape
    M = (I != 0)
    bg = ~M
    oh_ = [(I == c) for c in range(CHANNELS)]
    pres = [_dil(oh_[c], 1) for c in range(CHANNELS)]
    out = np.zeros((H, W), int)
    for c in range(1, CHANNELS):
        out[pres[c] & M & (I != c)] = c
    interior = M & ~_dil(bg, 1)
    big = interior & (_shift(interior, 1, 0) | _shift(interior, -1, 0) |
                      _shift(interior, 0, 1) | _shift(interior, 0, -1))
    F = _dil(big, 2) & M
    border = M & ~interior
    Fb = F & border
    for c in range(1, CHANNELS):
        Bc = border & oh_[c]
        Fbc = Fb & oh_[c]
        arm = np.zeros((H, W), bool)
        arm |= bg & _shift(Bc, -1, 0)
        arm |= bg & _shift(Bc, 1, 0)
        arm |= bg & _shift(Bc, 0, -1)
        arm |= bg & _shift(Bc, 0, 1)
        arm |= bg & _shift(Fbc, -2, 0)
        arm |= bg & _shift(Fbc, 2, 0)
        arm |= bg & _shift(Fbc, 0, -2)
        arm |= bg & _shift(Fbc, 0, 2)
        out[arm] = c
    return out


# --------------------------------------------------------------------------- #
# ONNX builder                                                                 #
# --------------------------------------------------------------------------- #
def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _build():
    nodes, inits = [], []
    one = oh.make_tensor("one", DATA_TYPE, [1, 1, 1, 1], [1.0])
    inits.append(one)

    # depthwise cross kernels
    cross3 = np.zeros((CHANNELS, 1, 3, 3), np.float32)
    for c in range(CHANNELS):
        cross3[c, 0, 0, 1] = cross3[c, 0, 2, 1] = 1.0
        cross3[c, 0, 1, 0] = cross3[c, 0, 1, 2] = 1.0
    inits.append(oh.make_tensor("cross3", DATA_TYPE, [CHANNELS, 1, 3, 3], cross3.ravel().tolist()))
    cross5 = np.zeros((CHANNELS, 1, 5, 5), np.float32)
    for c in range(CHANNELS):
        cross5[c, 0, 0, 2] = cross5[c, 0, 4, 2] = 1.0
        cross5[c, 0, 2, 0] = cross5[c, 0, 2, 4] = 1.0
    inits.append(oh.make_tensor("cross5", DATA_TYPE, [CHANNELS, 1, 5, 5], cross5.ravel().tolist()))
    cint3 = np.zeros((1, 1, 3, 3), np.float32)
    cint3[0, 0, 0, 1] = cint3[0, 0, 2, 1] = cint3[0, 0, 1, 0] = cint3[0, 0, 1, 2] = 1.0
    inits.append(oh.make_tensor("cint3", DATA_TYPE, [1, 1, 3, 3], cint3.ravel().tolist()))

    # opset-10 Slice takes starts/ends/axes as tensor inputs
    inits.append(oh.make_tensor("s0", INT64, [1], [0]))
    inits.append(oh.make_tensor("e1", INT64, [1], [1]))
    inits.append(oh.make_tensor("ax1", INT64, [1], [1]))
    inits.append(oh.make_tensor("s1", INT64, [1], [1]))
    inits.append(oh.make_tensor("eC", INT64, [1], [CHANNELS]))

    N = oh.make_node
    nodes += [
        N("ReduceSum", ["input"], ["real"], axes=[1], keepdims=1),
        N("Slice", ["input", "s0", "e1", "ax1"], ["X0"]),
        N("Sub", ["real", "X0"], ["M"]),
        N("MaxPool", ["X0"], ["bgmax"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        N("Sub", ["one", "bgmax"], ["nbg"]),
        N("Mul", ["M", "nbg"], ["interior"]),
        N("Conv", ["interior", "cint3"], ["crossint"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        N("Mul", ["interior", "crossint"], ["big"]),
        N("MaxPool", ["big"], ["Fpre"], kernel_shape=[5, 5], pads=[2, 2, 2, 2]),
        N("Sub", ["one", "interior"], ["nint"]),
        N("Mul", ["M", "nint"], ["border"]),
        N("Mul", ["Fpre", "border"], ["Fb"]),
        N("Mul", ["input", "border"], ["BorderColored"]),
        N("Mul", ["input", "Fb"], ["FbColored"]),
        N("Conv", ["BorderColored", "cross3"], ["dist1"], kernel_shape=[3, 3],
          pads=[1, 1, 1, 1], group=CHANNELS),
        N("Conv", ["FbColored", "cross5"], ["dist2"], kernel_shape=[5, 5],
          pads=[2, 2, 2, 2], group=CHANNELS),
        N("Add", ["dist1", "dist2"], ["armsum"]),
        N("Mul", ["armsum", "X0"], ["arm_colored"]),
        N("MaxPool", ["input"], ["pres"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        N("Sub", ["one", "input"], ["nX"]),
        N("Mul", ["pres", "nX"], ["swp1"]),
        N("Mul", ["swp1", "M"], ["swap_colored"]),
        N("Add", ["swap_colored", "arm_colored"], ["S"]),
        N("ReduceSum", ["arm_colored"], ["arm_any"], axes=[1], keepdims=1),
        N("Sub", ["one", "arm_any"], ["narm"]),
        N("Mul", ["X0", "narm"], ["chan0"]),
        N("Slice", ["S", "s1", "eC", "ax1"], ["Srest"]),
        N("Concat", ["chan0", "Srest"], ["output"], axis=1),
    ]
    return _model(nodes, inits)


def candidates(ex):
    prs = []
    for e in ex.get("train", []) + ex.get("test", []):
        a = np.array(e["input"], int)
        b = np.array(e["output"], int)
        if a.shape != b.shape or max(a.shape) > 30:
            return []
        prs.append((a, b))
    if not prs:
        return []
    for a, b in prs:
        try:
            if not np.array_equal(_solve(a), b):
                return []
        except Exception:
            return []
    return [("scrk1_onionswap", _build())]
