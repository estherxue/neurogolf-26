"""family_bpk005 — recompiled task5 / verify_045e512c (vc1_rays lineage).

Rays / sprite-stamp extension task (ARC 045e512c, 21x21, bg=0):
    the unique dense 3x3 sprite (the "middle") has small colour markers at offset
    +-4 in 2-3 of the 8 directions; paint pattern-shaped copies of the sprite along
    each marked ray at steps of 4, coloured by that direction's marker colour.

This is a memory-golf recompile of the incumbent one-hot Conv-canvas solver
(gw2f16_5, ~174KB, 26 nodes).  Same function, cheaper graph:

  * SINGLE-CHANNEL LABEL space throughout — the one-hot input is collapsed to a
    colour grid V=[1,1,30,30] by a 1x1 Conv([0..9]); no [1,10,H,W] canvases.
  * bbox found by the same 5x5 score conv (+1 inner 3x3 / -3 ring) -> argmax corner.
  * the 8 marker colours are read WITHOUT an 8-channel spatial tensor: a 3x3 MaxPool
    of V plus a GatherND at the 8 corner+4*dir sample points -> cd=[8].
  * ray propagation is a SINGLE [8,1,9,9] dilation-4 Conv of the sprite mask M
    (taps at k=1..4 per direction) -> MD=[1,8,30,30]; colouring + direction-merge is
    one Einsum('nchw,nc->nhw', MD, cd) -> R=[1,30,30] (no coloured 8-ch canvas).
  * output = Equal(V-max-R + ingrid, [1..10]) as a BOOL[1,10,30,30] straight into
    `output` (the `+ingrid` both offsets the label by 1 and zeroes the padding ring,
    so no separate mask multiply and no float one-hot intermediate).

Everything interior is float16.  Detection is behavioural (numpy mirror of the ONNX
numerics matched against train+test), so it only ever fires on this task.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh, numpy_helper as nh

from ng_utils_shim import GRID_SHAPE, IR_VERSION

FP32 = onnx.TensorProto.FLOAT
FP16 = onnx.TensorProto.FLOAT16
BOOL = onnx.TensorProto.BOOL
INT64 = onnx.TensorProto.INT64
H = 30
OPSET = [oh.make_opsetid("", 12)]

_DIRS8 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


# --------------------------------------------------------------------------- #
# graph accumulator                                                           #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def init(self, arr, dtype):
        n = self.nm("c")
        self.inits.append(nh.from_array(np.asarray(arr, dtype), n))
        return n

    def f16(self, arr):
        return self.init(arr, np.float16)

    def f32(self, arr):
        return self.init(arr, np.float32)

    def i64(self, arr):
        return self.init(arr, np.int64)

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _np_005(a):
    """Reference semantics (mirror of the ONNX numerics)."""
    a = np.asarray(a, int)
    Hd, Wd = a.shape
    if Hd > 30 or Wd > 30 or Hd < 4 or Wd < 4:
        return None
    N = (a > 0).astype(int)
    Np = np.pad(N, ((1, 3), (1, 3)))
    S = np.zeros((Hd, Wd), int)
    for u in range(5):
        for v in range(5):
            wgt = 1 if (1 <= u <= 3 and 1 <= v <= 3) else -3
            S += wgt * Np[u:u + Hd, v:v + Wd]
    mx = S.max()
    pos = np.argwhere(S == mx)
    if len(pos) != 1 or mx < 4:
        return None
    r0, c0 = map(int, pos[0])
    if r0 + 2 >= Hd or c0 + 2 >= Wd:
        return None
    B = np.zeros((Hd, Wd), int)
    B[r0:r0 + 3, c0:c0 + 3] = 1
    M = N * B
    R = np.zeros((Hd, Wd), int)
    for di, dj in _DIRS8:
        colors = []
        for u in range(3):
            for v in range(3):
                y, x = r0 + u + 4 * di, c0 + v + 4 * dj
                if 0 <= y < Hd and 0 <= x < Wd and a[y, x] > 0:
                    colors.append(int(a[y, x]))
        cd = max(colors) if colors else 0
        if cd == 0:
            continue
        for k in range(1, 8):
            for u in range(3):
                for v in range(3):
                    if M[r0 + u, c0 + v]:
                        y, x = r0 + u + 4 * k * di, c0 + v + 4 * k * dj
                        if 0 <= y < Hd and 0 <= x < Wd:
                            R[y, x] = cd
    return np.maximum(a, R)


def _build_005():
    g = _G()

    # value grid V (single channel colour labels) ---------------------------- #
    w0 = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    V32 = g.nd("Conv", ["input", g.f32(w0)], kernel_shape=[1, 1])
    Vh = g.nd("Cast", [V32], to=FP16)
    ingrid32 = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    ingridh = g.nd("Cast", [ingrid32], to=FP16)

    # nonbg mask ------------------------------------------------------------- #
    zero = g.f16([[[[0.0]]]])
    Nh = g.nd("Cast", [g.nd("Greater", [Vh, zero])], to=FP16)

    # 5x5 score conv -> unique bbox corner ----------------------------------- #
    k5 = np.full((1, 1, 5, 5), -3.0, np.float16)
    k5[0, 0, 1:4, 1:4] = 1.0
    S = g.nd("Conv", [Nh, g.f16(k5)], kernel_shape=[5, 5], pads=[1, 1, 3, 3])
    mx = g.nd("ReduceMax", [S], axes=[2, 3], keepdims=1)
    half = g.f16([[[[0.5]]]])
    Ph = g.nd("Cast", [g.nd("Greater", [S, g.nd("Sub", [mx, half])])], to=FP16)

    # sprite mask M = N inside the 3x3 box ----------------------------------- #
    B = g.nd("Conv", [Ph, g.f16(np.ones((1, 1, 3, 3), np.float16))],
             kernel_shape=[3, 3], pads=[2, 2, 0, 0])
    M = g.nd("Mul", [Nh, B])

    # ray-mask propagation MD[1,8,30,30] ------------------------------------- #
    w8 = np.zeros((8, 1, 9, 9), np.float16)
    for d, (di, dj) in enumerate(_DIRS8):
        for k in range(1, 5):
            aa, cc = 4 - k * di, 4 - k * dj
            w8[d, 0, aa, cc] = 1.0
    MD = g.nd("Conv", [M, g.f16(w8)], kernel_shape=[9, 9],
              dilations=[4, 4], pads=[16, 16, 16, 16])

    # 8 marker colours via MaxPool + GatherND -------------------------------- #
    Vmax = g.nd("MaxPool", [Vh], kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                strides=[1, 1])
    rows = g.f16(np.arange(H, dtype=np.float16))
    rowmarg = g.nd("ReduceSum", [Ph], axes=[0, 1, 3], keepdims=0)   # [30]
    colmarg = g.nd("ReduceSum", [Ph], axes=[0, 1, 2], keepdims=0)   # [30]
    r0 = g.nd("ReduceSum", [g.nd("Mul", [rowmarg, rows])], axes=[0], keepdims=1)  # [1]
    c0 = g.nd("ReduceSum", [g.nd("Mul", [colmarg, rows])], axes=[0], keepdims=1)  # [1]
    z1 = g.f16([0.0])
    base = g.nd("Concat", [z1, z1, r0, c0], axis=0)                  # [4] = [0,0,r0,c0]
    base = g.nd("Cast", [base], to=FP32)
    offs = np.array([[0, 0, 1 + 4 * di, 1 + 4 * dj] for (di, dj) in _DIRS8],
                    np.float32)
    pos = g.nd("Add", [g.f32(offs), g.nd("Reshape", [base, g.i64([1, 4])])])  # [8,4]
    pos = g.nd("Clip", [pos, g.f32(0.0), g.f32(float(H - 1))])
    idx = g.nd("Cast", [pos], to=INT64)
    cd = g.nd("GatherND", [Vmax, idx])                              # [8]
    cd = g.nd("Reshape", [cd, g.i64([1, 8])])
    cd = g.nd("Cast", [cd], to=FP16)

    # colour + direction merge (no coloured 8-ch canvas) --------------------- #
    R = g.nd("Einsum", [MD, cd], equation="nchw,nc->nhw")            # [1,30,30]

    LV = g.nd("Max", [Vh, R])                                        # [1,1,30,30]
    one = g.f16([[[[1.0]]]])
    # (LV+1) inside the grid, 0 in the padding ring -> label+1 vs [1..10]
    S2 = g.nd("Mul", [g.nd("Add", [LV, one]), ingridh])
    idxP = g.f16(np.arange(1, 11, dtype=np.float16).reshape(1, 10, 1, 1))
    g.nd("Equal", [S2, idxP], "output")                            # BOOL[1,10,30,30]

    x = oh.make_tensor_value_info("input", FP32, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", BOOL, GRID_SHAPE)
    used = {i for n in g.nodes for i in n.input}
    inits = [t for t in g.inits if t.name in used]
    m = oh.make_model(oh.make_graph(g.nodes, "bpk005", [x], [y], inits),
                      ir_version=IR_VERSION, opset_imports=OPSET)
    onnx.checker.check_model(m, full_check=True)
    return m


def _pairs(examples):
    out = []
    for split in ("train", "test"):
        for e in examples.get(split, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                return []
            out.append((a, b))
    return out


def candidates(examples):
    prs = _pairs(examples)
    if not prs:
        return
    for a, b in prs:
        o = _np_005(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return
    try:
        yield ("bpk005_rays045e512c", _build_005())
    except Exception:
        return
