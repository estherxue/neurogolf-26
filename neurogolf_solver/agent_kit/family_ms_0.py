"""family_ms_0 -- MAX-EFFORT deep-dive on the two highest-memory algorithmic tasks.

Assigned:
  task145 (hash 6455b5f5, "bisection"): recolor the largest-area black rectangle(s)
     -> 1 (blue) and the smallest-area black rectangle(s) -> 8 (cyan); red separators
     stay.  Current best: 15.54 pts / mem 12744.
  task285 (hash b775ac94, "missinglegs"): reconstruct 4-fold-mirror-symmetric creatures
     from one shown quadrant + the 2x2 centre colours.  Current best: 15.11 / mem 19286.

This module holds the *investigated* candidate builders.  `candidates()` routes a task
by its train fingerprint and only emits a builder when it verified strictly cheaper.

The `build145_rl` builder below is the best local reconstruction of the run-length /
fp16-area approach (see report for the floor analysis).  It is kept for measurement /
documentation; it is NOT routed by default because it does not beat the 12744 baseline
under local ORT 1.23.2 (the deployed baseline uses uint8 elementwise Max, which local
ORT does not implement, so any locally-loadable graph pays strictly more).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

U8 = onnx.TensorProto.UINT8
F16 = onnx.TensorProto.FLOAT16
F32 = onnx.TensorProto.FLOAT
BOOL = onnx.TensorProto.BOOL
I64 = onnx.TensorProto.INT64

GRID = [1, 10, 30, 30]


class G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0

    def _u(self, p="t"):
        self.n += 1
        return f"{p}{self.n}"

    def init(self, dt, dims, vals, name=None):
        name = name or self._u("k")
        self.inits.append(oh.make_tensor(name, dt, list(dims), list(vals)))
        return name

    def node(self, op, ins, out=None, **kw):
        out = out or self._u()
        self.nodes.append(oh.make_node(op, list(ins), [out], **kw))
        return out

    def model(self, opset=19):
        x = oh.make_tensor_value_info("input", F32, GRID)
        y = oh.make_tensor_value_info("output", BOOL, GRID)
        g = oh.make_graph(self.nodes, "g", [x], [y], self.inits)
        m = oh.make_model(g, opset_imports=[oh.make_opsetid("", opset)])
        m.ir_version = 10
        return m


def build145_rl():
    """Run-length (MaxPool markers) + fp16 area product; local-ORT compatible (opset 19,
    no uint8 elementwise Max)."""
    g = G()
    S = 20  # generator max grid dimension for this task
    # ---- black mask at SxS ----
    z_f32 = g.node("Slice", ["input",
                             g.init(I64, [4], [0, 0, 0, 0]),
                             g.init(I64, [4], [1, 1, S, S]),
                             g.init(I64, [4], [0, 1, 2, 3])], out="z_f32")   # [1,1,S,S] f32
    z_bool = g.node("Cast", [z_f32], out="z_bool", to=BOOL)                  # black
    zero = g.init(U8, [], [0]); one = g.init(U8, [], [1])
    two = g.init(U8, [], [2]); eight = g.init(U8, [], [8]); ten = g.init(U8, [], [10])
    pos_l = g.init(U8, [1, 1, 1, S], list(range(1, S + 1)))
    pos_r = g.init(U8, [1, 1, 1, S], list(range(S, 0, -1)))
    pos_u = g.init(U8, [1, 1, S, 1], list(range(1, S + 1)))
    pos_d = g.init(U8, [1, 1, S, 1], list(range(S, 0, -1)))
    # wall-position markers (0 at black, position at wall); prefix/suffix max via MaxPool
    ls = g.node("Where", [z_bool, zero, pos_l]); lm = g.node("MaxPool", [ls], kernel_shape=[1, S], pads=[0, S - 1, 0, 0], strides=[1, 1])
    rs = g.node("Where", [z_bool, zero, pos_r]); rm = g.node("MaxPool", [rs], kernel_shape=[1, S], pads=[0, 0, 0, S - 1], strides=[1, 1])
    us = g.node("Where", [z_bool, zero, pos_u]); um = g.node("MaxPool", [us], kernel_shape=[S, 1], pads=[S - 1, 0, 0, 0], strides=[1, 1])
    ds = g.node("Where", [z_bool, zero, pos_d]); dm = g.node("MaxPool", [ds], kernel_shape=[S, 1], pads=[0, 0, S - 1, 0], strides=[1, 1])
    lr = g.node("Add", [lm, rm]); ud = g.node("Add", [um, dm])              # uint8, = 20 - w / 20 - h
    scale = g.init(F16, [], [1.0]); zp = g.init(U8, [], [S])
    nw = g.node("DequantizeLinear", [lr, scale, zp])                        # fp16 = -width
    nh = g.node("DequantizeLinear", [ud, scale, zp])                        # fp16 = -height
    area = g.node("Mul", [nw, nh])                                          # fp16 = w*h
    ax = g.init(I64, [2], [2, 3])
    amax = g.node("ReduceMax", [area, ax], keepdims=1)
    afm = g.node("Where", [z_bool, area, amax])                            # black->area else big
    amin = g.node("ReduceMin", [afm, ax], keepdims=1)
    maxm = g.node("Equal", [area, amax])
    minm = g.node("Equal", [afm, amin])
    # ---- inside-grid bounding box (handles grids < SxS) ----
    ri = g.node("Cast", [g.node("MaxPool", [z_f32], kernel_shape=[S, S], pads=[0, 0, S - 1, 0], strides=[1, 1])], to=BOOL)  # [1,1,S,1]
    ci = g.node("Cast", [g.node("MaxPool", [z_f32], kernel_shape=[S, S], pads=[0, 0, 0, S - 1], strides=[1, 1])], to=BOOL)  # [1,1,1,S]
    inside = g.node("And", [ri, ci])                                        # [1,1,S,S] bool
    # ---- label grid ----
    base = g.node("Where", [z_bool, zero, two])                            # black->0 else 2(red)
    l1 = g.node("Where", [maxm, one, base])
    l2 = g.node("Where", [minm, eight, l1])
    core = g.node("Where", [inside, l2, ten])                              # outside->10
    lab30 = g.node("Pad", [core, g.init(I64, [8], [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]), ten])  # [1,1,30,30]
    ids = g.init(U8, [1, 10, 1, 1], list(range(10)))
    g.node("Equal", [lab30, ids], out="output")
    return g.model()


# --------------------------------------------------------------------------
# Fingerprint routing.  Both assigned tasks were investigated to FLOOR under
# local ORT 1.23.2, so no builder is emitted (shipping only verified wins).
# --------------------------------------------------------------------------
# Registry of (detector, builder) that only fires on an EXACT, VERIFIED-cheaper win.
RULES: list = []


def candidates(examples):
    out = []
    for det, build in RULES:
        try:
            if det(examples):
                out.append((build.__name__, build()))
        except Exception:
            pass
    return out
