"""family_cs2_2 — FINAL COMPLETE-SWEEP recompile.

Only one genuine win in this batch: task141 (623ea044, "shoot diagonal X-rays
from every least-colour cell, fill them with that colour").

The deployed out_blend6 net for task141 does NOT load in local ORT 1.23.2 — it
uses a `Max(13)` node whose fp16 typing has no ORT implementation
("NOT_IMPLEMENTED : Could not find an implementation for Max(13)"), so it scores
0 points. Any correct, loadable graph is therefore a strict win.

Rule (verify_623ea044):
    x0 = leastcolor(I)                      # least-common colour, tie -> smallest
    for each cell of colour x0, draw the two full diagonals through it
    (main: r-c const, anti: r+c const), clipped to the grid, and fill with x0.

Structure (single Conv, params-dominated):
    S       = seed mask (input channel x0)              [1,1,30,30]
    raycount= Conv(S, X-kernel[1,1,59,59], pad 29)      counts seeds on the two
              diagonals through each cell (radius 29 covers a 30x30 grid).
    ray     = raycount > 0
    valid   = any input channel set  (in-grid rectangle; suppresses rays that a
              seed casts into the padded region of a <30x30 grid)
    R       = ray AND valid
    output  = Where(R, onehot(x0), input)               # fill x0 on rays, keep I

All ops exist at opset 10 (Conv, ReduceSum/Max with axes attr, ArgMin first-index
tie-break == leastcolor's smallest-colour tie-break, Equal on int64, Greater/And
on the boolean masks, Where). fp32 working dtype keeps Conv/Greater on
ORT-implemented type combinations.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = onnx.TensorProto.FLOAT
I64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL
H = 30


class _G:
    def __init__(self):
        self.nodes, self.inits, self._k = [], [], 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def init(self, dt, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, dt, list(dims), list(vals)))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _build_141():
    g = _G()
    inp = "input"  # [1,10,30,30] fp32 one-hot

    # --- leastcolor(I) -> x0 (int64 scalar) ------------------------------ #
    count = g.nd("ReduceSum", [inp], axes=[2, 3], keepdims=1)      # [1,10,1,1]
    zero = g.init(F, [1], [0.0])
    big = g.init(F, [1], [1e9])
    present = g.nd("Greater", [count, zero])                       # bool
    masked = g.nd("Where", [present, count, big])                  # 0-count -> +inf
    x0 = g.nd("ArgMin", [masked], axis=1, keepdims=1)             # [1,1,1,1] i64

    # --- onehot(x0) over the 10 colour channels -------------------------- #
    arange = g.init(I64, [1, 10, 1, 1], list(range(10)))
    onehot_b = g.nd("Equal", [arange, x0])                         # bool [1,10,1,1]
    onehot = g.nd("Cast", [onehot_b], to=F)                        # fp32 [1,10,1,1]

    # --- seed mask S = input[x0] (Gather keeps no [1,10,30,30] tensor) --- #
    idx = g.nd("Reshape", [x0, g.init(I64, [1], [1])])            # [1] int64
    S = g.nd("Gather", [inp, idx], axis=1)                        # [1,1,30,30]

    # --- diagonal-line count via one Conv ------------------------------- #
    w = np.zeros((1, 1, 2 * H - 1, 2 * H - 1), np.float32)
    for a in range(2 * H - 1):
        w[0, 0, a, a] = 1.0                     # main diagonal
        w[0, 0, a, 2 * H - 2 - a] = 1.0         # anti diagonal
    W = g.init(F, w.shape, w.ravel().tolist())
    raycount = g.nd("Conv", [S, W], pads=[H - 1, H - 1, H - 1, H - 1])  # [1,1,30,30]

    half = g.init(F, [1], [0.5])
    ray = g.nd("Greater", [raycount, half])                       # bool
    cover = g.nd("ReduceMax", [inp], axes=[1], keepdims=1)        # [1,1,30,30]
    valid = g.nd("Greater", [cover, half])                        # bool in-grid
    R = g.nd("And", [ray, valid])                                 # bool

    g.nd("Where", [R, onehot, inp], out="output")                # [1,10,30,30]

    vi_in = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    vi_out = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "task141_diag", [vi_in], [vi_out], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# numpy reference (fingerprint gate)                                          #
# --------------------------------------------------------------------------- #
def _solve_141(grid):
    h, w = len(grid), len(grid[0])
    vals = [v for r in grid for v in r]
    x0 = min(sorted(set(vals)), key=lambda c: vals.count(c))
    seeds = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == x0]
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            for (sr, sc) in seeds:
                if (r - c) == (sr - sc) or (r + c) == (sr + sc):
                    out[r][c] = x0
                    break
    return out


def _fp_141(ex):
    seen = 0
    for split in ("train", "test"):
        for e in ex.get(split, []):
            g = e["input"]
            if not g or not g[0] or max(len(g), len(g[0])) > 30:
                continue
            if _solve_141(g) != e["output"]:
                return False
            seen += 1
    return seen > 0


_MODEL_141 = None


def candidates(ex):
    """Route by train fingerprint; only task141's diagonal-X rule is emitted."""
    global _MODEL_141
    out = []
    if _fp_141(ex):
        if _MODEL_141 is None:
            _MODEL_141 = _build_141()
        out.append(("cs2_141_diag", _MODEL_141))
    return out
