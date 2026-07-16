"""family_pb_6 — minimal-cost ONNX recompilations.

Targets (task, incumbent_pts): 150(20.07 vmirror), 331(18.19 stamp-conv),
95(19.16 dilate). Each rule is gated by a numpy reference that must reproduce
ALL train+test+arc-gen examples exactly before its ONNX model is yielded, so a
family never fires on the wrong task.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

F32 = TP.FLOAT
I32 = TP.INT32
I64 = TP.INT64


def _vi(name, dt, shape):
    return oh.make_tensor_value_info(name, dt, shape)


def _scalar(name, dt, val):
    return oh.make_tensor(name, dt, [], [val])


def _model(nodes, inits, opset=13, value_info=None):
    IN = oh.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    OUT = oh.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    g = oh.make_graph(nodes, "g", [IN], [OUT], inits, value_info=value_info or [])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", opset)])
    m.ir_version = 10
    return m


# --------------------------------------------------------------------------- #
# numpy references (operate on 2-D colour grids)
# --------------------------------------------------------------------------- #

def ref_150(a):  # vmirror = reverse columns
    return a[:, ::-1]


def ref_331(a):  # isolated fg(1) on bg(0): stamp orthogonal neighbours
    if set(np.unique(a).tolist()) - {0, 1}:
        return None
    out = a.copy()
    H, W = a.shape
    pts = np.argwhere(a == 1)
    for (i, j) in pts:
        if i + 1 < H: out[i + 1, j] = 8   # down
        if i - 1 >= 0: out[i - 1, j] = 2  # up
        if j + 1 < W: out[i, j + 1] = 6   # right
        if j - 1 >= 0: out[i, j - 1] = 7  # left
    for (i, j) in pts:
        out[i, j] = 1
    return out


def ref_95(a):  # dilate least-colour(5) by 8-neighbourhood -> 1
    if set(np.unique(a).tolist()) - {0, 5}:
        return None
    out = a.copy()
    H, W = a.shape
    pts = np.argwhere(a == 5)
    for (i, j) in pts:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and a[ni, nj] != 5:
                    out[ni, nj] = 1
    return out


# --------------------------------------------------------------------------- #
# ONNX builders
# --------------------------------------------------------------------------- #

def build_150():
    """Reverse columns of a square grid, re-anchored top-left.
    idx = [W-1, W-2, ..., W-30] (int32) via Range; Gather(axis=3). Indices >= W
    are negative -> wrap into the empty tail columns. size = sqrt(#cells) (square
    grids only)."""
    nodes = [
        oh.make_node("ReduceL2", ["input"], ["sf"], axes=[0, 1, 2, 3], keepdims=0),
        oh.make_node("Sub", ["sf", "one_f"], ["sm1f"]),
        oh.make_node("Cast", ["sm1f"], ["start"], to=I32),
        oh.make_node("Sub", ["start", "p30"], ["limit"]),
        oh.make_node("Range", ["start", "limit", "m1"], ["idx"]),
        oh.make_node("Gather", ["input", "idx"], ["output"], axis=3),
    ]
    inits = [_scalar("one_f", F32, 1.0), _scalar("p30", I32, 30),
             _scalar("m1", I32, -1)]
    vi = [_vi("sf", F32, []), _vi("sm1f", F32, []),
          _vi("start", I32, []), _vi("limit", I32, []), _vi("idx", I32, [30])]
    return _model(nodes, inits, opset=13, value_info=vi)


def build_331():
    """Single 3x3 Conv (900 weight + 10 bias). Isolated fg(1)/bg(0).
    ch1: keep centre. Stamp channels {2,8,6,7} require the centre cell to be
    IN-GRID background (ch0 centre = 1) so stamps never bleed past the grid
    edge into the all-zero outside region: value = fg_neighbour + ch0_centre - 1.
    ch0: bg centre minus 4 orthogonal fg neighbours (suppress stamped bg)."""
    W = np.zeros((10, 10, 3, 3), np.float32)
    B = np.zeros((10,), np.float32)
    # taps: [di,dj] center=(1,1). fg at (i+di-1, j+dj-1) contributes to (i,j).
    W[1, 1, 1, 1] = 1.0                 # centre keeps 1
    stamps = {2: (2, 1), 8: (0, 1), 6: (1, 0), 7: (1, 2)}
    for c, (di, dj) in stamps.items():
        W[c, 1, di, dj] = 1.0          # fg at that neighbour tap
        W[c, 0, 1, 1] = 1.0            # require in-grid bg at centre
        B[c] = -1.0                    # AND-gate: neighbour + ingrid - 1
    W[0, 0, 1, 1] = 1.0                 # bg passthrough
    for (di, dj) in stamps.values():
        W[0, 1, di, dj] = -1.0          # suppress bg where an orthogonal fg neighbour
    nodes = [oh.make_node("Conv", ["input", "w", "b"], ["output"],
                          kernel_shape=[3, 3], pads=[1, 1, 1, 1])]
    inits = [oh.make_tensor("w", F32, [10, 10, 3, 3], W.flatten().tolist()),
             oh.make_tensor("b", F32, [10], B.tolist())]
    return _model(nodes, inits, opset=13)


def build_95():
    """Single 3x3 Conv, no bias (900 params, 0 memory). special(5)/bg(0).
    ch5 centre passthrough; ch1 = any of 8 neighbours is 5 (ring); ch0 = bg
    centre minus 8-neighbour 5-count (suppress ring cells)."""
    W = np.zeros((10, 10, 3, 3), np.float32)
    W[5, 5, 1, 1] = 1.0                 # keep the 5
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            if (di, dj) == (1, 1):
                continue
            W[1, 5, di, dj] = 1.0       # ring: neighbour is 5
            W[0, 5, di, dj] = -1.0      # suppress bg on ring cells
    W[0, 0, 1, 1] = 1.0                 # bg passthrough
    nodes = [oh.make_node("Conv", ["input", "w"], ["output"],
                          kernel_shape=[3, 3], pads=[1, 1, 1, 1])]
    inits = [oh.make_tensor("w", F32, [10, 10, 3, 3], W.flatten().tolist())]
    return _model(nodes, inits, opset=13)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

RULES = [
    ("vmirror150", ref_150, build_150),
    ("stamp331", ref_331, build_331),
    ("dilate95", ref_95, build_95),
]


def _reproduces(ref, pairs):
    for a, b in pairs:
        try:
            r = ref(a)
        except Exception:
            return False
        if r is None:
            return False
        r = np.asarray(r)
        if r.shape != b.shape or not np.array_equal(r, b):
            return False
    return True


def candidates(examples):
    pairs = [(np.array(e["input"]), np.array(e["output"]))
             for e in examples["train"] + examples["test"]]
    out = []
    for name, ref, build in RULES:
        if _reproduces(ref, pairs):
            try:
                out.append((name, build()))
            except Exception:
                pass
    return out
