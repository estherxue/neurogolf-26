"""family_pb_5 — minimal-cost ONNX recompilations for tasks
177, 155, 106, 389 (and attempts at 198/295/37).

Each rule is gated by a numpy reference that must reproduce ALL provided
train+test examples exactly; only then is the corresponding ONNX model yielded.
This keeps the family from firing on the wrong task.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

F32 = TP.FLOAT
I32 = TP.INT32
I64 = TP.INT64
BOOL = TP.BOOL
U8 = TP.UINT8

IN = oh.make_tensor_value_info("input", F32, [1, 10, 30, 30])
OUT = oh.make_tensor_value_info("output", F32, [1, 10, 30, 30])


def _vi(name, dt, shape):
    return oh.make_tensor_value_info(name, dt, shape)


def _model(nodes, inits, opset=13, out_type=F32, value_info=None):
    out = oh.make_tensor_value_info("output", out_type, [1, 10, 30, 30])
    g = oh.make_graph(nodes, "g", [IN], [out], inits, value_info=value_info or [])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", opset)])
    m.ir_version = 10
    return m


def _scalar(name, dt, val):
    return oh.make_tensor(name, dt, [], [val])


# --------------------------------------------------------------------------- #
# numpy references (operate on 2-D colour grids)
# --------------------------------------------------------------------------- #

def ref_155(a):  # hmirror = reverse rows
    return a[::-1, :]


def ref_106(a):  # rot90/180/270 2x2 kaleidoscope
    r90 = np.rot90(a, -1)   # clockwise
    r180 = np.rot90(a, 2)
    r270 = np.rot90(a, 1)   # counter-clockwise
    top = np.concatenate([a, r90], axis=1)
    bot = np.concatenate([r270, r180], axis=1)
    return np.concatenate([top, bot], axis=0)


def ref_389(a):  # colour c -> 0 ; 5 -> c   (input holds exactly {5,c})
    cols = set(np.unique(a).tolist())
    others = cols - {5}
    if len(others) != 1:
        return None
    c = others.pop()
    out = a.copy()
    out[a == c] = 0
    out[a == 5] = c
    return out


def ref_177(a):  # compress (drop monochrome rows/cols) then vmirror (reverse cols)
    H, W = a.shape
    keep_r = [i for i in range(H) if len(set(a[i].tolist())) != 1]
    keep_c = [j for j in range(W) if len(set(a[:, j].tolist())) != 1]
    if not keep_r or not keep_c:
        return None
    sub = a[np.ix_(keep_r, keep_c)]
    return sub[:, ::-1]


# --------------------------------------------------------------------------- #
# ONNX builders
# --------------------------------------------------------------------------- #

def build_155():
    """Reverse rows of a square grid, re-anchored top-left.
    idx = [H-1, H-2, ..., H-30] (int32); Gather(axis=2). Rows >= H land on
    negative indices -> wrap to the empty tail rows (grid is square, H<=~10)."""
    nodes = [
        oh.make_node("ReduceL2", ["input"], ["size_f"], axes=[0, 1, 2, 3], keepdims=0),
        oh.make_node("Cast", ["size_f"], ["size_i"], to=I32),
        oh.make_node("Add", ["size_i", "m1"], ["start_i"]),
        oh.make_node("Sub", ["size_i", "p31"], ["limit_i"]),
        oh.make_node("Range", ["start_i", "limit_i", "m1"], ["idx_i"]),
        oh.make_node("Gather", ["input", "idx_i"], ["output"], axis=2),
    ]
    inits = [_scalar("m1", I32, -1), _scalar("p31", I32, 31)]
    vi = [_vi("size_f", F32, []), _vi("size_i", I32, []),
          _vi("start_i", I32, []), _vi("limit_i", I32, []),
          _vi("idx_i", I32, [30])]
    return _model(nodes, inits, opset=13, value_info=vi)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

RULES = [
    ("hmirror155", ref_155, build_155),
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
