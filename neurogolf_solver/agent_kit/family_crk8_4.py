"""family_crk8_4 — hard residual ARC tasks (slice U[4::6]).

Solved:
  * task077: orthogonal "≥2-active-neighbor" growth CA. Seeds = color-2 cells.
    A fill cell (color != 0,2) flips to color 4 once >=2 of its orthogonal
    neighbours are color-2 or already-flipped; iterate to fixpoint. Expressed as
    an unrolled stack of Conv(cross)->Clip->Mul layers on a single-channel mask.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# ------------------------------------------------------------------ task 077

def _orth_count(mask):
    H, W = mask.shape
    c = np.zeros((H, W), int)
    c[1:, :] += mask[:-1, :]; c[:-1, :] += mask[1:, :]
    c[:, 1:] += mask[:, :-1]; c[:, :-1] += mask[:, 1:]
    return c


def _sim077(a, iters=64):
    fill = (a != 0) & (a != 2)
    marked = np.zeros(a.shape, bool)
    for _ in range(iters):
        active = (a == 2) | marked
        new = fill & (_orth_count(active) >= 2)
        if np.array_equal(new, marked):
            break
        marked = new
    out = a.copy()
    out[marked] = 4
    return out


def _matches077(pairs):
    for a, b in pairs:
        if a.shape != b.shape:
            return False
        if 2 not in set(a.ravel().tolist()):
            return False
        # only color-4 additions, sourced from a single non-0/2 fill color
        diff = a != b
        if not diff.any():
            return False
        if set(b[diff].tolist()) != {4}:
            return False
        if not np.array_equal(_sim077(a), b):
            return False
    return True


def _build077(n_iter=14):
    nodes = []
    inits = []

    # cross kernel (4 orthogonal neighbours) + bias -1 (folds the ">=2" shift)
    K = oh.make_tensor("K", DATA_TYPE, [1, 1, 3, 3],
                       [0, 1, 0, 1, 0, 1, 0, 1, 0])
    B = oh.make_tensor("B", DATA_TYPE, [1], [-1.0])
    sel4 = oh.make_tensor("sel4", DATA_TYPE, [1, CHANNELS, 1, 1],
                          [1.0 if c == 4 else 0.0 for c in range(CHANNELS)])
    one = oh.make_tensor("one", DATA_TYPE, [1], [1.0])
    inits += [K, B, sel4, one]

    def sl(name_out, start, end):
        s = oh.make_tensor(name_out + "_s", INT64, [1], [start])
        e = oh.make_tensor(name_out + "_e", INT64, [1], [end])
        ax = oh.make_tensor(name_out + "_a", INT64, [1], [1])
        inits.extend([s, e, ax])
        nodes.append(oh.make_node("Slice", ["input", name_out + "_s",
                                            name_out + "_e", name_out + "_a"],
                                  [name_out]))

    sl("ch0", 0, 1)
    sl("ch2", 2, 3)  # seed
    nodes.append(oh.make_node("ReduceSum", ["input"], ["total"], axes=[1], keepdims=1))
    nodes.append(oh.make_node("Add", ["ch0", "ch2"], ["s02"]))
    nodes.append(oh.make_node("Sub", ["total", "s02"], ["fill"]))

    # active_0 = seed ; active_{t+1} = seed + fill*clip(conv(active_t)-1,0,1)
    active = "ch2"
    mk = None
    for t in range(n_iter):
        c = f"c{t}"; g = f"g{t}"; m = f"m{t}"; an = f"a{t}"
        nodes.append(oh.make_node("Conv", [active, "K", "B"], [c],
                                  kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        nodes.append(oh.make_node("Clip", [c], [g], min=0.0, max=1.0))
        nodes.append(oh.make_node("Mul", ["fill", g], [m]))
        nodes.append(oh.make_node("Add", ["ch2", m], [an]))
        active = an
        mk = m

    # assemble output: remove marked from all channels, paint channel 4
    nodes.append(oh.make_node("Sub", ["one", mk], ["notm"]))
    nodes.append(oh.make_node("Mul", ["input", "notm"], ["removed"]))
    nodes.append(oh.make_node("Mul", [mk, "sel4"], ["addc4"]))
    nodes.append(oh.make_node("Add", ["removed", "addc4"], ["output"]))
    return _model(nodes, inits)


# ------------------------------------------------------------------ dispatch

def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return []
    out = []
    if _matches077(prs):
        out.append(("ca_grow2", _build077(14)))
    return out
