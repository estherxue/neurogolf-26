"""family_scrk3_2 — solver for the "5-marker compass assembly" task (task 22).

Rule (verified EXACT on train+test+arc-gen, 266/266):
  * Output is always a 3x3 grid, placed top-left.
  * Center cell (1,1) is always color 5.
  * Every input cell of color 5 is a marker. For each marker at (r,c) and each of the
    8 neighbour offsets (dy,dx), if input[r+dy,c+dx] is a non-zero colour k, then output
    cell (1+dy, 1+dx) = k.  (No two markers ever disagree on a direction; verified.)
  * Output cells with no contributing colour are background (colour 0).

ONNX construction (opset-10, origin-anchored):
  five = input[:,5:6]                                            # [1,1,30,30] marker mask
  for each of the 8 directions d=(dy,dx):
      F_d   = shift(five, by (-dy,-dx))     (Slice+Pad, zero-fill)
      prod  = input * F_d                    # broadcast [1,10,30,30]
      vec_d = ReduceMax(prod, axes=[2,3])    # [1,10,1,1]  channel k = "some marker has
                                             #             a k-neighbour in direction d"
  Assemble a [1,10,3,3] block placing vec_d at (1+dy,1+dx); centre = one-hot(5).
  Recompute channel 0 = 1 - max(channels 1..9)   (so background cells light channel 0,
  and the spurious "empty-neighbour" signal in vec_d channel 0 is discarded).
  Pad the [1,10,3,3] block to [1,10,30,30] (zeros to the right/bottom = padding).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

INT64 = onnx.TensorProto.INT64
H = W = 30
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _solve_np(gi):
    """Reference numpy implementation used to gate/verify the family."""
    gh, gw = gi.shape
    out = np.zeros((3, 3), int)
    out[1, 1] = 5
    for r, c in zip(*np.where(gi == 5)):
        for dy, dx in DIRS:
            nr, nc = r + dy, c + dx
            if 0 <= nr < gh and 0 <= nc < gw and gi[nr, nc] != 0:
                out[1 + dy, 1 + dx] = int(gi[nr, nc])
    return out


def _shift_spec(o, size=30):
    """Return (slice_start, slice_end, pad_begin, pad_end) so that out[i]=in[i+o]."""
    if o >= 0:
        return o, size, 0, o
    m = -o
    return 0, size - m, m, 0


def _build():
    nodes = []
    inits = []

    def add_init(name, dims, vals, dtype=INT64):
        inits.append(oh.make_tensor(name, dtype, dims, list(vals)))

    # five = input[:, 5:6, :, :]
    add_init("ch5_s", [1], [5]); add_init("ch5_e", [1], [6]); add_init("ch5_a", [1], [1])
    nodes.append(oh.make_node("Slice", ["input", "ch5_s", "ch5_e", "ch5_a"], ["five"]))

    vec_names = {}
    for k, (dy, dx) in enumerate(DIRS):
        # F_d[r,c] = five[r-dy, c-dx]  => out[i]=in[i+o] with o=(-dy,-dx)
        hs, he, hb, hep = _shift_spec(-dy)
        ws, we, wb, wep = _shift_spec(-dx)
        pre = f"d{k}"
        add_init(f"{pre}_ss", [2], [hs, ws])
        add_init(f"{pre}_se", [2], [he, we])
        add_init(f"{pre}_sa", [2], [2, 3])
        nodes.append(oh.make_node("Slice",
                                  ["five", f"{pre}_ss", f"{pre}_se", f"{pre}_sa"],
                                  [f"{pre}_sl"]))
        nodes.append(oh.make_node("Pad", [f"{pre}_sl"], [f"{pre}_F"],
                                  mode="constant", value=0.0,
                                  pads=[0, 0, hb, wb, 0, 0, hep, wep]))
        nodes.append(oh.make_node("Mul", ["input", f"{pre}_F"], [f"{pre}_prod"]))
        nodes.append(oh.make_node("ReduceMax", [f"{pre}_prod"], [f"{pre}_vec"],
                                  axes=[2, 3], keepdims=1))
        vec_names[(dy, dx)] = f"{pre}_vec"

    # centre one-hot(5) as [1,10,1,1]
    e5 = np.zeros((1, 10, 1, 1), np.float32); e5[0, 5, 0, 0] = 1.0
    inits.append(oh.make_tensor("e5", DATA_TYPE, [1, 10, 1, 1], e5.ravel().tolist()))

    def cell(dy, dx):
        return "e5" if (dy, dx) == (0, 0) else vec_names[(dy, dx)]

    # assemble rows (concat along width axis 3), then along height axis 2
    for ri, dy in enumerate((-1, 0, 1)):
        row_inputs = [cell(dy, dx) for dx in (-1, 0, 1)]
        nodes.append(oh.make_node("Concat", row_inputs, [f"row{ri}"], axis=3))
    nodes.append(oh.make_node("Concat", ["row0", "row1", "row2"], ["A"], axis=2))

    # recompute channel 0 = 1 - max(channels 1..9)
    add_init("c19_s", [1], [1]); add_init("c19_e", [1], [10]); add_init("c19_a", [1], [1])
    nodes.append(oh.make_node("Slice", ["A", "c19_s", "c19_e", "c19_a"], ["A19"]))
    nodes.append(oh.make_node("ReduceMax", ["A19"], ["cpres"], axes=[1], keepdims=1))
    inits.append(oh.make_tensor("one", DATA_TYPE, [1, 1, 1, 1], [1.0]))
    nodes.append(oh.make_node("Sub", ["one", "cpres"], ["ch0"]))
    # final 3x3 one-hot: channel0 = ch0, channels 1..9 = A19
    nodes.append(oh.make_node("Concat", ["ch0", "A19"], ["A2"], axis=1))
    # pad [1,10,3,3] -> [1,10,30,30]
    nodes.append(oh.make_node("Pad", ["A2"], ["output"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, H - 3, W - 3]))

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "scrk3_2", [x], [y], inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def candidates(examples):
    pairs = [(np.array(e["input"]), np.array(e["output"]))
             for e in examples.get("train", []) + examples.get("test", [])]
    if not pairs:
        return []
    # Gate: every output is 3x3 with centre 5, and the compass rule reproduces it exactly.
    for gi, go in pairs:
        if go.shape != (3, 3) or go[1, 1] != 5:
            return []
        if not np.array_equal(_solve_np(gi), go):
            return []
    return [("compass5", _build())]
