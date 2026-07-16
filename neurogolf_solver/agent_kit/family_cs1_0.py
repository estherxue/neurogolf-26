"""family_cs1_0 — COMPLETE-SWEEP minimal recompile.

Six of the ten targeted incumbents (tasks 350, 338, 324, 216, 96, 77) were built
with uint8 `Min`/`Max` nodes, which ONNX Runtime 1.23.2 (the local/grading engine)
does NOT implement -> the graph fails to load and scores ZERO.  This module ships
a loadable, verified-exact replacement for each:

  * task350 (dbc1a6ce): a fresh hand-built float16 flood.  The incumbent's own
    MaxPool-flood + Min/Max structure, but the mask tensor is float16 instead of
    uint8 so Min/Max load.  Cheaper than the wrap (mem 14004, ~15.45 pts).
  * tasks 338/324/216/96/77: the incumbent graph, byte-for-byte, with every uint8
    Min/Max wrapped `Cast->float16 -> Min/Max -> Cast->uint8`.  Values are 0..255
    masks, so float16 is exact; results are identical to the (broken) incumbent,
    which already verifies exact on train+test+arc-gen and thousands of fresh
    ARC-GEN samples.  float16 wrapping (vs int32) keeps the added tensors small.

The four remaining targets (367, 74, 379, 89) already LOAD and sit at a tight
per-cell working-set floor (dense label grid + genuine bounded-flood chain); no
strictly-cheaper legal form was found, so they are skipped.

candidates() fingerprints the incoming task by its train pairs and yields only the
matching task's graph (gated: family_test re-checks train+test exact before scoring).
"""
from __future__ import annotations

import json
import os

import numpy as np
import onnx
from onnx import helper as oh
from onnx import TensorProto as TP

from ng_utils_shim import IR_VERSION, tasks_dir

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX = os.path.join(_HERE, "out_blend6", "onnx")

F16 = TP.FLOAT16

# task_num -> generator hash (documentation / provenance)
_HASH = {350: "dbc1a6ce", 338: "d5d6de2d", 324: "d07ae81c",
         216: "8efcae92", 96: "4290ef0e", 77: "36fdfd69"}

# tasks whose fix is: reuse the incumbent graph, wrap uint8 Min/Max in float16.
_WRAP = [338, 324, 216, 96, 77]


def _wrap_minmax_f16(model):
    """Replace every uint8-unsupported Min/Max node with
    Cast(f16)->Min/Max->Cast(uint8).  Lossless for 0..255 integer masks."""
    g = model.graph
    new_nodes = []
    k = 0
    for n in g.node:
        if n.op_type in ("Min", "Max"):
            ins = []
            for i in n.input:
                ci = f"__cs1w{k}"
                k += 1
                new_nodes.append(oh.make_node("Cast", [i], [ci], to=F16))
                ins.append(ci)
            tmp = f"__cs1w{k}"
            k += 1
            new_nodes.append(oh.make_node(n.op_type, ins, [tmp]))
            new_nodes.append(oh.make_node("Cast", [tmp], [n.output[0]], to=TP.UINT8))
        else:
            new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    return model


def _build_wrapped(task_num):
    m = onnx.load(os.path.join(_ONNX, f"task{task_num:03d}.onnx"))
    return _wrap_minmax_f16(m)


def _build_350():
    """dbc1a6ce: a background cell becomes cyan(8) iff it lies strictly between two
    blue(1) cells in its row OR in its column; blue cells are preserved.  MaxPool
    floods (kernel = full extent, one-sided pad) give inclusive 'blue reachable in
    this direction'; Min = both sides, Max = row-or-column.  All in float16 so the
    Min/Max load.  `Where(mask, colour8_vec, input)` keeps every non-cyan cell
    (blue, black, and off-grid) exactly as the input, so no extra masking needed."""
    nodes, inits = [], []

    def C(name, dt, dims, vals):
        inits.append(oh.make_tensor(name, dt, dims, np.asarray(vals).ravel().tolist()))
        return name

    C("s", TP.INT64, [4], [0, 1, 0, 0])
    C("e", TP.INT64, [4], [1, 2, 26, 24])
    nodes.append(oh.make_node("Slice", ["input", "s", "e"], ["c"]))
    nodes.append(oh.make_node("Cast", ["c"], ["m"], to=F16))
    nodes.append(oh.make_node("MaxPool", ["m"], ["fL"],
                              kernel_shape=[1, 24], pads=[0, 23, 0, 0], strides=[1, 1]))
    nodes.append(oh.make_node("MaxPool", ["m"], ["fR"],
                              kernel_shape=[1, 24], pads=[0, 0, 0, 23], strides=[1, 1]))
    nodes.append(oh.make_node("Min", ["fL", "fR"], ["sH"]))
    nodes.append(oh.make_node("MaxPool", ["m"], ["fU"],
                              kernel_shape=[26, 1], pads=[25, 0, 0, 0], strides=[1, 1]))
    nodes.append(oh.make_node("MaxPool", ["m"], ["fD"],
                              kernel_shape=[26, 1], pads=[0, 0, 25, 0], strides=[1, 1]))
    nodes.append(oh.make_node("Min", ["fU", "fD"], ["sV"]))
    nodes.append(oh.make_node("Max", ["sH", "sV"], ["sp"]))
    nodes.append(oh.make_node("Greater", ["sp", "m"], ["g"]))  # between AND not-blue
    C("p", TP.INT64, [4], [0, 0, 4, 6])
    C("ax", TP.INT64, [2], [2, 3])
    C("zpad", TP.BOOL, [1], [0])
    nodes.append(oh.make_node("Pad", ["g", "p", "zpad", "ax"], ["gp"], mode="constant"))
    C("f8", TP.FLOAT, [1, 10, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    nodes.append(oh.make_node("Where", ["gp", "f8", "input"], ["output"]))

    x = oh.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])
    y = oh.make_tensor_value_info("output", TP.FLOAT, [1, 10, 30, 30])
    g = oh.make_graph(nodes, "cs1_350", [x], [y], inits)
    return oh.make_model(g, ir_version=IR_VERSION,
                         opset_imports=[oh.make_operatorsetid("", 18)])


_BUILDERS = {350: _build_350}
for _t in _WRAP:
    _BUILDERS[_t] = (lambda tn: (lambda: _build_wrapped(tn)))(_t)


# --------------------------------------------------------------------------- #
# routing: fingerprint a task by its exact train pairs                         #
# --------------------------------------------------------------------------- #
def _load_train_sigs():
    sigs = {}
    try:
        tdir = tasks_dir()
    except Exception:
        return sigs
    for tn in _BUILDERS:
        try:
            ex = json.load(open(tdir / f"task{tn:03d}.json"))
            sigs[_sig(ex["train"])] = tn
        except Exception:
            pass
    return sigs


def _sig(train):
    return repr([(e["input"], e["output"]) for e in train])


_SIGS = _load_train_sigs()


def candidates(example):
    tn = _SIGS.get(_sig(example.get("train", [])))
    if tn is None:
        return
    try:
        yield (f"cs1_{_HASH[tn]}", _BUILDERS[tn]())
    except Exception:
        return
