"""family_cs2_7 — FINAL COMPLETE-SWEEP recompile.

One genuine win in this sweep:

  * task 171 (hash 6f8cd79b) — "draw the border ring".  The out_blend6 incumbent
    produces correct outputs but its graph contains a Conv whose `pads` attribute
    carries a negative value.  ORT 1.23.2 runs it, yet the official scorer
    (`neurogolf_utils.calculate_memory` -> `onnx.checker.check_model(full_check=True)`)
    raises `[ShapeInferenceError] Attribute pads must not contain negative values`,
    so the model is UNSCORABLE and effectively scores 0.  A clean recompile is a
    strict win (0 -> ~16.98 pts).

The rule (verify_6f8cd79b): the input is an all-zero H x W grid (H, W in 3..9);
the output is the same grid with its outer border ring painted colour 8 and the
interior left as background 0.  In the one-hot [1,10,30,30] encoding the only
information present is the grid extent (channel-0 == 1 over the top-left HxW
block).  We:
  * slice channel 0 to a fixed 9x9 crop (max grid size) -> region mask G;
  * a 4-neighbour Conv (pads=[1,1,1,1], all non-negative) counts in-grid
    neighbours; interior = (count == 4); border = G - interior.  Both planes are
    identically zero outside the grid, so no separate gating is needed;
  * a rank-2 Einsum expands the two planes into the 10-channel one-hot block
    (border -> channel 8, interior -> channel 0), kept at 9x9 in fp16;
  * Pad restores the 30x30 canvas (zeros elsewhere).

Verified: exact on all local train+test and on 2500 fresh generator samples.

The other eleven tasks in this sweep (143, 122, 48, 221, 43, 153, 320, 178, 267,
289, 372) already load and score ~18.3-18.4 pts in the ORT-1.23.2 grader — at or
near the log-wall floor for their structure, with no genuinely cheaper graph
available — and are skipped.

candidates() fingerprints the incoming example against task 171's own train+test
pairs and yields the recompiled graph on a match (evaluate() re-gates on
train+test exact anyway).
"""
from __future__ import annotations

import json
import os

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as oh
from onnx import numpy_helper as nh

_HERE = os.path.dirname(os.path.abspath(__file__))
_F16 = TP.FLOAT16

# task_num -> hash (single border-ring recompile win from an unscorable 0-pt graph)
_TARGETS = {
    171: "6f8cd79b",
}


def _build_171():
    def c16(name, arr):
        return nh.from_array(np.asarray(arr, np.float16), name)

    def ci64(name, arr):
        return nh.from_array(np.asarray(arr, np.int64), name)

    # E[10,2]: col0 -> one-hot channel 8 (border colour); col1 -> one-hot channel 0 (interior/bg)
    E = np.zeros((10, 2), np.float16)
    E[8, 0] = 1.0
    E[0, 1] = 1.0
    # 4-neighbour kernel (self excluded)
    W4 = np.array([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], np.float16)  # [1,1,3,3]

    inits = [
        ci64("sl_st", [0, 0, 0]), ci64("sl_en", [1, 9, 9]), ci64("sl_ax", [1, 2, 3]),
        c16("W4", W4), c16("four", [4.0]), c16("E", E),
        ci64("pad", [0, 0, 0, 0, 0, 0, 21, 21]),
    ]
    nodes = [
        oh.make_node("Slice", ["input", "sl_st", "sl_en", "sl_ax"], ["g32"]),
        oh.make_node("Cast", ["g32"], ["G"], to=_F16),
        oh.make_node("Conv", ["G", "W4"], ["cnt"], pads=[1, 1, 1, 1]),
        oh.make_node("Equal", ["cnt", "four"], ["isint"]),
        oh.make_node("Cast", ["isint"], ["interior"], to=_F16),
        oh.make_node("Sub", ["G", "interior"], ["border"]),
        oh.make_node("Concat", ["border", "interior"], ["P"], axis=1),
        oh.make_node("Einsum", ["E", "P"], ["block"], equation="oc,bchw->bohw"),
        oh.make_node("Pad", ["block", "pad"], ["output"], mode="constant"),
    ]
    g = oh.make_graph(
        nodes, "border171",
        [oh.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [oh.make_tensor_value_info("output", _F16, [1, 10, 30, 30])],
        inits,
    )
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 18)])
    m.ir_version = 9
    return m


_BUILDERS = {
    171: _build_171,
}


def _sig(example):
    return json.dumps([example.get("train", []), example.get("test", [])], sort_keys=True)


_REGISTRY = {}          # signature -> task_num


def _build_registry():
    if _REGISTRY:
        return
    from ng_utils_shim import tasks_dir
    tdir = tasks_dir()
    for tn in _TARGETS:
        try:
            ex = json.load(open(tdir / f"task{tn:03d}.json"))
        except Exception:
            continue
        _REGISTRY[_sig(ex)] = tn


def candidates(example):
    _build_registry()
    tn = _REGISTRY.get(_sig(example))
    if tn is None:
        return
    try:
        yield (f"cs2_{_TARGETS[tn]}", _BUILDERS[tn]())
    except Exception:
        return
