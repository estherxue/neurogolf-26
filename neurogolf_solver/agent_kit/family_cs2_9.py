"""family_cs2_9 — FINAL COMPLETE-SWEEP recompile for tasks
68,200,126,332,59,355,147,300,214,362,253,146.

Routing: candidates(example) fingerprints the task's train pairs (md5 of the
train grids) and yields only that task's winning graph.

Wins (all three incumbents score 0 *locally* under ORT 1.23.2 + the grader's
own score_network path; we emit an arithmetically-identical graph that loads
and is scorable):

  * task214 (8e5a5113): incumbent has ir_version=13, but ORT 1.23.2 refuses to
    load any model with IR > 11 ("Unsupported model IR version: 13") -> fails to
    load -> 0.  Fix: set ir_version=10 (its opset is 13, needs only IR>=7).
    Byte-identical graph otherwise.  0 -> 18.74.  2500/2500 fresh gen.

  * task200 (8403a5d5): the first Conv uses NEGATIVE pads [-9,0,-20,-20] (a
    1x1-kernel channel-collapse that also crops the 30x30 grid to row 9,
    cols 0..9).  onnx.checker.check_model(full_check=True) — invoked inside the
    grader's calculate_memory — rejects negative Conv pads with a
    ShapeInferenceError, so the incumbent is UNSCORABLE -> 0.  Fix: crop the
    input with a Slice (rows[9:10], cols[0:10]) then run the same 1x1 Conv with
    pads=0.  Identical output.  0 -> 18.06.  2500/2500 fresh gen.

  * task146 (662c240a): the checksum Conv uses NEGATIVE pads [0,0,-21,0]
    (kernel 3x3, strides [3,30]; the -21 bottom crop keeps only input rows 0..8,
    yielding 3 band-checksums).  Same negative-pad check_model rejection -> 0.
    Fix: run the Conv with pads=0 over the full grid (10 band-checksums) then
    Slice the first 3.  First 3 outputs are bit-identical; the extra full-conv
    intermediate is only [1,1,10,1] = 10 floats.  0 -> 18.71.  2500/2500 fresh gen.

Skips (incumbent loads, is exact, and is at the floor for its structure —
18.5..18.8 pts, mem<600, params<300; no fundamentally cheaper structure):
  68, 126, 332, 59, 355, 147, 300, 362, 253.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import onnx
from onnx import helper as oh
from onnx import numpy_helper as nh

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX_DIR = os.path.join(_HERE, "out_blend6", "onnx")


def _sig(example):
    s = json.dumps([[ex["input"], ex["output"]] for ex in example["train"]],
                   sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


# md5(train) -> task number
_SIG2TASK = {}


def _register_sig():
    """Populate _SIG2TASK from the local task json files (train pairs)."""
    tdir = None
    d = os.environ.get("NG_DATA_DIR")
    if d:
        for cand in (os.path.join(d, "tasks"), d):
            if os.path.isdir(cand) and any(
                    f.startswith("task") for f in os.listdir(cand)):
                tdir = cand
                break
    if tdir is None:
        return
    for t in (214, 200, 146):
        p = os.path.join(tdir, f"task{t:03d}.json")
        if os.path.isfile(p):
            ex = json.load(open(p))
            _SIG2TASK[_sig(ex)] = t


_register_sig()


def _add_init(g, name, arr):
    g.initializer.append(nh.from_array(np.asarray(arr), name))


def _build_214():
    """ir_version 13 -> 10 so ORT 1.23.2 will load it (max IR = 11)."""
    m = onnx.load(os.path.join(_ONNX_DIR, "task214.onnx"))
    m.ir_version = 10
    onnx.checker.check_model(m, full_check=True)
    return m


def _build_200():
    """Replace the negative-pad Conv (unscorable) with Slice(crop)->Conv(pads=0)."""
    m = onnx.load(os.path.join(_ONNX_DIR, "task200.onnx"))
    g = m.graph
    new = []
    for nd in g.node:
        if nd.op_type == "Conv" and nd.output[0] == "seed32":
            _add_init(g, "_cs29_st", np.array([9, 0], dtype=np.int64))
            _add_init(g, "_cs29_en", np.array([10, 10], dtype=np.int64))
            _add_init(g, "_cs29_ax", np.array([2, 3], dtype=np.int64))
            new.append(oh.make_node(
                "Slice", ["input", "_cs29_st", "_cs29_en", "_cs29_ax"],
                ["_cs29_crop"]))
            new.append(oh.make_node(
                "Conv", ["_cs29_crop", nd.input[1]], ["seed32"], pads=[0, 0, 0, 0]))
        else:
            new.append(nd)
    del g.node[:]
    g.node.extend(new)
    onnx.checker.check_model(m, full_check=True)
    return m


def _build_146():
    """Replace the negative-pad checksum Conv with full Conv(pads=0)->Slice."""
    m = onnx.load(os.path.join(_ONNX_DIR, "task146.onnx"))
    g = m.graph
    new = []
    for nd in g.node:
        if nd.op_type == "Conv" and nd.output[0] == "checks":
            ks = list(next(a.ints for a in nd.attribute if a.name == "kernel_shape"))
            st = list(next(a.ints for a in nd.attribute if a.name == "strides"))
            new.append(oh.make_node(
                "Conv", [nd.input[0], nd.input[1]], ["_cs29_full"],
                kernel_shape=ks, strides=st, pads=[0, 0, 0, 0]))
            _add_init(g, "_cs29_st", np.array([0], dtype=np.int64))
            _add_init(g, "_cs29_en", np.array([3], dtype=np.int64))
            _add_init(g, "_cs29_ax", np.array([2], dtype=np.int64))
            new.append(oh.make_node(
                "Slice", ["_cs29_full", "_cs29_st", "_cs29_en", "_cs29_ax"],
                ["checks"]))
        else:
            new.append(nd)
    del g.node[:]
    g.node.extend(new)
    onnx.checker.check_model(m, full_check=True)
    return m


_BUILDERS = {214: _build_214, 200: _build_200, 146: _build_146}


def candidates(example):
    task = _SIG2TASK.get(_sig(example))
    if task is None:
        return
    b = _BUILDERS.get(task)
    if b is None:
        return
    try:
        yield (f"cs29_{task}", b())
    except Exception:
        return
