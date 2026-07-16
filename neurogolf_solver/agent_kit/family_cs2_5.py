"""family_cs2_5 — FINAL COMPLETE-SWEEP recompile for tasks
149,228,194,288,239,195,32,97,120,139,193,287.

Routing: candidates(example) fingerprints the task's train pairs (md5 of the
train grids) and yields only that task's winning graph.

Wins:
  * task139 (60b61512): the out_blend6 incumbent's final node is `ConvInteger`,
    which ORT 1.23.2 has NO kernel for (NOT_IMPLEMENTED) -> the incumbent fails
    to load -> scores 0 locally.  ConvInteger here is a 1x1 integer convolution
    (a per-pixel linear channel remap of features -> 10 one-hot output channels,
    with a `pads` that also expands the 9x9 working grid up to 30x30).  We patch
    it to Cast(features->FLOAT) -> Conv(float W) -> FLOAT output, byte-identical
    arithmetic (small integer values are exact in float), so the graph loads and
    is exact.  Verified 2500/2500 fresh generator samples.  17.46 pts vs 0.

Skips (incumbent already loads, is exact, and is at the floor for its structure):
  * 32,97,120,193,287: single-node graphs, mem=0, 910 params (18.19 pts).  Their
    true rules (local recolor conv / D4 max-symmetrization) genuinely need a
    dense per-channel/per-position table; any multi-node rewrite would add named
    intermediates whose static 30x30 memory dwarfs the 910-param saving.
  * 149,228,194,288,239,195: load, exact, ~18.1 pts, already tiny (params<200,
    mem<900).  No fundamentally cheaper structure.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import onnx
from onnx import helper as oh
from onnx import numpy_helper as nh
from onnx import TensorProto as TP

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
    for env in ("NG_DATA_DIR",):
        d = os.environ.get(env)
        if d:
            for cand in (os.path.join(d, "tasks"), d):
                if os.path.isdir(cand) and any(
                        f.startswith("task") for f in os.listdir(cand)):
                    tdir = cand
                    break
    if tdir is None:
        return
    for t in (139,):
        p = os.path.join(tdir, f"task{t:03d}.json")
        if os.path.isfile(p):
            ex = json.load(open(p))
            _SIG2TASK[_sig(ex)] = t


_register_sig()


def _patch_convinteger_to_float(model):
    """Rewrite the ConvInteger node (unimplemented in ORT) to
    Cast(features->FLOAT) -> Conv(float weight) -> FLOAT output."""
    g = model.graph
    # locate the int8 weight of the ConvInteger to float-cast it once
    conv = next(n for n in g.node if n.op_type == "ConvInteger")
    feats, wname = conv.input[0], conv.input[1]
    pads = list(next(a.ints for a in conv.attribute if a.name == "pads"))

    # float copy of the weight; drop the now-unused int8 initializer
    w_arr = None
    keep = []
    for init in list(g.initializer):
        if init.name == wname:
            w_arr = nh.to_array(init).astype(np.float32)
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    wf = "_cs25_Wf"
    g.initializer.append(nh.from_array(w_arr, wf))

    new_nodes = []
    for n in g.node:
        if n.op_type == "ConvInteger":
            new_nodes.append(oh.make_node("Cast", [feats], ["_cs25_featf"],
                                          to=TP.FLOAT))
            new_nodes.append(oh.make_node("Conv", ["_cs25_featf", wf],
                                          [n.output[0]], pads=pads))
        else:
            new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    g.output[0].type.tensor_type.elem_type = TP.FLOAT
    onnx.checker.check_model(model, full_check=True)
    return model


def _build_139():
    m = onnx.load(os.path.join(_ONNX_DIR, "task139.onnx"))
    return _patch_convinteger_to_float(m)


_BUILDERS = {139: _build_139}


def candidates(example):
    task = _SIG2TASK.get(_sig(example))
    if task is None:
        return
    b = _BUILDERS.get(task)
    if b is None:
        return
    try:
        yield (f"cs25_{task}", b())
    except Exception:
        return
