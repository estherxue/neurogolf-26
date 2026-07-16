"""family_cs1_7 — COMPLETE-SWEEP minimal recompile for tasks
199,336,388,30,161,41,163,206,112,302.

Routing: candidates(example) fingerprints the task's train pairs (md5 of the
train grids) and yields only that task's winning graph.

Wins so far:
  * task030 (1caeab9d) and task161 (6cdd2623): the out_blend6 incumbents use a
    `Max` node over uint8/int8 tensors, which ORT 1.23.2 has NO kernel for
    (NOT_IMPLEMENTED) -> the incumbent fails to load -> scores 0 locally.  We
    patch the `Max` nodes to run over FLOAT16 (a type ORT does implement) and
    cast the result back to the original integer dtype, leaving all other
    semantics byte-identical.  The patched graph loads and is exact, so it beats
    the incumbent's 0.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import onnx
from onnx import helper as oh
from onnx import TensorProto as TP

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX_DIR = os.path.join(_HERE, "out_blend6", "onnx")

F16 = TP.FLOAT16


def _sig(example):
    s = json.dumps([[ex["input"], ex["output"]] for ex in example["train"]],
                   sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


# md5(train) -> task number
_SIG2TASK = {
    "fe99e78111127e50cdfd71b47dc92588": 199,
    "9bf3ac8f61dfb2a6426f45dbe5d611b0": 336,
    "57f7b5e43f6cbd64875027705da38780": 388,
    "a3523314e1fef3ac2510f04bce9aa8e9": 30,
    "b8d3d31899921cefa7e41bfee175f6f2": 161,
    "37fbd1d82005371466440a6f94172e45": 41,
    "2d18a85f04291341c90fca6a49f60b67": 163,
    "b798b311c02084eec5520c6686e2b261": 206,
    "316db8442e30469df68fe22e456b3cbe": 112,
    "5a985d917cb5c00e4dcec2f0f9398da0": 302,
}


def _dtype_of(model, name):
    """Best-effort element type of a tensor by name (init or value_info)."""
    for init in model.graph.initializer:
        if init.name == name:
            return init.data_type
    g = onnx.shape_inference.infer_shapes(model, strict_mode=False).graph
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        if vi.name == name and vi.type.HasField("tensor_type"):
            return vi.type.tensor_type.elem_type
    return None


def _patch_max_to_f16(model):
    """Rewrite every Max node whose inputs are non-float (ORT has no int8/uint8
    Max kernel) to: Cast(inputs->f16) -> Max -> Cast(result->original dtype)."""
    g = model.graph
    new_nodes = []
    ctr = [0]

    def uniq(p):
        ctr[0] += 1
        return f"_cs17_{p}{ctr[0]}"

    for node in list(g.node):
        if node.op_type != "Max":
            new_nodes.append(node)
            continue
        out_name = node.output[0]
        out_dt = _dtype_of(model, out_name)
        # cast every input to f16
        f16_ins = []
        for inp in node.input:
            cn = uniq("cin")
            new_nodes.append(oh.make_node("Cast", [inp], [cn], to=F16))
            f16_ins.append(cn)
        max_out = uniq("max")
        new_nodes.append(oh.make_node("Max", f16_ins, [max_out]))
        # cast back to the original output dtype so downstream is unchanged
        new_nodes.append(oh.make_node("Cast", [max_out], [out_name], to=out_dt))

    del g.node[:]
    g.node.extend(new_nodes)
    onnx.checker.check_model(model, full_check=True)
    return model


def _load_incumbent(task_num):
    return onnx.load(os.path.join(_ONNX_DIR, f"task{task_num:03d}.onnx"))


def _build_patched(task_num):
    return _patch_max_to_f16(_load_incumbent(task_num))


# task -> builder
_BUILDERS = {
    30: lambda: _build_patched(30),
    161: lambda: _build_patched(161),
}


def candidates(example):
    task = _SIG2TASK.get(_sig(example))
    if task is None:
        return
    b = _BUILDERS.get(task)
    if b is None:
        return
    try:
        yield (f"cs17_{task}", b())
    except Exception:
        return
