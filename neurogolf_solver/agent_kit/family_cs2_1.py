"""family_cs2_1 — FINAL COMPLETE-SWEEP recompile.

Two out_blend6 incumbents are *correct in logic* but score 0 in the ORT-1.23.2
grader because they use uint8 Min/Max, for which ORT 1.23.2 ships no kernel, so
the InferenceSession fails to load:

  * task 226 (hash 941d9a10) — two uint8 Max nodes ('ch5' 2-input, 'union'
    4-input).
  * task 303 (hash c1d99e64) — two uint8 Min nodes.

Both are fixed by a numerically-identical recompile: Cast every Min/Max input to
fp16, do the Min/Max in fp16 (all label values 0..255 are exact in fp16), Cast
the result back to uint8.  Same result bit-for-bit; fp16 is the smallest
supported dtype so the added named tensors are minimal.  Since the raw incumbent
scores 0 (unloadable), the patched graph is a strict win.

The remaining ten tasks in this sweep (159, 260, 156, 343, 134, 345, 75, 325,
217, 308) already load and score ~17.6–17.7 pts — at/near the log-wall floor with
no genuinely cheaper structure available — and are skipped.

candidates() fingerprints the incoming example against each target task's own
train+test pairs and yields the patched graph for the matching task (evaluate()
re-gates on train+test exact anyway).
"""
from __future__ import annotations

import json
import os

import onnx
from onnx import TensorProto as TP
from onnx import helper as oh
from onnx import shape_inference

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX = os.path.join(_HERE, "out_blend6", "onnx")

# task_num -> hash (both are uint8 Min/Max recompile wins from 0 pts)
_TARGETS = {
    226: "941d9a10",
    303: "c1d99e64",
}

# small-int dtypes with no Min/Max kernel in ORT 1.23.2
_BAD_MINMAX = {TP.UINT8, TP.INT8, TP.UINT16, TP.INT16, TP.UINT32, TP.UINT64}


def _patch(model):
    """Numerically-identical model with uint8 Min/Max lifted to fp16."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    mi = shape_inference.infer_shapes(m)
    dt = {vi.name: vi.type.tensor_type.elem_type
          for vi in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output)}
    for init in m.graph.initializer:
        dt[init.name] = init.data_type

    new_nodes = []
    k = 0
    for n in m.graph.node:
        if n.op_type in ("Min", "Max") and dt.get(n.output[0]) in _BAD_MINMAX:
            odt = dt[n.output[0]]
            cast_ins = []
            for inp in n.input:
                cn = f"_mm_c{k}"; k += 1
                new_nodes.append(oh.make_node("Cast", [inp], [cn], to=TP.FLOAT16))
                cast_ins.append(cn)
            tmp = f"_mm_t{k}"; k += 1
            new_nodes.append(oh.make_node(n.op_type, cast_ins, [tmp]))
            new_nodes.append(oh.make_node("Cast", [tmp], [n.output[0]], to=odt))
            continue
        new_nodes.append(n)

    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    return m


def _sig(example):
    return json.dumps([example.get("train", []), example.get("test", [])], sort_keys=True)


_REGISTRY = {}          # signature -> (task_num, hash)


def _build_registry():
    if _REGISTRY:
        return
    from ng_utils_shim import tasks_dir
    tdir = tasks_dir()
    for tn, h in _TARGETS.items():
        p = tdir / f"task{tn:03d}.json"
        try:
            ex = json.load(open(p))
        except Exception:
            continue
        _REGISTRY[_sig(ex)] = (tn, h)


def candidates(example):
    _build_registry()
    hit = _REGISTRY.get(_sig(example))
    if not hit:
        return
    tn, h = hit
    try:
        raw = onnx.load(os.path.join(_ONNX, f"task{tn:03d}.onnx"))
        yield (f"cs2_{h}", _patch(raw))
    except Exception:
        return
