"""family_cs2_4 — FINAL COMPLETE-SWEEP minimal recompile.

Two out_blend6 incumbents are *correct in logic* but score 0 in the ORT-1.23.2
grader because the InferenceSession refuses to load them:

  * task301 (beb8660c): a uint8 Min(13) node — ORT 1.23.2 has no uint8 kernel
    for Min(13), so session construction fails ("Could not find an
    implementation for Min(13)").
  * task183 (77fdfe62): ir_version 13, which exceeds ORT 1.23.2's max
    supported IR version (11) — the model is rejected before any op runs.
    It *also* carries a uint8 Max(13) node with the same missing-kernel issue.

Both are fixed by a numerically-identical recompile:

  * uint8 Min/Max  ->  Cast operands to fp16, Min/Max in fp16, Cast back to
    uint8.  All label/coordinate values here are small non-negative integers
    that are exact in fp16, so the result is bit-for-bit identical.
  * ir_version is clamped to 10 (the value the sibling loadable incumbents
    use) so the model passes ORT's IR-version check.

Everything else — every op, initializer, and shape — is untouched, so the
patched graph produces exactly the incumbent's intended output.  Each is exact
on train+test+arc-gen and on 2000 fresh generator samples, and scores strictly
more than the broken incumbent's 0.

candidates() fingerprints the incoming example against each target task's own
train/test pairs and yields the patched graph for the matching task
(evaluate() re-gates on train+test exact anyway).
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

# task_num -> hash; both are broken-incumbent minimal-recompile wins.
_TARGETS = {
    301: "beb8660c",
    183: "77fdfe62",
}

# small-int dtypes with no Min/Max kernel in ORT 1.23.2
_BAD_MINMAX = {TP.UINT8, TP.INT8, TP.UINT16, TP.INT16, TP.UINT32, TP.UINT64}


def _patch(model):
    """Return a numerically-identical model that loads and scores under ORT 1.23.2."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    if m.ir_version > 10:
        m.ir_version = 10
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
