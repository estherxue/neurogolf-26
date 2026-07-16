"""family_cs1_6 — COMPLETE-SWEEP minimal recompile (tasks 185,392,99,224,237).

Each of these five incumbent out_blend6 graphs already encodes the correct rule but
FAILS TO LOAD under ORT 1.23.2: they route mask logic through Min/Max on uint8/int8
tensors, and ORT's CPU kernels only implement Min/Max for float/int32/int64. A graph
that does not load scores 0.

Fix: wrap every Min/Max whose (inferred) output dtype is an unsupported small int
(uint8/int8/uint16/int16) with Cast->int32 on the inputs and Cast back on the output.
Because every operand lies inside its original integer range, the int32 Min/Max is
bit-identical to the (unavailable) small-int Min/Max — the patched graph is provably
equivalent to the incumbent wherever the incumbent was correct. IR version is pinned
to 10 (237 shipped ir=13, above ORT 1.23.2's ceiling of 11).

Each patched graph verified: loads under ORT 1.23.2, exact on all local
train+test+arc-gen, and exact on 2000 FRESH generator samples per task. Incumbents
score 0 (unloadable) -> every one of these is a strict win.

The 5 sibling tasks in this sweep (238,35,55,88,213) already load and sit at
mem+params ~= 1850-1935 (points ~17.4, near the algorithmic floor); no strictly
cheaper form was found, so they are SKIPPED.
"""
from __future__ import annotations

import pathlib

import onnx
from onnx import helper as oh
from onnx import shape_inference
from onnx import TensorProto as TP

_HERE = pathlib.Path(__file__).resolve().parent
_ONNX = _HERE / "out_blend6" / "onnx"

# tasks whose incumbent is unloadable (uint8/int8 Min/Max) and thus a free win.
_TARGETS = {185: "7837ac64", 392: "f8c80d96", 99: "444801d8",
            224: "928ad970", 237: "99fa7670"}

# int/uint dtypes ORT CPU does not implement for Min/Max.
_BAD_DT = {TP.UINT8, TP.INT8, TP.UINT16, TP.INT16}
_I32 = TP.INT32


def _patch(m):
    """Cast unsupported small-int Min/Max through int32; pin IR to 10."""
    m.ir_version = 10
    mi = shape_inference.infer_shapes(m)
    dt = {v.name: v.type.tensor_type.elem_type
          for v in list(mi.graph.value_info) + list(mi.graph.input)
          + list(mi.graph.output)}
    dt.update({i.name: i.data_type for i in m.graph.initializer})

    new_nodes = []
    ctr = 0
    for n in m.graph.node:
        if n.op_type in ("Min", "Max") and dt.get(n.output[0]) in _BAD_DT:
            out_dt = dt[n.output[0]]
            cast_ins = []
            for inp in n.input:
                cn = f"__cs16_c{ctr}"
                ctr += 1
                new_nodes.append(oh.make_node("Cast", [inp], [cn], to=_I32))
                cast_ins.append(cn)
            mid = f"__cs16_m{ctr}"
            ctr += 1
            new_nodes.append(oh.make_node(n.op_type, cast_ins, [mid]))
            new_nodes.append(oh.make_node("Cast", [mid], [n.output[0]], to=out_dt))
        else:
            new_nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    return m


def _sig(examples):
    """Order-independent signature of the train pairs (exact grid contents)."""
    pairs = []
    for e in examples.get("train", []):
        pairs.append((tuple(map(tuple, e["input"])), tuple(map(tuple, e["output"]))))
    return frozenset(pairs)


def _load_targets():
    """Load each target task json to build its train signature -> patched model."""
    from ng_utils_shim import tasks_dir
    tdir = tasks_dir()
    import json
    out = {}
    for tn in _TARGETS:
        fj = tdir / f"task{tn:03d}.json"
        onx = _ONNX / f"task{tn:03d}.onnx"
        if not fj.is_file() or not onx.is_file():
            continue
        ex = json.load(open(fj))
        out[_sig(ex)] = (tn, onx)
    return out


_ROUTES = None


def candidates(examples):
    global _ROUTES
    if _ROUTES is None:
        try:
            _ROUTES = _load_targets()
        except Exception:
            _ROUTES = {}
    hit = _ROUTES.get(_sig(examples))
    if hit is None:
        return
    tn, onx = hit
    try:
        m = _patch(onnx.load(str(onx)))
    except Exception:
        return
    yield (f"cs16_{_TARGETS[tn]}", m)
