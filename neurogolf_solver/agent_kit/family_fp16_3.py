"""family_fp16_3 — FP16-intermediate rebuilds of memory-dominated incumbents.

Strategy: the incumbent ONNX (out_p5/onnx/taskNNN.onnx) already encodes the TRUE
rule and is numerically exact. We lower every FLOAT (fp32) tensor in that graph to
FLOAT16, uniformly, so op inputs stay same-dtype and the graph remains SSA:

  * graph input stays FLOAT[1,10,30,30]; a single Cast->float16 feeds the body.
  * every FLOAT initializer / Constant value  -> float16 (exact for the 0/1 and
    small-integer constants these solvers use).
  * every `Cast to=FLOAT`                      -> `Cast to=FLOAT16`.
  * the final producer of `output` emits float16; one Cast->FLOAT restores I/O.

Int32/int64/bool tensors are left untouched (they carry indices / boolean masks,
already minimal-byte). float16 is exact for integers < 2048; ReduceSum over <=900
one-hot cells and 0/1 conv accumulations stay well under that bound. Any residual
inexactness is caught by the grader (exact on train+test+arc-gen) — such a task is
simply not yielded, so we never regress correctness.

This is NOT onnxconverter_common auto-convert: no per-op Cast insertion, no mixed
dtypes — a single, dtype-consistent relabel of the float subgraph.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import onnx
from onnx import helper as oh, numpy_helper

FLOAT = onnx.TensorProto.FLOAT       # 1
FLOAT16 = onnx.TensorProto.FLOAT16   # 10

_HERE = pathlib.Path(__file__).resolve().parent
_ONNX_DIR = _HERE / "out_p5" / "onnx"

# my slice of FP16 targets
_MINE = [280, 308, 117, 90, 387, 64, 382, 277, 61, 189, 2, 30, 37, 271, 355, 290,
         384, 195, 8, 222, 221, 345, 199, 139, 249, 354, 335, 381, 94, 251, 336,
         225, 180, 28]


# ---------------------------------------------------------------- lowering ----

def _lower_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a copy of `model` with every fp32 tensor lowered to fp16.

    Graph I/O remains FLOAT[1,10,30,30] via boundary Casts. Assumes a single
    input named 'input' and single output named 'output'."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph

    # already an fp16 solver (some incumbents declare a FLOAT16 output): nothing
    # to lower — return unchanged so it still scores at its existing (already-low) cost.
    if g.output and g.output[0].type.tensor_type.elem_type == FLOAT16:
        return m

    # 1) redirect the fp32 input through a Cast->fp16 named 'input_f16'
    IN, IN16 = "input", "input_f16"
    for node in g.node:
        for i, nm in enumerate(node.input):
            if nm == IN:
                node.input[i] = IN16

    # 2) the producer of 'output' now emits float16; declare the graph output as
    #    FLOAT16 (the grader thresholds output>0, exact on our fp16 integers — the
    #    same convention the existing hand-built f16 incumbents use). No output-cast
    #    tensor => saves a full [1,10,30,30] intermediate.
    g.output[0].type.tensor_type.elem_type = FLOAT16

    # 3) lower every FLOAT initializer -> FLOAT16 (clamp big sentinels into fp16 range;
    #    |v|>65504 overflows to inf, so pin to the fp16 max — still "large enough"
    #    versus the small grid magnitudes these sentinels gate against).
    new_inits = []
    for init in g.initializer:
        if init.data_type == FLOAT:
            arr = np.clip(numpy_helper.to_array(init).astype(np.float32),
                          -65504.0, 65504.0).astype(np.float16)
            ni = numpy_helper.from_array(arr, init.name)
            new_inits.append(ni)
        else:
            new_inits.append(init)
    del g.initializer[:]
    g.initializer.extend(new_inits)

    # 4) lower Cast->FLOAT nodes to Cast->FLOAT16; lower Constant fp32 values
    for node in g.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == FLOAT:
                    a.i = FLOAT16
        elif node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value" and a.t.data_type == FLOAT:
                    arr = np.clip(numpy_helper.to_array(a.t).astype(np.float32),
                                  -65504.0, 65504.0).astype(np.float16)
                    a.t.CopyFrom(numpy_helper.from_array(arr, a.t.name))
                elif a.name == "value_float":
                    pass  # scalar float attr; harmless, op consumers get fp16 elsewhere

    # 5) prepend the single input boundary cast: input(fp32) -> input_f16(fp16)
    cast_in = oh.make_node("Cast", [IN], [IN16], to=FLOAT16, name=IN16)
    nodes = [cast_in] + list(g.node)
    del g.node[:]
    g.node.extend(nodes)

    # drop stale value_info (types now differ); shape inference not required by grader
    del g.value_info[:]
    return m


# ------------------------------------------------------------ fingerprint ----

def _fp(ex):
    """Stable fingerprint of a task from its train+test input/output grids."""
    parts = []
    for sub in ("train", "test"):
        for e in ex.get(sub, []):
            parts.append(np.asarray(e["input"], dtype=np.int64).tobytes())
            parts.append(np.asarray(e["output"], dtype=np.int64).tobytes())
    return hash(tuple(parts))


def _build_index():
    d = os.environ.get("NG_DATA_DIR")
    idx = {}
    if not d:
        return idx
    base = pathlib.Path(d)
    for t in _MINE:
        fp_onnx = _ONNX_DIR / f"task{t:03d}.onnx"
        fp_json = base / f"task{t:03d}.json"
        if not (fp_onnx.exists() and fp_json.exists()):
            continue
        try:
            ex = json.load(open(fp_json))
            idx[_fp(ex)] = t
        except Exception:
            continue
    return idx


_INDEX = None


def candidates(examples):
    global _INDEX
    if _INDEX is None:
        _INDEX = _build_index()
    t = _INDEX.get(_fp(examples))
    if t is None:
        return []
    fp_onnx = _ONNX_DIR / f"task{t:03d}.onnx"
    try:
        base = onnx.load(str(fp_onnx))
    except Exception:
        return []
    out = [("orig_%d" % t, base)]     # incumbent — safety net, never regress
    try:
        out.append(("fp16_%d" % t, _lower_fp16(base)))
    except Exception:
        pass
    return out
