"""family_cs1_1 — COMPLETE-SWEEP minimal recompile.

Six out_blend6 incumbents are *correct in logic* but score 0 in the ORT-1.23.2
grader because they use ops the runtime / scorer rejects:

  * uint8 Min/Max  (tasks 370, 9, 328, 382, 279) — ORT 1.23.2 has no uint8
    kernel for Min(13)/Max(13); the InferenceSession fails to load.
  * a Conv with NEGATIVE pads [1,1,-11,-11]  (task 278) — the scorer's
    onnx.checker.check_model(full_check=True) rejects negative pad attributes,
    so score_network raises and the model is unscorable.

Both are fixed by a *numerically identical* recompile:

  * uint8 Min/Max -> Cast(fp16) inputs, Min/Max in fp16 (all label values
    0..255 are exact in fp16), Cast back to uint8.  Same result, and fp16 is
    the smallest supported dtype so the added named tensors are minimal.
  * negative-pad Conv -> the same Conv with pads clamped to >=0, followed by a
    Slice that crops exactly the rows/cols the negative pad would have dropped
    (pads[..,-k] == crop k from that end).  Mathematically identical.

Each patched graph is bit-for-bit equal to the raw incumbent's runtime output
(verified on 3000 samples for 278) and exact on train+test+arc-gen and on 2000
fresh generator samples (278: 1999/2000 — the single miss is the incumbent's
own learned-conv approximation, reproduced faithfully).

candidates() fingerprints the incoming example against each target task's own
train pairs and yields the patched graph for the matching task (evaluate()
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

# task_num -> hash (for naming); all six are minimal-recompile wins.
_TARGETS = {
    370: "e8dc4411",
    9: "06df4c85",
    328: "d22278a0",
    382: "f15e1fac",
    279: "b2862040",
    278: "b27ca6d3",
}

# uint8 / small-int dtypes with no Min/Max kernel in ORT 1.23.2
_BAD_MINMAX = {TP.UINT8, TP.INT8, TP.UINT16, TP.INT16, TP.UINT32, TP.UINT64}
_BIG = 1 << 30


def _patch(model):
    """Return a numerically-identical model that loads and scores under ORT 1.23.2."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    mi = shape_inference.infer_shapes(m)
    dt = {vi.name: vi.type.tensor_type.elem_type
          for vi in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output)}
    for init in m.graph.initializer:
        dt[init.name] = init.data_type

    new_nodes = []
    extra_inits = []
    k = 0
    for n in m.graph.node:
        # ---- uint8 Min/Max -> fp16 Min/Max ----
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

        # ---- negative-pad Conv -> clamped Conv + Slice crop ----
        if n.op_type == "Conv":
            pads = None
            for a in n.attribute:
                if a.name == "pads":
                    pads = list(a.ints)
            if pads is not None and len(pads) == 4 and any(p < 0 for p in pads):
                hb, wb, he, we = pads          # H_begin, W_begin, H_end, W_end
                clamped = [max(0, p) for p in pads]
                keep = [a for a in n.attribute if a.name != "pads"]
                conv_out = f"_cv_full{k}"; k += 1
                cv = oh.make_node("Conv", list(n.input), [conv_out], pads=clamped)
                cv.attribute.extend(keep)
                new_nodes.append(cv)
                starts = [max(0, -hb), max(0, -wb)]
                ends = [(-max(0, -he) if he < 0 else _BIG),
                        (-max(0, -we) if we < 0 else _BIG)]
                sn, en, an = f"_cv_s{k}", f"_cv_e{k}", f"_cv_a{k}"; k += 1
                extra_inits += [
                    oh.make_tensor(sn, TP.INT64, [2], starts),
                    oh.make_tensor(en, TP.INT64, [2], ends),
                    oh.make_tensor(an, TP.INT64, [2], [2, 3]),
                ]
                new_nodes.append(oh.make_node("Slice", [conv_out, sn, en, an], [n.output[0]]))
                continue

        new_nodes.append(n)

    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    m.graph.initializer.extend(extra_inits)
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
        yield (f"cs1_{h}", _patch(raw))
    except Exception:
        return
