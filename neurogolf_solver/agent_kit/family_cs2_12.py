"""family_cs2_12 — FINAL COMPLETE-SWEEP minimal recompile.

Three out_blend6 incumbents are *correct in logic* but score 0 under the
ORT-1.23.2 grader because they never survive scoring:

  * task266 (a9f96cdd): first Conv carries negative pads [1,1,-26,-24].
    ORT runs it fine, but score_network's ``onnx.checker.check_model(full_check=True)``
    rejects a Conv whose ``pads`` contain negative values -> memory/params are
    never computed and the model scores 0.
  * task357 (e179c5f4): a uint8 Min(13) node ("path_col") and a final
    ConvInteger(10) projection.  ORT 1.23.2 has no uint8 Min kernel and no
    ConvInteger kernel, so the InferenceSession refuses to load -> 0.
  * task3   (017c7c7b): a final ConvInteger(10) projection -> same load
    failure -> 0.

Each is fixed by a numerically-identical recompile:

  * Negative-pad Conv -> crop the input to the convolution's true receptive
    field with a Slice, then run the same Conv with the negative pads clamped
    to 0.  Bit-for-bit identical output over the whole grid.
  * uint8 Min -> Cast operands to fp16, Min in fp16, Cast back.  All values
    are tiny non-negative integers (exact in fp16).
  * ConvInteger(x, w[, x_zp]) -> Cast x to fp16, subtract x_zp (fp16), Conv
    against an fp16 copy of w.  ConvInteger's 1x1 integer kernels here operate
    on {0,1,2}-valued codes and {-1,0,1} int8 weights, all exact in fp16, and
    its zero-point padding coincides with Conv's zero padding after the
    subtraction, so the result matches exactly.  The final output tensor
    becomes fp16; the grader only reads ``out > 0`` so this is irrelevant.

Everything else — every other op, initializer and shape — is untouched.  Each
patched graph is exact on train+test+arc-gen and on 2000 fresh generator
samples, and scores strictly more than the broken incumbent's 0:

    task266  0 -> 17.85    task357  0 -> 18.38    task3  0 -> 19.09

candidates() fingerprints the incoming example against each target task's own
train/test pairs and yields the patched graph for the matching task
(evaluate() re-gates on train+test exact anyway).
"""
from __future__ import annotations

import json
import os

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as oh
from onnx import numpy_helper as nh
from onnx import shape_inference

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX = os.path.join(_HERE, "out_blend6", "onnx")

# task_num -> hash; broken-incumbent minimal-recompile win.
_TARGETS = {
    266: "a9f96cdd",
    357: "e179c5f4",
    3: "017c7c7b",
}

# small-int dtypes with no Min/Max kernel in ORT 1.23.2
_BAD_MINMAX = {TP.UINT8, TP.INT8, TP.UINT16, TP.INT16, TP.UINT32, TP.UINT64, TP.INT64}


def _init_map(g):
    return {i.name: i for i in g.initializer}


def _patch(model):
    """Return a numerically-identical model that loads and scores under ORT 1.23.2."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    if m.ir_version > 10:
        m.ir_version = 10
    g = m.graph

    mi = shape_inference.infer_shapes(m)
    dt = {vi.name: vi.type.tensor_type.elem_type
          for vi in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output)}
    shp = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
           for vi in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output)}
    inits = _init_map(g)
    for init in g.initializer:
        dt[init.name] = init.data_type

    new_nodes = []
    k = 0
    for n in g.node:
        # --- uint8/int Min/Max -> fp16 ---
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

        # --- ConvInteger -> fp16 Conv ---
        if n.op_type == "ConvInteger":
            x, w = n.input[0], n.input[1]
            zp = n.input[2] if len(n.input) > 2 else None
            pads = ks = None
            for a in n.attribute:
                if a.name == "pads":
                    pads = list(a.ints)
                if a.name == "kernel_shape":
                    ks = list(a.ints)
            xin = f"_ci_x{k}"; k += 1
            new_nodes.append(oh.make_node("Cast", [x], [xin], to=TP.FLOAT16))
            if zp is not None:
                zpf = f"_ci_z{k}"; k += 1
                new_nodes.append(oh.make_node("Cast", [zp], [zpf], to=TP.FLOAT16))
                xm = f"_ci_m{k}"; k += 1
                new_nodes.append(oh.make_node("Sub", [xin, zpf], [xm]))
                xin = xm
            warr = nh.to_array(inits[w]).astype(np.float16)
            wn = f"_ci_w{k}"; k += 1
            g.initializer.append(nh.from_array(warr, wn))
            new_nodes.append(oh.make_node("Conv", [xin, wn], [n.output[0]],
                                          kernel_shape=ks, pads=pads))
            g.output[0].type.tensor_type.elem_type = TP.FLOAT16
            continue

        # --- Conv with negative pads -> crop input to receptive field ---
        if n.op_type == "Conv":
            pads = ks = None
            for a in n.attribute:
                if a.name == "pads":
                    pads = list(a.ints)
                if a.name == "kernel_shape":
                    ks = list(a.ints)
            if pads is not None and any(p < 0 for p in pads):
                pt, pl, pb, pr = pads
                kh, kw = ks
                out_h, out_w = shp[n.output[0]][2], shp[n.output[0]][3]
                need_h = out_h - pt + kh - 1
                need_w = out_w - pl + kw - 1
                cs = nh.from_array(np.array([0, 0], np.int64), f"_cs{k}")
                ce = nh.from_array(np.array([need_h, need_w], np.int64), f"_ce{k}")
                ca = nh.from_array(np.array([2, 3], np.int64), f"_ca{k}")
                g.initializer.extend([cs, ce, ca])
                cropped = f"_crop{k}"; k += 1
                new_nodes.append(oh.make_node(
                    "Slice", [n.input[0], cs.name, ce.name, ca.name], [cropped]))
                nn = oh.make_node("Conv", [cropped] + list(n.input[1:]),
                                  [n.output[0]], kernel_shape=ks, pads=[pt, pl, 0, 0])
                new_nodes.append(nn)
                continue

        new_nodes.append(n)

    del g.node[:]
    g.node.extend(new_nodes)

    used = {i for nd in g.node for i in nd.input}
    keep = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
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
