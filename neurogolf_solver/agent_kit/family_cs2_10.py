"""family_cs2_10 — FINAL COMPLETE-SWEEP minimal recompile.

Two out_blend6 incumbents are *correct in logic* (exact on train+test+arc-gen and
the ORT-1.23.2 session runs them fine) but score 0 in the grader because their
final node is a ConvTranspose with NEGATIVE pad values:

  * task384 (f25fbde4): ConvTranspose pads=[0,0,-22,-20]  (upsample [1,1,4,5] by
    2 -> [1,10,8,10], then extend the output canvas to 30x30).
  * task127 (54d9e175): ConvTranspose pads=[1,1,-23,-19], group=5, strides=4
    (crop 1 row/col off the begin, extend the end to reach 30x30).

`onnx.checker.check_model(full_check=True)` — which the grader calls inside
score_network/calculate_memory — rejects any ConvTranspose whose `pads` contain a
negative value ("Attribute pads must not contain negative values"), so scoring
raises and the task earns 0 points even though the network is exact.

Fix (numerically identical, verified in ORT 1.23.2): a negative pad on a
ConvTranspose output means "extend that side of the output canvas with zeros",
which is exactly a trailing constant-0 Pad. So we split the node:

  ConvTranspose(pads=p)  ->  ConvTranspose(pads=max(p,0))  ->  Pad(zeros = -min(p,0))

The clamped ConvTranspose keeps every non-negative (cropping) pad, and the Pad
adds back the zero-extension the negative pads asked for. This was checked to be
bit-for-bit identical to the original negative-pad node on random inputs for both
tasks' exact geometries. The only new named intermediate is the ConvTranspose's
full (un-extended) output; everything else is untouched.

candidates() fingerprints the incoming example against the target task's own
train/test pairs and yields the patched graph for the matching task
(evaluate() re-gates on train+test exact anyway).
"""
from __future__ import annotations

import json
import os

import onnx
from onnx import TensorProto as TP
from onnx import helper as oh

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX = os.path.join(_HERE, "out_blend6", "onnx")

# task_num -> hash; broken-incumbent (negative-pad ConvTranspose) minimal recompile.
_TARGETS = {
    384: "f25fbde4",
    127: "54d9e175",
}


def _pads_attr(node):
    for a in node.attribute:
        if a.name == "pads":
            return list(a.ints)
    return None


def _patch(model):
    """Return a numerically-identical model whose ConvTranspose pads are all >= 0."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    if m.ir_version > 10:
        m.ir_version = 10

    new_nodes = []
    new_inits = []
    k = 0
    for n in m.graph.node:
        if n.op_type == "ConvTranspose":
            pads = _pads_attr(n)
            if pads is not None and any(p < 0 for p in pads):
                # pads layout for 2 spatial dims: [begin_H, begin_W, end_H, end_W]
                clamped = [max(p, 0) for p in pads]      # cropping pads kept on the conv
                ext = [max(-p, 0) for p in pads]         # negative pads -> zero extension
                # rebuild ConvTranspose with clamped pads, writing to a temp name
                tmp = f"_ct_full{k}"
                nn = onnx.NodeProto()
                nn.CopyFrom(n)
                del nn.output[:]
                nn.output.append(tmp)
                for a in nn.attribute:
                    if a.name == "pads":
                        del a.ints[:]
                        a.ints.extend(clamped)
                new_nodes.append(nn)
                # Pad node: [b_N,b_C,b_H,b_W, e_N,e_C,e_H,e_W]
                pad_vals = [0, 0, ext[0], ext[1], 0, 0, ext[2], ext[3]]
                pname = f"_ct_pads{k}"
                new_inits.append(oh.make_tensor(pname, TP.INT64, [8], pad_vals))
                new_nodes.append(oh.make_node(
                    "Pad", [tmp, pname], [n.output[0]], mode="constant"))
                k += 1
                continue
        new_nodes.append(n)

    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    m.graph.initializer.extend(new_inits)
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
