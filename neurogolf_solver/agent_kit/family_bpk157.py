"""family_bpk157 — cheaper rebuild of the task157 (6a1e5592, "footprints") incumbent.

The incumbent gw2f16_157 casts the whole [1,10,30,30] float32 input to fp16,
materialising an 18000-byte named tensor whose ONLY consumers are three Slices.
We move the Cast *after* the Slices: slice the free float32 `input`, then cast the
tiny slices to fp16. Semantics are identical, so the graph stays exact on every
example; the 18000-byte tensor disappears (replaced by ~4800 bytes of small
float32 slices).  Everything downstream is untouched.
"""
from __future__ import annotations

import onnx
from onnx import helper as oh, TensorProto as TP

import family_gw2_f16 as _gw2


def _transform(model):
    g = model.graph
    # locate the whole-input cast: input -> input__gw2f16 (fp16)
    cast = next(n for n in g.node
                if n.op_type == "Cast" and n.input and n.input[0] == "input")
    big = cast.output[0]
    consumers = [n for n in g.node if big in n.input]
    assert all(n.op_type == "Slice" for n in consumers), "unexpected consumer"

    new_nodes = []
    k = 0
    for n in g.node:
        if n is cast:
            continue  # drop the big cast
        if n in consumers:
            k += 1
            f32_out = f"bpk_sl{k}"
            # slice the free float32 `input` instead of the fp16 copy
            sl = oh.make_node("Slice", ["input"] + list(n.input[1:]), [f32_out],
                              name=f"bpk_slice{k}")
            cst = oh.make_node("Cast", [f32_out], [n.output[0]],
                               name=f"bpk_cast{k}", to=TP.FLOAT16)
            new_nodes.append(sl)
            new_nodes.append(cst)
        else:
            new_nodes.append(n)

    del g.node[:]
    g.node.extend(new_nodes)
    onnx.checker.check_model(model, full_check=True)
    return model


def _build():
    for _k, (tid, a, _b) in _gw2._E.items():
        if tid == 157:
            return _transform(_gw2._d(a))
    return None


_MODEL = None


def candidates(examples):
    global _MODEL
    fp = _gw2._fp(examples.get("train", []))
    e = _gw2._E.get(fp)
    if not e or e[0] != 157:
        return []
    if _MODEL is None:
        _MODEL = _build()
    return [("bpk157", _MODEL)]
