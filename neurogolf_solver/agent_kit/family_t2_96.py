"""family_t2_96 — repaired + measured task096 solver (candidate "t2_96").

The out_blend12 incumbent (golfe5_fp16_96) does NOT load under the local
ORT 1.23.2 hard gate: it contains uint8 `Max`/`Min` nodes and ORT 1.23.2 has
no uint8 kernel for Max(13)/Min(13), so the session fails to initialize
(evaluate() -> points 0.0). This module rebuilds the *exact* same algorithm
with the only change being Max/Min -> Where(Greater/Less), which is
mathematically identical for the integer operand ranges involved.

Dominant-tensor dissection (see report): the two ReduceSum maps
`row_sum_full` / `col_sum_full` (FLOAT [10,30] = 1200 B each) plus
`padded_color` (UINT8 [30,30] = 900 B) and `radius_i32` (INT32 [11,11] = 484 B)
are FLOORED — the FLOAT model-input contract forces the reduces to emit
float [10,30], any pre-cast of the [1,10,30,30] input to a smaller dtype
materialises a >=9000 B tensor, `radius_i32` must stay int32 as a Gather index,
and `padded_color` must be 30x30 uint8 to one-hot into the [1,10,30,30] output.
So memory is left at the incumbent floor; the win here is loadability.
"""
import os
import onnx
from onnx import helper

_ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "out_blend12", "onnx", "task096.onnx")


def _repair(model):
    """Replace uint8 Max/Min (no ORT 1.23.2 kernel) with Where(Greater/Less).

    Max(a,b) = Where(Greater(a,b), a, b);  Min(a,b) = Where(Greater(a,b), b, a).
    A single Greater is shared when two nodes have identical operands, so only
    the strictly necessary comparison tensors are added.
    """
    g = model.graph
    new, cmp_of = [], {}
    for n in g.node:
        if n.op_type in ("Max", "Min") and len(n.input) == 2:
            a, b = n.input
            out = n.output[0]
            key = (a, b)
            gt = cmp_of.get(key)
            if gt is None:
                gt = out + "_gt"
                new.append(helper.make_node("Greater", [a, b], [gt], name=gt))
                cmp_of[key] = gt
            if n.op_type == "Max":
                new.append(helper.make_node("Where", [gt, a, b], [out], name=out))
            else:
                new.append(helper.make_node("Where", [gt, b, a], [out], name=out))
        else:
            new.append(n)
    del g.node[:]
    g.node.extend(new)
    return model


def candidates(example):
    return [("t2_96", _repair(onnx.load(_ONNX)))]
