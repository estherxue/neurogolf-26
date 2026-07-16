#!/usr/bin/env python
"""STEP-3 dtype surgery on incumbent task096.onnx.

FINDING (see report): the incumbent is already at its dtype FLOOR under
ORT-1.23.2. The step-2 profiler u8_slack=2533 is a MIRAGE -- every flagged
tensor is pinned by a hard constraint the entropy profiler does not model:

  * row_sum_full / col_sum_full  (1200B f32 each): ReduceSum of the float32
    graph input. Reductions preserve dtype -> f32. Making them u8/f16 requires
    a smaller-dtype view of the 9000-elem input; the cheapest such cast is
    9000-18000B, which DWARFS the 1200B saved. Empirically, an f16 rewrite
    blows mem 7260 -> 24080 (input16 cast alone = 18000B). u8 ReduceSum/ReduceMax
    have no kernel; MaxPool(u8) exists but needs a u8 input (same 9000B cast).
  * radius_i32 (484B int32): Gather/Scatter/Slice INDEX. ORT rejects u8/int8
    index tensors ("Invalid tensor data type"), so int32 is the floor.
  * channel_count/non_bg_count/top_colors_0 (f32): dtype propagates from the
    f32 reductions/TopK.  6x ArgMax + TopK indices: forced int64.
  * code_i32/radius_raw/radius_idx/maxd/starts2/ends2 (int32): flow into
    Slice/Scatter indices -> int32 forced.

The ONLY genuinely reducible tensors are code_pre_i32 and length_i32, each of
which feeds a single downstream Cast, so their int32 (4B/elem) can collapse to
u8 (1B/elem). This surgery captures that (~35B, mem 7260->7225, +0.0045 pts).
It is BELOW the merge acceptance threshold (+0.01) and is provided only as a
strictly-gate-passing artifact; the ~5150 step-2 target is unreachable.

Transforms (semantics byte-identical; code in [0,55], all ops exact in u8):
  - drop  Cast(code_pre u8 -> int32)               [code_pre_i32, 20B]
  - node81 Where -> u8 branches (zero_u8)          [code_i32 20B -> code_u8 5B]
  - insert Cast(code_u8 -> int32) = code_i32       [for the Div/radius path]
  - node96 Mod(code_u8, eight_u8) -> length_u8     [length_i32 20B eliminated]
  - drop  Cast(length_i32 -> length_u8)
"""
import os, onnx
from onnx import helper as oh, TensorProto as TP

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(KIT, "out_blend19", "onnx", "task096.onnx")
DST = os.path.join(KIT, "_cands4", "task096.onnx")

def surgery():
    m = onnx.load(SRC); g = m.graph
    del g.value_info[:]                       # let ORT re-infer types
    keep = []
    for n in g.node:
        o = list(n.output)
        if n.op_type == "Cast" and o == ["code_pre_i32"]:
            continue                           # drop: was code_pre u8 -> int32
        if n.op_type == "Cast" and o == ["length_u8"] and list(n.input) == ["length_i32"]:
            continue                           # drop: length_i32 -> length_u8
        if n.op_type == "Where" and o == ["code_i32"]:
            # Where(present_bool, code_pre_i32, zero_i32) -> u8 form
            n.input[1] = "code_pre"            # u8
            n.input[2] = "zero_u8"             # u8
            n.output[0] = "code_u8"
            keep.append(n)
            keep.append(oh.make_node("Cast", ["code_u8"], ["code_i32"], to=TP.INT32))
            continue
        if n.op_type == "Mod" and o == ["length_i32"]:
            n.input[0] = "code_u8"
            n.input[1] = "eight_u8"
            n.output[0] = "length_u8"
            keep.append(n)
            continue
        keep.append(n)
    del g.node[:]; g.node.extend(keep)
    onnx.checker.check_model(m)
    onnx.save(m, DST)
    return DST

if __name__ == "__main__":
    p = surgery()
    print("wrote", p)
