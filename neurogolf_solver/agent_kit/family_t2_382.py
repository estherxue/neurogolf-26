"""family_t2_382 -- LOADABILITY repair + golf on task382 (hash f15e1fac).

WHY THE INCUMBENT IS BEATABLE (unspent headroom):
  out_blend12/onnx/task382.onnx does NOT load under the local ORT 1.23.2 hard gate.
  Its color-combine node is `Max` over five *uint8* tensors, and ORT 1.23.2 has no
  uint8 kernel for Max(13):
      NOT_IMPLEMENTED : Could not find an implementation for Max(13) node 'color_index'
  So the incumbent's real (gated) score is 0.0 -- any loadable exact model beats it.

INCUMBENT DISSECTION (75 nodes, opset18, params=96, mem=5606 => nominal 16.35 pts,
  but real 0.0 because it never initializes):
    input  float32 [1,10,30,30] one-hot
    output BOOL    [1,10,30,30]  (grader compares the FULL uncropped one-hot:
      run_network thresholds (out>0); convert_to_numpy zeros every channel beyond
      the actual height x width grid -> our out-of-grid cells MUST be all-zero,
      which is why an "outside" sentinel of 255 is load-bearing.)
  Per-tensor bytes (each named intermediate once, MAX shape; io free):
    color_index_30 [1,1,30,30] u8 = 900   <-- DOMINANT tensor (free-output pad:
        Equal(color_index_30, channel_ids[1,10,1,1]) -> output; 255 matches no
        channel so out-of-grid cells become all-zero). This is the uint8 floor.
    row_pattern/col_pattern/red_value/color_index [1,1,20,20] u8 = 400 x4
    row_any_sum/col_any_sum [1,3,30] f32 = 360 x2 (Einsum presence; f32 forced by
        the f32 one-hot input -- casting input to f16 would name an 18000B copy).
    + banks / 1-D coord vectors, all already at the u8/bool/i32 floor.

THE REPAIR (this module):
  Replace the unimplemented 5-input uint8 `Max(row_pattern, col_pattern, red_value,
  row_outside, col_outside)` with ORT-1.23.2-supported uint8 ops:
    * The three colour layers are pairwise DISJOINT (each output cell is exactly one
      of black/red/cyan, and only one of the row/col cyan patterns is ever active),
      so their per-cell Max == their Add.  Add(u8) is implemented.
    * The two "outside" layers are a 255-or-0 sentinel; Max with them == force 255
      wherever the cell's row OR column is outside the grid.  A cell is inside iff
      its row has a black pixel (x0_row_any) AND its column has one (x0_col_any),
      so two nested Where(u8) with the (small) x0 bool conditions reproduce it:
          rc     = Add(row_pattern, col_pattern)          # cyan layer  [1,1,20,20]
          sum3   = Add(rc, red_value)                     # + red       [1,1,20,20]
          inner  = Where(x0_col_any_1d, sum3, 255)        # col outside -> 255
          color_index = Where(x0_row_any, inner, 255)     # row outside -> 255
  This drops the two tiny row/col_outside Where nodes and adds three [1,1,20,20]
  u8 intermediates (rc, sum3, inner).  Everything downstream (Pad -> color_index_30
  -> Equal -> output) is unchanged, so the transform is bit-identical to the
  incumbent's intended algorithm -- only the op set changes.

  Measured (this module, via the real scorer): loads, exact on train+test and 3000+
  fresh generator samples; mem+params ~6862 -> ~16.17 pts.  That is STRICTLY more
  than the incumbent's real 0.0 (and is the max achievable: the dominant 900B pad is
  the free-output uint8 floor, the 720B f32 presence pair is forced by the f32 input,
  and uint8 has no cheaper multi-way max than Add-of-disjoint + Where-sentinel).
"""
from __future__ import annotations

import os
import onnx
from onnx import helper as H, TensorProto as TP

_INCUMBENT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "out_blend12", "onnx", "task382.onnx"
)


def _build():
    m = onnx.load(_INCUMBENT)
    g = m.graph

    # Drop the two "outside" Where nodes (18,19) and the unimplemented uint8 Max (72).
    drop = {"row_outside", "col_outside", "color_index"}
    kept = [n for n in g.node if not (set(n.output) & drop)]

    # Rebuild the color-combine with ORT-1.23.2-supported uint8 ops.
    #   x0_row_any    [1,1,20,1] bool   (produced upstream, node 12)
    #   x0_col_any_1d [1,1,20]   bool   (produced upstream, node 9)
    #   outside_u8    = 255 (existing initializer, the sentinel / pad constant)
    new = [
        H.make_node("Add", ["row_pattern", "col_pattern"], ["rc_382"]),
        H.make_node("Add", ["rc_382", "red_value"], ["sum3_382"]),
        H.make_node("Where", ["x0_col_any_1d", "sum3_382", "outside_u8"], ["inner_382"]),
        H.make_node("Where", ["x0_row_any", "inner_382", "outside_u8"], ["color_index"]),
    ]

    # Insert the new nodes right before the Pad that consumes `color_index`.
    pad_idx = next(i for i, n in enumerate(kept) if "color_index" in n.input)
    kept[pad_idx:pad_idx] = new

    del g.node[:]
    g.node.extend(kept)
    return m


def candidates(ex):  # noqa: ARG001
    return [("t2_382", _build())]
