"""family_t2_379: enriched recompile attempt for task379 (hash ecdecbb3, "projection").

RESULT: FLOORED. No candidate strictly beats the incumbent
(out_blend12/onnx/task379.onnx = fp16_379|pool|ryo), so candidates() returns [].

TASK (generator tasks/task_ecdecbb3.py):
  Full cyan(8) lines (all horizontal rows, then optionally transposed) act as walls.
  Isolated red(2) dots each cast a beam PERPENDICULAR to the wall orientation toward
  every wall reachable along that axis (blocked by the nearest wall). At each
  wall-intersection a 3x3 cyan frame is stamped with its centre punched red; the beam
  fills red from the dot up to the frame. width,height in randint(12,20) -> max grid
  20x20; <=1 dot per (canonical) column; <=2 lines.

INCUMBENT DISSECTION (82 nodes, params=153, memory=7640, cost=7793, points=16.04):
  Per-tensor bytes (each named intermediate once, MAX shape; input/output free):
    padded  [1,1,30,30] u8   = 900   <-- DOMINANT single tensor
    11x     [1,1,20,20]      = 4400  (the working masks, below)
    rMax,cMax,e1,e2,e3,e5 [1,1,30/1] f32 = 6x120 = 720
    ~19x [1,1,*,20] f16      = 40 each (beam-bound 1D chain)  = 760
    misc f16/bool/u8 vectors = ~1000
  The 11 [1,1,20,20] tensors and why each is structurally required:
    beam range-check: ge=GE(coords,lo), le=LE(coords,hi), TRAIL=And(ge,le)
    box (per-line):   box0=And(rb_0,DIL_0), box1=And(rb_1,DIL_1), BOX=Or(box0,box1)
    map layering:     base, m_trail, m_box   (3-stage uint8 Where chain)
    orientation:      m_box_T=Transpose(m_box), MAP=Where(horB,m_box,m_box_T)

WHY IT IS AT THE FLOOR (what I tried with the enriched arsenal):
  * DOMINANT `padded` (900) is minimal for the free-output one-hot pattern
    output = Equal(padded[1,1,30,30]u8, colorvec[10,1,1]) -> [1,10,30,30].
    The only alternative, Equal(MAP20)->[1,10,20,20] then Pad->output, costs a
    [1,10,20,20]=4000 intermediate. 30*30*1 uint8 is the floor; cannot shrink.
  * Working masks already at the floor: 20x20 (generator max grid = 20) x 1 byte
    (uint8/bool). Cannot use a smaller static region (no dim_param allowed; metric
    takes MAX shape) and cannot drop below 1 byte.
  * beam range-check needs 2 comparisons + And (Clip won't help: ONNX Clip min/max
    must be SCALAR, but bounds are per-column [1,1,1,20]; sign/abs/cumsum variants all
    still need >=3 2D tensors).
  * box needs 2 outer products + Or because the dot<->line reaching is per-line-paired
    (a dot above both lines must ring only the top line); merging via a single
    dilated-centre MaxPool re-introduces two paired outer products for the centres.
  * map layering (line/black background, +red beam, +cyan frame) is a minimal 3-stage
    Where chain; arithmetic (8*isCyan+2*isRed) expands to ~9 2D tensors -- worse.
  * orientation: input is [1,10,30,30] (36000B) so it cannot be transposed to a
    canonical axis cheaply; transposing the finished 20x20 map (m_box_T) + select
    (MAP) is 2 tensors and is the cheapest orientation handling.
  * The 6x f32 [1,1,30,1] reduce vectors (720) are forced: ReduceMax/Einsum on the
    float32 input emit float32; producing f16 would require Cast(input)->f16
    [1,10,30,30]=18000. Casting the outputs afterwards keeps the f32 node output.
  Arsenal tricks that do NOT apply: QLinearConv binary detect / adjoint-conv paint /
  static-Gather are for fixed-template or stamping ops; this is a variable-geometry
  beam cast. FREE-OUTPUT (Equal->output) and the u8/bool/f16 dtype floor are ALREADY
  used by the incumbent, which is a well-optimized fp16|pool model and near-optimal.

  Conclusion: incumbent memory 7640 is at the structural floor for this transform;
  no restructuring found that reduces the named-intermediate total, so no candidate
  can score STRICTLY more than 16.04. Reporting floored per the harness contract.
"""
from __future__ import annotations


def candidates(examples):  # noqa: ARG001 - floored, nothing beats the incumbent
    return []
