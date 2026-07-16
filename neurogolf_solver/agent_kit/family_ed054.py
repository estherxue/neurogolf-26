"""family_ed054 -- ENRICHED-ARSENAL deep-rebuild study of task054 ("crosshair
projection").  Candidate name: "ed054".

CONCLUSION (see the block comment below): task054 is at its practical FLOOR.
The incumbent ``gw2f16_54`` (out_blend10/onnx/task054.onnx) scores

    real grader:  memory=25201  params=193  cost=25394  points=14.8577

and every dominant byte is structurally forced or a genuine, non-foldable,
input-dependent pipeline stage.  None of the enriched-arsenal primitives
(runtime-bias QLinearConv binary detect, adjoint-conv paint, static Gather,
uint8/bool masks, CumSum spans) can be applied to strictly lower the cost
without either (a) increasing memory or (b) requiring a full re-derivation of
an intricate variable-structure algorithm whose best-case saving (~+0.1..+0.3)
does not justify the very high risk of failing the exact-match-on-3000-gen gate.

candidates() therefore returns the incumbent itself (train+test exact) so the
module is a valid, runnable solver; it TIES the incumbent (it does not beat it).

--------------------------------------------------------------------------------
WHY IT IS FLOORED  (full accounting)
--------------------------------------------------------------------------------
Real cost model (neurogolf_utils.calculate_memory / calculate_params):
  * intermediate memory = elements * np.dtype.itemsize  (uint8=1, bool=1,
    fp16=2, int64=8) summed over every NAMED node output, MAX(static, trace).
  * params = element COUNT of initializers/Constant values (dtype-free, cheap).
  * input/output are excluded (free, any dtype).

Incumbent memory by dtype (total 25201):
  float32 3640 | uint8 10437 | bool 6368 | float16 1092 | int64 3152 | int32 512

1. 3600-byte float32 [1,1,30,30] Conv output -- LOCKED.
   The grader's convert_to_numpy() ALWAYS feeds the input as float32 one-hot
   [1,10,30,30].  Collapsing 10 one-hot channels to a color-index grid is a
   1x1 Conv with weights [0..9]; a Conv over a float32 input necessarily emits
   float32 [1,1,30,30] = 3600 bytes, then Cast->uint8 (900).  Every alternative
   is worse:  ArgMax(axis=1) -> int64 [1,1,30,30] = 7200;  Cast(input,uint8) ->
   uint8 one-hot [1,10,30,30] = 9000;  QLinearConv needs a uint8 input which
   would first cost a 9000-byte QuantizeLinear of the one-hot.  3600 is minimal.

2. ~13500 bytes = 15 full [1,1,30,30] uint8/bool grids -- GENUINE stages.
   color grid, background/box detection, marker detection, per-marker arm masks,
   ScatterND/ScatterElements stamp results, final grid.  cost == TOTAL bytes, so
   batching the 4 per-marker blocks into one dim-4 tensor does NOT help (4x fewer
   tensors x 4x bigger = same bytes).

3. Arms are PER-BOX-BOUNDED and cannot be globally vectorized.  In 300/300 gen
   samples multiple boxes share rows AND columns, so a marker's row/col arm must
   stop at the background edge of its own box; a global row/col projection would
   bleed across boxes.  This forces the per-marker (<=4-slot) architecture with a
   CumSum span per marker -- exactly what the incumbent does.  A rebuild lands on
   the same shape.

4. int64 3152 -- LOCKED.  ArgMax (opset) always outputs int64; converting the
   ~82 int64 scalars to int32 requires an added Cast per ArgMax (net-neutral or
   worse, because the int64 output still exists).  The ScatterND index tensors
   ([4,8,4]=1024, [4,3], [2,4], the [4,8]/[4,8,1] adds) MUST be int64 per the
   ONNX ScatterND spec; the arithmetic funnels into that int64 sink, so an int32
   shadow only ADDS a cast back to int64 -- NET ZERO.

5. float16 1092 -- LOCKED.  These are the CumSum span computations; CumSum needs
   a float type and float16 is the minimal one.

6. No constant folding: reachability shows 100% of node outputs depend on
   'input'.  No dead outputs.

Adjoint-conv paint (the arsenal's stamping primitive) could in principle replace
the fixed 8-cell diagonal stamp with markers (x)) kernel, saving ~1000-1500 bytes
of int64 index machinery (~+0.07), BUT it does not graft onto the incumbent's
index-based design (it would need a marker-INDICATOR grid, which the incumbent
only holds as int64 positions -- materialising it costs another 900-byte grid +
int64 scatter), and the box-bounded variable-length arms are NOT convolvable.

Net: the reliable, gate-passing improvement available is ~0 bytes.
--------------------------------------------------------------------------------
"""
from __future__ import annotations

import os
import numpy as np
import onnx

_INCUMBENT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "out_blend10", "onnx", "task054.onnx"
)


def _fingerprint(ex):
    """Cheap structural fingerprint so this module only fires on task054 when run
    across all 400 tasks by family_test.py."""
    tr = ex.get("train", [])
    if len(tr) != 3:
        return False
    shapes = [(len(p["input"]), len(p["input"][0])) for p in tr]
    if any(s != (30, 30) for s in shapes):
        return False
    # task054's three train inputs use backgrounds {8, 1, 8} and contain the
    # crosshair template + rectangular boxes.  Match on the exact first-train grid.
    a = np.array(tr[0]["input"])
    return a.shape == (30, 30) and a[0, 0] == 8 and a[1, 2] == 1 and a[5, 5] == 3


def candidates(ex):
    if not _fingerprint(ex):
        return []
    if not os.path.exists(_INCUMBENT):
        return []
    model = onnx.load(_INCUMBENT)
    return [("ed054", model)]
