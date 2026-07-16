"""family_cs1_3 — COMPLETE-SWEEP minimal-recompile pass over tasks
202, 174, 265, 359, 208, 14, 162, 131, 50, 117.

Goal: for each task find a *strictly cheaper* correct graph than the out_blend6
incumbent, or SKIP because the incumbent already sits at the algorithmic floor.

Cost model:  points = 25 - ln(mem + params),  mem = Σ bytes of NAMED intermediates
at max runtime shape (input/output FREE), params = initializer/Constant elements.
GatherND/ScatterND indices are constrained to tensor(int64) (verified against the
ONNX op schema + local ORT 1.23.2), ArgMax/ReduceSum have no low-precision output
option, and Slice preserves the float32 dtype of the one-hot input.

Verdict after dissecting every incumbent (real, itemsize-correct costs):

    task  incumbent cost / pts     driver tensors                         verdict
    ----  --------------------     ------------------------------------   ------
    202   4481 / 16.592            3× [1,1,30,30] (mark/band/cg, 900 ea)  SKIP
    174   4347 / 16.623            multi-object [3,10,10]/[3,10,5] set     SKIP
    265   4357 / 16.620            gray float-slice 1296 + pad30 900       SKIP
    359   4284 / 16.637            2× float per-color counts (1200 ea)     SKIP
    208   4189 / 16.660            17² float-slice 1156 + mask30 900       SKIP
    014   4117 / 16.677            18² float-slice 1296 + pad30 900        SKIP
    162   4068 / 16.689            ch0 float-slice 1600 + pad30 900        SKIP
    131   3894 / 16.733            ScatterND int64 idx [1,50,4]=1600       SKIP
    050   3850 / 16.744            cyan float-slice 900 + mask30 900       SKIP
    117   3795 / 16.759            13² masks + reflection working set      SKIP

Two structural costs dominate almost every task and are provably at the floor:

  (1) THE READ.  Extracting one colour channel out of the float32 one-hot input
      requires a float32 tensor cropped to the true grid: size²·4 bytes is the
      minimum (Slice keeps float32; ReduceSum/ArgMax over channels give the SAME
      or larger footprint — ArgMax is int64 = size²·8).  1600 (20²), 1296 (18²),
      1156 (17²), 900 (15²) are exactly size²·4 — already minimal.

  (2) THE WRITE.  The free one-hot output [1,10,30,30] must come from
      Equal(label30, cidx) or Where(mask30, colourvec, input); either way the
      30×30 label/mask is a named tensor ≈900 bytes (uint8/bool).  Padding a
      *smaller* one-hot up to 30×30 is strictly worse (10 channels → ≥2250 B).

  Between the read and the write each incumbent carries only a genuine, minimal
  working set (a 3×3/2×2 morphological opening for 162/265, two directional
  MaxPool spreads per axis for 50, per-column/row modes for 359, an object
  gather/matmul stack for 174, a mandatory int64 ScatterND index block for 131,
  a reflection/copy set for 117).  None of these compress under the legal arsenal
  without either raising the tensor count or breaking exactness.

Levers explicitly checked and rejected:
  * QLinearConv to read a channel as uint8 (skip the float slice): impossible —
    QLinearConv needs a quantised (uint8) INPUT, so the whole [1,10,30,30] input
    would have to be cast to uint8 first (9000 B) before any crop.  Net loss.
  * int32 GatherND/ScatterND indices (halve 131's 1600 B): rejected by the ONNX
    checker AND ORT 1.23.2 — indices are tensor(int64) only.  Test-reproduced.
  * float16 per-colour counts for 359: getting f16 counts needs a f16 cast of the
    input (18000 B) or a post-hoc cast that keeps the f32 count anyway.  Net loss.
  * Downsampling 359/202 below 30×30: rejected — 359's stripes make width up to
    sum(wides) ≈ 30 and 202's scored grids reach 30×30; both genuinely need 30².

Because the hard gates forbid shipping any unverified or regressive graph, and no
strictly-cheaper *and* exactly-correct reformulation exists for any of the ten,
this family yields nothing: the sweep confirms out_blend6 is already at the floor
for this batch and the incumbents are retained unchanged.
"""
from __future__ import annotations


def candidates(examples):
    """No strictly-cheaper graph was found for any task in this batch; every
    incumbent is at its structural floor (see module docstring).  Yield nothing
    so the harness keeps the out_blend6 model for each task (no regression)."""
    return
    yield  # pragma: no cover  (marks this a generator; never reached)
