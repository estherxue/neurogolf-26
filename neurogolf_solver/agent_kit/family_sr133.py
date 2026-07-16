"""family_sr133 — SIMPLER-EQUIVALENT-RULE hunt for task133 (57aa92db, "vc2_scale").

OUTCOME: NO_SIMPLER.

Generator (task_57aa92db): 2..4 sprites share ONE template shape (a
`continuous_creature` inside a 3x3 box, pixel[0] = "signature" cell drawn in the
common `pcolor`, other cells drawn in that sprite's own body colour).
  * sprite 0 (magnifier bmag=1) is drawn IN FULL in the input  -> the "legend".
  * every other sprite is drawn in the input as only TWO bmag x bmag blocks:
    the pcolor signature block + one adjacent body block.
The output redraws EVERY sprite in full = the legend's shape upscaled by that
sprite's bmag, anchored at its signature block, in its body colour.

The exact inverse therefore needs, and only needs, these irreducible statistics:
  (1) the legend pattern  (relative body-cell offsets from the signature cell);
  (2) per partial sprite: its position (signature-block corner),
  (3) its scale bmag, and (4) its body colour.
All four are necessary to reconstruct an ARBITRARY shape at an arbitrary
scale/position/colour; none is derivable from the others.

Empirically verified minimal sufficient statistics on 4000 fresh pairs:
  * pcolor == the colour appearing in the MOST 8-connected components (always >=2);
  * every OTHER colour forms EXACTLY ONE component  -> per-colour bounding box is
    unambiguous, so NO flood-fill / component labelling is required;
  * legend == the object with pcolor drawn as a single cell + full body.
=> position, scale (= body/signature block side) and colour all read straight off
   per-colour bounding boxes; the pattern is one small window read at the legend.

The DEPLOYED incumbent (out_blend6/onnx/task133.onnx) ALREADY realises exactly
this closed form, and golfs it near-optimally:
  * per-colour row/col presence + ArgMax  -> bounding boxes (no flood);
  * a single 9x9 read at the legend        -> the template pattern `pat`;
  * Resize(pat, 2/3/4)                      -> the 4 possible upscaled kernels;
  * ScatterND seeds + QLinearConv           -> stamp every partial at once;
  * grids stored as uint8, coordinate maths in float16.
Real scorer: params=1016, mem=20306, cost=21322, points=15.03.

Where the 20306 bytes live (measured): gridf(float 30x30)=3600, rowany/colany
(1200 each), and TWELVE uint8 30x30 grids (seed1..4, stamp1..4, grid, gflat,
outgrid) = 10800. The 4 seed + 4 stamp surfaces are one-per-distinct-scale
(bmag in {1,2,3,4}); they are the irreducible reconstruction floor. Collapsing
them to a single runtime-variable upscale (family_vc2_3._upk style) replaces four
cheap fixed Resize kernels with 30x30 MatMul upscales per component -> strictly
MORE named 30x30 float intermediates, i.e. higher cost. Dropping to fewer than 4
scale branches is impossible because bmag ranges over four values. The float32
gridf is forced by Conv's output dtype before the uint8 cast.

No SIMPLER sufficient statistic and no cheaper equivalent structure exists: the
rule is already the direct inverse expressed with minimal state, and the ONNX is
already dtype-golfed to uint8/float16 with flood-fill eliminated. Any faithful
re-build lands at >= the same reconstruction floor, and a standard (non-QLinear)
Conv stamp would push grids back to float32 and raise cost. Hence no candidate is
offered.

candidates() yields nothing (no strictly-cheaper equivalent found).
"""
from __future__ import annotations


def candidates(examples):
    return
    yield  # pragma: no cover  (keeps this a generator, emits no candidate)
