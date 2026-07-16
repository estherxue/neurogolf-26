"""family_golf8_2 -- AGGRESSIVE-GOLF reformulation attempt over golf slice [2::6].

Targets (golf_targets.json[2::6], sorted-asc, lowest-scoring = most headroom):
  396 framecrop, 110/286/61/381/102/259/121/198/305/63/125 g73_f16 (mosaic-repair),
  268 slitlight, 324 diagray, 379 t379, 111 objmarker, 34 t34, 178 crk2_compress,
  242 symhole, 159 boxfill, 128 t128, 189 t189, 12 stamp12, 48 conn2_flood,
  30 align, 156 rectfill, 169 sizrecolor, 137 concentric_rings, 290 cropswap,
  91 crop91, 293 crossswap, 306 tile10.

RESULT OF ANALYSIS: no SAFE cost improvement is achievable for any of these with the
allowed opset-10 op set, so this module intentionally emits nothing (returns []).
It is a documented no-op: it never displaces the existing (generalizing) baselines,
which is important because emitting a memorizing/over-fit solver that passes the
provided pairs but fails the private set would LOWER the real leaderboard score.

Why nothing golfs (evidence gathered while building this module):

1. The one genuinely-cheap lever -- a single KxK Conv input->output (params-only,
   ZERO intermediate memory, ~16-18 pts) -- requires the task to be a LOCAL and
   LINEARLY-SEPARABLE function of a KxK one-hot patch. Empirically, every same-size
   target is NON-LOCAL: patch->label has CONFLICTS at K3/K5 (same 5x5 patch maps to
   two different output colors), i.e. the rule depends on context beyond 5x5
   (global/long-range). The only two exceptions both fail:
     * t293 crossswap: local (no conflict) but NOT linearly separable at K3/K5;
       separable only at K7 by memorizing (held-out 77/79 wrong). Its true rule is
       a GLOBAL band-thickness comparison (thicker crossing band wins).
     * t12 stamp12: local + separable at K5, but the perceptron fit MEMORIZES:
       held-out error stays high at 50/70/85% coverage (35/40 wrong @85%). The
       true stamp rule is not a single linear threshold (center-color vs arm-color
       coupling is a product, not a sum).

2. Structural pipelines that ARE exact + generalizing still cannot beat baseline,
   because per-pixel RECOLORING with a runtime (data-dependent) palette forces
   [1,10,30,30] float intermediates (36000 B each), and every such task needs ~5-6
   of them. Concretely, a fully-correct structural solver for t12 (verified EXACT on
   all 265 provided pairs, 0 conflicts, and it GENERALIZES -- it is the true 5x5
   plus->X(A)+ring(B) stamp rule) measures params=64, memory=263780 -> 12.52 pts,
   which is BELOW the 12.67 baseline. The ~200k memory floor (>=5 full one-hot
   tensors for the palette extract + paint) sits right at the baselines already.

3. The rest are structurally uncheapenable with static shapes:
     * g73_f16 (110,286,61,381,102,259,121,198,305,63,125): mosaic/periodic repair
       with a per-example VARIABLE period; OR-of-shifts replication needs many full
       tensors and exceeds baseline.
     * data-dependent CROPS (396,111,178,242,159,189,290,91,259,121): output is a
       variable-size/variable-location sub-region -> not a static graph.
     * GLOBAL classification (48 connectivity of two blobs; 156/159 fill-color from
       object SIZE; 293 band thickness): needs reductions/measurement, not local.
     * multi-directional data-dependent BEAMS (34,268,324,379): rays whose count and
       direction depend on the data; each direction+color needs its own propagation
       chain -> many intermediates, above baseline.
     * 306 tile10: replicate one content-panel across a 4-line-separated periodic
       grid; OR-of-period-shifts costs more than the 13.95 baseline.

Kept as a valid, side-effect-free module so the harness runs cleanly.
"""
from __future__ import annotations


def candidates(examples):
    # No safe, cheaper-than-baseline, generalizing reformulation found for this
    # slice. Emit nothing rather than ship an over-fit (memorizing) solver.
    return []
