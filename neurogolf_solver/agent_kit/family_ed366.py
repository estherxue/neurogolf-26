"""ENRICHED-ARSENAL deep-rebuild attempt for task366 (arc-gen hash e6721834).

VERDICT: FLOORED with the enriched arsenal. candidates() returns [] (no
cost-beating, exact graph found) so this module never regresses the pool.
Incumbent = out_blend10/onnx/task366.onnx: 14.686 pts, memory 29962 B,
params 203, 401 nodes, 404 named tensors, gen 251/251 exact locally.

------------------------------------------------------------------------------
THE TRUE RULE (decoded from the generator, tasks/task_e6721834.py; numpy
reference validated to 249/251 local + ~98.5% on fresh gen):

The grid is split into two equal halves (left|right if horiz, top|bottom if
vert). One half ("fore") is drawn with solid FORECOLOR rectangles (boxes),
each box containing (idx+1) same-colored DOTS. The other half ("non-fore")
holds ONLY those dots (no rectangles), on its own background. The two halves
have different background colors. num_boxes = 2 or 3; box sizes 2..7 each dim;
each box has a unique dot-COUNT (1,2,3) but box COLORS may collide.

OUTPUT = the non-fore half moved to the top-left, KEEPING its background and
dots, with the forecolor rectangle of EACH dot-cluster reconstructed (drawn
under the dots). The rectangle's size and its dot-offsets are read from the
MATCHING fore box (matched by dot pattern), then re-anchored onto the
non-fore dots (whose absolute positions are INDEPENDENT of the fore side).

Required sub-steps for exactness (all verified necessary against the data):
  1. split-axis detection (both dims are often even -> must pick the axis whose
     two halves each have a single clean background; wrong axis mixes both bgs);
  2. forecolor = most-frequent non-bg color in the fore half, absent from the
     other half (the 2x2-block heuristic FAILS: dots break up small boxes);
  3. connected-component box finding in the fore half;
  4. two-sided dot-pattern matching with the JOINT CONSTRAINT:
       * ~32.7% of samples have a COLOR COLLISION (two boxes share a dot color)
         -> matching by color alone is ambiguous; must disambiguate by
         dot-COUNT + relative arrangement, matching multi-dot boxes first so a
         1-dot box cannot steal a 2-dot box's dot (reproduced directly here);
       * ~1.4% have ADJACENT boxes whose forecolor rectangles fuse into one
         connected component -> must be DECOMPOSED back into separate boxes;
  5. variable-size rectangle stamping at DATA-DEPENDENT positions (one per box);
  6. data-dependent shift of the non-fore half to the top-left (the non-fore
     half may be the bottom/right half).

------------------------------------------------------------------------------
WHY THE ARSENAL DOES NOT CRUSH IT (unlike e191):

* COST STRUCTURE (profiled, ORT_DISABLE_ALL, all local examples): the incumbent
  is NOT a fat-static-evidence-map case. It is a lean DYNAMIC-shape symbolic
  solver whose tensors are charged their ACTUAL runtime sizes:
    - half-grids at 15x17  = 255 B (bool/uint8), 510 B (f16),
    - box regions at 6x6   = 36 B, 6-vectors = 6 B, plus many scalars,
    - exactly ONE full [30,30] float32 = 3600 B.
  404 tensors -> 29962 B. There is no single fat tensor to collapse; the cost
  is spread across many tiny tensors that are already near-optimal.

* e191 won because its incumbent had THREE fat STATIC [1,8,23,23] tensors
  (21 KB) doing Conv-f16 -> Equal -> Cast, which QLinearConv-binary-detect +
  adjoint-conv-paint collapsed into two integer convs. task366 has no such
  structure -- the arsenal's QLinearConv / adjoint primitives operate in STATIC
  [1,C,H,W] conv space where every [30,30] uint8 tensor costs 900 B and every
  [30,30] float costs 3600 B.

* A from-scratch STATIC arsenal graph is therefore charged ~3.5x per tensor
  (900 B vs the incumbent's 255 B dynamic half-grids). To merely reach 29962 B
  it would need <=~33 uint8 [30,30] tensors total, yet the algorithm above
  (component labels + per-color masks + per-box rectangle stamps + collision
  decomposition + variable-axis + shift) needs many DOZENS of [30,30] masks --
  it starts OVER budget before any matching logic. Matching the incumbent would
  require re-implementing its dynamic-shape design, i.e. a reconstruction with
  zero cost headroom, not a golf.

* EXACTNESS WALL: reaching even LOCAL exactness (gate b, required to ship)
  needs 100% on the merged-component + collision joint-constraint cases. The
  best numpy reference here (frequency-forecolor + decreasing-count joint match)
  still fails 2/251 local + ~1.5% fresh on exactly those cases; a static ONNX
  encoding of connected-component decomposition + joint collision matching is
  far beyond what fits under 29962 B.

CORROBORATION: bitpack-wave3 previously got 0 wins on task366; the sr-hunt
found NO_SIMPLER ("366 joint-constraint needed, color collisions 32.7%"). This
independent re-analysis reproduces the same 32.7% collision joint-constraint
and the merged-component wall from the generator itself.

Net: the enriched arsenal (QLinearConv binary-detect, adjoint-conv paint,
static-Gather, uint8/bool) targets fat static evidence maps; task366's cost is
an irreducible spread of tiny dynamic tensors for genuine joint-constraint
matching. No strictly-cheaper exact graph exists within reach -> FLOORED.
"""
from __future__ import annotations


def candidates(ex):
    """No cost-beating exact graph found for task366 (see module docstring).

    Returns [] for every task so the integrator keeps the incumbent
    (out_blend10/onnx/task366.onnx, 14.686 pts) unchanged -- zero regression.
    """
    return []
