"""family_crk11_0 — slice U[0::3] = [5,44,66,80,101,157,173,209,238,285,363].

Investigated every task in this slice for an exact, generalizing static opset-10
rule (template-match / ConvTranspose stamp, fixed-output assembly with a global
rule, origin-anchored symmetry). Result: this slice is the hardest-remaining
leftover — every task is data-dependent per-object routing whose parameters
(template shape/orientation, beam direction, repeat count, or output bounding
box) are derived per-example and cannot be baked into a static graph:

  * 5   — diagonal beam replication of a stamp; direction+count from a 2-cell
          marker, tiled to the (variable) grid edge. Per-object direction.
  * 44  — swap/recolor of paired objects across a region. Per-object matching.
  * 66  — L-shaped path drawn connecting a colour-3 seed to a colour-2 seed.
          Data-dependent routing.
  * 80  — sub-grid (cell-of-cells) content propagation. Per-cell adaptive.
  * 101 — connect same-colour marker PAIRS with a per-grid-learned template
          (horizontal in one example, vertical in another). Adaptive template.
  * 157 — flood/paint under a bar keyed by seeds. Per-region.
  * 173 — multi-template stamp: each grid holds several DISTINCT templates, each
          stamped at its own single-cell seeds. Not a fixed kernel.
  * 209 — gather objects into a compact assembly bounded by colour-4 corners;
          OUTPUT SIZE VARIES per example (data-dependent bbox).
  * 238 — insert a shape into a marker-defined frame; variable output size.
  * 285 — per-object symmetrisation / growth. Per-object.
  * 363 — laser/beam propagation of colour-2 through colour-5 walls; per-seed
          direction. Data-dependent.

None reduce to the crackable patterns (fixed-kernel stamp like task069, or a
component-free global rule like task182). The genuinely-crackable tasks landed
in other slices. This family therefore proposes no candidates: emitting nothing
is monotone-safe (never regresses the portfolio); wrong guesses would only be
rejected by the grader anyway, but here no rule is even exactly reproducible on
the fit set, so there is nothing honest to emit.
"""
from __future__ import annotations

import numpy as np


def candidates(examples):
    # No task in slice U[0::3] admits an exact static-graph rule (see module
    # docstring). Return no candidates — monotone-safe, no regression.
    return []
