"""family_fcrk_0 — crack attempt over unsolved slice U[0::3].

Slice tasks: [5, 44, 66, 80, 101, 157, 173, 209, 238, 285, 363].

Every task in this slice was read pair-by-pair. All of them require
DATA-DEPENDENT object localization / routing / per-example template stamping,
none of which reduces to a fixed, origin-anchored global op with CONSTANT
weights that generalizes to the held-out arc-gen set:

  5   : stamp a per-example 3x3 template repeatedly in the direction of an
        arrow marker; template + direction vary per example.
  44  : swap the contents of two 5-bordered boxes (6<->8 etc.); box positions
        and payload colors are data-dependent.
  66  : draw an L / straight path of 3s connecting a 3-marker to a 2-marker;
        endpoints vary per example (connect-dots).
  80  : propagate a marker across a grid-of-cells lattice; source cell varies.
  101 : grow / extend a 1-region from seed shapes; seed geometry varies.
  157 : reflect the lower 5-object into the upper 2-region as color 1; the
        reflection axis is the data-dependent 2/5 region boundary.
  173 : stamp a per-example template shape at marker locations; template varies.
  209 : crop the object region then symmetrize; crop window is data-dependent
        (output SIZE changes per example).
  238 : crop two objects and merge them into a small fixed panel; crop location
        data-dependent (output size changes).
  285 : reflect each object across a per-object local axis; object positions
        vary.
  363 : template-match a "diamond-around-a-5" (or horizontal-bar) seed and stamp
        color 2 at every matching site; the seed/template itself VARIES between
        examples (diamond in pair0/2, bar in pair1), so no constant Conv weight
        exists.

For each, a held-out self-check on the untouched 30% of arc-gen would fail, so
nothing is emitted (over-proposing wrong static rules only wastes grader calls).
"""
from __future__ import annotations

import numpy as np


def candidates(examples):
    # No member of this slice is expressible as a static, origin-anchored
    # opset-10 graph with constant weights that generalizes. Emit nothing.
    return []
