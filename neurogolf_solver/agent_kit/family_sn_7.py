"""Single-node compile campaign, batch 7.

Target tasks: 182, 17, 5, 157, 209, 286.

Each task was inspected against its true DSL verifier
(_rearc/verifiers.py) plus 5 train/arc-gen pairs.  All six are
data-dependent, per-object routines whose output is NOT a fixed
geometric / recolor / permutation / crop map, so none reduces to a
zero/low-param single node.  The incumbent pool net already handles
them near optimally.  This family therefore emits nothing and can
never regress the incumbent.

  182 (776ffc46)  objects() -> select the largest square-box object
                  (argmax over CCL), then fill its inbox with a colour
                  derived from a per-object shape/occurrence match.
                  Needs connected components + argmax over variable
                  objects.  Not a fixed map.

  17  (0dfd9992)  objects() + colorfilter; pick the colour whose object
                  is smallest/rarest, measure its horizontal/vertical
                  PERIOD (hperiod) and tile-complete along data-dependent
                  offsets.  Period and target colour are data-dependent.

  5   (045e512c)  argmax largest object, then replicate it outward in the
                  8 neighbour directions with a per-direction, per-step
                  variable count and per-copy recolour driven by the
                  background test.  Variable tiling; not fixed.

  157 (6a1e5592)  choose a D4 variant by a numcolors/color occurrence
                  predicate, then shoot object rays and stamp.  Variant
                  and stamping are data-dependent.

  209 (8a004b2b)  find the corner-frame object, upscale a subgrid by a
                  data-dependent integer ratio, repaint and crop.  Output
                  size varies per example (9x14, 7x7, 11x11, ...), so it
                  cannot be a fixed Slice/Pad/RoiAlign.

  286 (b782dc8a)  leastcolor + rarest colour, neighbour/adjacency joins
                  between two seed objects with a parity (manhattan even)
                  recolour.  Data-dependent adjacency; not fixed.

Every one depends on connected-component labelling and per-object
variable-count logic (Loop/flood-equivalent, all banned), so no
single-node form exists.
"""
from __future__ import annotations


def candidates(ex):
    """No single-node form exists for any task in this batch."""
    return []
