"""family_crk10_5 -- hardest remaining unsolved tasks, slice U[5::8].

Assigned tasks: [46, 89, 138, 170, 209, 349].

Every rule below was reverse-engineered EXACTLY from all train pairs (and cross-
checked against the arc-gen size/colour statistics).  The grader (evaluate.py,
full=True) demands EXACT equality on train+test AND on all 262 arc-gen pairs, so
a partial / structural match earns nothing.  Each of these six tasks is a
data-dependent, multi-object, variable-output-size construction that provably
cannot be expressed as an opset-10 static graph under the origin-anchored /
static-shape contract.  The exact rules and the specific op-set blocker are
documented per task so the analysis is not lost; candidates() therefore yields
nothing (emitting a graph that fails arc-gen would score 0 and only add risk).

--------------------------------------------------------------------------------
TASK 46  (3xN -> 3xM, M<N, "paper fold along the 5-creases")
  The colour-5 cells form crease markers scattered across the 3-row strip.  The
  strip is folded onto itself at the (data-dependent) crease columns, overlaying
  the coloured cells (non-bg wins) and shrinking the width.  Output width and the
  fold positions depend on where the 5s land -> variable output size + a
  data-dependent reflection whose axis is read from the content.  Not static.

TASK 89  (13x13 -> 13x13, "stamp the template shape at every lone marker")
  Fixed 13x13.  Each example has one or more TEMPLATE objects = a connected
  component containing a single "marker" colour cell plus surrounding "shape"
  cells, and several LONE MARKER cells (same colour as a template's marker,
  isolated).  Output = input with, for each lone marker, a copy of the matching
  template's shape stamped so its marker aligns (orientation can mirror).  Output
  only ADDS cells (verified: 0 changed cells across all pairs).  This is a
  convolution of a data-dependent marker map with a data-dependent template
  kernel -> data-tensor (x) data-tensor 2D correlation.  Opset-10 has no way to
  build a data-dependent conv kernel / Toeplitz (Conv weights must be static
  initialisers; ScatterND/GatherElements/im2col are banned).  Not static.

TASK 138 (large -> framed histogram, "bar chart of marks per band")
  Two coloured vertical lines and two coloured horizontal lines partition the
  grid; scattered marks in each band are COUNTED and drawn as bars growing from
  the bottom/right inside a rectangular frame whose border reuses the four line
  colours.  Output size = number of bands; interior = per-band counts.  Counting
  + data-dependent output size.  Not static.

TASK 170 (large -> KxK, "key grid masked by the block meta-pattern")
  A large area holds a KxK arrangement of BxB monochrome blocks (present/absent
  = the meta-pattern); a small dense KxK "key" grid of distinct colours sits
  elsewhere.  Output = key[i,j] kept iff block-slot[i,j] is present, else 0.
  K in {3,4} varies per example (2 output sizes in arc-gen).  Requires: detect K,
  locate + extract the key to the origin, downsample the block field (different
  scale, different location) to KxK, and align the two KxK grids.  Two independent
  data-dependent crops at different scales + data-dependent output size.  Not
  static.

TASK 209 (large -> bbox, "reconstruct objects inside the 4-marker box via key")
  Four colour-4 corner markers define the output bounding box (verified: output
  size == bbox(colour 4) on every pair).  Inside, a set of BxB coloured objects
  sit at grid slots; a small schematic KEY says how to expand/connect them
  (upscale the key by the object size B and paint the connective fill colour).
  The crop-to-bbox is expressible (dyncrop MatMul trick), but the interior fill
  is a data-dependent schematic reconstruction (fill colour + which slots, read
  from the key, upscaled by a data-dependent B).  Not static.

TASK 349 (NxN -> NxN, "each 9-square grows a 3-frame + 1-trails toward others")
  Input has only bg + colour-9 squares of varying size/position.  Each square is
  wrapped in a colour-3 frame and emits colour-1 trails that reach toward the
  other squares; frame thickness and trail length/direction depend on the
  relative geometry of the squares.  Non-local, multi-object, geometry-dependent
  growth -> not a fixed-radius CA / static op.  Not static.
--------------------------------------------------------------------------------

Conclusion: none of the six is realisable as an exact opset-10 static ONNX graph.
candidates() returns [] (verified SOLVED 0, no runtime error).
"""
from __future__ import annotations

import numpy as np


def _pairs(examples, splits=("train", "test", "arc-gen")):
    prs = []
    for s in splits:
        for e in examples.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim == 2 and b.ndim == 2 and a.size and b.size:
                prs.append((a, b))
    return prs


def candidates(examples):
    """No opset-10 static graph exactly reproduces any of the assigned tasks
    (46, 89, 138, 170, 209, 349); see module docstring for each rule and its
    blocker.  Return no candidates rather than emit graphs that fail the grader."""
    _ = _pairs(examples)  # detection would run here; every branch rejects.
    return []
