"""family_sr366 -- simpler-equivalent-rule HUNT for task 366 (arc-gen hash e6721834,
name "pool_only|ryo", generator category "compositing").

OUTCOME: NO_SIMPLER.  No closed-form (CumSum/Gather per-region recolor/copy)
inverse exists; the transform is an irreducible per-box template-matching /
compositing operation, which is exactly why the incumbent needs ~401 nodes
(TopK + ArgMax + ScatterElements + Conv + 52 And / 45 Gather ...).  `candidates`
returns [] so this module contributes nothing and never regresses the pool.

--------------------------------------------------------------------------------
WHAT THE TASK ACTUALLY IS (verified by dissecting 6000+ fresh generator pairs)
--------------------------------------------------------------------------------
The input is two width x height half-grids placed side-by-side.  Orientation:
horizontal iff in_width > in_height (in_width = 2*width), else vertical
(in_height = 2*height).  Grader ignores any grid whose max dim > 30, so horiz
grids with width > 15 (in_width 32/34) are never scored.

  * TEMPLATE half: solid `forecolor` rectangles ("boxes"), each with 1..3 dots
    of a per-box colour painted at random interior cells.  num_boxes is 2 or 3.
  * SKETCH half: the SAME boxes (a subset -- box 0 always, boxes 1/2 each w.p.
    2/3) but drawn as ONLY their dots on plain background (no rectangle).
  * OUTPUT (width x height, single grid): the SKETCH half with every present
    box's forecolor rectangle STAMPED back in, dots kept on top.  Output
    background = sketch-half background.

So reconstructing a box requires reading its (tall, wide) + dot layout from the
TEMPLATE half and re-anchoring it onto the SKETCH dots.  forecolor = the single
most-common non-background colour of the template half; the sketch half never
contains forecolor.

--------------------------------------------------------------------------------
WHY THERE IS NO SIMPLER RULE (measured, not asserted)
--------------------------------------------------------------------------------
1. NOT a per-cell / per-region recolor of either half.  The stamped rectangle
   extends arbitrarily beyond the dots' bounding box, so its extent is unknown
   without the template.  Output != either input half (0/2000 exact for a plain
   copy; backgrounds differ, positions differ).

2. Template boxes CANNOT be separated by connectivity.  The generator's overlap
   test uses strict `<`, so two forecolor rectangles on the same side may share
   an edge -> one merged connected component with a non-rectangular (L/T) union
   and no per-box (tall, wide).  Interior dots also punch holes.  A bounding-box
   read of forecolor components therefore yields wrong box dimensions.

3. Box identity is ambiguous.  Each box's dots share ONE colour, but colours are
   drawn independently per box, so two boxes collide on colour 32.7% of grids
   (template side) and 12.7% of grids have TWO sketch-side boxes sharing a colour
   -- so "group sketch dots by colour" is not injective.  Dot COUNT (1/2/3)
   distinguishes boxes, but a box's dots are spatially scattered (not connected),
   so they cannot be grouped locally by count either.

4. Genuine inverse ambiguity for the single-dot box (box 0): its 1-pixel
   signature matches every same-coloured dot on the sketch side, including dots
   that belong to box 1 / box 2.  Resolving which dot is box 0's needs the joint
   placement of the other boxes -- a constraint-satisfaction step, not a local
   map.  These are the "inherent ambiguity" cases the incumbent resolves with
   TopK/ArgMax/ScatterElements.

5. Best-effort closed-ish reconstructions plateau well short of exact:
     * per-colour group + full dot-geometry match ....... ~ 83% of decodable grids
     * per-template full-signature correlation stamp ..... ~ 51% exact (2912/5709)
   Neither reaches the exactness gate (all local + >=3000 fresh) required to beat
   the incumbent, and both are already heavier in ops than a "simple" rule.

CONCLUSION: the output is a template-matching composite, not a simple per-region
recolor/copy expressible with CumSum/Gather.  No equivalent simpler rule found;
the 14.69-pt incumbent (out_blend6/onnx/task366.onnx) is retained.
"""
from __future__ import annotations


def candidates(ex):
    """No simpler-equivalent rule exists for task 366 (see module docstring).

    Returns [] so the family harness records zero and never displaces the
    incumbent compositing network.
    """
    return []
