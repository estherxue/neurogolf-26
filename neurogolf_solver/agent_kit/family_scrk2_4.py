"""family_scrk2_4 — slice U[4::5] = tasks [44, 79, 107, 157, 182, 233, 319].

Deep per-pair analysis + computational locality/transform probes over the full
train+test+arc-gen sets show NONE of these 7 tasks reduce to an EXACT, generalizing
opset-10 static-graph rule within the arsenal (pointwise recolor, transpose/flip,
CA/flood doubling, component labeling, fixed conv stamps, KxK->LUT). Findings:

  * 44  (same-size 30x30): "container fill" — each hollow 5-frame's interior is
        filled with the color of a DISTANT loose object matched to it, and the
        loose object is erased. Object<->container assignment is a GLOBAL matching
        (fill color comes from outside the frame), not a local/CA rule.
        Locality probe: K=3 still leaves 64 cell conflicts over 25870 windows
        (pure overfit, would fail the private set).
  * 79  (14x14 -> 3x3): output = the 3x3 shape that occurs MOST OFTEN among the
        scattered 3x3 objects. Requires template counting + extracting a
        data-located 3x3 to the origin (data-dependent translation) — not
        expressible with a static origin-anchored graph.
  * 107 (5x5 -> 5n x 5n, n=2,3,4 varies per example): fractal/reflected upscale
        whose scale factor VARIES per grid. No fixed static output shape / upscale
        can produce a data-dependent scale. Not expressible.
  * 157 (same-size): mirror/symmetry completion — holes in the top 2-region are
        marked with 1 driven by the reflected bottom 5-shapes. Content-anchored
        vertical reflection over VARIABLE grid sizes is position-unsafe (flip
        sends content to the far edge for <30 grids). Locality probe: K<=3 stays
        heavily conflicted. Not expressible.
  * 182 (same-size 20x20): template recolor — the 5-box encloses a template shape
        of color C; every loose object of color 1 whose SHAPE matches the template
        is recolored to C, others are left alone. Shape matching against a
        variable, variably-placed template is a global op; not local (K<=3
        conflicts remain, windows explode). Not expressible.
  * 233 (var -> var): the large bordered 2-rectangle is cropped out and small
        scattered markers are overlaid at matching interior positions. Variable
        data-dependent crop + global overlay. Not expressible.
  * 319 (var -> small var): select ONE object (by an odd-one-out / property rule)
        and place it top-left. Data-dependent variable crop + translation. Not
        expressible.

candidates() therefore emits nothing for this slice. (It self-validates any rule
against ALL provided examples and would yield only on an exact match, so it never
emits a candidate the grader would reject — but no such rule exists here.)
"""
import numpy as np


def candidates(examples):
    # No exact, generalizing static-graph rule exists for any task in this slice.
    # Kept as a no-op so the family scores 0 without ever proposing a rejectable model.
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for g in ("train", "test", "arc-gen") for e in examples.get(g, [])]
    if not prs:
        return []
    return []
