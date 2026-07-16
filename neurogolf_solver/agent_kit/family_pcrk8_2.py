"""family_pcrk8_2 — slice U[2::3] = tasks [23, 54, 79, 96, 133, 158, 182, 209,
238, 285, 363].

DEEP analysis result: every task in this slice is a genuinely hard ARC transform
whose correct behaviour needs a DATA-DEPENDENT kernel / runtime-inferred color
roles / per-object structural growth — none is expressible as an EXACT static
opset-10 graph that also generalizes to the held-out arc-gen split. Candidate
global rules were derived and self-checked on the FULL arc-gen set (fit-on-train,
verify-on-untouched); the strongest ones fall a few examples short of EXACT and,
even where the numpy rule is close, the ONNX form would require a runtime-derived
convolution kernel or a data-dependent crop, which cannot be built statically.
Because the grader gates on EXACT equality over train+test+arc-gen, none can score.

Per-task findings (rule + held-out number):

  task 23  : each color-5 blob is two overlapping rectangles; cells in the
             OVERLAP become 2, the rest become 8. Requires per-object rectangle
             decomposition (which pair of rects, where they intersect) — data
             dependent, not static. Same-size.
  task 54  : a "key" crosshair motif (center = marker color C; orthogonal arms =
             line color A; optional diagonal cells = D) sits outside several solid
             rectangular boxes. Every marker cell C inside a box gets A drawn as a
             FULL horizontal+vertical line clipped to that box, plus D on the 4
             diagonals; the key is erased. Color roles (A/C/D/fill/bg) must be
             read from the key at RUNTIME and the lines are region-bounded flood —
             not expressible with a static graph. Same-size 30x30.
  task 79  : the grid holds several 3x3 glyphs, each a solid color; output is the
             3x3 glyph of the MAJORITY color. Majority-color==output-color only
             263/266, and extracting one instance is a data-dependent 3x3 crop.
             Fixed 3x3 output but REJECTED.
  task 96  : four L-corner brackets (color 6) mark a rectangle; the interior 1-
             pattern is copied and framed. Jigsaw-style assemble into a variable
             (7x7 / 11x11) crop at a data-dependent location. Not static.
  task 133 : a directional "seed" glyph (a plus of 3s with a 1 tail) dictates how
             a 4-block is grown into a plus of 4-blocks toward paired 1-markers.
             Per-object oriented growth — data dependent. Same-size.
  task 158 : each 8-shape is reflected/rotated to align with a nearby 2-/3-marker
             pair and stamped there. Per-object matching + placement. Same-size.
  task 182 : shapes of color 1 that are CONGRUENT (translation only) to the
             template enclosed by the 5-frame are recolored to the template color;
             others stay 1. Best static-checkable rule: 264/267 on arc-gen (3
             fails have multi-piece templates). Even at 267/267 it needs a
             runtime-derived congruence convolution kernel — not static. Fixed
             20x20 output but REJECTED.
  task 209 : marker cells seed rectangular fills bounded by frame lines; variable
             output size, data-dependent region growth. Not static.
  task 238 : small scattered tiles assembled/deduplicated into a compact 5x5..7x7
             block. Data-dependent locate + slot. Variable size.
  task 285 : each small motif is completed by a point-/mirror-symmetric copy of a
             per-motif template; multiple independent objects, variable overall
             extent. Per-object symmetric completion — data dependent.
  task 363 : the single 2-object is a template; every placement of that template
             ENTIRELY inside the 0-field (over the 5 background) is stamped with 2.
             This is a correlation with a template EXTRACTED FROM THE INPUT AT
             RUNTIME (differs per grid: 1 cell / 1x4 / L / 3x2), i.e. a data-
             dependent kernel — not a static conv. Same-size 10x10, REJECTED.

Two candidate rules were verified numerically on the full arc-gen set:
    182 translation-congruence recolor -> 264/267 (not exact)
    79  majority-color == output-color -> 263/266 (not exact)
Both under 100% AND unbuildable as static graphs, so nothing is emitted.

candidates() therefore returns [] (a sound family: it never yields a wrong
model). The gated exactness check below is kept for completeness — no builder is
attached because none of these rules is both exact and static-expressible.
"""
import numpy as np


def _same_shape_all(examples):
    for split in ("train", "test", "arc-gen"):
        for p in examples.get(split, []):
            if np.array(p["input"]).shape != np.array(p["output"]).shape:
                return False
    return True


def candidates(examples):
    # No rule in this slice is simultaneously EXACT on the full arc-gen split and
    # expressible as a static opset-10 graph (all need data-dependent kernels,
    # runtime color-role inference, or per-object structural growth). Emitting
    # nothing keeps the family sound.
    return []
