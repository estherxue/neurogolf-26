"""family_scrk5_0 - assigned unsolved slice U[0::4] = [5,44,76,96,133,158,182,219,264,363].

Every task in this slice was examined in depth over ALL train + test + arc-gen pairs.
Each requires NON-LOCAL, data-dependent object reasoning (template extraction, shape
matching, object attraction/movement, oriented stamping, or content-dependent output size)
that CANNOT be expressed as an EXACT, generalizing, GRID-AGNOSTIC (no-crop) opset-10 static
graph. The grader gates on EXACT equality over all 261-262 arc-gen examples, so anything
approximate or overfit scores 0.

Mechanical checks performed (all negative):
  * simple origin-anchored transforms (identity, transpose) -> no match on any pair.
  * per-cell KxK neighborhood -> color LUT for K in {3,5,7}: INCONSISTENT at K=3/5 for every
    task. The only "consistent" hit is 363 at K=7, but with 26497 distinct 7x7 patches
    ~= the total cell count -> every patch is unique, i.e. the table memorizes and cannot
    generalize to the hidden private set. Not a real local rule.

Per-task findings (why each is not a static grid-agnostic graph):

  5   (VAR, same-shape): each object carries a directional "marker line" (e.g. 3.3.3 or 222)
      that encodes a direction AND a repeat count; the object's shape is then STAMPED that
      many times along that direction. Count/direction are per-object and data-dependent.

  44  (FIXED 10x10): scattered colored objects are MOVED into the interior of the nearest
      5-bordered box (object attraction). Non-local per-object translation to a
      data-dependent target cell.

  76  (FIXED 13x13): one complete "template" 4-object carries satellite markers (1,2,3) in a
      fixed relative layout; every other partial 4-object gets those markers stamped
      according to its own ORIENTATION. Orientation- and template-dependent stamping.

  96  (VAR -> 7x7 / 11x11): reconstruct a symmetric emblem by overlaying scattered fragments;
      OUTPUT SIZE is content-dependent (7 vs 11 for the same 18x18 input) and requires a crop.
      Forbidden (no-crop) AND not a static function of input size.

  133 (VAR, same-shape): a "key" object shows how a base color grows arms around a small
      marker; each other object is expanded into that grown shape. Per-object, marker-relative,
      non-local.

  158 (VAR, same-shape): a 3x3 template object is stamped onto each isolated marker cell with a
      per-marker orientation/reflection. Template extraction + oriented stamping.

  182 (FIXED 20x20): the shape inside a 5-box is the template; every scattered color-1 object
      whose SHAPE MATCHES the template is recolored to the template's color, others stay 1.
      Requires translation-invariant shape matching + component labeling. Template is
      data-dependent; number/positions of objects vary. Not expressible exactly & generalizing.

  219 (VAR, same-shape): the first 8-object is a complete template; each later partial 8-object
      matches a prefix and the MISSING remainder is filled in color 1. Template completion,
      non-local, per-object alignment.

  264 (VAR -> 9x9): assemble a 9x9 emblem from scattered framed pieces; output size is
      content-dependent and requires a crop. Forbidden and not a static size function.

  363 (FIXED 10x10): a color-2 "diamond" seed pattern is stamped at every location matching a
      specific 5/hole configuration; which empty cells become 2 depends on a non-local
      multi-cell template match (K=3/5 LUT inconsistent; K=7 only "works" by memorizing unique
      patches -> no generalization).

No exact, private-safe, grid-agnostic candidate exists for these under the current arsenal, so
this family proposes nothing (emitting wrong models would only fail the EXACT grader gate).
"""

import numpy as np  # noqa: F401


def candidates(examples):
    return []
