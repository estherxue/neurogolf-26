"""family_crk11_1 -- crack attempt over the U[1::3] slice of unsolved.json:
tasks [18, 46, 76, 89, 118, 158, 191, 219, 255, 319, 366].

Investigation summary
---------------------
Every task in this slice was checked (numpy references) against the full battery
of origin-anchored, statically-expressible global rules:
  * pointwise colour map / recolor          -> no
  * transpose / flip / rot180 / identity    -> no
  * overlay-with-own-symmetry (LR/UD/rot/T)  -> no
  * enclosed-background fill (flood-from-border, 4- & 8-conn) -> no
  * enclosed-by-wall fill (wall colour 2)    -> no
  * fixed template-match + stamp             -> no (family_templatematch yields 0)
  * constant output                          -> no

Each remaining task is a data-dependent PER-OBJECT routing / selection /
reflection transform, whose correct positions depend on the (variable) grid
content and therefore cannot be encoded in a static opset-10 graph:
  18  objects relocated to marker-defined destinations (per-object translation)
  46  per-object recolor + data-dependent crop
  76  per-object reflection/stamp, data-dependent geometry
  89  shape reflected through a data-dependent marker axis
  118 background cells enclosed *between* colour-2 markers recolored (per-object)
  158 fill/connect between paired markers (data-dependent paths)
  191 template box stamped at data-dependent trigger locations
  219 horizontal rays drawn from shapes (per-object)
  255 largest empty rectangle filled with colour 3 (data-dependent region)
  319 select-a-subgrid-by-property (data-dependent, output shape changes)
  366 panel matching / overlay (data-dependent, output shape changes)

No crackable member was found, so candidates() proposes nothing.  Emitting no
candidate is the correct (monotone) action: it never regresses any other
family, and the harness keeps whatever incumbent is cheaper.
"""
from __future__ import annotations


def candidates(examples):
    # Nothing in this slice is expressible as an exact static-graph global rule.
    return []
