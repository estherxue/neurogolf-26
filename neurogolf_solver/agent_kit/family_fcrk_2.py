"""family_fcrk_2 -- crack wave for unsolved slice U[2::3].

Slice investigated: tasks 23, 54, 79, 96, 133, 170, 201, 233, 264, 349, 367.

Verdict after reading every train/test pair: none reduce to a generalizing,
origin-anchored *static* graph.  Each is a data-dependent per-object operation
(routing / stamping / selection / assembly) whose output geometry depends on
positions or sizes that are not knowable at graph-build time:

  023  overlay decomposition: two shapes both drawn as 5 are split into 8 vs 2
       per source object (5x5-neighborhood lookup is consistent on the 4 given
       pairs but that is a memorized table, not a rule -- fails to generalize).
  054  draw a full-length cross through a data-placed marker inside each box.
  079  select ONE 3x3 motif among scattered shapes -> fixed 3x3 out (which one
       is data-dependent).
  096  nest the scattered motifs as concentric D4-symmetric rings ordered by
       size -> assembly, data-dependent.
  133  grow / wrap each seed object -> per-object, data-dependent.
  170  a block-presence 3x3 pattern MASKS a separate 3x3 colour key; both live
       at data-dependent positions/scales.
  201  assemble a corner-framed overlay of two shapes -> data-dependent.
  233  tile small 3x3 stamps into a grid -> assembly, data-dependent.
  264  same tile-assembly family -> data-dependent.
  349  concentric nested-square ring growth around each seed -> data-dependent.
  367  fill the interiors of DRAWN rectangle outlines with 4.  This is NOT the
       clean "enclosed hole" rule (already covered by family_flood): it selects
       intended rectangle interiors while leaving equally-enclosed leftover
       background empty, and it fills raw-grid edge pockets.  Verified against
       flood-fill (border-4-conn), corner-seed, largest-8-comp and 4-ray casts
       -- all fail byte-exact.  Not expressible as a static, size-agnostic graph.

So this module contributes no new solves; it yields nothing.  A candidates()
that returns [] is the honest, false-positive-free result (the shared grader is
exact, so any emitted model must be byte-exact on train+test+arc-gen).
"""
from __future__ import annotations


def candidates(examples):
    return []
