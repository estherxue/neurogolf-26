"""Single-node compile campaign, batch 5.

Target tasks: 25, 349, 285, 101, 187, 319, 364, 18, 54, 158.

Every task in this batch was inspected against its true DSL verifier and 5+
train/arc-gen pairs.  All ten are data-dependent, per-object routines whose
output cannot be written as a single fixed geometric / recolor / permutation /
crop map:

  25  (1a07d186)  frontiers + per-object gravitate toward walls (variable count)
  349 (db93a21d)  objects -> shoot rays DOWN, backdrop fill (data-dependent)
  285 (b775ac94)  per-object neighbour-mirror stamping (variable objects)
  101 (447fd412)  self-tiling + occurrence matching + upscale template
  187 (7b6016b9)  fill only NON-bordering closed boxes with 2, recolor bg->3
  319 (ce602527)  two objects, upscale, occurrence test picks which to crop
  364 (e509e548)  recolor each object by shape class (line/box/other)
  18  (0e206a2e)  D4-symmetric template completion (occurrence driven)
  54  (264363fd)  seed object shoots lines then subgrid crop (data-dependent)
  158 (6aa20dc0)  largest-object backdrop mirror completion (variable)

None reduce to a fixed Gather/Transpose/Slice/Pad/Einsum map: the output color
of a cell depends on the shape/closure/position of the object it belongs to,
which requires connected-component labelling and per-object variable-count
logic (Loop/flood-equivalent) that the incumbent pool net already handles near
optimally.  Hence this family emits nothing and never regresses the incumbent.
"""
from __future__ import annotations


def candidates(ex):
    """No single-node form exists for any task in this batch."""
    return []
