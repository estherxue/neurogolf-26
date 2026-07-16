"""family_pcrk8_0 — slice U[0::3] = tasks
[5, 44, 66, 80, 101, 143, 170, 191, 219, 255, 319, 366].

DEEP analysis result: every task in this slice is a genuinely hard, object-level
or data-dependent ARC transformation with NO exact static-graph rule that
generalizes to a held-out arc-gen split. Each candidate rule was derived and
self-checked against the structure; none is exact + grid-agnostic + static, and
the grader gates on EXACT equality over all train+test+arc-gen, so none can
score. Emitting a wrong/overfit candidate is worse than emitting nothing, so this
family returns [] (with a strict exactness gate that never fires).

Verified up front (scripted over all train+test pairs): none of the 12 tasks is
identity / global color-map / transpose / flipH / flipW / rot180. Same-size:
{5,44,66,80,101,143,191,219,255} same H,W in==out; {170,319,366} shrink.

Per-task findings (full derivation in the agent report):

  task 5   : a small stencil glyph (e.g. an 8-box) plus a directional 1/marker;
             the glyph is STAMPED repeatedly along the marker's direction a
             data-dependent number of times. Direction+count vary -> not static.
  task 44  : hollow 5x5 boxes; each box interior is FILLED with the color of an
             adjacent single-color marker (66/88), and the marker is consumed.
             Per-object pairing of box<->marker -> data-dependent.
  task 66  : two 2-cell markers (2 and 3); an L-shaped PATH of 3s is routed to
             connect them, hugging walls of 8. Path routing -> data-dependent.
  task 80  : a 5x5 tiled lattice (8 gridlines) with a few decorated cells (3/6
             frames); decorated cells are reflected/propagated across the lattice.
             Which cells + how far vary per grid -> data-dependent, variable size.
  task 101 : a template glyph (1s+2s) is "grown" out of 2-colored seed markers,
             oriented per seed. Data-dependent placement/orientation.
  task 143 : one object is recolored to 5 (the object matching the color-5 legend
             glyph's shape / the designated one); the selected object varies by
             example -> per-object shape match, does not generalize as a static op.
  task 170 : big 5x5 blocks form a 3x3 (or larger) occupancy mask; a tiny digit
             stencil is masked by it and cropped to a 3x3/4x4 output. Both the
             block layout and the digits are data-dependent -> variable crop.
  task 191 : a fixed 5x5 glyph (1s+4s) is STAMPED (with orientation) at locations
             where scattered noise-4 pixels partially match its 4-pattern.
             Data-dependent placement/orientation -> not static.
  task 219 : left-anchored / staircase 8-shapes each emit a horizontal ray of 1s;
             ray row + length depend on the shape geometry -> data-dependent.
  task 255 : a 3-colored plus/cross with 3 edge-reaching arms is drawn over the
             0-cells of a data-derived empty region; the region shape varies per
             grid (masks are NOT identical across examples) -> not static.
  task 319 : select ONE object (of a data-dependent color) and crop its bounding
             box; selected color + output size vary per example -> data-dependent.
  task 366 : the grid is two halves (wall/background split); objects from one half
             are MOVED/stamped onto marker clusters in the other half. Split axis,
             object positions, and destinations all vary -> data-dependent, and
             output size varies with the half size.

FORBIDDEN alternatives (per the task spec) were explicitly avoided: no
per-neighborhood KxK LUT/perceptron (overfits unseen neighborhoods), and no
fixed-size crop for the variable-size tasks (170/319/366 span many output sizes).
"""
import numpy as np


def _exact_on_all(examples, fn):
    """Return True iff `fn` reproduces EVERY provided example exactly (all splits).
    Used as a strict soundness gate: a candidate is emitted only if it is exact on
    train+test+arc-gen. None of the derived rules pass, so nothing is emitted."""
    for split in ("train", "test", "arc-gen"):
        for p in examples.get(split, []):
            gi = np.array(p["input"])
            go = np.array(p["output"])
            try:
                pred = fn(gi)
            except Exception:
                return False
            if pred.shape != go.shape or not np.array_equal(pred, go):
                return False
    return True


def candidates(examples):
    # No exact, generalizing, static-graph rule exists for any task in this slice.
    # Return nothing so the family stays sound (never emits a wrong candidate).
    return []
