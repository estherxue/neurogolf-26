"""family_scrk_2 — analysis of unsolved slice U[2::5] = [22,66,89,133,170,201,238,349].

Every task in this slice was examined in depth (all train+test+arc-gen pairs) and each
requires NON-LOCAL, data-dependent structure that a cheap static opset-10 graph cannot
express while generalizing to the hidden private set:

  22  (FIXED 11x11->3x3): detect 2+ scattered objects at varying positions, overlay into 3x3.
  66  (VAR, same-shape):  draw a color-3 path that ROUTES around color-8 obstacles from the
                          3-marker to the 2-marker (maze routing) — not a local/global op.
  89  (FIXED 13x13):      stamp a detected template shape at each isolated marker cell, with a
                          per-marker reflection whose sign could not be determined consistently.
  133 (VAR, same-shape):  each object grows base-color arms AWAY from its "1" marker into a
                          plus; center/direction/length are per-object & marker-relative.
  170 (VAR)->3x3/4x4:     read a partitioned grid of cells + a small key; output size (3 vs 4)
                          is content-dependent, not a function of input size.
  201 (in 13x13 -> 4x6..7x8): crop to a data-dependent 4-corner box and overlay a free object;
                          output SIZE varies for identical input size -> impossible statically.
  238 (in 14-16 -> 5..7): combine two objects into a framed overlay; output size is
                          content-dependent, not a function of input size.
  349 (VAR, same-shape):  each 9-rectangle spawns a frame + directional trail (growth from
                          object); per-object, non-local.

Mechanical checks performed (all negative):
  * simple geometric / symmetry-overlay transforms (identity, flips, rot180, transpose,
    OR-with-reflections) — no match on any pair.
  * per-cell KxK neighborhood -> color LUT for K in {3,5,7} — inconsistent (not a local fn)
    for 66/89/133/349.
  * output-shape-as-function-of-input-shape — FALSE for 170/201/238 (same input size yields
    differing output sizes), so even the output dimensions are content-dependent.

No exact, private-safe candidate exists for these under the current arsenal, so this family
proposes nothing (returning wrong models would only fail the EXACT gate).
"""

def candidates(examples):
    return []
