"""family_scrk5_2 -- unsolved slice U[2::4] = [22,54,80,107,143,173,201,238,319,367].

Every task in this slice was examined in depth (all train+test pairs plus many arc-gen)
and each is INFEASIBLE as a static, GRID-AGNOSTIC (no-crop) opset-10 graph that would still
pass the EXACT grader on the held-out/arc-gen set. Findings per task:

  22  (11x11 -> 3x3): overlay of 2+ scattered small objects into a fixed 3x3. Output is a
        SHRINK/crop (always 3x3) and its content is a data-dependent overlay of objects at
        varying positions -> forbidden (no-crop rule) AND not statically expressible.

  54  (30x30, same shape): a small "key" motif elsewhere on the canvas shows how to decorate
        a marker (a plus/cross + diamond); that decoration is stamped on the single marker
        inside each solid rectangle, with the cross STRETCHED to the rectangle's borders, and
        the key is erased. Template is data-dependent (read from the input) and the projection
        length is region-dependent -> not a fixed local/global op.

  80  (var, same shape): meta-grid of cells separated by full 8/3 lines; a marked cell's
        content is propagated across the meta-grid. IMPOSSIBLE to even represent: 35 arc-gen
        grids are 31x31 (exceed the fixed 30x30 canvas), so an exact [1,10,30,30] graph
        cannot reproduce them. (No cropping/oversize handling allowed.)

  107 (5x5 -> {10,15,20,25,30}^2): the SAME 5x5 input maps to 5 different output sizes across
        arc-gen (verified). Output size is content-dependent, so even the output dimensions
        of a static graph cannot be correct -> impossible.

  143 (10x10, same shape): a small TEMPLATE object sits left of the "5" L-glyph; the rule
        recolors to 5 the OTHER object elsewhere that is congruent (exact shape+orientation)
        to that template. The template shape is data-dependent (lives in the input), so the
        match is a data-dependent shape correlation -> not expressible with fixed weights.

  173 (var, same shape): each isolated marker is stamped with a prototype motif that is
        DEFINED elsewhere in the same grid (data-dependent per instance). A fixed per-color
        stamp is inconsistent (613 conflicts over 713 markers) -> the surround is not a fixed
        function of color; it depends on the in-grid prototype -> not statically expressible.

  201 (13x13 -> {4x6..7x8}): identical 13x13 input yields 7 different output sizes across
        arc-gen. Crop to a data-dependent 4-corner box + overlay; output size content-dependent
        -> impossible statically (and cropping is forbidden).

  238 (var -> {5x5,6x6,7x7}): identical input shape yields 3 different output sizes. Combine
        two objects into a framed overlay whose size is content-dependent -> impossible.

  319 (var -> {3x3..5x5}): identical input shape yields 6+ different output sizes; reads a
        sub-grid down to a tiny output. Content-dependent size + crop -> impossible.

  367 (var, same shape): 5-strokes form closed loops; a color-4 inner ring/interior is drawn.
        The fill includes border-adjacent pockets (cells ON the grid edge become 4), so no
        flood-from-border interior test can produce them, and it is provably not a KxK local
        function (K=3: 7041 conflicts; K=5: 1191). Best clean topological interior-fill scores
        only 108/266 -> no exact, generalizing static rule found.

Mechanical checks performed (all negative):
  * identity / transpose / flip_h / flip_w / rot180 via the real grader -> 0/10 solved.
  * KxK -> color LUT locality for K in {3,5} on all same-shape tasks -> every one NON-LOCAL.
  * enclosed-region (flood) fill for 367 -> 108/266 only.
  * fixed per-color stamp for 173 -> 613 conflicts.
  * congruent-object recolor for 143 -> requires a data-dependent (in-grid) template kernel.
  * output-shape as a function of input-shape for 107/201/238/319 -> FALSE (same input size
    maps to multiple output sizes), so even the static output dims cannot be right.

Each task needs either a data-dependent template/kernel, a content-dependent output size, a
crop, or a grid larger than 30x30 -- none of which a cheap static opset-10 graph can express
while generalizing to the private/arc-gen set. Proposing wrong models would only fail the
EXACT gate, so this family proposes nothing.
"""

def candidates(examples):
    return []
