"""family_pcrk9_2 — deep re-examination of unsolved slice U[2::4].

Tasks examined (1-based task ids): 23, 66, 89, 133, 170, 201, 238, 319, 367.

Every train pair (and many arc-gen) was read for each task and multiple
candidate rules were tested numerically. Each task was found to require a
capability that a translation-equivariant, static, opset-10 ONNX graph
cannot encode in a way that survives the held-out arc-gen gate:

  23  — recolor 5 -> {8,2} is a *tiling partition* of the blob into disjoint
        2x2 squares (covered cells -> 8, leftover cells -> 2). Verified it is
        NOT "any 2x2" (that mislabels shared cells) but a non-overlapping
        packing -> a global combinatorial matching, not a local conv.
  66  — connect two 2-cell markers (colors 2 and 3) with a rectilinear path.
        Per-instance routing between two arbitrarily placed markers on up to
        20x20; turn position is data-dependent. Not static-expressible.
  89  — extract the data-dependent template object (a shape around a center
        marker) and stamp it at each single-pixel marker, with per-marker
        mirror/orientation. Data-dependent kernel + multi-object placement.
  133 — one "master" object defines a template; every other seed object is
        completed to that template, recolored to the seed color and scaled by
        the seed's unit-block size. Data-dependent template + variable scale.
  170 — output is an NxN (N in {3,4}) key-palette masked by a presence grid
        obtained by downsampling a variable-size / variable-position block
        region. Two data-dependent regions at data-dependent scale; output
        size itself varies -> not a fixed crop, not static-expressible.
  201 — reflect two shapes into a 4-corner colored frame; output size varies
        (6x8, 7x8, ...) and depends on the data. Per-object reflect+assemble.
  238 — same genre as 201: mirror an object into a 4-sided colored box; output
        size varies (6x6, 5x5, ...). Per-object reflect+assemble.
  319 — segment the objects, SELECT one by a data-dependent criterion, and
        output just that object cropped (variable output size, 5x5 / 5x3 /...).
        Data-dependent selection + variable crop.
  367 — flood/enclosure fill (background -> 4). Exhaustively tested vertical/
        horizontal ray-sandwich (edge-open and edge-wall variants), 4/8-conn
        border flood, and per-edge flood; none reproduce the pair where a
        box interior flush against the RIGHT grid edge fills while an
        identically-walled pocket against the TOP edge does not, AND a large
        cross-enclosed region stays unfilled. The true exterior definition is
        subtle and not translation-equivariant; also would require an
        iterative flood unroll. Not confidently encodable.

Emitting overfit models for any of these would be rejected by the grader's
held-out gate, so this family intentionally yields nothing.
"""


def candidates(examples):
    return []
