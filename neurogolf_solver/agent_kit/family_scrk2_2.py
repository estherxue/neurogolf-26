"""family_scrk2_2 -- analysis of the unsolved slice U[2::5] = [22,66,96,138,173,209,264,366].

Every task in this slice was reverse-engineered from all train+test+arc-gen pairs (266 pairs
each).  Each one requires NON-LOCAL, per-object, data-dependent structure that a cheap static
opset-10 graph cannot express while generalizing to the hidden private set.  In particular the
opset-10 arsenal lacks GatherND/ScatterND and cannot apply distinct data-dependent 2-D
translations to different connected components simultaneously, which is the crux of most of
these tasks.

  22  (FIXED 11x11 -> 3x3): several small objects, each anchored to a colour cell, are gathered
                            and merged onto a shared 3x3 frame by their per-object anchor offset.
                            Non-local object gather -> not a fixed crop / pointwise map.

  66  (VAR, same-shape):    draw a colour-3 path that ROUTES around colour-8 obstacles from the
                            3-marker to the 2-marker (maze / shortest-path).  Path-finding is not
                            a bounded local or global reduction op.

  96  (VAR -> odd square):  multiple partial "ring" fragments are overlaid into ONE concentric,
                            4-fold-symmetric figure; output side (7 vs 11) = number of nested
                            rings, i.e. content-dependent, not a function of input size.

  138 (VAR):                four border lines (top/bottom/left/right, distinct colours) frame the
                            grid; scattered interior markers are accumulated into a per-row/col
                            cumulative bar chart.  Counting/accumulation with content-dependent
                            output size.

  173 (VAR, same-shape):    a small point-symmetric TEMPLATE (plus / X / colinear triple) is
                            inferred from one complete instance and stamped at every lone
                            "centre" marker.  The kernel is READ FROM THE INPUT (data-dependent
                            convolution weights) and its geometry differs per grid and per centre
                            colour -> not a fixed local LUT.

  209 (VAR):                crop to a data-dependent 4-corner box; inside, each solid block is
                            extended by a "successor"-colour block dictated by a separate 2xN key
                            legend (colour-chain expansion).  Output size varies for equal input
                            size.

  264 (in 14-16 -> 9x9):    ~9 scattered 3x3 objects are gathered and arranged into a 3x3-of-3x3
                            layout keyed by each object's zone/direction relative to the centre.
                            Per-object gather with content-dependent placement.

  366 (VAR -> half):        grid = two equal abutting panels with different backgrounds; one holds
                            multi-cell OBJECTS, the other holds MARKERS.  Rule (verified value-
                            EXACT on all 266 numpy pairs): translate each object so its special-
                            colour cells land exactly on the matching marker group, stamp the whole
                            object, drop non-matching objects.  Fully cracked, but ~half the
                            objects carry MULTIPLE special cells, so it needs connected-component
                            segmentation followed by a *distinct per-object* 2-D translate+scatter.
                            Simultaneous distinct data-dependent shifts of different components are
                            not expressible with the static MatMul-shift / Conv / ConvTranspose /
                            doubling-CA arsenal under opset-10 (no GatherND/ScatterND), so no cheap
                            private-safe ONNX graph exists for it.

Mechanical checks performed (all negative for a static solver):
  * geometric / symmetry transforms (identity, flips, rot180, transpose, OR-with-reflections)
    match no pair on any task.
  * per-cell KxK -> colour LUT (K in {3,5,7}) is inconsistent for 66/96/138/173/366 (not a
    local function).
  * output-shape as a function of input-shape is FALSE for 96/138/209/264 (equal input sizes
    yield differing output sizes) -> even the output dimensions are content-dependent.
  * 366 was taken furthest: an exact numpy oracle (panel split + component match + exact-landing
    translate) reproduces all 266 pairs, confirming the rule but also confirming it needs
    per-component data-dependent scatter that opset-10 cannot provide.

Emitting an approximate model would only fail the EXACT gate (and risk the private set), so this
family proposes nothing.
"""

def candidates(examples):
    return []
