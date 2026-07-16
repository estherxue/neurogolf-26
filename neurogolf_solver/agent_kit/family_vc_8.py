"""family_vc_8 — assigned tasks task076 (36d67576), task191 (7df24a62), task366 (e6721834).

VERDICT: all three are the same "hidden template stamping" family and are genuinely
too data-dependent for a static opset-10 graph. No model is emitted (emitting a
heuristic that is not grader-exact AND held-out-exact is worse than emitting nothing).

Decoded rule (verified: a pure-numpy reproduction of the full DSL matched
verify_36d67576 EXACTLY on all 266 train+test+arc-gen grids, so the decode is correct):

  1. Take the LARGEST diagonal connected multicolor object = the TEMPLATE
     (DSL: x1 = argmax(objects(I,F,T,T), size)).
  2. Its "key" sub-pattern = the template cells whose color also appears in the
     small scattered fragments (x4 = palette of the other objects).
  3. For each of the 8 dihedral orientations of the template, normalize its key
     pattern and find every exact-color `occurrences` of it in the grid.
  4. At each occurrence, stamp the FULL oriented template (aligning key-ulcorner),
     merge all stamps, and paint onto the grid.
  task366 first splits the grid into two panels (data-dependent vsplit/hsplit chosen
  by comparing #hline vs #vline frontiers), orders the panels by numcolors to pick a
  base panel vs a template panel, then does the same stamp search across 4 orientations.

Why it cannot be a static graph (concrete non-static DSL steps):

  * `argmax(objects(...), size)` — selecting the largest connected component requires
    CC labeling + per-component cell counting + argmax over a DATA-DEPENDENT number of
    components. In opset-10 with Loop/Scan/NonZero/Unique/Compress all BANNED, there is
    no static segment-reduction/argmax over a runtime label set. (Bounded masked-maxpool
    label propagation could approximate labeling but not exact largest-by-size selection
    across arbitrary geometries.) The template kernel is therefore runtime-variable in
    BOTH shape and color content (verified: 266 distinct normalized templates over 266
    grids for 076; 185 over 267 for 191 — no fixed-kernel shortcut), needed at 8
    orientations, and must exact-multicolor-match without false-firing on dense noise.
  * task366 additionally: `branch(greater(#hlines,#vlines), vsplit, hsplit)` picks the
    split AXIS at runtime and `order(panels, numcolors)` picks which panel is the base
    — both fully data-dependent (all four axis x base-role combinations occur across
    arc-gen), so neither the split geometry nor the panel roles are static.

The convolution stamping trick (conv match -> transposed-conv stamp) absorbs the
variable stamp COUNT/POSITION/per-hit-orientation, but the largest-CC template
selection (and 366's data-dependent split) is the wall that no arsenal primitive
clears exactly within the banned-op set.
"""


def candidates(examples):
    return
    yield  # unreachable; keeps this a generator
