"""family_pcrk9_1 — deep re-examination of unsolved slice U[1::4].

Tasks: 18, 54, 80, 118, 158, 191, 233, 285, 366.

Every train pair (and many arc-gen) re-read. Each task requires
per-object variable placement / data-dependent stamping / reflection /
cross-panel routing whose kernel or geometry is READ FROM THE INPUT
(differs per example), so it is NOT expressible as a translation-
equivariant static ONNX graph that generalizes to held-out arc-gen.
Emitting anything here would be an overfit the grader's held-out gate
rejects, so this module intentionally yields nothing.

Per-task findings (why rejected):
- 18:  objects routed to distinct colored target markers (per-object
       destination lookup). Variable placement.
- 54:  a template (diamond/plus) is DRAWN IN THE INPUT and stamped at
       each single marker. Kernel is data-dependent (train0 != train1),
       so ConvTranspose w/ static weights cannot reproduce it.
- 80:  data-dependent tile pattern propagated to plus-neighbor grid cells.
- 118: recolor 5-cells whose local neighborhood matches an input-given
       2-template (data-dependent template match; LUT would overfit).
- 158: multiple distinct templates (from input) stamped at markers.
- 191: 4/5-size template (from input, size varies) stamped at marker
       clusters that match its interior 4-subpattern. Data-dependent.
- 233: variable output (0/265 same-shape); complex per-object decode+tile.
- 285: each small object reflected into quadrants keyed by its own corner
       marker colors — reflection about the object's LOCAL axis, not the
       origin, so not expressible for arbitrarily placed objects.
- 366: two stacked panels; markers cross-stamped between panels then one
       variable-size half cropped (0/265 same-shape). Cross-panel + var crop.

Checked for fixed-size shortcuts: 54 and 191 are always 30x30 / 23x23 but
the transform is still data-dependent stamping (fixed canvas doesn't help).
233 and 366 have >30 distinct output sizes -> no fixed crop.
"""


def candidates(examples):
    return []
