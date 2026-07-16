"""family_scrk_4 -- slice U[4::5] of unsolved.json = tasks [44,79,101,143,175,209,264,366].

Every task in this slice was fully reverse-engineered (exact rules below), but each
requires a PER-EXAMPLE, data-dependent template / shape-match / assembly step that
cannot be expressed as an EXACT static opset-10 graph without banned ops
(Loop/Scan/NonZero/...).  The arsenal's stamp/CA/reflection/crop primitives only
help when the template or displacement is TASK-fixed; here it is read from each
input.  So no exact, generalizing candidate is emitted (over-proposing a wrong
static approximation would just be rejected by the exact grader anyway).

Reverse-engineered rules (for the record / future attempts):

- 44  JIGSAW: 5-boxes each have a hole; loose colour blobs teleport to fill the
      hole whose SHAPE matches the blob.  Needs per-blob shape matching -> not static.
- 79  MAJORITY-MOTIF: field of 3x3 motifs (several identical copies + odd ones).
      Output(3x3) = the motif that repeats the most.  Needs motif detection+count
      +argmax+locate+crop -> per-example, not static.
- 101 SCALE-STAMP: a per-example multicolour template with 2-anchor cells; each
      k*k block of 2s = one template pixel scaled by k.  Redraw the 1-cells scaled.
      Kernel = the (per-example) template -> data-dependent Conv kernel, not static.
- 143 SHAPE-MATCH RECOLOR: a legend object sits by the 5-frame at the origin; the
      unique OTHER object congruent to the legend is recoloured to 5.  Needs
      correlation with the per-example legend shape -> data-dependent kernel.
- 175 SYMMETRIC DENOISE: 21x21 concentric-ring field with 0-holes; holes filled so
      the result is transpose-symmetric + ring-consistent.  Not a single symmetry
      fill (tested T/flip/rot/antiT/D4 -- none exact); reconstruction is global.
- 209 BOX-CROP + OVERLAY: crop the 4-corner-marked box, then overlay a second
      small legend pattern.  Assembly of two regions -> not static.
- 264 ASSEMBLY -> 9x9: composes multiple scattered sub-blocks into a fixed 9x9.
      Per-example placement -> not static.
- 366 TWO-PANEL STAMP: one panel holds several per-example motifs (keyed by an
      embedded colour), the other holds markers; each motif is stamped at its
      marker.  Reads per-example motif -> data-dependent kernel, not static.
"""


def candidates(examples):
    return []
