"""family_fcrk_1 — crack attempt over unsolved slice U[1::3].

Tasks examined: [18, 46, 76, 89, 118, 158, 191, 219, 255, 319, 366].

Every one of these is a DATA-DEPENDENT transform that cannot be expressed as a
static, origin-anchored graph (verified by reading all train/test pairs):

  18  — collect scattered fragments and assemble them at a data-dependent site.
  46  — remove '5' markers, shift/recolour bars; width shrinks per-content.
  76  — per-object completion/extension (routing keyed on each object's shape).
  89  — marker-stamp: a fixed motif stamped at each '2' marker, but the stamp is
        REFLECTED/rotated per marker (orientation is data-dependent).
  118 — recolour 5->8 to complete '2'-marker crosses symmetrically; the fill
        geometry depends on each cross's arm lengths (a 5x5 window is only
        *consistent* by memorising 836 distinct windows -> overfits held-out).
  158 — stamp a 3x3 key at each marker, ROTATED so its colours align (per-marker
        orientation) -> data-dependent.
  191 — draw a 5x5 block of 1s around each cluster of '4' markers, preserving the
        cluster's own 4s; #clusters and their content vary.
  219 — extend each partial staircase/checkerboard object with colour 1 to match
        a reference object; per-object, count varies.
  255 — recolour the LARGEST connected 0-component to 3 (only 221/600 of the 0s);
        connected-components / flood-fill -> data-dependent.
  319 — select one object (the symmetric one) and CROP to it; output size and crop
        position are data-dependent.
  366 — split grid into two halves and route one half's motif onto the other
        half's marker positions; data-dependent.

None fit the crackable shapes (fixed template-match+stamp, global symmetry about
the origin, interior-fill of a clean rectangle). No candidate is emitted; the
harness rejects wrong guesses anyway, so returning nothing is the correct,
non-overfitting result.
"""
from __future__ import annotations


def candidates(examples):
    return []
