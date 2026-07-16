"""family_cs2_0 — FINAL COMPLETE-SWEEP recompile pass.

Targets: 381, 51, 63, 348, 70, 45, 61, 109, 33, 323, 295, 36.

Result: NO strictly-cheaper correct graph found. All twelve incumbents in
out_blend6/onnx are already structurally minimal for their (complex) rules:

  * 381,51,63,70,45,33,36  — cost 1639..1928 (17.4..17.6 pts): at floor.
    Object-detection / component rules (objects+colorfilter, shoot-from-cell,
    frontier-box, crop-by-variance).  No fixed permutation/crop shortcut; the
    graphs are already tiny single-channel-label pipelines.

  * 348,109,295,61,323 — cost 4243..6423 (16.2..16.7 pts).  Each already uses
    the single-channel uint8 label trick (Einsum widths, Mod/QLinear periodic
    grids, Where triangles) and terminates in one Pad to [1,1,30,30] label
    (pad sentinel 255) followed by a FREE Equal->one-hot output.  The rules are
    genuinely non-fixed (periodic replication with input-derived modulus,
    mirror-selection, diagonal shoot into a half-height canvas), so no low-rank
    Einsum / single-Gather remap reproduces them.

The only remaining lever is moving the terminal [30,30] label Pad (3600 B) into
the one-hot domain (Equal at true size, then Pad the small one-hot to the free
output).  That saves ~2000-2600 B on 348/109/295 but is a pure representation
shuffle inside the log-wall marginal regime (+0.5..0.67 pts, i.e. < "halving
memory") — not a fundamentally cheaper structure — and it risks shape-inference
blow-up to [1,10,30,30].  Per the cost brief this does not qualify as a win, so
nothing is emitted.

candidates() yields nothing; family_test / evaluate keep the incumbents.
"""
from __future__ import annotations


def candidates(example):
    return []
