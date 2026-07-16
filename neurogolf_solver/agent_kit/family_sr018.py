"""family_sr018 — SIMPLER-EQUIVALENT-RULE hunt for task18 / 0e206a2e.

RESULT: NO_SIMPLER.

Hunt hypothesis (from orchestrator):
    "the clone body = a direct function of the 3 markers — paste the reference
     translated by the marker-centroid delta, with rotation determined by a
     simple marker-geometry invariant — one Conv correlation + one stamp."

What the generator actually does (decoded from task_0e206a2e.py):
    Each of 1-2 sprites is a 4-connected creature: body cells share the MODE
    colour, 3 cells are 3 DISTINCT marker colours.  Both sprites reuse the SAME
    4-colour palette.  Each sprite appears twice — ORIGINAL drawn fully, CLONE
    drawn under one of 4 fixed D4 transforms showing ONLY its 3 markers.  Output
    = the clone(s) drawn FULLY (rotated body + markers); the originals are dropped.

The exact inverse map (verified below): for each reference sprite, the 3 distinct
marker colours give 3 point correspondences ref->clone; those 3 non-collinear
correspondences UNIQUELY determine the D4 element + translation, which is then
applied to the whole reference sprite.  This is *mathematically identical* to the
incumbent's "try the 4 D4 orientations, Conv-correlate the 3 markers, stamp with
ConvTranspose".  The incumbent already IS the minimal closed form.

Empirical tests on fresh generator samples (see report):
  - affine-from-3-correspondences decode is EXACT on 1979/2000 = 98.95%, with
    ~1.7% samples having a non-unique parse — this is the generator's own
    irreducible ambiguity (diagonal marker triple symmetric under transpose),
    provably not a function of the input, matching the incumbent's 2954/3000.
  - reference-blob count is {1: 47%, 2: 53%}: the MAJORITY of inputs contain TWO
    same-palette sprites, so the markers must first be grouped with their bodies.
    That grouping needs a flood-fill/connected-component separation — the exact
    step that dominates the incumbent's 673-node / 23668-byte cost.  There is no
    single-Conv / single-stamp shortcut, because a lone correlation cannot tell
    which of the two identically-coloured bodies a given marker belongs to.

Why the hunt's simplification does NOT reduce cost:
  - "rotation from a marker-geometry invariant" still needs per-colour marker
    POSITION extraction (ArgMax) + a 2x2 affine solve (division) + a
    data-dependent SCATTER of every body pixel to a rotated cell.  In ONNX
    (no Loop/Scan/NonZero), that scatter is precisely a data-dependent
    ConvTranspose — the same op the incumbent uses — and the per-colour position
    tensors are extra NAMED intermediates that would only INCREASE the memory the
    cost model charges, not decrease it.
  - the D4 element being chosen by algebra vs by trying 4 orientations is a minor
    part of the graph and does not touch the dominant flood-fill / kernel /
    stamping memory.

Conclusion: the inverse map has NO closed form simpler than the incumbent's; the
incumbent already implements the minimal-sufficient reconstruction and correctly
returns the deterministic parse on the irreducibly-ambiguous minority.  No
candidate can strictly beat the incumbent's 14.9 pts, so none is emitted.
"""
from __future__ import annotations


def candidates(example):
    # NO_SIMPLER — no rule beats the incumbent; emit nothing.
    return []
