"""family_dc_0 — MINIMAL DYNCROP recompile attempt for tasks 177, 310, 365, 201.

Result: NO win shipped.  After measuring the incumbent out_blend6 graphs and the
theoretical floor of the minimal-dyncrop arsenal, all four incumbents already sit
at (or below) that floor, or the task is not a pure dyncrop.  Emitting anything
would either regress cost or require a graph I cannot verify strictly cheaper, so
candidates() returns [] (the HARD GATE forbids shipping a regression).

Per-task findings (cost = params + sum of named-intermediate bytes; input/output free):

  177  verify_7468f01a = compress(I) then vmirror.
       Empirically (265/265 local) the non-uniform rows/cols are contiguous, so
       compress == bbox-crop, and bbox-crop+vmirror == output.  A dynamic Slice
       would give the crop for free but produces dim_param (dynamic) value_info ->
       calculate_memory() returns None -> REJECTED.  The only static path is the
       fixed-K OneHot row/col selection matrices, which is EXACTLY what the
       incumbent does (K=8): two OneHot[8,30] (960 each) dominate.  Incumbent
       already collapses to a single-channel value crop then Equal->bool crop
       (the cheap trick).  cost=3249, at the floor.  No cheaper static form exists
       (fp16/bool selection matrices break Einsum dtype; single value image is
       [1,1,30,30]=3600 > savings).

  310  verify_c909285e = pick the min-area single-colour box-outline object
       (partition/per-colour, toindices==box), then subgrid.  Selection is already
       done in tiny scalar tensors from per-colour row/col statistics (no flood
       fill).  The crop is a direct dynamic Slice of the one-hot input ->
       [1,10,8,8] float = 2560, which is CHEAPER than the OneHot-einsum crop
       (2816) for a 30-wide grid.  cost=3215, at the floor.

  365  verify_e50d258f = pick the 4-connected non-bg component with the most
       colour-2 cells, then subgrid.  Fixed 10x10 input, output cap 6x6.  A true
       connected-component / flood-fill needs many [1,1,10,10] iterates that blow
       the budget.  The incumbent avoids CC with a corner/TopK scheme entirely in
       <400-byte tensors plus a Gather-based crop (888 bytes, cheaper than the
       OneHot-einsum crop 1944 for small K).  cost=3818; its tiny-tensor selection
       is already leaner than any CC-free rebuild I could verify.

  201  verify_846bdb03 = NOT a dyncrop.  Output is never a subgrid of the input
       (0/266 pairs); the graph recolours/paints content (paint/shift/recolor in
       the verifier).  Out of scope for this arsenal.

If a future generator change lowers an output-size cap or exposes an interior
uniform row, revisit 177 (bbox-crop path) — the fingerprint check below is kept
for that purpose.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# fingerprint gate kept for documentation / future reuse (177 bbox-crop form)  #
# --------------------------------------------------------------------------- #
def _ref_177(a):
    """compress(I) then vmirror — the true rule for task 177."""
    a = np.asarray(a, int)
    keep_r = [i for i in range(a.shape[0]) if len(set(a[i].tolist())) != 1]
    keep_c = [j for j in range(a.shape[1]) if len(set(a[:, j].tolist())) != 1]
    if not keep_r or not keep_c:
        return None
    return a[np.ix_(keep_r, keep_c)][:, ::-1]


def candidates(examples):
    """Return []: every target is already at/below the arsenal cost floor or is
    not a pure dyncrop.  Shipping any graph here would regress cost or be
    unverifiable-cheaper, which the HARD GATE forbids."""
    return []
