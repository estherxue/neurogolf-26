"""family_sn_6 — SINGLE-NODE COMPILE campaign, batch 6.

Tasks surveyed: 233, 366, 191, 110, 204, 80, 169, 189, 240, 242.

Each task was read from its true DSL verifier (_rearc/verifiers.py) and checked
against all train + test + arc-gen pairs for three fixed-map families: a global
per-color recolor (Gather axis=1), a fixed D4 transform, and a fixed (mirror,
offset) crop. NONE matched any task. Every rule is data-dependent — it needs
connected-component labelling, per-object variable-count logic, or a
content-selected crop location — so no zero-memory single-node graph reproduces
it across the generator distribution, and shipping a fixed graph would regress
the incumbent pool net on fresh samples (forbidden by the hard gates).

  233 (97a05b5b): solid square with marker-shaped holes; forced exact-cover fill
                  then dynamic top-left crop. Variable objects/crop. (family_r233
                  already builds the full data-dependent graph.)
  366 (e6721834): frontier split, occurrence-driven object stamping, variable
                  crop. Input/output shapes both vary. Data-dependent.
  191 (7df24a62): partition + D4 occurrence completion + trim. Content-driven.
  110 (484b58aa): hperiod-based periodic tiling of a minority-color stamp.
                  Period depends on the input; per-object. Data-dependent.
  204 (868de0fa): CCL objects -> fill even-height squares with 2, odd with 7.
                  Output color of a cell depends on its object's shape/size (CCL).
  80  (39e1d7f9): compress + objects + frontier-count upscale + subgrid crop.
                  Highly data-dependent.
  169 (6e82a1ae): CCL objects -> recolor by cell count (2->3, 3->2, 4->1).
                  Per-object size; needs connected-component labelling.
  189 (7c008303): subgrid of largest object, upscaled small-object stamp overlay.
                  Content-selected crop (no fixed offset). Data-dependent.
  240 (9d9215db): 4-fold D4 symmetrization + per-object ray shooting. Iterative,
                  data-dependent.
  242 (9ecd008a): pick vmirror/hmirror minimizing zeros in the zero-region, then
                  subgrid of that region. Content-selected mirror + crop.

Empirical confirmation (all pairs): per-color recolor = False and fixed D4
transform = [] for every same-shape task (191,110,204,80,169,240); fixed
(mirror,offset) crop = [] for every crop task (189,242,366; 233 varies). Hence
this family routes nothing.
"""
from __future__ import annotations


def candidates(ex):
    """No single-node fixed-map form exists for any task in this batch."""
    return []
