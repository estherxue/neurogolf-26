"""family_sn_1 — SINGLE-NODE COMPILE campaign.

Tasks surveyed: 177, 310, 340, 365, 42, 69, 79, 85, 90, 105.

Result: every one of these tasks implements a *data-dependent* rule, not a fixed
geometric/recolor/permutation/crop map, so none admits a zero-memory single-node
form that would pass fresh generator samples:

  177 (7468f01a): compress(I) then vmirror  -> content-based crop, variable out size.
  310 (c909285e): subgrid of the smallest square frame object     -> dynamic crop.
  340 (d687bc17): gravitate trimmed objects toward matching colors -> per-object move.
  365 (e50d258f): subgrid of the block with most color-2 cells     -> dynamic crop.
   42 (22233c11): per-object upscaled diagonal frontiers, fill 8   -> per-object CCL.
   69 (321b1fc6): stamp normalized max-color object at each corner -> per-object CCL.
   79 (39a8645d): extract most-common normalized object shape      -> dynamic content.
   85 (3bdb4ada): partition-based parity fill                       -> data-dependent.
   90 (3eda0437): occurrences/periodic pattern fill                 -> data-dependent.
  105 (4612dd53): least-color box frontier fill                     -> data-dependent.

Because a single fixed node cannot reproduce any of these across the generator
distribution, this family routes nothing. Shipping a fixed graph here would
regress the incumbent on fresh samples, which the hard gates forbid.
"""


def candidates(ex):
    # Fingerprint the task's train pairs; none of the surveyed tasks reduces to a
    # fixed single-node map, so no candidate graph is emitted.
    return []
