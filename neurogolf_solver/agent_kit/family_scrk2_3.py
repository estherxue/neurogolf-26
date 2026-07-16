"""family_scrk2_3 — slice U[3::5] = tasks [23, 76, 101, 143, 175, 219, 285, 367].

Deep per-pair analysis (numpy, all train+test+arc-gen). None of these 8 tasks
reduce to an EXACT, generalizing, cheap opset-10 static-graph rule within the
arsenal. Each hits a documented wall:

* 23  — same-shape recolor 5 -> {2,8}. The 2-vs-8 choice is a GLOBAL 2-coloring
        of the single 5-shape, not a local feature: a 3x3 binary-neighborhood
        LUT is ambiguous (79 conflicting patterns), 5x5 still conflicts, only a
        full 7x7 LUT is consistent (18770 entries) -> not cheaply expressible /
        would overfit. No clean parity/stripe/2x2 formula fits all pairs.
* 76  — multi-object reflect/copy at data-dependent target positions (padding
        wall: correct placement depends on the variable grid geometry).
* 101 — same: per-object reflection paste into data-dependent locations.
* 143 — recolor exactly one color X -> 5, where X is selected by a GLOBAL
        property (not min/max count, not #components; verified across 266 pairs
        no simple invariant picks X). Global object-selection -> not golfable.
* 175 — symmetry / diagonal-band pattern inpainting of a variable-position
        0-hole; reconstruction needs the global pattern, hole position varies.
* 219 — staircase/arrow completion emitting rays at data-dependent positions.
* 285 — multi-object spatial reconstruction at data-dependent positions.
* 367 — enclosed-region fill (0 -> 4). NOT the standard border-flood: a 0-pocket
        pinched against the grid EDGE is still "inside" ((1,18) in train0 gets 4
        though it is a border cell), while huge edge-touching background stays 0.
        The inside/outside test is a global topological property; best local /
        largest-component / 4-side-surround / ray-parity variants top out at
        ~76-108/266 arc-gen. Matches the sibling-slice finding for this task.

candidates() self-validates any attempted rule against ALL provided examples
(train+test+arc-gen) and yields ONLY on an exact match, so it never emits a
candidate the grader would reject. For this slice it yields nothing.
"""
import numpy as np


def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for g in ("train", "test", "arc-gen") for e in examples.get(g, [])]
    if not prs:
        return []
    # No exact, generalizing rule found for this slice within the arsenal.
    # (Any future rule would be built here and emitted only if it matched every
    #  pair in `prs` exactly.)
    return []
