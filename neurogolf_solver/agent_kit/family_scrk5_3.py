"""family_scrk5_3 -- deep-dig attempts for the hardest-unsolved slice U[3::4]:
tasks [23, 66, 89, 118, 157, 175, 209, 255, 349].

Every train pair (plus a sample of the 262 arc-gen pairs) was read and the exact
generator rule reverse-engineered for each task.  The grader gates on EXACT
equality over ALL arc-gen pairs, and the round forbids cropping / size-global
patching, so an approximate or overfit graph scores 0.  None of these nine can
be written as an exact, generalising opset-10 static graph under those rules;
the per-task findings (what the rule actually is, and precisely why it is not
statically expressible) are recorded below.

Rather than overfit, candidates() only emits models a numpy reference reproduces
EXACTLY on every provided pair, using origin-anchored primitives that genuinely
generalise (identity, H<->W transpose, bijective channel recolour).  None of the
nine match those, so the slice yields nothing new.

Per-task infeasibility
----------------------
23  Recolour a single 5-drawn shape into {2,8} (colours {0,5}->{0,2,8}).  Rule is
    bounded-local (0 conflicts on 7x7 windows, but 1161 conflicts on 3x3), so it
    is NOT a compact 3x3 conv/LUT and a dense 7x7 ternary LUT is un-enumerable.
    The 8-vs-2 split is not a clean local predicate (a cell inside a 2x2 all-5
    block can be either), so no small computed formula reproduces it exactly.

66  Connect-two-markers: a 2-cell "3" seed and a 2-cell "2" target are joined by
    an L-shaped path drawn in colour 3, first leg along the seed's orientation.
    Path length and the turn location are data-dependent and unbounded -> not a
    static graph.

89  Template stamping: a small multi-cell template (carrying an anchor colour) is
    copied onto every matching single-cell marker, aligned on the anchor and
    reflected.  Per-object, variable count, variable placement -> object level.

118 Plus-completion: clusters of colour-2 form (partially hidden) plus/cross signs;
    each is restored to a FULL symmetric plus of radius L drawn in 8.  Verified
    bounded-local (0 conflicts on 7x7).  A tensor formulation (per-cell arm
    reduction + symmetric fill) reaches 235/267 pairs but cannot reach 100%:
    L is a GLOBAL arm-length shared by all crosses, centres are hidden (the centre
    cell is often an erased 5-gap, and a pure horizontal 2-line with no vertical
    seed still grows a vertical arm), and colinear arm-cells belonging to DIFFERENT
    crosses are locally indistinguishable from a real centre.  These ambiguities
    are irreducible for any static local operator -> not exactly expressible.

157 The colour-2 top block's holes are filled with 1 and the count/shape of the
    bottom 5-blobs drives how far the 1s propagate.  Counting + shape transfer
    across the grid -> non-local, not static.

175 Nearest-seed / distance-field recolour with diagonal expansion of a 6-field;
    the recolour of every interior cell depends on a global distance-to-seed
    computation -> not expressible as a bounded static graph.

209 Objects are gathered and re-tiled into a composite whose SIZE varies per
    example (9x14, 7x7, 11x11, 8x14).  Requires cropping to a data-dependent
    output size, which is explicitly forbidden this round.

255 A thick plus/branch-shaped region kept empty of noise is filled with 3.  The
    fill equals the 1-erosion of the empty complement, but random noise leaves
    accidental empty 3x3 pockets that survive erosion and leaves ragged one-cell
    edges adjacent to the reserved strip; recovering the exact latent rectangle-
    cross from the noise-confounded complement is not a clean local operation and
    fails EXACT on arc-gen.

349 Each colour-9 block is wrapped in a 3-border and given 1-tails / arrows whose
    length and direction depend on the block and free space.  Per-object frame +
    variable-length tail construction -> object level, not static.
"""
from __future__ import annotations

import numpy as np

from builders import identity, transpose_hw, recolor_gather
from ng_utils_shim import CHANNELS


def _pairs(examples):
    out = []
    for split in ("train", "test"):
        for e in examples.get(split, []):
            out.append((np.array(e["input"]), np.array(e["output"])))
    return out


def _bijective_recolor(pairs):
    """Length-10 gather index if one consistent per-colour bijection maps every
    input cell colour to its output colour on ALL pairs (same shape), else None."""
    fwd = {}
    for ai, ao in pairs:
        if ai.shape != ao.shape:
            return None
        for ci in range(CHANNELS):
            vals = ao[ai == ci]
            if vals.size == 0:
                continue
            u = np.unique(vals)
            if u.size != 1:
                return None
            oc = int(u[0])
            if fwd.get(ci, oc) != oc:
                return None
            fwd[ci] = oc
    inv = {}
    for ci, oc in fwd.items():
        if oc in inv and inv[oc] != ci:
            return None  # not invertible -> not a channel gather
        inv[oc] = ci
    src = list(range(CHANNELS))
    for oc in range(CHANNELS):
        src[oc] = inv.get(oc, oc)
    return src


def candidates(examples):
    pairs = _pairs(examples)
    if not pairs:
        return
    # 1) identity
    if all(ai.shape == ao.shape and np.array_equal(ai, ao) for ai, ao in pairs):
        yield ("identity", identity())
        return
    # 2) H<->W transpose (origin-safe)
    if all(ai.shape[::-1] == ao.shape and np.array_equal(ai.T, ao) for ai, ao in pairs):
        yield ("transpose", transpose_hw())
        return
    # 3) bijective channel recolour
    src = _bijective_recolor(pairs)
    if src is not None and src != list(range(CHANNELS)):
        cmap = {src[oc]: oc for oc in range(CHANNELS)}
        if all(np.array_equal(np.vectorize(lambda v: cmap.get(int(v), int(v)))(ai), ao)
               for ai, ao in pairs):
            yield ("recolor", recolor_gather(src))
            return
    return
