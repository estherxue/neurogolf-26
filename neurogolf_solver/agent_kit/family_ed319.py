"""family_ed319 -- ENRICHED-ARSENAL deep-rebuild attempt of task319 (koji, hash ce602527).

INCUMBENT: out_blend10/onnx/task319.onnx  -- 316 nodes, opset 12.
  Real grader:  memory=21834  params=269  cost=22103  points=14.9965  (report says 15.0).
  Per-tensor memory floor is dominated by:
     input_u8  [1,10,30,30] uint8 = 9000   (43% of all memory; shared Cast of `input`)
     3 object planes [1,30,30] uint8       =  2700
     ~150-tensor tail (signature packing + 4-edge clip masks + 38-entry fallback
     lookup) that resolves the discriminator                    ~10000

TASK MECHANICS (fully reverse-engineered from the generator, tasks/task_ce602527.py):
  * Grid filled with a background colour.  Two Conway sprites (idx0, idx1), each
    3..5 wide/tall, drawn in colours[0]/colours[1].  Sprite idx0 is ALSO drawn 2x
    magnified (nearest-neighbour, each pixel -> 2x2 block) in `magcolor`, placed so
    exactly ONE edge strip is clipped off the grid (some_hidden must be true).
  * OUTPUT = the small sprite idx0's tight bbox, painted in colours[0] on bg
    (one-hot: channel colours[0]=1 on sprite pixels, channel bg=1 elsewhere in the
    bbox, everything outside the bbox all-zero).  So the job is: identify which of
    the two small sprites was the one that got magnified, and emit its bbox.

CLEAN DISCRIMINATOR I DERIVED AND VERIFIED (numpy, `predict` below is the reference):
  For each non-bg colour s, magnify(crop(s)) 2x and test a boundary-constrained
  clip-match against every other colour's full-grid mask: does there exist a
  placement where magnify(s) clipped to the ACTIVE grid == that colour exactly,
  with off-grid strips only on grid-touching sides?  Score s by the MIN off-grid
  strip WIDTH over valid placements.  c0 = argmin(score) with ties broken by
  LARGER pixel-area, then larger colour id.
     - LOCAL: 267/267 exact (train+test+arc-gen).
     - FRESH generator: 20000 samples -> 99.905% exact vs the generator;
       19 mismatches, of which 15 match the incumbent and only 4 (0.02%) match
       neither (genuine inherent-ambiguity where the incumbent wins via its
       memorised 38-entry fallback_sig_table).  => gate-c "match gen-or-incumbent"
       passes at 99.98%.
  KEY finding: the larger-pixel-area tie-break REPLACES the incumbent's entire
  memorised fallback table (nodes 273-285 + fallback_sig_table[38,3] +
  fallback_slots[38]) -- it matches the generator on 33/33 of the ambiguous cases
  where the incumbent needs its lookup.  (Simpler variants fail: plain
  clip-match-existence + larger-area = 0.85% mismatch; the min-clip-WIDTH cost is
  essential, and it is NOT a function of pixel-count/area/bbox-excess -- those all
  give 0.9-15% error -- it genuinely requires the boundary-constrained placement
  search because of re-crop, i.e. clipping one edge can empty a perpendicular
  bbox row/col.  The train examples use odd/multi-block clips (magcol=-1,-2,-3;
  magrow bottom-clip of 2 or 4), which break every downscale/2x2-block shortcut,
  so the full magnify+search is mandatory to pass gate (b).)

ONNX COST ASSESSMENT (result: no VERIFIED model that strictly beats 22103):
  The discriminator needs, per candidate, a placement-search clip-match with a
  min-off-grid-WIDTH cost (existence alone = 0.85% mismatch, fails gate-c; the
  WIDTH cost is what drops it to ~0.1%).  Routes considered:
    - Explicit unrolled patch-space search (~5x5 offsets x Slice/Equal/ReduceMin
      x 2-3 candidates) is ~200 small-tensor ops ~= 13k bytes.  Added to per-object
      planes (input_u8 [1,10,30,30]=9k is not avoidable more cheaply -- gathering 4
      active channels costs 3600 as float or still needs a full cast) and one-hot
      output (~3-5k), this OVERSHOOTS 22103.
    - Best path found (NOT built/verified, est. ~15-18k => ~+0.2..+0.4 pt): 6x
      QLinearConv correlations (each candidate s vs each other colour as target),
      exact-clip-match detected as corr == target_area, ArgMax -> placement, then
      DERIVE the off-grid strip width from (placement, 2*hs/2*ws, board_h/w) with
      scalar ops; score = min width; c0 = argmin, larger-area, larger-colour.  The
      6 conv maps (~[1,1,~40,40] int8 ~= 9.6k) plus 3 penalty signals + planes +
      output land ~15-18k on paper.  Risks that stopped me from shipping it as a
      VERIFIED win: (a) exact-match-under-clipping detection via a signed penalty
      signal + threshold is fiddly at opset-12 QLinearConv requant semantics;
      (b) multi-valid-placement handling (ArgMax picks one; min-width needs all);
      (c) width decode + it must stay exact on all 267 local incl. the odd/multi
      block-clip train cases AND >=3000 fresh AND invariant across ORT opt levels.
      The margin over 22103 is thin enough that any one of these going 1-2k over
      erases the win, so I did not claim it without a measured, gated model.
    - Replacing ONLY the incumbent's fallback table with the larger-area rule is a
      valid but marginal surgery (~-1.2k bytes, +~0.05 pt); removing the fallback
      outright FAILS a local example (agi (4,1)) -- it is load-bearing -- so it must
      be rewired into the exact decision node, high-risk for +0.05.
  Net: with the enriched arsenal (runtime-bias QLinearConv binary detect,
  adjoint-conv paint, static Gather, u8/i8 masks) the OUTPUT and the EXISTENCE test
  are cheap, but the min-clip-WIDTH cost this discriminator needs pushes a clean
  rebuild to ~the incumbent's cost.  I could not produce a model that both stays
  exact on the odd-clip train cases AND measures strictly below 22103.  Reported as
  floored; the ~15-18k conv route above is the concrete next attempt.

The reference discriminator (verified correct) is kept below for future use; this
family emits no candidate because none strictly beats the incumbent's 22103 cost.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Reference (numpy) discriminator -- verified 267/267 local, 99.905% fresh gen. #
# Not emitted as ONNX: see module docstring for the cost analysis.             #
# --------------------------------------------------------------------------- #
def _bbox(m):
    ys, xs = np.where(m)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _crop(m):
    if m.sum() == 0:
        return m[:0, :0]
    r0, r1, c0, c1 = _bbox(m)
    return m[r0:r1 + 1, c0:c1 + 1]


def _magnify(S):
    return np.repeat(np.repeat(S, 2, 0), 2, 1)


def _min_clip(Ms, magmask):
    """Min off-grid strip width over boundary-constrained placements where
    Ms clipped to the active grid == magmask exactly; None if no valid placement."""
    H, W = magmask.shape
    h, w = Ms.shape
    if magmask.sum() == 0:
        return None
    mr0, mr1, mc0, mc1 = _bbox(magmask)
    total = int(magmask.sum())
    top, bot = mr0 == 0, mr1 == H - 1
    lft, rgt = mc0 == 0, mc1 == W - 1
    best = None
    for dr in range(mr0 - h + 1, mr1 + 1):
        for dc in range(mc0 - w + 1, mc1 + 1):
            gr0, gr1 = max(0, dr), min(H, dr + h)
            gc0, gc1 = max(0, dc), min(W, dc + w)
            if gr0 >= gr1 or gc0 >= gc1:
                continue
            if (dr < 0 and not top) or (dr + h > H and not bot):
                continue
            if (dc < 0 and not lft) or (dc + w > W and not rgt):
                continue
            win = Ms[gr0 - dr:gr1 - dr, gc0 - dc:gc1 - dc]
            if int(win.sum()) != total:
                continue
            if mr0 < gr0 or mr1 >= gr1 or mc0 < gc0 or mc1 >= gc1:
                continue
            if not np.array_equal(win, magmask[gr0:gr1, gc0:gc1]):
                continue
            off = max(0, -dr) + max(0, dr + h - H) + max(0, -dc) + max(0, dc + w - W)
            if best is None or off < best:
                best = off
    return best


def predict(grid):
    """Reference solver: returns the output grid (HxW ints) or None."""
    g = np.array(grid)
    vals, cnts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(cnts)])
    nb = [int(v) for v in vals if v != bg]
    if len(nb) != 3:
        return None
    crops = {v: _crop((g == v).astype(int)) for v in nb}
    masks = {v: (g == v).astype(int) for v in nb}
    scores = {}
    for s in nb:
        Ms = _magnify(crops[s])
        best = None
        for mg in nb:
            if mg == s:
                continue
            c = _min_clip(Ms, masks[mg])
            if c is not None and (best is None or c < best):
                best = c
        if best is not None:
            scores[s] = best
    if not scores:
        return None
    c0 = min(scores, key=lambda s: (scores[s], -int(crops[s].sum()), -s))
    sprite = crops[c0]
    return np.where(sprite == 1, c0, bg)


def candidates(ex):
    # No ONNX form of the min-clip-WIDTH discriminator is strictly cheaper than the
    # incumbent's 22103-cost graph (see module docstring).  Emit nothing so the
    # incumbent pool net is retained for task319.
    return []
