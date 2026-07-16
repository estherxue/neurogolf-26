"""family_sr158 -- SIMPLER-EQUIVALENT-RULE hunt for task158 (hash 6aa20dc0).

TASK
----
Input: one fully-rendered 3x3 reference sprite (idx0, magnification 1) plus, for
each additional magnified sprite, ONLY its two diagonal color-corner blocks
(base cells (0,0)=color0 and (2,2)=color1 of the sprite, each rendered as a solid
mag x mag block, after the sprite's h/v flip). Output: every sprite fully
rendered (the reference template nearest-neighbor-upscaled by its magnification,
flipped, placed at its location).

HYPOTHESIS TESTED (the hunt's premise)
--------------------------------------
"Is magnification recoverable per-sprite from a simple statistic (bbox size
ratio), making reconstruction one Resize/GridSample per scale bucket instead of
QLinearConv stamp banks?"

FINDING
-------
The magnification statistic IS real and trivially recoverable:

    mag(sprite) == the pixel side-length of its solid corner-marker block.

Each shown corner is a solid mag x mag square, so its bounding-box side == mag.
A complete closed-form inverse follows (see solve() below):
  1. background = most common color;
  2. reference template R = the unique fully-rendered multi-color 3x3 component
     (only idx0 is full; marker corners are single-color separate components);
  3. for each marker: mag = corner-block side; orientation = the one flip of R
     (of {R, hflip R, vflip R, rot180 R}) whose two color-corners match the two
     marker blocks; stamp = R_flipped kron ones(mag,mag) placed at the square.

Verified: magnification recovered correctly in 100% of 22000+ fresh samples.
Full-grid reconstruction is exact on ~99.97% of fresh samples; the residual
~0.03% are INHERENT multi-marker pairing ambiguities (a shared corner block
validly pairs with two different markers forming two non-overlapping 6x6 squares
with correct diagonal offset and empty interiors) -- undecidable from the input
alone, so no solver (incumbent included) can be exact on them. This residual is
orthogonal to the magnification hypothesis.

WHY NO CHEAPER ONNX (outcome = NO_SIMPLER)
------------------------------------------
1. The incumbent (out_blend6/onnx/task158.onnx) ALREADY exploits exactly this
   statistic: it carries separate per-magnification matched filters
   w_pair1/w_pair2/w_pair3 (sizes 5x5 / 8x8 / 11x11 = the 3*mag corner-pair
   signatures) and matching stamp banks stamp_idx1/2/3 -- i.e. it recovers mag
   from corner-block size and stamps. There is no un-exploited slack to harvest.
2. The hypothesized "single Resize/GridSample per scale bucket" cannot replace
   the stamp banks. Resize uniformly rescales the WHOLE grid; the sprites have
   MIXED magnifications at arbitrary positions and orientations, so
   reconstruction fundamentally requires per-location, per-orientation,
   per-scale placement. That placement (ConvTranspose / Gather stamping) is
   irreducible and is the stamp bank. Knowing mag does not collapse the three
   scale banks into one Resize because a Resize cannot place different-scale
   sprites at different locations.
3. Cost is dominated by named intermediates (mem ~= 26178) from streaming the
   ~25x25 grid through int8 detection + stamping stages -- already tightly
   int8/QLinearConv-optimized. A from-scratch rebuild would at best re-derive
   the incumbent's architecture with more, not fewer, intermediates.

Hence the equivalent simpler statistic exists and is verified, but it yields NO
construction cheaper than the 14.74-pt incumbent. candidates() proposes nothing
(no regression, no win).

The pure-numpy reference solver is retained for documentation/reproducibility;
it is NOT an ONNX candidate (uses connected components, a banned-op family).
"""

import numpy as np


def _label8(mask):
    H, W = mask.shape
    lbl = np.zeros((H, W), int)
    cur = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lbl[i, j] == 0:
                cur += 1
                st = [(i, j)]
                lbl[i, j] = cur
                while st:
                    y, x = st.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lbl[ny, nx] == 0:
                                lbl[ny, nx] = cur
                                st.append((ny, nx))
    return lbl, cur


def solve(gi):
    """Closed-form inverse via the mag=corner-bbox-side rule. Reference only."""
    g = np.array(gi)
    vals, cnts = np.unique(g, return_counts=True)
    b = int(vals[cnts.argmax()])
    lbl, n = _label8(g != b)
    comps = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        comps.append(dict(r0=int(r0), c0=int(c0), h=int(r1 - r0 + 1),
                          w=int(c1 - c0 + 1), n=len(ys),
                          cols=set(g[ys, xs].tolist())))
    base = None
    for c in comps:
        if c['h'] == 3 and c['w'] == 3 and len(c['cols']) > 1:
            base = c
            break
    if base is None:
        return None
    R = g[base['r0']:base['r0'] + 3, base['c0']:base['c0'] + 3].copy()
    flips = [R, R[:, ::-1], R[::-1, :], R[::-1, ::-1]]
    out = g.copy()
    blocks = []
    for c in comps:
        if c is base:
            continue
        if c['h'] == c['w'] and c['n'] == c['h'] * c['w'] and len(c['cols']) == 1:
            blocks.append((c['r0'], c['c0'], c['h'], next(iter(c['cols']))))
    used = [False] * len(blocks)
    for i, (r, cc, m, col) in enumerate(blocks):
        if used[i]:
            continue
        for j, (r2, c2, m2, col2) in enumerate(blocks):
            if j == i or used[j] or m2 != m:
                continue
            if abs(r2 - r) != 2 * m or abs(c2 - cc) != 2 * m:
                continue
            sr, sc = min(r, r2), min(cc, c2)
            sq = g[sr:sr + 3 * m, sc:sc + 3 * m]
            if int((sq != b).sum()) != 2 * m * m:  # square holds only the 2 markers
                continue
            pi = (0 if r == sr else 2, 0 if cc == sc else 2)
            pj = (0 if r2 == sr else 2, 0 if c2 == sc else 2)
            chosen = None
            for F in flips:
                if F[pi] == col and F[pj] == col2:
                    chosen = F
                    break
            if chosen is None:
                continue
            used[i] = used[j] = True
            out[sr:sr + 3 * m, sc:sc + 3 * m] = np.kron(chosen, np.ones((m, m), int))
            break
    return out.tolist()


def candidates(example):
    """No equivalent-yet-cheaper ONNX than the incumbent exists (see module docstring)."""
    return []
