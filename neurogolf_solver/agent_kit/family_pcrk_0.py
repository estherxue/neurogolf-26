"""family_pcrk_0 — slice U[0::6] = tasks [5, 66, 101, 157, 201, 264, 367].

Deep analysis result: every task in this slice is a genuinely hard, multi-object
or data-dependent ARC transformation with NO exact static-graph rule that
generalizes to a held-out arc-gen split. Candidate rules were derived and
self-checked; none are exact on the full arc-gen set, so (because the grader gates
on EXACT equality over all train+test+arc-gen) none can score.

Per-task findings (see the agent report for the full derivation):

  task 5   : arrow/marker points a direction; a small stencil shape is stamped
             repeatedly along that direction a data-dependent number of times.
             Position/direction/count all data-dependent -> not static.
  task 66  : two 2-cell markers (color 2 and color 3); an L-shaped PATH of 3s is
             routed connecting them. Path routing is data-dependent.
  task 101 : a template glyph (1s+2s) is "grown" out of 2-colored seed markers,
             oriented per seed. Data-dependent placement/orientation.
  task 157 : holes in a top 2-block become 1 and are extended downward with a
             shape taken from per-column 5-blobs. Data-dependent, per-object.
  task 201 : a 4-corner frame with two colored edges is filled by a point-symmetric
             (180deg + color-swap) completion of a template. Output size VARIES
             (6 sizes) -> variable crop, cannot fix-size.
  task 264 : jigsaw — 8 scattered 3x3 color-coded tiles assembled into a fixed 9x9
             mandala. Output is fixed 9x9 but tile locating+slotting is data-dependent.
  task 367 : "double-wall channel" fill: 0-cells lying in the narrow gap between two
             parallel 5-line segments are painted 4. Requires line/object pairing;
             best local/flood approximations top out ~198/266 arc-gen (not exact).

The rule below (flood-fill: 0-regions not touching top/bottom edge -> 4) was the
best structural candidate for 367 (198/266). It is kept only as a GATED attempt:
candidates() emits it ONLY if it reproduces EVERY provided example exactly. It does
not, so nothing is emitted. This keeps the family sound (no wrong candidates).
"""
import numpy as np


def _flood_tb_fill(gi):
    """367 candidate: 0-regions (4-conn, grid edge = wall) that touch the top or
    bottom row are 'exterior'; everything else (interior) is painted color 4.
    Implemented as a fixed-point flood from the top/bottom rows through 0-cells."""
    free = (gi == 0)
    H, W = gi.shape
    reach = np.zeros((H, W), bool)
    reach[0, :] |= free[0, :]
    reach[-1, :] |= free[-1, :]
    for _ in range(H * W):  # fixed point; H*W iters always suffices
        nxt = reach.copy()
        nxt[1:, :] |= reach[:-1, :]
        nxt[:-1, :] |= reach[1:, :]
        nxt[:, 1:] |= reach[:, :-1]
        nxt[:, :-1] |= reach[:, 1:]
        nxt &= free
        if np.array_equal(nxt, reach):
            break
        reach = nxt
    out = gi.copy()
    out[free & ~reach] = 4
    return out


def _exact_on_all(examples, fn):
    for split in ("train", "test", "arc-gen"):
        for p in examples.get(split, []):
            gi = np.array(p["input"])
            go = np.array(p["output"])
            pred = fn(gi)
            if pred.shape != go.shape or not np.array_equal(pred, go):
                return False
    return True


def candidates(examples):
    # Only same-size, color-preserving-shape rule has a builder here (367 flood).
    # Gate strictly on exactness so no wrong candidate ever leaks.
    try:
        if _exact_on_all(examples, _flood_tb_fill):
            # Would build+yield the ONNX doubling-flood model here. Not reached:
            # the rule is not exact on any task's full arc-gen split.
            pass
    except Exception:
        pass
    return []
