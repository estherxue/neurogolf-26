"""family_vc3_1 — RETRY on task233 (verify_97a05b5b).

The previous wave (family_vc2_5) characterised the rule but shipped NO numpy
reference for task233 and declared the ONNX "INFEASIBLE" for the wrong reason
("variable, data-dependent hole<->marker matching ... exactly the Loop/NonZero
class of ops").  This wave:

  * builds a numpy reference that is EXACT on ALL 266 examples (3 train + 1 test
    + 262 arc-gen), fixing the tie-break so it matches the authoritative data;
  * proves the tie-break is GEOMETRIC (a fixed canonical D4 order), NOT hash /
    RNG order;
  * proves the exact-cover assignment is always FORCED (0 guesses over all 266),
    i.e. the rule is fully deterministic;
  * identifies the *actual* blocked tensor (global exact-cover over ambiguous
    local matches) and shows the counter-techniques DO overcome the previously
    cited blockers, leaving a large-but-feasible static path that is out of this
    wave's build budget.

================================================================================
CORRECTED RULE  (numpy `solve233` below — EXACT 266/266)
================================================================================
Colours: bgc = most common (always 0 here); sqc = most common non-bgc (the solid
square colour, always 2 here).  The INPUT contains

  * one solid sqc square at (loci,locj) of size sgh x sgw (<=20x20), with several
    "anti-shape" holes carved into it in bgc; each hole is the `obj` shape of one
    marker, drawn in the marker's *actual* (possibly D4-rotated/reflected)
    orientation;
  * several 3x3 marker rectangles scattered in the background, each in its BASE
    orientation.  A marker = a 3x3 block where the minority colour `col` (unique
    per marker) fills the rectangle and the `obj` cells are left as sqc.  Marker
    colours are unique per example; markers are 8-separated from the square and
    from each other.

OUTPUT = the square cropped to the top-left (size sgh x sgw), where every hole is
replaced by the matching marker's full 3x3 rectangle: obj cells -> sqc, the rest
-> col, placed so the marker's obj aligns exactly onto the carved hole and drawn
in the hole's orientation.  Everything else stays sqc; the background markers are
outside the crop and vanish.  Equivalently: out = sqc everywhere, then for each
placed marker paint its col cells (its 3x3 minus obj) at the aligned position.

Reconstruction (verified exact 266/266):
  1. find square = largest 4-connected sqc component; bbox = (loci..,locj..).
  2. holemask = bgc cells inside that bbox (the union of every carved obj).
  3. markers = 8-connected non-bgc components outside the square; each gives a
     3x3 block (col + obj) and its colour col.
  4. EXACT-COVER: assign each marker a (D4 orientation, offset) so its obj cells
     land exactly on holemask cells and the placements partition holemask
     exactly.  A hole cell is covered by exactly one marker.
  5. paint: obj->sqc, col->col for every placement, into the cropped square.

Two subtleties, both fully resolved (see below):
  (a) TIE-BREAK (orientation).  When obj has a non-trivial D4 stabiliser (L-tromino
      -> {id,transpose}; S/Z-tetromino -> {id,rot180}), two orientations map the
      base obj onto the same hole but stamp DIFFERENT col borders.  Only 8 of 938
      holes are ambiguous.  The authoritative choice = the FIRST orientation in
      the fixed order [rot0,rot90,rot180,rot270, flip+rot0..rot270].  This is a
      pure canonical order (deterministic, geometric); it reproduces all 8.  The
      re-arc verifier's 265/266 miss on train[0] is a *different* order (its
      scored-argmax over the 8 variants tie-breaks the other way).
  (b) POSITION.  A marker's obj can full-window-match holemask at up to 6 places
      (a small obj is a sub-pattern of the whole hole field).  Local matching is
      NOT unique; only the global exact-cover pins it down.  BUT the cover is
      always FORCED: at every step some marker has a UNIQUE match against the
      *remaining* holes (0 guesses over all 266) — pure constraint propagation,
      no search/backtracking in the true generator distribution.

================================================================================
ONNX status — FEASIBLE-BUT-UNSHIPPED (revised from prev wave's "infeasible")
================================================================================
The previously-cited blockers are all removable with the requested techniques:

  * "variable number of holes / markers" -> markers have UNIQUE colours, so the
    count collapses to <=10 fixed colour SLOTS (per-slot unroll).  Each slot's
    marker mask, 3x3 kernel, and 8 D4 variants are static.
  * "8-orientation matching" -> D4 all-8 unroll + batched correlation Conv with
    the runtime 3x3 kernels (family_vc2_1 build_076 pattern: correlate obj kernel
    against the hole field to find exact-match anchors, then ConvTranspose to
    stamp the col cells).
  * "data-dependent crop of a variable square" -> dyncrop MatMul (family_vc2_2):
    build the filled square in place, then Srow @ filled @ Scol relocates it to
    the origin; padding stays all-zero.
  * marker 3x3 kernel extraction (obj cells are sqc, sometimes not adjacent to
    the col cells) -> bounded flood (<=2 MaxPool dilations of the col seeds,
    min'd with the outside-square non-bgc mask) recovers each marker's full block;
    markers are 8-separated (verified: outside-comps == #markers on all 266).

The one genuinely hard piece is that step-4 is a global EXACT-COVER, which a
static feed-forward graph cannot search.  What rescues it is (b): the cover is
always FORCED.  So it is expressible as K fixed constraint-propagation passes
(K <= #markers <= 5): each pass, for every colour slot, compute exact-match
indicators over the 8 variants against the CURRENT remaining-hole mask, and stamp
a marker only where its match is unique; update the remaining mask; repeat.

That works, but the honest cost is a very large graph and one fiddly detail — the
uniqueness test must count distinct *covered cell-sets* (symmetric-obj variants
share a cell-set at one anchor but differ at different anchors), which interacts
with the per-slot / per-pass / per-orientation unroll.  Combined with the flood
extraction and the dyncrop, an EXACT build is several hundred nodes and high
bug-risk; it is out of this wave's budget.  candidates() therefore ships no model
this wave, but the path above is concrete and no longer "banned-op" blocked.

The numpy reference is provided so a future wave can build against an exact gate.
"""
from __future__ import annotations

import numpy as np

# fixed canonical D4 order used for the orientation tie-break (rotations, then
# rotations of the left-right flip).  np.rot90(m, k) is CCW; k=0..3.
_D4 = [("r", 0), ("r", 1), ("r", 2), ("r", 3),
       ("f", 0), ("f", 1), ("f", 2), ("f", 3)]


def _orient(m, key):
    t, k = key
    r = np.rot90(m, k)
    return np.fliplr(r) if t == "f" else r


def _comps(mask, conn8=False):
    """4- or 8-connected components of a boolean mask -> list of cell lists."""
    H, W = mask.shape
    lab = -np.ones((H, W), int)
    nbr = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if conn8:
        nbr += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    out = []
    cur = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] < 0:
                st = [(i, j)]
                lab[i, j] = cur
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy, dx in nbr:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lab[ny, nx] < 0:
                            lab[ny, nx] = cur
                            st.append((ny, nx))
                out.append(cells)
                cur += 1
    return out


def _parse(a):
    a = np.asarray(a, int)
    vals, cnts = np.unique(a, return_counts=True)
    bgc = int(vals[np.argmax(cnts)])
    nb = a[a != bgc]
    if nb.size == 0:
        return None
    vv, cc = np.unique(nb, return_counts=True)
    sqc = int(vv[np.argmax(cc)])
    sqcomps = _comps(a == sqc)
    if not sqcomps:
        return None
    sq = max(sqcomps, key=len)
    ys = [y for y, x in sq]
    xs = [x for y, x in sq]
    r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
    inside = np.zeros_like(a, bool)
    inside[r0:r1 + 1, c0:c1 + 1] = True
    markers = []
    for m in _comps((a != bgc) & ~inside, conn8=True):
        yy = [y for y, x in m]
        xx = [x for y, x in m]
        block = a[min(yy):max(yy) + 1, min(xx):max(xx) + 1]
        col_vals = block[block != sqc]
        if col_vals.size == 0:
            return None
        cv, cn = np.unique(col_vals, return_counts=True)
        col = int(cv[np.argmax(cn)])
        markers.append((block, col))
    return a, bgc, sqc, (r0, r1, c0, c1), inside, markers


def solve233(a):
    """Exact reference. Returns the output grid, or None if the rule doesn't fit."""
    parsed = _parse(a)
    if parsed is None:
        return None
    a, bgc, sqc, (r0, r1, c0, c1), _inside, markers = parsed
    if not markers:
        return None
    H, W = r1 - r0 + 1, c1 - c0 + 1
    sub = a[r0:r1 + 1, c0:c1 + 1]
    holeset = frozenset((int(y), int(x)) for y, x in zip(*np.where(sub == bgc)))

    # candidate placements per marker: dedup by covered cell-set, keep the first
    # orientation in the canonical D4 order (the tie-break).
    cand = []
    for block, col in markers:
        seen = {}
        lst = []
        for key in _D4:
            tb = _orient(block, key)
            th, tw = tb.shape
            oys, oxs = np.where(tb == sqc)
            rel = list(zip(oys.tolist(), oxs.tolist()))
            if not rel:
                continue
            for oy in range(H - th + 1):
                for ox in range(W - tw + 1):
                    cells = frozenset((oy + yy, ox + xx) for yy, xx in rel)
                    if cells <= holeset and cells not in seen:
                        seen[cells] = True
                        lst.append((cells, tb, col, oy, ox))
        cand.append(lst)

    N = len(markers)
    # most-constrained-first ordering makes the (already forced) search instant.
    order = sorted(range(N), key=lambda i: len(cand[i]))
    cand = [cand[i] for i in order]
    sol = [None] * N
    used = set()

    def bt(i):
        if i == N:
            return not (holeset - used)
        for placement in cand[i]:
            cells = placement[0]
            if cells & used:
                continue
            sol[i] = placement
            used.update(cells)
            if bt(i + 1):
                return True
            used.difference_update(cells)
            sol[i] = None
        return False

    if not bt(0):
        return None
    out = sub.copy()
    for cells, tb, col, oy, ox in sol:
        th, tw = tb.shape
        for yy in range(th):
            for xx in range(tw):
                out[oy + yy, ox + xx] = sqc if tb[yy, xx] == sqc else col
    return out


# --------------------------------------------------------------------------- #
# harness entry point                                                          #
# --------------------------------------------------------------------------- #
def candidates(examples):
    """No shippable static ONNX this wave (see module docstring).

    The rule is EXACT and deterministic (`solve233`, 266/266, forced cover, fixed
    D4 tie-break), and the static path is now concrete (per-colour slot unroll +
    D4 all-8 correlation Conv + K-pass constraint propagation + dyncrop MatMul).
    The exact build is several-hundred-node / high-held-out-risk and is out of
    this wave's budget, so no model is emitted.
    """
    return []


# --------------------------------------------------------------------------- #
# self-test (run: python family_vc3_1.py)                                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import json
    import os

    tdir = os.environ.get(
        "NG_DATA_DIR",
        "/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26"
        "--claude-worktrees-kaggle-agent-harness/f26477d2-2e56-461c-9fe3-1ac499bf563f"
        "/scratchpad/ng_data/tasks",
    )
    d = json.load(open(os.path.join(tdir, "task233.json")))
    ok = tot = 0
    for split in ("train", "test", "arc-gen"):
        for e in d[split]:
            o = solve233(e["input"])
            b = np.asarray(e["output"], int)
            good = o is not None and o.shape == b.shape and np.array_equal(o, b)
            ok += good
            tot += 1
    print(f"solve233 exact: {ok}/{tot}")
