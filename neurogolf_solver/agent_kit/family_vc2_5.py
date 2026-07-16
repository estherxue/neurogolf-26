"""family_vc2_5 — RETRY on task096 (verify_4290ef0e) and task233 (verify_97a05b5b).

Both re-arc verifiers are off-by-1 on NeuroGolf data (265/266).  This module
reverse-engineered the true generators and produced EXACT numpy references
(the corrected rules), then evaluated static-ONNX feasibility.

================================================================================
task096 / 4290ef0e  — CORRECTED RULE (numpy reference exact 266/266)
================================================================================
Generator generate_4290ef0e builds a D4-symmetric "frame mandala" of size
(2d+1)x(2d+1), d in {3,4,5}: d concentric square frames, one per foreground
colour, plus an optional single centre dot.  Frame at radius r (1..d) is a
canonical shape CAN(r,L): four L-corners at the box corners, each with two arms
of length L (=linlen) pointing inward.  Output = overlay of all frames centred
at (d,d), background elsewhere, centre cell overwritten by the dot colour if a
count-1 colour is present.

The INPUT scatters each colour's FULL frame at a random location, possibly
truncated by the grid edge — but truncation is always a *rectangular crop* of
the (D4-symmetric) full frame.  So each colour's input fragment is a sub-window
of exactly CAN(r,L).

Reconstruction (see solve096 below, verified 266/266 on train+test+arc-gen):
  * centre dot = the count-1 colour (if any); frames = the rest; d = #frames.
  * for each frame colour, feasible (r,L) = all templates CAN(r,L), r in 1..d,
    that contain the fragment as a rectangular sub-window.
  * colours whose feasible r is UNIQUE claim that radius; the (rare) ambiguous
    single-elbow fragment [[1,1],[1,0]] takes the one leftover radius.
    (No backtracking needed — a per-frame + leftover rule is exact on all 266.)
  * draw CAN(r, max feasible L for that r) centred at (d,d); paint centre dot.

The re-arc verifier's bug: its order()/argmax mirror-selection (x9/x22) mis-ranks
frames when two frames tie on max(h,w)+maxwidth, scrambling one test grid.

--- ONNX status: static path IDENTIFIED but not emitted. ---
Feasible in principle via: (a) finite-d unroll d in {3,4,5} (technique 2) giving
3 fixed 7/9/11 output sizes, each with a FIXED centre so the D4 unfold of the
top-left quadrant `quad` is a static Slice/Gather/Concat; (b) per-radius-slot
colour+arm detection by correlating each colour mask against the CAN(r,L)
templates (technique 5-style oriented correlations) + a leftover one-hot for the
ambiguous elbow; (c) variable-length arm stamping = Less(coord_ramp, L) masks at
the fixed corner (k,k).  The blocker to *shipping* it is cost/size, not
expressibility: ~10 colours x ~20 templates x 3 d-branches of [1,1,30,30]
correlation chains plus the leftover-radius one-hot resolution is a many-hundred-
node graph with high held-out risk (the leftover rule must stay exact when >1
elbow appears).  Not completed within this wave's budget.

================================================================================
task233 / 97a05b5b  — CORRECTED RULE (understood; verifier off by 1)
================================================================================
Generator generate_97a05b5b: OUTPUT `go` is the solid square (colour sqc) of size
sgh x sgw with several "anti-shape" markers painted in.  Each marker = a small
object `obj`; in `go` it is drawn as f(rectangle of `col` with the obj cells left
as sqc), i.e. the col-coloured complement of obj inside its bbox, at a random
position/orientation f (a D4 element).  The INPUT carries: the square at a random
position (loci,locj) with bgc holes carved in the obj shapes, plus the marker
rectangles scattered in the background in their base orientation.  Solving =
match each square hole to a scattered marker (by obj shape under D4), then paint
that marker's col anti-shape into the square in the hole's orientation; output is
the square cropped to the top-left.

The verifier's bug: the D4-orientation argmax (x38 over x17-scored rapply of the
8 dihedral variants) breaks a tie the wrong way on train[0], mis-placing the
"4" marker.  NeuroGolf's authoritative orientation = the one matching the hole's
actual carved orientation.

--- ONNX status: INFEASIBLE (specific un-buildable tensors). ---
1. The output is a data-dependent crop of the square at variable position
   (loci,locj) AND variable size (sgh x sgw, ~100 distinct sizes up to 20x20).
   That is the dyncrop MatMul machinery — heavy but expressible.
2. The un-buildable part: per-hole marker assignment.  Each of a *variable*
   number of holes at *data-dependent* positions must be matched to one of a
   variable number of scattered markers under all 8 D4 orientations, then that
   marker's colour+anti-shape stamped in the matched orientation.  This is an
   all-pairs, data-dependent (hole x marker x 8-orientation) assignment whose
   result feeds back into position-dependent painting.  There is no fixed-shape
   static tensor that realises "for THIS hole pick THE matching marker's colour
   and orientation" without per-object gather-by-data over a runtime-variable
   object count — exactly the Loop/NonZero/Sequence class of ops that are banned.
   The finite-k / finite-orientation unrolls (techniques 2/5) do not help because
   the hole COUNT and hole POSITIONS are unbounded and data-dependent, not a
   small finite set.

Both numpy references are provided below so a future wave can build from exact
gates; candidates() does not emit a model (no shippable static ONNX this wave).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# task096 exact numpy reference (verified 266/266 train+test+arc-gen)          #
# --------------------------------------------------------------------------- #
def _canonical(r, L):
    n = 2 * r + 1
    F = np.zeros((n, n), int)
    for (ci, cj) in [(0, 0), (0, 2 * r), (2 * r, 0), (2 * r, 2 * r)]:
        di = 1 if ci == 0 else -1
        dj = 1 if cj == 0 else -1
        for t in range(L):
            F[ci, cj + dj * t] = 1
            F[ci + di * t, cj] = 1
    return F


_CAN = {(r, L): _canonical(r, L) for r in range(1, 8) for L in range(2, r + 2)}


def _frag(I, c):
    ys, xs = np.where(I == c)
    m = np.zeros((ys.max() - ys.min() + 1, xs.max() - xs.min() + 1), int)
    m[ys - ys.min(), xs - xs.min()] = 1
    return m


def _subwin(frag, full):
    fh, fw = frag.shape
    nh, nw = full.shape
    if fh > nh or fw > nw:
        return False
    for oi in range(nh - fh + 1):
        for oj in range(nw - fw + 1):
            if np.array_equal(full[oi:oi + fh, oj:oj + fw], frag):
                return True
    return False


def solve096(I):
    I = np.array(I)
    vals, cnts = np.unique(I, return_counts=True)
    bg = vals[np.argmax(cnts)]
    colors = [int(c) for c in vals if c != bg]
    frames, center = [], None
    for c in colors:
        if int((I == c).sum()) == 1:
            center = c
        else:
            frames.append(c)
    d = len(frames)
    feas = {}
    for c in frames:
        fm = _frag(I, c)
        feas[c] = [(r, L) for r in range(1, d + 1) for L in range(2, r + 2)
                   if _subwin(fm, _CAN[(r, L)])]
        if not feas[c]:
            return None
    assign, claimed, amb = {}, set(), []
    for c in frames:
        rs = set(r for r, _ in feas[c])
        if len(rs) == 1:
            r = rs.pop()
            assign[c] = r
            claimed.add(r)
        else:
            amb.append(c)
    missing = [r for r in range(1, d + 1) if r not in claimed]
    if len(amb) != len(missing):
        return None
    for c in amb:
        opts = [r for r, _ in feas[c] if r in missing and r not in claimed]
        if not opts:
            return None
        r = opts[0]
        assign[c] = r
        claimed.add(r)
        missing.remove(r)
    if len(set(assign.values())) != d:
        return None
    S, cx = 2 * d + 1, d
    out = np.full((S, S), bg)
    for c in frames:
        r = assign[c]
        L = max(L for rr, L in feas[c] if rr == r)
        F = _CAN[(r, L)]
        off = cx - r
        ys, xs = np.where(F == 1)
        out[ys + off, xs + off] = c
    if center is not None:
        out[cx, cx] = center
    return out


# --------------------------------------------------------------------------- #
# harness entry point                                                          #
# --------------------------------------------------------------------------- #
def candidates(examples):
    """No shippable static ONNX this wave (see module docstring).

    task096: static path identified (finite-d unroll + template correlation +
             D4 unfold) but not built — cost/held-out risk, out of budget.
    task233: infeasible — variable-count, data-dependent hole<->marker D4
             matching cannot be a fixed-shape static tensor pipeline.
    """
    return []
