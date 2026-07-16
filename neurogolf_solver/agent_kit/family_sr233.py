"""family_sr233 — SIMPLER-EQUIVALENT-RULE hunt for task233 (verify_97a05b5b).

RESULT: NO_SIMPLER.  The hypothesised simpler rule is NOT equivalent, and the
genuine sub-simplifications that DO hold do not reduce the dominant cost below
the incumbent (r233 / pool70, 14.55 pts, mem 33637).

Task recap
----------
A big solid red(=2) square has bgc(=0) "anti-shape" holes carved into it, each
hole being one D4 orientation of a scattered 3x3 marker (unique minority colour
`col`, sits on a red frame outside the square).  Output = the square, every hole
filled by the matching marker's 3x3 block (obj cells -> red, frame cells -> col).

Hypotheses tested on the 266 embedded arc-gen pairs + 3000+ fresh generator
samples (7 independent seed streams):

  H1  colour(hole) = the marker colour with obj-count n_c, and
      n_c = 9 - |{V==c}|                          -> EXACT 266/266, fresh 100%.
      (Cheap: no D4 correlation needed to pick the colour; counts 4..8 are
      sampled WITHOUT replacement so they are distinct per sprite -> a lookup.)
  H2  square bbox = bbox of red cells NOT within Chebyshev-2 of any coloured
      (non 0/2) cell                              -> EXACT 266/266.
  H3  "per-hole INDEPENDENT template matching suffices, no propagation"
      (the headline hypothesis)                   -> REFUTED.
        * one-pass forced condition (per-colour match-count == free-shape D4
          stabiliser size) holds on only ~30% of fresh samples;
        * naive independent per-colour matching reconstructs only ~65%;
        * the incumbent's iterative constraint-propagation `_ref` solves 12/12
          of the seeds where independent matching fails.
      Cause: adjacent 3x3 blocks (irows/icols may touch, margin 0) let holes of
      DIFFERENT sprites become 8-adjacent and merge; and count-4/5 sprites have
      a 2x3 / 3x2 obj bbox whose 3x3 block anchor is genuinely orientation-
      dependent (two shifted windows both contain all n_c holes).  Resolving
      these requires the joint exact-cover / D4 placement the incumbent does.

Why no cheaper compile
----------------------
Exact placement still needs the per-(colour,orientation) 3x3 correlation.  Folded
to channels that is a [1,64,30,30] fp16 activation = 115 200 B in a SINGLE named
tensor, and the grader's memory = SUM of every named intermediate's bytes
(neurogolf_utils.calculate_memory).  That one tensor alone already exceeds the
incumbent's 33 637 B budget, so the D4-placement core cannot be reproduced under
budget, and the cheap H1/H2 colour+box shortcuts do not touch that core.  Hence
no equivalent rule compiles strictly cheaper than the incumbent.

`candidates()` therefore emits nothing (returning a losing/incorrect candidate
would only waste grader time).  The numpy `_ref_lean` below is the exact H1+H2
colour/box reference and the trivial-placement reconstruction; it is kept for
transparency and documents precisely where the simpler rule stops being exact.
"""
from __future__ import annotations

import numpy as np

COLORS = [1, 3, 4, 5, 6, 7, 8, 9]


def _conv_same(B, k):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    kh, kw = k.shape
    P = np.pad(B, ((kh // 2, kh // 2), (kw // 2, kw // 2)))
    return np.einsum('ijkl,kl->ij', sw(P, (kh, kw)), k)


def _ref_lean(V):
    """H1 (colour = 9-|V==c|) + H2 (5x5-dilation box) + TRIVIAL 3x3-bbox
    placement.  Exact only when every obj bbox is a full 3x3 (~66% of samples);
    returned so callers can see exactly where independence breaks."""
    V = np.asarray(V, int)
    if V.ndim != 2 or max(V.shape) > 30:
        return None
    H, W = V.shape
    ncnt = {c: 9 - int((V == c).sum()) for c in COLORS if (V == c).any()}
    if not ncnt:
        return None
    red = (V == 2)
    colored = (V != 0) & (V != 2)
    dil = _conv_same(colored.astype(int), np.ones((5, 5), int)) > 0
    boxred = red & ~dil
    if not boxred.any():
        return None
    ys, xs = np.where(boxred)
    R0, R1, C0, C1 = ys.min(), ys.max(), xs.min(), xs.max()
    inside = np.zeros((H, W), bool)
    inside[R0:R1 + 1, C0:C1 + 1] = True
    B = ((V == 0) & inside).astype(int)
    c3 = _conv_same(B, np.ones((3, 3), int))
    c5 = _conv_same(B, np.ones((5, 5), int))
    center = (c3 == c5) & (c3 >= 4) & inside
    cnt2col = {n: c for c, n in ncnt.items()}
    colorval = np.zeros((H, W), int)
    for y, x in zip(*np.where(center)):
        v = int(c3[y, x])
        if v not in cnt2col:
            return None
        colorval[y, x] = cnt2col[v]
    frame = _conv_same(colorval, np.ones((3, 3), int))
    out = np.full((R1 - R0 + 1, C1 - C0 + 1), 2, int)
    sf = frame[R0:R1 + 1, C0:C1 + 1]
    sb = B[R0:R1 + 1, C0:C1 + 1]
    m = (sf > 0) & (sb == 0)
    out[m] = sf[m]
    return out


def candidates(examples):
    # NO_SIMPLER: the simpler independent-matching rule is not equivalent, and the
    # D4-placement core cannot be compiled under the incumbent's memory budget.
    return []
