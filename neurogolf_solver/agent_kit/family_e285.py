"""ENTROPY rebuild attempt of task285 (arc-gen b775ac94) -- NO-SHIP REPORT.

Verdict: no ONNX materialization beating the incumbent (19706 cost / 15.11
pts) was found under the allowed-op set (no TopK/Gather/ScatterElements/i64
coordinate machinery).  candidates() intentionally returns [].  This module
preserves the verified task analysis and a numpy-exact reference solver so a
future attempt can start from facts, not guesses.

RULE (verified against generator source + 3000 gens, seed 31337):
  Square grid 12..30, black bg, 1..3 groups.  Each group: 4-connected sprite
  S (4..8 px in 5x5, ALWAYS contains (0,0); common.continuous_creature grows
  via 4-neighbours only, despite the "diagonal" comment) + anchor (br,bc),
  br,bc in [5, size-6].  Four reflections ("angles") of S around the 2x2 core
  rows br..br+1 x cols bc..bc+1, four DISTINCT colors per group (groups may
  reuse each other's colors).  At most ONE angle absent (color 0), never the
  shown one.  INPUT = one full angle (shows) + the (0,0) core pixel of every
  present angle.  OUTPUT = all present angles fully drawn.  Copies never
  overlap; different groups are >=2 apart in Chebyshev distance (8-nbhd
  legality check on the OUTPUT).

STRUCTURAL FACTS (each verified with 0 violations on 3000 fresh gens):
  F1. Core cells == exactly the non-bg cells having a 4-adjacent non-bg cell
      of a DIFFERENT color.
  F2. Vertical "diff-dominos" (two vertically adjacent non-bg cells of
      different colors) occur ONLY inside cores: top cell at (br,bc) and/or
      (br,bc+1).  Horizontal diff-dominos: left cell at (br,bc)/(br+1,bc).
  F3. Every group has >=1 vertical AND >=1 horizontal diff-domino (a 2x2
      block missing at most one cell always keeps one full row + one full
      column pair, and in-group colors are distinct).
  F4. Exact block corner: BLK(r,c) = [Vdom(r,c) or Vdom(r,c+1)] AND
      [Hdom(r,c) or Hdom(r+1,c)]  fires exactly at (br,bc), all absent-corner
      cases included; cross-group mixing impossible (>=2 separation).
  F5. solve_exact() below (F1/F4 detection + per-angle 4-connected flood from
      each core cell + reflected paint): 0/3000 mismatches.

MEASURED BASELINES (3000 fresh gens, seed 31337):
  incumbent out_blend14/onnx/task285.onnx: 16/3000 grids wrong.
    (It does NOT run on local ort 1.23.2: u8 Min/Max kernels NOT_IMPLEMENTED;
     measured after replacing its u8 Min/Max with Greater+Where equivalents.)
  incumbent true-byte cost: 19286 mem + 420 params = 19706 -> 15.11 pts.
    (_cost.py REPORTS 44636 because its BYT dict lacks UINT8 and defaults
     u8 planes to 4 bytes/element -- do not trust it for u8-heavy models.)
  2019/3000 grids have >=2 groups; 1862 of those have column-overlapping
  group boxes -> any "row-collapsed" mirror trick contaminates ~62% of grids
  (measured: 1849/3000 wrong for the cheapest such pipeline).

WHY THE REBUILD LOST (cost floors, true-byte accounting, each full-plane
node output = 900B u8 / 3600B f32):
  A decode      : Conv(one-hot,0..9)->f32 3600 + Cast u8 900       = 4500
  B detection   : m255(QLC w=255) 900 + Concat[g;m255] 1800 +
                  4 sat-QLC domino convs 3600 (+tiny reduces)      = 6300
                  (bands/bg force a mask channel; every 1-channel or
                   1-conv variant provably confuses bg with colors)
  D recolor     : BS=rm4(x)cm4 outer 900 + GR 4-tap QLC 900 +
                  aSUM Mul 900 + QC one 6x6 MaxPool 900            = 3600
                  (disjoint-window quadrant coloring: anchors placed at
                   blk+{0,6}^2 carry c_k; single spread window [p,p+5]^2
                   covers each quadrant from exactly one anchor)
  E compose     : sb Greater 900 + fin Where(sb,QC,g) 900          = 1800
                  (QC == g on input cells, so fin needs no bg mask)
  C mirror pass : THE BLOCKER.  V/H completion around per-group lines:
    - row/col-collapsed 1-D dilated QLC kernels (30x1 marks) cost only
      ~2700/axis but are WRONG (1849/3000): a line row spread across all
      columns pairs with other groups' pixels in shared columns.
    - the exact rule needs per-offset coupling  paint(p) <= OR_row
      X(p-(2row+1)) AND LINE2D(p-(row+1)) : every materialization found
      (5-channel QLC + reshape + dilated-MaxPool collapse; shifted-product
      planes; value-coded target-row with window matching; per-column
      depthwise runtime kernels) costs 5.5k..20k PER AXIS.  Budget for both
      axes at incumbent parity: 19706-16200 = 3.5k.  Gap ~3x structural.
  Total best-correct sketch ~= 28-30k (14.6-14.7 pts) < incumbent 15.11.

IDEAS RULED OUT (so nobody re-burns time):
  - one-hot channel tricks: any [1,10,30,30] intermediate = 9000B(bool/u8)
    or 36000B(f32); per-channel convs/pools all blow the budget.
  - 2x2 "3+ distinct colors" core tests from counts/sums/squares: {c,c,c'}
    edge-straddle windows are indistinguishable from 3-distinct cores by
    (n, sum, max, min); Sidon/B3 codings exceed u8 after window sums.
  - point-reflection via dilated-(2,2) conv with kernel=BLK plane over the
    180-flip: EXACT and cheap (2700) but covers only the point quadrant;
    V/H mirrors are not point maps and cannot compose from them.
  - per-column depthwise QLC with runtime one-hot shift kernels on the
    flipped grid: correct only for single-line columns; two groups sharing
    a column (common) breaks it, and it costs ~7k/axis anyway.
  - scalar extraction of (br,bc) per group without TopK/Gather: f16 code
    plane + iterative ReduceMax masking ~7k; marginal-projection pairing
    fails on same-row groups (frequent on small grids).
  - color relay via pair-sums (c_tgt = domino_sum - c_src): the sum lives
    only at cols bc/bc+1 and cannot be spread side-correctly (bc vs bc+1
    windows overlap).

The incumbent (a TopK/Gather/ScatterElements coordinate machine: decode ->
TopK32 cells -> score "has same-color neighbour AND different-color edge
neighbour" -> TopK3 anchors -> 8-neighbour pattern MatMul LUT for the other
core offsets -> 45-cell patch gather + 2x masked-MaxPool connectivity ->
TopK9 members -> 3 reflected index sets -> ScatterElements(max)) stays the
best known model for this task at 19706 / 15.11 pts.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- reference
def _cores(g: np.ndarray) -> np.ndarray:
    cc = np.zeros_like(g, bool)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = np.zeros_like(g)
        if dr == 1:
            n[:-1] = g[1:]
        elif dr == -1:
            n[1:] = g[:-1]
        elif dc == 1:
            n[:, :-1] = g[:, 1:]
        else:
            n[:, 1:] = g[:, :-1]
        cc |= (g > 0) & (n > 0) & (n != g)
    return cc


def _blocks(g: np.ndarray):
    pts = list(zip(*np.nonzero(_cores(g))))
    out = set()
    for p in pts:
        comp = [q for q in pts
                if abs(q[0] - p[0]) <= 1 and abs(q[1] - p[1]) <= 1]
        out.add((min(q[0] for q in comp), min(q[1] for q in comp)))
    return sorted(out)


def _cell(br, bc, k, row, col):
    return (br - row if k in (0, 1) else br + 1 + row,
            bc - col if k in (0, 2) else bc + 1 + col)


def solve_exact(gi: np.ndarray) -> np.ndarray:
    """Numpy-exact reference: 0/3000 mismatches on fresh gens (seed 31337)."""
    out = gi.copy()
    H, W = gi.shape
    for br, bc in _blocks(gi):
        c = [gi[br, bc], gi[br, bc + 1], gi[br + 1, bc], gi[br + 1, bc + 1]]
        S = np.zeros((5, 5), bool)
        for k in range(4):                     # 4-conn flood per present angle
            if c[k] == 0:
                continue
            seen, stack = {(0, 0)}, [(0, 0)]
            while stack:
                row, col = stack.pop()
                S[row, col] = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in seen:
                        r_, c_ = _cell(br, bc, k, nr, nc)
                        if 0 <= r_ < H and 0 <= c_ < W and gi[r_, c_] == c[k]:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
        for k in range(4):
            if c[k] == 0:
                continue
            for row in range(5):
                for col in range(5):
                    if S[row, col]:
                        r_, c_ = _cell(br, bc, k, row, col)
                        out[r_, c_] = c[k]
    return out


def candidates(examples):
    """No-ship: no allowed-op ONNX build beat the incumbent (see docstring)."""
    return []
