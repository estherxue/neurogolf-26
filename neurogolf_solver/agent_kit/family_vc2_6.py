"""family_vc2_6 — RETRY of task255 (verify_a64e4611) and task363 (verify_e5062a87).

Both were assigned as "verifier off-by-1..3, fix the tie-break, then emit static ONNX".
After exhaustive analysis (see report) BOTH reduce to a non-local, data-dependent
tie-break that (a) has no clean numpy rule exact on all 265 train+test+arc-gen and
(b) is not expressible as a static opset-10 graph. The grader gate requires EXACT on
ALL train+test+arc-gen (evaluate.py: any train/test miss => ok=False, 0 pts), so a
verifier-faithful model (264/265 or 263/265) is rejected outright.

Therefore this family emits no candidates. Kept as a documented negative result.

--- task255 (a64e4611) ---
Verifier-faithful numpy replica is EXACT on 264/265 (matches verify_a64e4611 on all
265). Mechanism reproduced exactly:
  stage1: pad grid by 1 (bg=0). Fill 3 into the inner 12 cells of every 3-tall x
          14-wide all-empty window (horizontal) and every 14-tall x 3-wide window
          (vertical). This EQUALS the verifier's occurrences-of-3x14-rectangle stamp
          (confirmed bit-exact vs x26 on all 265).
  stage2: draw a 3-frame on the padded border, then tip-erode with two local patterns
          (single-3 with up/left/right==0; horizontal-3-pair with above/left/right==0)
          in all 4 orientations, iterated (rot90 chain x8). Confirmed bit-exact vs the
          full verifier on all 265.
The ONLY authoritative divergence is train[2] (6 cells): the verifier leaves two thin
1-wide protrusions that stick one cell past the main filled cross
(col16 rows16-18, col0 rows19-21); the NeuroGolf output straightens the cross edges
and removes them. This "remove thin appendages / straighten to the maximal empty
rectangle" is a NON-LOCAL property: the 4-neighbour signatures of the removed cells
(0330,3330,3300) are IDENTICAL to signatures of KEPT cells elsewhere, so no local
CA pattern separates them (verified: a von-Neumann rule that removes them also removes
legitimate band edges -> train0 loses all 221 fill cells). It needs connected-component
width / maximal-rectangle reasoning, which is not static-ONNX-expressible and for which
no clean numpy rule exact on all 265 exists.

--- task363 (e5062a87) ---
Verifier stamps the least-colour motif at EVERY location whose footprint is all-0.
Exact on 261/261 arc-gen + test, fails 2 hand-authored train examples, but the two
failures need DIFFERENT and mutually-inconsistent tie-breaks:
  train[1]: 1-D (horizontal) motif. occ={(1,3),(5,6)} (non-overlapping); output keeps
            only (5,6) — the one collinear with the original motif's row. => axis-
            restricted extension.
  train[0]: 2-D (diamond) motif. occ has two overlap conflicts; output keeps the
            farther-from-original of each conflicting pair (greedy non-overlap).
No single rule fits both: 'stamp-all', 'nearest greedy', 'farthest greedy' each miss
>=1 (farthest fixes train0 but not train1; the axis rule fixes train1 but not train0).
The selection is a sequential, motif-topology-dependent greedy — per-object-hard,
neither cleanly numpy-expressible nor statically ONNX-expressible.
"""

def candidates(examples):
    return []
