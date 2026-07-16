"""family_rb002 — task2 (ARC 00d62c1b, "honeypots"): PROVABLY UNSOLVABLE exactly.

REBUILD verdict: the generator's output is NOT a function of its input, so no static
opset-10 ONNX (nor any deterministic function of the grid) can be exact on fresh
generator output.  candidates() therefore intentionally emits NOTHING — shipping a
train+test-exact rule (flood-fill / solid-rect fill) would pass the fixed local 262
arc-gen set but score 0 on the fresh PRIVATE set, re-creating the exact overfit trap
this rebuild exists to eliminate.

=========================  GROUND-TRUTH GENERATOR RULE  =========================
tasks/task_00d62c1b.py draws, on a size x size grid (size in [6,20]):
  * `static`  : random green (3) pixels, density 0.05.
  * `honeypots`: num_pots in [1,8] non-overlapping rectangles (wide,tall in [3,8]).
     Each pot's BORDER is drawn green as a rectangle RING MINUS ITS 4 CORNERS
     (vertical walls rows 1..tall-2 at cols 0,wide-1 ; horizontal walls cols
     1..wide-2 at rows 0,tall-1).  Each pot's INTERIOR (the solid rectangle
     rows 1..tall-2 x cols 1..wide-2) is drawn YELLOW (4) in the output, BLACK (0)
     in the input.
  * finally a SINGLE row-major, in-place pass sets any still-black cell to yellow
     iff `is_surrounded` == all 4 orthogonal neighbours are > 0 (out-of-grid = -1).

So   output_yellow = (pot interiors, filled by DRAWING) ∪ (is_surrounded cells).
The canonical/incumbent solver is a global flood-fill (fill every black cell that
cannot reach the border through black).  Measured divergence from the true generator
over fresh samples:  flood-fill grid-fail 5.5% (matches the reported 6.3%; all 8
known-fail cases are flood OVER-fills, truth ⊆ flood).  Best possible LOCAL rule
(fill only enclosed SOLID rectangles with full green edge-walls) still grid-fails
2.92% (175/6000).  With 262 all-or-nothing examples,  P(pass) ≈ 0.97^262 ≈ 5e-4 → 0.

===============================  WHY UNSOLVABLE  ===============================
The residual error is a hard floor of "pockets": a black solid rectangle enclosed by
full green edge-walls that is NOT a drawn pot (its walls are borrowed from crossing
walls of other pots + static).  A real 3-tall / 3-wide pot has a THIN multi-cell
interior that is filled ONLY by the drawing step; the identical thin corridor formed
incidentally is left black because the local single-pass `is_surrounded` DEADLOCKS on
a closed 1-wide corridor (each cell keeps a still-black neighbour).  A pocket and a
genuine pot are therefore BYTE-IDENTICAL locally.

Proof 1 (output is not a function of input; valid generate() calls):
  generate(size=7, brows=[2],bcols=[2],wides=[4],talls=[3], rows=[],cols=[])   # real pot
  generate(size=7, rows=[2,2,4,4,3,3],cols=[3,4,3,4,2,5], brows=[],...)         # same walls, no pot
  -> IDENTICAL 7x7 input; outputs differ (interior yellow vs. black).

Proof 2 (in-distribution local collision, both from the real random path):
  seed 4298 (region r3-4 c5) and seed 5349 (region r3-4 c10) produce the IDENTICAL
  6x5 neighbourhood window yet opposite fill labels (filled vs. not).  A bounded
  receptive-field net (which is all a static opset-10 conv/dilation stack can be)
  cannot separate them.

Distinguishing pocket from pot would require reconstructing the exact non-overlapping
pot cover of the grid — a combinatorial parse that (a) is itself not unique (Proof 1)
and (b) is not expressible in opset-10 without the banned Loop/NonZero/etc.  Hence no
static ONNX can meet the "exact on fresh generator output" bar.
"""


def candidates(examples):
    # Intentionally emit nothing: task2 has no correct static-ONNX solver (see module docstring).
    return
    yield  # pragma: no cover  (keeps this a generator)
