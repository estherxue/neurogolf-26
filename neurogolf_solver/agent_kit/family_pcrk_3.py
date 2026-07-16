"""family_pcrk_3 — CRACK slice U[3::6] = tasks [44, 80, 133, 173, 233, 349].

Deep analysis outcome: every task in this slice reduces to a *data-dependent
object-correspondence / template-routing* transformation that cannot be encoded
as a single generalizing static opset-10 graph under the banned-op set
(no Loop/Scan/NonZero/Unique/Compress/Sequence). Per-task exact rules and the
reason each is not statically expressible are documented below. Verified with
numpy reference solvers (rules are correct; the *routing* is the blocker).

Task 44  (fixed 10x10):
  Hollow 5-boxes each have an interior "hole" (0-region ringed only by 5s).
  Scattered elsewhere are single-color blobs. Each hole is filled with the color
  of the blob whose *normalized shape* is identical, and that blob is erased.
  -> connected-component labelling + shape-key JOIN (match hole-shape==blob-shape,
     gather that blob's color). A content-addressable lookup over a variable number
     of components. Not expressible without dynamic gather/labelling. numpy ref: 260/266.

Task 80  (5 fixed sizes, cell-grid):
  Grid partitioned by 8/3-lines into a matrix of KxK cells; a few cells carry a
  colored sub-pattern. The pattern is *propagated* to other cells following the
  arrangement logic of the seed cells (row/col broadcast of a moving sub-pattern).
  -> data-dependent multi-cell copy whose source/target set depends on cell contents.

Task 133 (variable size):
  A small "key" cross defines a directional expansion template (arm colors). Each
  Nx2 / 2xN colored blob is expanded into a cross/plus stamped per the key.
  -> template read from grid at runtime + per-object stamping. Data-dependent kernel.

Task 173 (variable size):
  One or more template objects (a plus / an X) each with a distinct CENTER color C
  and ARM color A. Every isolated marker cell of color C is replaced by the full
  template stamped around it; partial templates are completed.
  -> the stamp kernel (shape+colors) is read from the grid and routed by center
     color. Data-dependent per-marker convolution. Not a fixed conv.

Task 233 (variable -> variable crop):
  Extract the large 2-box; fill its interior holes with the 3x3/NxN pattern found
  in a small template outside; output size == box interior size.
  -> data-dependent crop to an object whose size varies + pattern transfer.

Task 349 (5 fixed sizes):
  Each square block of 9s gets a concentric 3-frame and emits a 1-beam to a grid
  edge; frame/beam geometry depends on each block's size and position.
  -> per-object framing + data-dependent beam direction/length. Not static.

Conclusion: no generalizing static candidate is emitted for this slice; any
train-only fit would memorize component/template routing and be rejected on the
held-out arc-gen split. candidates() returns nothing.
"""


def candidates(examples):
    return []
