"""family_sr054 — SIMPLER-EQUIVALENT-RULE hunt for task54 (gw2f16_54, star/flags
reconstruction). Result: NO_SIMPLER.

Incumbent: out_blend6/onnx/task054.onnx — 286 nodes, mem 25201, 14.86 pts,
fp16-tuned. It already uses the hinted technique: 8x CumSum (segment-bounded line
spans) + 21x ArgMax / 33x Gather / 22x Equal (arbitrary color-role identification).

The generator (task_264363fd) draws boxes (flags) of `box` color, a star TEMPLATE
in a bg corner (encoding the color scheme + which axes are active), and single-cell
`s0` marker dots inside the boxes. The output erases the corner template, and for
each marker projects a `s1`-colored crosshair spanning THAT marker's box (horiz
line if `horiz`, vert line if `vert`) plus a small star (3x3 `tbg` if tbg!=-1,
center `s0`, 1-out arms `s1`).

WHAT WAS TESTED (fresh pairs sampled from the real generator + 266 local cases):

1. FULL INVERSE (faithful, exact). Border-histogram bg; box = most-frequent
   non-bg; s0 = colour of a box-adjacent non-bg/non-box cell; template = the
   remaining non-bg/non-box cells (not box-adjacent); star centre = template
   centroid; vert/horiz/s1 read from the distance-2 template arms; tbg from a
   distance-1 diagonal; per-box rectangle from connected components; crosshair
   filled to the box's own extent; star stamped. -> EXACT on 4000/4000 fresh and
   266/266 local. This confirms the spec but is NOT structurally simpler than the
   incumbent: it needs the SAME arbitrary colour-role identification (histograms/
   ArgMax) AND the SAME segment-bounded two-axis fill the incumbent already does
   with CumSum spans.

2. GLOBAL row/col fill (the hoped-for "per-cell closed form from row/col marker
   positions" — a box cell becomes s1 iff any marker shares its row (horiz) or
   column (vert), IGNORING box boundaries). -> WRONG on 1961/2000 fresh samples:
   fresh 2-box grids share columns (both boxes span ~cols 1-18) so a vertical
   line bleeds into the neighbouring box; local train[1] shows the same for an
   explicit case. The box-boundary constraint is load-bearing, which is exactly
   why a segment-bounded CumSum span (not a plain row/col OR) is required.

3. Single-correlation / nearest-wall-distance closed forms: N/A — the output
   colour of a cell depends on the arbitrary per-grid colour assignment (bg, box,
   s0, s1, tbg) AND on segment-bounded membership; no fixed per-cell function of
   position or of a single correlation reproduces it.

CONCLUSION: the inverse map is faithful but of the same algorithmic class as the
incumbent (colour-role ID + CumSum-span line fill + local star stamp + template
erase). No meaningfully simpler sufficient rule exists; any cheaper build would be
pure memory-golf of the same algorithm against an already-fp16-tuned 25201-byte
incumbent, which is outside this hunt's mandate and not promising. -> NO_SIMPLER.
"""


def candidates(example):
    # No simpler equivalent rule found; do not challenge the incumbent.
    return []
