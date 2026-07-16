"""family_dc_2 — minimal-dyncrop recompile attempt for tasks 243, 378, 102, 62.

VERDICT: none of the four targeted tasks are dynamic-crop tasks.  Reading each
true rule (verify_<hash> in _rearc/verifiers.py) shows every output is the FULL
input grid mutated in place (paint/fill), i.e. output shape == input shape on all
local samples.  There is no data-dependent subgrid extraction, so the dyncrop
arsenal (bbox scalars + Slice/GatherND/row-select MatMul) does not apply.

  243 / 9edfc990  -> paint(I, recolor(ONE, zero-objs adjacent to color-1))   same-size
  378 / ec883f72  -> fill(I, color, diagonal shoots off 2nd-largest partition)  same-size
  102 / 44d8ac46  -> fill(I, TWO, square deltas of objects)                   same-size
   62 / 2bcee788  -> paint(fill(I, THREE, largest partition), mirrored small)  same-size

All four confirmed same-size against the generator task data.  No dyncrop win to
ship; candidates() emits nothing.
"""


def candidates(examples):
    return []
