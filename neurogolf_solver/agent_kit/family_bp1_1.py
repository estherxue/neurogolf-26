"""family_bp1_1 — recompile attempt for the memory-dominated tasks 233/158/286/54.

Mandate: beat out_blend4/onnx/task{233,158,286,54}.onnx on points = 25 - ln(mem+params).

Full dissection result (see module docstring below) — NO safe, verifiable, strict
improvement exists for any of the four without a from-scratch bitpacked/downsampled
rearchitecture of the whole per-task morphology pipeline.  Every cheap lever the
arsenal offers was checked and is already spent by the pool nets:

  * dedup identical initializers ....... 0 duplicates in all 4 graphs
  * prune unused initializers .......... 0 unused in all 4 graphs
  * remove dead nodes .................. 0 dead nodes in all 4 graphs
  * Constant -> initializer rescue ..... 0 Constant nodes in all 4 graphs
  * fp32 -> fp16/uint8 on intermediates  the only fp32 heavies (task233
        safe_name_55/57) are Conv outputs; narrowing them would require casting
        the free [1,10,30,30] fp32 `input` to fp16 first -> a NEW 18000-byte
        intermediate, i.e. strictly worse.  All other big intermediates are
        already bool/uint8/fp16 on the 30x30 canvas (1 byte/elem, irreducible).
  * broadcast/scalar compression of params — the only axis-collapsible
        initializers are Conv KERNELS (task233 safe_name_52 [1,10,3,3] all-equal
        spatially; task158 w_row_q [1,1,1,23] all-equal; task54 safe_name_26).
        Collapsing a conv kernel's spatial axes changes the window sum semantics,
        so it is NOT value-preserving; it cannot be collapsed.
  * generate-instead-of-store (task286 `em` is a [25,25] checkerboard, task286
        no identity) — regenerating it costs a 625-byte bool intermediate +
        coordinate tensors, strictly worse than the 625 stored params it replaces
        (params and memory count 1:1 in the cost).

Why the log makes even a perfect param-zeroing worthless here (arsenal's own rule
"5% off a 20k net = +0.05 — redesign instead"):

    task     mem     params  cost    pts     zero-ALL-params -> pts   gain
    233    33637      763    34400  14.55         14.57 (33637)       +0.02
    158    26178     2305    28483  14.74         14.79 (26178)       +0.05
    286    26064      845    26909  14.80         14.82 (26064)       +0.02
    54     25201      193    25394  14.86         14.86 (25201)       +0.00

Memory dominates ~30-130x over params for all four.  A meaningful +0.5 pts needs
cost cut by e^0.5 = 1.65x, i.e. ~10k bytes of intermediates eliminated — only
reachable by bitpacking the ~30 900-byte working masks (8 masks/byte) or shrinking
the working canvas below 30x30.  The generators produce grids up to 30x30 (verified
in the data), so the canvas is not provably shrinkable, and a bitpacked rebuild of
these particular pipelines (mirror-stamp occurrence matching for 233, scaled-stamp
occurrence rendering for 158, seed-growth flood for 286, crosshair projection for
54 — among the most complex verifiers in the suite) is a per-task from-scratch
effort with high risk of failing the >=1500-fresh-sample no-overfit gate, and no
onnxsim / onnxconverter_common / onnxscript surgery tooling is installed locally.

Decision: emit NO candidate rather than ship a within-noise / unverifiable change.
`candidates()` therefore yields nothing; the harness keeps the existing pool nets.
Kept as an importable, side-effect-free module so family_test runs cleanly.
"""
from __future__ import annotations


def candidates(examples):
    """No verified strict improvement found for tasks 233/158/286/54.

    See the module docstring for the full dissection.  Yielding nothing guarantees
    the harness never ships a regression for these tasks.
    """
    return
    yield  # pragma: no cover  (makes this a generator, never reached)
