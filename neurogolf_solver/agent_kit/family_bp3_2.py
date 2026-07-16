"""family_bp3_2 — recompile attempt for tasks 133, 285, 101, 187.

GOAL (from the brief): turn each fp16 [1,10,30,30]-canvas solver into a
label-space + downsampled graph, strictly beating the current out_blend6 cost.

OUTCOME: all four are SKIPPED. Every one of these four current models is ALREADY
in label / cropped-work space (measured below) — the easy 20x "kill the 10-channel
canvas" win is already captured. What remains is irreducible, and the cheaper
label-space rewrites I costed all come in ABOVE the current graphs. Per the hard
gate "NEVER ship a regression or an unverified graph", candidates() emits nothing.

Measured current cost (static shape-inference of out_blend6/onnx/taskNNN.onnx —
reproduces the grader's memory exactly):

  task133 (57aa92db): mem 20306, params 1016  -> 15.03 pts
      149 named tensors; already a single-channel uint8/bool 30x30 label grid
      (top tensors: gridf f32 3600, then a long tail of 900-byte [1,1,30,30]
      bool/uint8 label grids: seed1-4, stamp1-4, outgrid...). Grid is a genuine
      variable 10..30 square-ish canvas (generator width/height=randint(10,30)),
      so it CANNOT be downsampled below 30. The task is sprite-template
      reconstruction + per-sprite magnify(1..4) + recolor + signature-pixel
      stamping. Cost is num-of-30x30-label-grids, not channel width. A faithful
      rebuild still needs the same handful of 30x30 label stamps; no headroom.

  task285 (b775ac94): mem 19286, params 420   -> 15.11 pts
      113 named tensors; 4-fold rotational sprite completion on a square 12..30
      grid. Already label-space (top: cf f32 3600, gf f16 1800 are the argmax /
      conv label extraction; rest are <=900 label grids). Uses TopK + a 45-wide
      rotation LUT. The two big tensors are the unavoidable float label-extraction
      buffers; retyping them to uint8 breaks the Conv/MatMul that consume them.

  task101 (447fd412): mem 16961, params 909    -> 15.21 pts
      354 named tensors but each tiny (max 1428) — already computes on a CROPPED
      17x21 work area (row_offsets[17], col_offsets[21]), exactly lever (2).
      Blue/red creature template detected from box 0, stamped+magnified into the
      other boxes. Cost is the large node COUNT of the detect-and-stamp logic,
      not tensor size; nothing to shave without a new algorithm.

  task187 (7b6016b9): mem 15285, params 505    -> 15.33 pts
      91 named tensors; ALREADY bit-packed (BitShift/BitwiseAnd/BitwiseOr on
      uint32) — lever (3) is already applied. Task = recolor: background black->
      green(3), box interiors black->red(2), borders+lines keep colour, on a
      20..25 grid (flip/xpose). The clean "flood-fill enclosure" rewrite was
      prototyped and REJECTED for two reasons, both verified against the ARC-GEN
      generator (3000 fresh samples):
        (a) enclosure != box-interior: 49/3000 samples have background pockets
            enclosed by LINES (not boxes) that must stay green, so a plain
            border-flood mislabels them red. Matching the generator needs true
            box detection, more logic than the current graph.
        (b) even ignoring (a), the flood needs up to 18 sequential dilation
            steps; each step is >=6 ops and every ONNX op output is a distinct
            named tensor that the memory rule charges (a packed uint32[30] row
            vector is 120 B), so ~18*6*120 ~= 13 KB JUST for the loop, on top of
            label extraction — over the 15285 budget. "Deep chains on tiny
            tensors are free" is false under this cost model: compute is free,
            but each intermediate NAME is charged once.

Fingerprint routing is implemented (route_task) so this module slots into the
family harness, but candidates() returns [] for every task: no verified strictly
cheaper graph was found, and shipping a regression / unverified graph is barred.
"""
from __future__ import annotations

import numpy as np

# Hashes of the four tasks this module owns (task_hash_map.json).
_HASH = {"133": "57aa92db", "285": "b775ac94", "101": "447fd412", "187": "7b6016b9"}


def _fp(example):
    """Cheap structural fingerprint of a task's train pairs -> task id or None.

    Uses only generator-stable, size-invariant features so it routes the four
    owned tasks without matching anything else. (Routing is real even though
    every branch currently yields no candidate.)
    """
    train = example.get("train", [])
    if not train:
        return None
    ins = [np.array(p["input"]) for p in train]
    outs = [np.array(p["output"]) for p in train]

    def cols(a):
        return set(int(v) for v in np.unique(a))

    same_shape = all(i.shape == o.shape for i, o in zip(ins, outs))
    if not same_shape:
        return None

    # 187: output background is green(3) and interiors red(2); input is black-bg
    # with no green and (almost) no red; output adds a large green field.
    def frac(a, v):
        return (a == v).mean()
    is187 = all(
        3 not in cols(i) and frac(o, 3) > 0.2 and 3 in cols(o) and 2 in cols(o)
        for i, o in zip(ins, outs)
    )
    if is187:
        return "187"

    # 101: creatures are blue(1)/red(2); output strictly adds blue where input
    # was black (template completion), same colour palette {0,1,2}.
    def pal(a):
        return cols(a) - {0}
    is101 = all(pal(o) <= {1, 2} and pal(i) <= {1, 2} and
                (o != i).any() and (i[o == 0] == 0).all() for i, o in zip(ins, outs))
    if is101:
        return "101"

    # 285: square grids, output is a strict superset of input pixels (angles
    # added), same colours in and out.
    is285 = all(i.shape[0] == i.shape[1] and
                ((i == 0) | (o == i)).all() and (o != i).any()
                for i, o in zip(ins, outs))
    if is285:
        return "285"

    # 133: output a strict superset of input, variable-size canvas, a single
    # "signature" colour appears as many isolated pixels.
    is133 = all(((i == 0) | (o == i)).all() and (o != i).any() for i, o in zip(ins, outs))
    if is133:
        return "133"

    return None


def route_task(example):
    """Public: fingerprint -> owned task id (str) or None."""
    return _fp(example)


def candidates(example):
    """Dispatch cheaper rebuilds for the four owned tasks.

    Currently returns [] for all of them: no rebuild beat the (already
    label/work-space, already bit-packed) out_blend6 graphs without a cost
    regression, and the hard gate forbids shipping a regression or an
    unverified graph. See module docstring for the per-task cost analysis.
    """
    _ = route_task(example)  # routing is live; no verified cheaper graph to emit
    return []
