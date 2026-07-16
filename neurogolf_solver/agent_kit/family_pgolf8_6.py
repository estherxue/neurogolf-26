"""family_pgolf8_6 -- cheaper EXACT reformulations for GOLF slice golf_targets[6::7].

The slice [6::7] holds the 34 lowest-points (= highest-memory) survivors. After a
full audit every one of them is already golfed with the recommended toolkit
(10x10 work-area crop, single-channel [1,1,10,10] fields, Hillis-Steele doubling,
free `output` write). Concretely verified this session:

  * t392 crk9_concentric (1.76M): searches all NC*NC doubled centres -> ~7 float
    [NC,NC,10,10] tensors.  Cannot be shrunk without an analytic centre (risky,
    not value-exact).  Already at the minimal S=10.
  * t37  pgolf37_diagconnect (166k): per-colour diagonal prefix-OR needs all 9
    colour channels -- a single-channel value/nearest reformulation is NOT the
    true rule (fails 39/266 arc-gen when a foreign dot sits between a pair), so it
    is REJECTED.  9 channels are load-bearing.
  * the remaining 31 are variable-size structural transforms already at/near the
    sub-0.1 headroom floor of the SUM-of-intermediates cost model.

The ONE clean, value-exact, generalizing win is on

  t200  pin200  (fixed 10x10, single seed at (9,dc), colour D != 5):
    the incumbent's colour-reassembly tail spends FIVE [1,10,10,10] tensors
    (bg10, lineC, markC, o10a, out10).  Colours 0 and 5 are FIXED channels and the
    only data-dependent channel is the line colour D.  So we build the fixed
    channels (background + colour-5 markers) with a single Concat and add just one
    variable-colour term (line mask * one-hot(D)) -- THREE [1,10,10,10] tensors
    instead of five (-8000 bytes).  Same geometry (reused, already grader-verified
    on all train+test+arc-gen), same numerics, strictly cheaper.

Detection and geometry are reused verbatim from family_golf8_5 (the incumbent), so
this only swaps the final recolour; the grader re-checks EXACTness on all
train+test+arc-gen, and the leaner tail fires ONLY when the incumbent's numpy
reference reproduces every pair.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

import family_golf8_5 as base
from family_golf8_5 import _G, _pin_solve_np, _model

INT64 = onnx.TensorProto.INT64
FLOAT = onnx.TensorProto.FLOAT


def _build_pin_lean():
    """pin200 with a 3-big-tensor recolour tail (incumbent uses 5)."""
    g = _G()
    N = 10
    # ---- geometry: identical to family_golf8_5._build_pin ----------------
    g.cst("sp0", np.array([0, 0], np.int64), INT64)
    g.cst("spN", np.array([N, N], np.int64), INT64)
    g.cst("spax", np.array([2, 3], np.int64), INT64)
    g.node("Slice", ["input", "sp0", "spN", "spax"], ["S"])
    g.cst("c0", np.array([0], np.int64), INT64)
    g.cst("c1", np.array([1], np.int64), INT64)
    g.cst("cax", np.array([1], np.int64), INT64)
    g.node("Slice", ["S", "c0", "c1", "cax"], ["ch0"])
    g.node("ReduceSum", ["S"], ["realmask"], axes=[1], keepdims=1)
    g.node("Sub", ["realmask", "ch0"], ["dot"])                # single 1 at (9,dc)

    cur = "dot"
    for k in (1, 2, 4, 8):
        s = g.trans(cur, -k, 0, N, f"vup{k}")
        g.node("Max", [cur, s], [f"V{k}"]); cur = f"V{k}"
    for k in (2, 4, 8):
        s = g.trans(cur, 0, k, N, f"lrt{k}")
        g.node("Max", [cur, s], [f"L{k}"]); cur = f"L{k}"
    L = cur                                                    # line mask [1,1,10,10]

    t0 = g.trans("dot", -9, 1, N, "t0")
    cur = t0
    for k in (4, 8):
        s = g.trans(cur, 0, k, N, f"trt{k}")
        g.node("Max", [cur, s], [f"T{k}"]); cur = f"T{k}"
    Tm = cur
    b0 = g.trans("dot", 0, 3, N, "b0")
    cur = b0
    for k in (4, 8):
        s = g.trans(cur, 0, k, N, f"brt{k}")
        g.node("Max", [cur, s], [f"B{k}"]); cur = f"B{k}"
    Bm = cur
    g.node("Max", [Tm, Bm], ["M5"])                           # colour-5 markers [1,1,10,10]

    # ---- leaner recolour tail: 3 big tensors (lineD, base10, out10) ------
    e0 = np.zeros((1, 10, 1, 1), np.float32); e0[0, 0] = 1.0
    g.cst("e0", e0)
    g.node("ReduceMax", ["S"], ["cvA"], axes=[2, 3], keepdims=1)  # 1 at ch0 and ch D
    g.node("Sub", ["cvA", "e0"], ["cvD"])                     # one-hot at line colour D
    g.node("Mul", [L, "cvD"], ["lineD"])                      # [1,10,10,10], only ch D

    g.node("Sub", ["realmask", L], ["bg1"])                   # [1,1,10,10]
    g.node("Sub", ["bg1", "M5"], ["bgm"])                     # background mask
    g.cst("z", np.zeros((1, 1, N, N), np.float32))            # 100-param zero channel
    g.node("Concat", ["bgm", "z", "z", "z", "z", "M5", "z", "z", "z", "z"],
           ["base10"], axis=1)                                # [1,10,10,10] fixed channels
    g.node("Add", ["base10", "lineD"], ["out10"])            # add variable colour D
    g.node("Pad", ["out10"], ["output"], mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, 20, 20])
    return _model(g.nodes, g.inits)


def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return []
    # fire only when the incumbent's verified numpy reference reproduces every pair
    for a, b in prs:
        pred = _pin_solve_np(a)
        if pred is None or pred.shape != b.shape or not (pred == b).all():
            return []
    return [("pin200_lean", _build_pin_lean())]
