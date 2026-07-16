"""crk5_0 -- PERIODIC PATTERN: reconstruct-then-shift (data-dependent period).

The input carries a 2-D spatially periodic pattern that fills a TOP-LEFT square
region; the remainder of the grid (the complement of that square, i.e. the
bottom-right L) is covered by a single solid "occluder" colour.  The output is
the SAME-size grid in which (a) the occluded region is reconstructed by extending
the periodic pattern and (b) the whole, now-complete, pattern is cyclically
shifted LEFT by one column (output[i,j] = Pfull[i, j+1]).

Both periods (horizontal q, vertical p) vary per example and so are recovered AT
INFERENCE TIME by exact autocorrelation, and the hole is filled with a doubling-OR
of data-dependent shift matrices -- exactly the ``family_dynperiod`` machinery.
The two extra pieces here are:

  * the occluder colour is NOT fixed across the task, so it is detected at runtime
    as the colour of the bottom-right real corner cell (always inside the
    occluded L), realised as a one-hot over channels via a last-real-row x
    last-real-col mask;
  * a final static one-column left shift (Slice+Pad) realises the +1 phase change.

Everything is origin-anchored and static-shape.  Detection mirrors the ONNX
semantics in numpy EXACTLY and only emits when every train+test+arc-gen pair is
reproduced, so wrong hypotheses are never scored.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

import family_dynperiod as dp
from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE
G = HEIGHT  # 30
QMAX = dp.QMAX

_G = dp._G
_model = dp._model
_period_scalar = dp._period_scalar
_fill_axis = dp._fill_axis

# Observed period in this family is <=3; cap the (data-dependent) period search at a
# small value with margin to slash intermediate-memory cost.  Both the ONNX builder
# (_period_scalar) and the numpy reference (_period) read dp.QMAX, so they stay in
# lock-step.  Detection still validates EXACTness on every pair before emitting.
dp.QMAX = 6


# --------------------------------------------------------------------------- #
# ONNX builder                                                                #
# --------------------------------------------------------------------------- #
def build():
    g = _G()
    half = g.f([1, 1], [0.5])
    halfc = g.f([1, 1, 1, 1], [0.5])
    one4 = g.f([1, 1, 1, 1], [1.0])

    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    color = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)          # [1,1,30,30]

    # ---- runtime occluder colour = colour of bottom-right real cell -------- #
    rowsum = g.nd("ReduceSum", [realmask], axes=[3], keepdims=1)           # [1,1,30,1]
    colsum = g.nd("ReduceSum", [realmask], axes=[2], keepdims=1)           # [1,1,1,30]
    realrow = g.nd("Cast", [g.nd("Greater", [rowsum, halfc])], to=F)
    realcol = g.nd("Cast", [g.nd("Greater", [colsum, halfc])], to=F)
    rr_next = g.nd("Pad", [g.nd("Slice", [realrow, g.i64([1]), g.i64([G]), g.i64([2])])],
                   mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 1, 0])
    lastrow = g.nd("Mul", [realrow, g.nd("Sub", [one4, rr_next])])          # [1,1,30,1]
    rc_next = g.nd("Pad", [g.nd("Slice", [realcol, g.i64([1]), g.i64([G]), g.i64([3])])],
                   mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 0, 1])
    lastcol = g.nd("Mul", [realcol, g.nd("Sub", [one4, rc_next])])          # [1,1,1,30]
    corner = g.nd("Mul", [lastrow, lastcol])                               # [1,1,30,30]
    h_onehot = g.nd("ReduceSum", [g.nd("Mul", ["input", corner])], axes=[2, 3], keepdims=1)
    #                                                                        # [1,10,1,1]

    holeplane = g.nd("ReduceSum", [g.nd("Mul", ["input", h_onehot])], axes=[1], keepdims=1)
    known = g.nd("Sub", [realmask, holeplane])                            # real & not hole
    ones_c = g.f([1, CHANNELS, 1, 1], [1.0] * CHANNELS)
    X = g.nd("Mul", ["input", g.nd("Sub", [ones_c, h_onehot])])           # hole ch zeroed

    q_h = _period_scalar(g, known, color, 3, halfc)
    q_v = _period_scalar(g, known, color, 2, halfc)

    rowidx = g.f([G, 1], list(range(G)))
    colidx = g.f([1, G], list(range(G)))
    Hf = _fill_axis(g, X, q_h, 3, rowidx, colidx, half)
    Vf = _fill_axis(g, Hf, q_v, 2, rowidx, colidx, half)

    # cyclic-ish LEFT shift by one column: out[...,j] = Vf[...,j+1]
    S = g.nd("Pad", [g.nd("Slice", [Vf, g.i64([1]), g.i64([G]), g.i64([3])])],
             mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 0, 1])
    g.nd("Mul", [S, realmask], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy reference (mirrors the ONNX semantics EXACTLY)                          #
# --------------------------------------------------------------------------- #
def _sim_grid(a, h):
    X = dp._onehot(a)
    realmask = X.sum(0)
    color = sum(c * X[c] for c in range(CHANNELS))
    known = realmask * (np.abs(color - h) > 0.5)
    Xc = X.copy()
    Xc[h] = 0.0
    q_h = dp._period(known, color, 1)
    q_v = dp._period(known, color, 0)
    Hf = dp._fill(Xc, q_h, 1)
    Vf = dp._fill(Hf, q_v, 0)
    S = np.zeros_like(Vf)
    S[..., :G - 1] = Vf[..., 1:]
    out = S * realmask[None]
    sel = out > 0.5
    cnt = sel.sum(0)
    if (cnt[realmask > 0.5] != 1).any():
        return None
    if (cnt[realmask <= 0.5] != 0).any():
        return None
    return np.argmax(sel, axis=0)


def _corner(a):
    H, W = a.shape
    return int(a[H - 1, W - 1])


# --------------------------------------------------------------------------- #
# detection / entry point                                                      #
# --------------------------------------------------------------------------- #
def candidates(ex):
    det = dp._pairs(ex, ("train", "test"))
    allp = dp._pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []
    if any(a.shape != b.shape for a, b in allp):
        return []
    if all(np.array_equal(a, b) for a, b in allp):
        return []

    def ok(plist):
        for a, b in plist:
            h = _corner(a)
            if not (0 <= h < CHANNELS):
                return False
            pred = _sim_grid(a, h)
            if pred is None:
                return False
            H, W = b.shape
            if not np.array_equal(pred[:H, :W], b):
                return False
        return True

    if not ok(det) or not ok(allp):
        return []

    try:
        model = build()
        onnx.checker.check_model(model, full_check=True)
    except Exception:
        return []
    return [("crk5_0_pshift", model)]
