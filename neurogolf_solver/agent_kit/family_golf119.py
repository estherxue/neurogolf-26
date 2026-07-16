"""family_golf119 -- cheaper EXACT solver for task 119 (bouncing diagonal-ray billiard CA).

The baseline (bounce_ray_n12, family_crk6_0.build119 with N=12) runs the 4-direction
billiard CA over the full 30x30 tensor, so every intermediate is [1,1,30,30] = 3600 B.

Task 119 is FIXED-SIZE: every train+test+arc-gen input AND output is the same square
SxS (S=12 here). Per the fixed-size rule this lets us CROP the whole computation to the
real SxS region: we Slice the input to [1,10,S,S], run the byte-identical billiard CA on
[1,1,S,S] intermediates (S*S*4 = 576 B for S=12, a 6.25x memory cut vs 3600 B), then Pad
the SxS result back to [1,10,30,30]. The algorithm is unchanged (same reflection CA, same
step count), so it stays exact for any grid the generator produces at this fixed size.

We keep the same step count N=12 as the accepted baseline (observed max propagation depth
across all 266 pairs is 10, so N=12 keeps the baseline's +2 safety buffer). Correctness is
identical to the baseline; only the working resolution shrinks.

Gate: we require every train+test+arc-gen pair to be the same square SxS, colour 3 to be the
only introduced colour, and the full-strength numpy reference (family_crk6_0.solve119 at a
generous N) to reproduce every pair exactly -- so we never fire on the wrong task.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS,
)
import family_crk6_0 as base

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE

_DIRS = base._DIRS
_RH = base._RH
_RV = base._RV

N119 = 12  # same step count as the accepted baseline (observed max depth 10, +2 buffer)


class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, F, list(dims),
                                         [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        return n

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _shift(g, t, dr, dc, S):
    """Shift content of a [1,K,S,S] tensor by (dr,dc): out[r,c]=t[r-dr,c-dc], zero fill."""
    pads = [0, 0, max(dr, 0), max(dc, 0), 0, 0, max(-dr, 0), max(-dc, 0)]
    p = g.nd("Pad", [t], mode="constant", value=0.0, pads=pads)
    rs, cs = max(-dr, 0), max(-dc, 0)
    return g.nd("Slice", [p, g.i64([rs, cs]), g.i64([rs + S, cs + S]), g.i64([2, 3])])


def build119_crop(S, N=N119):
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0])
    half = g.f([1, 1, 1, 1], [0.5])
    coeff = g.f([1, CHANNELS, 1, 1], [-1, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    # crop input to the real SxS region
    xc = g.nd("Slice", ["input", g.i64([0, 0]), g.i64([S, S]), g.i64([2, 3])])
    wall = g.nd("Slice", [xc, g.i64([2]), g.i64([3]), g.i64([1])])
    ball = g.nd("Slice", [xc, g.i64([8]), g.i64([9]), g.i64([1])])
    real = g.nd("ReduceSum", [xc], axes=[1], keepdims=1)  # all-ones over the crop

    rowwall = g.nd("ReduceSum", [wall], axes=[3], keepdims=1)
    roww = g.nd("ReduceSum", [real], axes=[3], keepdims=1)
    eqrow = g.nd("Cast", [g.nd("Greater", [rowwall, g.nd("Sub", [roww, half])])], to=F)
    posrow = g.nd("Cast", [g.nd("Greater", [roww, half])], to=F)
    Hw = g.nd("Min", [g.nd("ReduceSum", [g.nd("Mul", [eqrow, posrow])], axes=[2, 3], keepdims=1), one])
    colwall = g.nd("ReduceSum", [wall], axes=[2], keepdims=1)
    colh = g.nd("ReduceSum", [real], axes=[2], keepdims=1)
    eqcol = g.nd("Cast", [g.nd("Greater", [colwall, g.nd("Sub", [colh, half])])], to=F)
    poscol = g.nd("Cast", [g.nd("Greater", [colh, half])], to=F)
    Vw = g.nd("Min", [g.nd("ReduceSum", [g.nd("Mul", [eqcol, poscol])], axes=[2, 3], keepdims=1), one])

    blocked = {}; fw = {}
    for name, d in _DIRS.items():
        nd0, nd1 = -d[0], -d[1]
        sw = _shift(g, wall, nd0, nd1, S)
        sr = _shift(g, real, nd0, nd1, S)
        blocked[name] = g.nd("Min", [g.nd("Add", [sw, g.nd("Sub", [one, sr])]), one])
        fw[name] = sw

    se = g.nd("Mul", [ball, g.nd("Min", [g.nd("Add", [_shift(g, ball, 1, 1, S), _shift(g, ball, -1, -1, S)]), one])])
    sw_ = g.nd("Mul", [ball, g.nd("Min", [g.nd("Add", [_shift(g, ball, 1, -1, S), _shift(g, ball, -1, 1, S)]), one])])
    B = {"SE": se, "NW": se, "SW": sw_, "NE": sw_}
    visited = None
    for _ in range(N):
        free = {n: g.nd("Mul", [B[n], g.nd("Sub", [one, blocked[n]])]) for n in _DIRS}
        hit = {n: g.nd("Mul", [B[n], fw[n]]) for n in _DIRS}
        newB = {}
        for name, d in _DIRS.items():
            cont = _shift(g, free[name], d[0], d[1], S)
            ht = g.nd("Mul", [Hw, _shift(g, hit[_RH[name]], d[0], d[1], S)])
            vt = g.nd("Mul", [Vw, _shift(g, hit[_RV[name]], d[0], d[1], S)])
            s = g.nd("Add", [g.nd("Add", [cont, ht]), vt])
            newB[name] = g.nd("Mul", [g.nd("Min", [s, one]), real])
        B = newB
        cur = g.nd("Min", [g.nd("Add", [g.nd("Add", [B["SE"], B["NW"]]),
                                        g.nd("Add", [B["SW"], B["NE"]])]), one])
        visited = cur if visited is None else g.nd("Min", [g.nd("Add", [visited, cur]), one])

    mark = g.nd("Mul", [g.nd("Mul", [visited, g.nd("Sub", [one, ball])]), g.nd("Sub", [one, wall])])
    delta = g.nd("Mul", [mark, coeff])
    outc = g.nd("Add", [xc, delta])
    pad = 30 - S
    g.nd("Pad", [outc], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, pad, pad])
    return _model(g)


def _pairs(examples):
    out = []
    for sec in ("train", "test", "arc-gen"):
        for e in examples.get(sec, []):
            try:
                a = np.array(e["input"], int); b = np.array(e["output"], int)
            except Exception:
                continue
            if a.ndim == 2 and b.ndim == 2 and a.size and b.size:
                out.append((a, b))
    return out


def candidates(examples):
    prs = _pairs(examples)
    if not prs:
        return []
    # fixed-size square gate: every input and output share one square SxS
    shapes = {a.shape for a, _ in prs} | {b.shape for _, b in prs}
    if len(shapes) != 1:
        return []
    S = next(iter(shapes))[0]
    if shapes != {(S, S)} or not (1 <= S <= 30):
        return []
    # only colour 3 is introduced
    intro3 = False
    for a, b in prs:
        d = a != b
        if d.any():
            if (b[d] != 3).any():
                return []
            intro3 = True
    if not intro3:
        return []
    # faithful full-strength reference must reproduce every pair exactly
    for a, b in prs:
        if not np.array_equal(base.solve119(a, 40), b):
            return []
    try:
        model = build119_crop(S, N119)
    except Exception:
        return []
    return [("bounce_ray_crop_n12", model)]
