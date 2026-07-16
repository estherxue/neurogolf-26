"""family_golf3_0 -- CHEAPER exact re-solvers for a slice of golf targets.

Each candidate re-derives the rule from train+test+arc-gen pairs, validates a
numpy mirror of the *exact* ONNX semantics on every available pair, and only then
emits a minimal opset-10 graph.  The integrator auto-picks the cheapest correct
solver, so we just need these to be exact and cheaper than the incumbent.

Golf levers used here:
  * data-independent index/identity/shift matrices baked as INITIALIZERS
    (params, free of intermediate-memory), not Constant nodes (which would also
    be charged as memory);
  * single-channel [1,1,30,30] intermediates wherever possible;
  * data-dependent 2x tiling expressed as two MatMuls against [1,1,30,30]
    selection matrices  Mw = I + shift(+w),  Mh = I + shift(-h);
  * write the final answer straight into the FREE `output` tensor via Add/Concat.

Targets (rule -> incumbent points):
  388  coltile_f8: per-column 8-fill of the zeros that share a column with a
       non-background cell, then 2x2 tile of the grid                  (12.53)
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE
H, W = HEIGHT, WIDTH


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                       #
# --------------------------------------------------------------------------- #
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

    def i64(self, vals, dims=None):
        n = self.nm("i")
        dims = dims if dims is not None else [len(vals)]
        self.inits.append(oh.make_tensor(n, INT64, list(dims),
                          [int(v) for v in np.asarray(vals).ravel()]))
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


def _slice(g, src, starts, ends, axes, steps=None):
    ins = [src, g.i64(starts), g.i64(ends), g.i64(axes)]
    if steps is not None:
        ins.append(g.i64(steps))
    return g.nd("Slice", ins)


# --------------------------------------------------------------------------- #
# pairs                                                                        #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


# ===========================================================================
# 388  coltile_f8: column-fill background with F, then 2x2 tile
# ===========================================================================
def _detect_388(prs):
    """Return fill colour F if every pair matches the rule, else None."""
    # all outputs must be exactly 2x the input in both dims
    for a, b in prs:
        if b.shape != (2 * a.shape[0], 2 * a.shape[1]):
            return None
    # derive F from the first pair
    a0, b0 = prs[0]
    colnz = (a0 != 0).any(axis=0)
    Fs = set()
    for r in range(a0.shape[0]):
        for c in range(a0.shape[1]):
            if a0[r, c] == 0 and colnz[c]:
                Fs.add(int(b0[r, c]))
    if len(Fs) != 1:
        return None
    Fc = Fs.pop()
    if not (1 <= Fc <= 9):
        return None
    return Fc


def _mir_388(a, Fc):
    h, w = a.shape
    colnz = (a != 0).any(axis=0)
    b = a.copy()
    b[(a == 0) & colnz[None, :]] = Fc
    return np.tile(b, (2, 2))


def _build_388(g, Fc):
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    negone = g.f([1, 1, 1, 1], [-1.0])
    colidx = g.f([1, 1, 1, W], list(range(W)))
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    zero1 = g.f([1, 1, H, W], np.zeros((H, W), np.float32))
    # identity and difference matrices (data-independent => initialisers)
    Ident = g.f([1, 1, H, W], np.eye(H, W, dtype=np.float32))
    Dmat = g.f([1, 1, H, W],
               (np.arange(H)[:, None] - np.arange(W)[None, :]).astype(np.float32))

    # ---- grid width w and height h (from the raw input) -------------------
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)        # [1,1,30,30]
    colany = g.nd("ReduceMax", [realmask], axes=[2], keepdims=1)          # [1,1,1,30]
    maxc = g.nd("ReduceMax", [g.nd("Mul", [colany, colidx])], axes=[3], keepdims=1)
    w = g.nd("Add", [maxc, one])                                          # [1,1,1,1]
    rowany = g.nd("ReduceMax", [realmask], axes=[3], keepdims=1)          # [1,1,30,1]
    maxr = g.nd("ReduceMax", [g.nd("Mul", [rowany, rowidx])], axes=[2], keepdims=1)
    h = g.nd("Add", [maxr, one])                                          # [1,1,1,1]

    # ---- tiling selection matrices  Mw = I + [D == -w], Mh = I + [D == h] --
    castDw = g.nd("Cast", [g.nd("Less",
                  [g.nd("Abs", [g.nd("Add", [Dmat, w])]), half])], to=F)  # [1,1,30,30]
    Mw = g.nd("Add", [Ident, castDw])
    castDh = g.nd("Cast", [g.nd("Less",
                  [g.nd("Abs", [g.nd("Sub", [Dmat, h])]), half])], to=F)
    Mh = g.nd("Add", [Ident, castDh])

    # ---- 2x2 tile of the raw input ---------------------------------------
    inW = g.nd("MatMul", ["input", Mw])                                   # [1,10,30,30]
    tiled = g.nd("MatMul", [Mh, inW])                                     # [1,10,30,30]

    # ---- fill positions computed on the tiled grid -----------------------
    rmask_t = g.nd("ReduceSum", [tiled], axes=[1], keepdims=1)            # [1,1,30,30]
    ch0_t = _slice(g, tiled, [0], [1], [1])                              # [1,1,30,30]
    nz_t = g.nd("Sub", [rmask_t, ch0_t])                                 # nonbg cells
    colnz_t = g.nd("ReduceMax", [nz_t], axes=[2], keepdims=1)             # [1,1,1,30]
    fillpos = g.nd("Mul", [ch0_t, colnz_t])                              # [1,1,30,30]
    negfill = g.nd("Mul", [fillpos, negone])

    # delta: -fillpos on channel 0, +fillpos on channel F, zero elsewhere
    parts = []
    for ch in range(CHANNELS):
        if ch == 0:
            parts.append(negfill)
        elif ch == Fc:
            parts.append(fillpos)
        else:
            parts.append(zero1)
    delta = g.nd("Concat", parts, axis=1)                                # [1,10,30,30]
    g.nd("Add", [tiled, delta], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []
    out = []

    def emit(name, mirror, build):
        try:
            for a, b in prs:
                o = mirror(a)
                if o is None or o.shape != b.shape or not np.array_equal(o, b):
                    return
            g = _G()
            m = build(g)
            onnx.checker.check_model(m, full_check=True)
            out.append((name, m))
        except Exception:
            pass

    # 388 -- column 8-fill then 2x2 tile
    Fc = None
    try:
        Fc = _detect_388(prs)
    except Exception:
        Fc = None
    if Fc is not None:
        emit("g30_coltile388", lambda a: _mir_388(a, Fc),
             lambda g: _build_388(g, Fc))

    return out
