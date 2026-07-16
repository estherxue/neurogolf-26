"""family_sgolf2_5 -- cheaper EXACT re-derivations for golf slice [5::7].

Each solver is byte-for-byte numerically identical to an existing accepted
baseline, but removes ONE dominant full-width intermediate tensor
([1,10,30,30]=36000 B or [1,9,30,30]=32400 B) via a value-exact rewrite:

  * golf_edgelines (T161): the interior-occurrence count of every colour was
    computed as ReduceSum(input * intmask) -- a [1,10,30,30] intermediate.  The
    interior mask is SEPARABLE (introw[r] * intcol[c]), so the same count is
    obtained with two fused MatMuls (input @ intcolT then introwT @ .), whose
    intermediates are only [1,10,30,1]=1200 B.  Removes the 36000 B tensor (and
    the 3600 B intmask).

  * golf_rays (T237): the colour-value grid  sum_c c*input[c]  was computed as
    ReduceSum(input * idxvec) -- a [1,10,30,30] intermediate.  The identical
    reduction is a 1x1 Conv (channel weights 0..9), which fuses the multiply-add
    and emits a [1,1,30,30] output directly.  Removes the 36000 B tensor.

  * concentric_rings (T137): the 9-colour slice inp9 = input[:,1:10] ([1,9,30,30]
    =32400 B) was used only to derive the dot-presence mask and the present
    colour.  Both are recovered without it: mask = sum_all(input) - channel0, and
    present9 = slice(ReduceMax(input,axes=[2,3]), 1:10) ([1,9,1,1]).  Removes the
    32400 B tensor.

Every rewrite preserves the exact float-then-threshold semantics, so the graph
is proposed only when the ORIGINAL family's numpy mirror reproduces every
train+test+arc-gen pair EXACTLY (detection reuses the source detectors).  Wrong
tasks never fire and the grader re-validates arc-gen, so it generalises.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

import family_golf3_4 as g34
import family_crk10_4 as c104

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE
H, W, C = HEIGHT, WIDTH, CHANNELS
_BIG = g34._BIG


# --------------------------------------------------------------------------- #
# T161 golf_edgelines -- separable-MatMul interior count (no [1,10,30,30])     #
# --------------------------------------------------------------------------- #
def build_edgelines():
    g = g34._G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])

    realrow = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", ["input"], axes=[1, 3], keepdims=1), half])], to=F)  # [1,1,30,1]
    realcol = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", ["input"], axes=[1, 2], keepdims=1), half])], to=F)  # [1,1,1,30]

    pad_r = g.nd("Pad", [realrow], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 1, 0])
    shiftup = g.nd("Slice", [pad_r, g.i64([1]), g.i64([H + 1]), g.i64([2])])
    brow = g.nd("Mul", [realrow, g.nd("Sub", [one, shiftup])])
    pad_c = g.nd("Pad", [realcol], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 0, 1])
    shiftleft = g.nd("Slice", [pad_c, g.i64([1]), g.i64([W + 1]), g.i64([3])])
    bcol = g.nd("Mul", [realcol, g.nd("Sub", [one, shiftleft])])

    introw = g.nd("Mul", [g.nd("Mul", [realrow, g.nd("Cast", [g.nd("Greater", [rowidx, half])], to=F)]),
                          g.nd("Sub", [one, brow])])                                      # [1,1,30,1]
    intcol = g.nd("Mul", [g.nd("Mul", [realcol, g.nd("Cast", [g.nd("Greater", [colidx, half])], to=F)]),
                          g.nd("Sub", [one, bcol])])                                      # [1,1,1,30]

    # interior occurrence count per channel, WITHOUT a [1,10,30,30] product:
    #   ic[ch] = sum_r introw[r] * sum_c input[ch,r,c]*intcol[c]
    intcolT = g.nd("Transpose", [intcol], perm=[0, 1, 3, 2])                              # [1,1,30,1]
    rowsum = g.nd("MatMul", ["input", intcolT])                                           # [1,10,30,1]
    introwT = g.nd("Transpose", [introw], perm=[0, 1, 3, 2])                              # [1,1,1,30]
    ic = g.nd("MatMul", [introwT, rowsum])                                                # [1,10,1,1]

    total = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    present = g.nd("Cast", [g.nd("Greater", [total, half])], to=F)
    notbg = g.f([1, C, 1, 1], [0.0] + [1.0] * 9)
    gate = g.nd("Mul", [present, notbg])
    negic = g.nd("Mul", [ic, g.f([1, 1, 1, 1], [-1.0])])
    scoregate = g.nd("Mul", [g.nd("Sub", [gate, one]), g.f([1, 1, 1, 1], [_BIG])])
    score = g.nd("Add", [negic, scoregate])                                               # [1,10,1,1]
    markeridx = g.nd("ArgMax", [score], axis=1, keepdims=1)

    midx1 = g.nd("Reshape", [markeridx, g.i64([1])])
    mgrid = g.nd("Gather", ["input", midx1], axis=1)                                      # [1,1,30,30]

    toprow = g.nd("Slice", [mgrid, g.i64([0]), g.i64([1]), g.i64([2])])
    botrow = g.nd("ReduceSum", [g.nd("Mul", [mgrid, brow])], axes=[2], keepdims=1)
    leftcol = g.nd("Slice", [mgrid, g.i64([0]), g.i64([1]), g.i64([3])])
    rightcol = g.nd("ReduceSum", [g.nd("Mul", [mgrid, bcol])], axes=[3], keepdims=1)

    vmask = g.nd("Mul", [toprow, botrow])
    hmask = g.nd("Mul", [leftcol, rightcol])
    vline = g.nd("Mul", [vmask, realrow])
    hline = g.nd("Mul", [hmask, realcol])
    linemask = g.nd("Cast", [g.nd("Greater", [g.nd("Add", [vline, hline]), half])], to=F)
    realmask = g.nd("Mul", [realrow, realcol])

    markercolor = g.nd("Cast", [markeridx], to=F)
    Gf = g.nd("Sub", [g.nd("Mul", [markercolor, linemask]), g.nd("Sub", [one, realmask])])
    Gint = g.nd("Cast", [Gf], to=INT64)
    idxvec = g.i64(list(range(C)), dims=[1, C, 1, 1])
    g.nd("Cast", [g.nd("Equal", [Gint, idxvec])], "output", to=F)
    return g34._model(g)


# --------------------------------------------------------------------------- #
# T237 golf_rays -- 1x1 Conv colour-value grid (no [1,10,30,30])               #
# --------------------------------------------------------------------------- #
def build_rays():
    g = g34._G()
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])

    # sum_c c*input[c] via a 1x1 Conv (weight = channel indices 0..9); the Conv
    # fuses the multiply-add so no [1,10,30,30] product is materialised.
    wv = g.f([1, C, 1, 1], list(range(C)))
    colorgrid = g.nd("Conv", ["input", wv], kernel_shape=[1, 1], pads=[0, 0, 0, 0])       # [1,1,30,30]

    realrow = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", ["input"], axes=[1, 3], keepdims=1), half])], to=F)
    realcol = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", ["input"], axes=[1, 2], keepdims=1), half])], to=F)
    realmask = g.nd("Mul", [realrow, realcol])
    pad_c = g.nd("Pad", [realcol], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 0, 1])
    shiftleft = g.nd("Slice", [pad_c, g.i64([1]), g.i64([W + 1]), g.i64([3])])
    bcol = g.nd("Mul", [realcol, g.nd("Sub", [one, shiftleft])])

    presence = g.nd("Cast", [g.nd("Greater", [colorgrid, half])], to=F)
    markercolor = g.nd("ReduceSum", [colorgrid], axes=[3], keepdims=1)
    markercol = g.nd("ReduceSum", [g.nd("Mul", [presence, colidx])], axes=[3], keepdims=1)

    geo = g.nd("Sub", [one, g.nd("Cast", [g.nd("Less", [colidx, markercol])], to=F)])
    Hc = g.nd("Mul", [g.nd("Mul", [markercolor, geo]), realcol])

    filled = markercolor
    for step in (1, 2, 4, 8, 16):
        pad = g.nd("Pad", [filled], mode="constant", value=0.0,
                   pads=[0, 0, step, 0, 0, 0, 0, 0])
        shifted = g.nd("Slice", [pad, g.i64([0]), g.i64([H]), g.i64([2])])
        iszero = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [filled]), half])], to=F)
        filled = g.nd("Add", [filled, g.nd("Mul", [iszero, shifted])])
    edgecol = g.nd("Mul", [filled, realrow])
    edgeC = g.nd("Mul", [edgecol, bcol])

    OUT = g.nd("Add", [edgeC, g.nd("Mul", [Hc, g.nd("Sub", [one, bcol])])])
    Gf = g.nd("Sub", [g.nd("Mul", [OUT, realmask]), g.nd("Sub", [one, realmask])])
    Gint = g.nd("Cast", [Gf], to=INT64)
    idxvec = g.i64(list(range(C)), dims=[1, C, 1, 1])
    g.nd("Cast", [g.nd("Equal", [Gint, idxvec])], "output", to=F)
    return g34._model(g)


# --------------------------------------------------------------------------- #
# T137 concentric_rings -- drop the [1,9,30,30] inp9 slice                     #
# --------------------------------------------------------------------------- #
def build_t137():
    N = oh.make_node
    Rg = (np.arange(HEIGHT, dtype=np.float32)[None, None, :, None]
          * np.ones((1, 1, 1, WIDTH), np.float32))
    Cg = (np.ones((1, 1, HEIGHT, 1), np.float32)
          * np.arange(WIDTH, dtype=np.float32)[None, None, None, :])
    inits = [
        oh.make_tensor("Rgrid", F, [1, 1, HEIGHT, WIDTH], Rg.flatten().tolist()),
        oh.make_tensor("Cgrid", F, [1, 1, HEIGHT, WIDTH], Cg.flatten().tolist()),
        oh.make_tensor("BIG", F, [1, 1, 1, 1], [1e6]),
        oh.make_tensor("HALF", F, [1, 1, 1, 1], [0.5]),
        oh.make_tensor("ONE", F, [1, 1, 1, 1], [1.0]),
        oh.make_tensor("sl1", INT64, [1], [1]),
        oh.make_tensor("sl10", INT64, [1], [10]),
        oh.make_tensor("slax", INT64, [1], [1]),
    ]
    nodes = []
    nodes.append(N("ReduceSum", ["input"], ["gmask"], axes=[1], keepdims=1))         # real cells
    # dots mask = all colours - background channel-0  (no [1,9,30,30] slice)
    nodes.append(N("Slice", ["input", "slax_z", "sl1", "slax"], ["ch0"]))            # channel 0
    nodes.append(N("Sub", ["gmask", "ch0"], ["mask"]))                               # dots
    # present colour one-hot over channels 1..9 without slicing the 9-plane volume
    nodes.append(N("ReduceMax", ["input"], ["presAll"], axes=[2, 3], keepdims=1))    # [1,10,1,1]
    nodes.append(N("Slice", ["presAll", "sl1", "sl10", "slax"], ["present9"]))       # [1,9,1,1]
    # penalty field for cells that are NOT dots
    nodes.append(N("Sub", ["ONE", "mask"], ["invmask"]))
    nodes.append(N("Mul", ["invmask", "BIG"], ["penBIG"]))
    nodes.append(N("Mul", ["Rgrid", "mask"], ["rmask"]))
    nodes.append(N("Mul", ["Cgrid", "mask"], ["cmask"]))
    nodes.append(N("Add", ["rmask", "penBIG"], ["rminf"]))
    nodes.append(N("Sub", ["rmask", "penBIG"], ["rmaxf"]))
    nodes.append(N("Add", ["cmask", "penBIG"], ["cminf"]))
    nodes.append(N("Sub", ["cmask", "penBIG"], ["cmaxf"]))
    nodes.append(N("ReduceMin", ["rminf"], ["minr"], axes=[2, 3], keepdims=1))
    nodes.append(N("ReduceMax", ["rmaxf"], ["maxr"], axes=[2, 3], keepdims=1))
    nodes.append(N("ReduceMin", ["cminf"], ["minc"], axes=[2, 3], keepdims=1))
    nodes.append(N("ReduceMax", ["cmaxf"], ["maxc"], axes=[2, 3], keepdims=1))
    nodes.append(N("Add", ["minr", "maxr"], ["rsum"]))
    nodes.append(N("Add", ["minc", "maxc"], ["csum"]))
    nodes.append(N("Mul", ["HALF", "rsum"], ["Cr"]))
    nodes.append(N("Mul", ["HALF", "csum"], ["Cc"]))
    nodes.append(N("Sub", ["maxr", "minr"], ["rspan"]))
    nodes.append(N("Sub", ["maxc", "minc"], ["cspan"]))
    nodes.append(N("Max", ["rspan", "cspan"], ["span"]))
    nodes.append(N("Mul", ["HALF", "span"], ["s"]))
    nodes.append(N("Sub", ["Rgrid", "Cr"], ["dRr"]))
    nodes.append(N("Sub", ["Cgrid", "Cc"], ["dCc"]))
    nodes.append(N("Abs", ["dRr"], ["aRr"]))
    nodes.append(N("Abs", ["dCc"], ["aCc"]))
    nodes.append(N("Max", ["aRr", "aCc"], ["D"]))
    nodes.append(N("Mod", ["D", "s"], ["modv"], fmod=1))
    nodes.append(N("Less", ["modv", "HALF"], ["ringb"]))
    nodes.append(N("Cast", ["ringb"], ["ring"], to=F))
    nodes.append(N("Mul", ["ring", "gmask"], ["ring_in"]))
    nodes.append(N("Mul", ["ring_in", "present9"], ["colored"]))                     # [1,9,30,30]
    nodes.append(N("Sub", ["ONE", "ring_in"], ["notring"]))
    nodes.append(N("Mul", ["gmask", "notring"], ["ch0out"]))
    nodes.append(N("Concat", ["ch0out", "colored"], ["output"], axis=1))
    inits.append(oh.make_tensor("slax_z", INT64, [1], [0]))
    return c104._model(nodes, inits)


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
def _emit(out, name, builder):
    try:
        m = builder()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return
    out.append((name, m))


def candidates(examples):
    out = []
    # -- golf3_4 rules (edgelines, rays): reuse its arc-gen-inclusive gate ------
    prs = g34._pairs(examples)
    if prs and not all(np.array_equal(a, b) for a, b in prs):
        if g34._matches(prs, g34._ref_edgelines):
            _emit(out, "golf_edgelines", build_edgelines)
        if g34._matches(prs, g34._ref_rays):
            _emit(out, "golf_rays", build_rays)

    # -- concentric_rings: validate the source mirror on ALL sections -----------
    cprs = []
    for s in ("train", "test", "arc-gen"):
        for e in examples.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim == 2 and b.ndim == 2 and a.size and b.size \
                    and max(a.shape) <= 30 and max(b.shape) <= 30:
                cprs.append((a, b))
    if cprs and c104._t137_detect(cprs):
        _emit(out, "concentric_rings", build_t137)

    return out
