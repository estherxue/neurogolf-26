"""family_sgolf_6 -- CHEAPER, value-EXACT FIXED-SIZE crop variants of incumbent solvers.

Technique (ANTI-OVERFIT SAFE, per the fixed-size-crop rule): for a task whose every
train+test+arc-gen INPUT and OUTPUT is the same square SxS, the generator is fixed-size,
so the hidden/private grids are the SAME size.  We may therefore rebuild the byte-identical
incumbent graph at working resolution S (monkeypatching the incumbent's grid-size globals so
its coordinate matrices / index vectors shrink to SxS), then wrap it: Slice input->[1,10,S,S],
run the unchanged node sequence, Pad the SxS result back to [1,10,30,30].  The ALGORITHM and
every step count are UNCHANGED -- only the working canvas shrinks -- so correctness is identical
for any grid the generator can produce at this fixed size.  Every intermediate drops from
30x30 to SxS and the coordinate initializers drop from 30x30 to SxS, cutting cost -> more points.

Targets (all FIXED-size squares; incumbent references gate so we never fire on the wrong task):
  * task 99  container_fill (family_crk6_0.build99, G-sized triangular/shift MatMuls) -- 10x10.
  * task 62  crk2_4_t62     (family_crk2_4.build_62, H/W index vectors + reflection MatMuls) -- 10x10.
  * task 165 col_rays165    (family_crk8_3.build_lines165, G-sized row-index constant)  -- 20x20.

Each candidate is emitted only when the incumbent family's OWN numpy reference reproduces every
pair exactly AND every input/output is one fixed square, then validated EXACT by the shared
harness (train+test+arc-gen) before it can score.  S = the detected fixed side (the smallest
exact crop); the harness's exactness gate confirms full propagation at that resolution.
"""
from __future__ import annotations

import copy
import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

import family_crk6_0 as c60
import family_crk2_4 as c24
import family_crk8_3 as c83


# --------------------------------------------------------------------------- #
# generic helpers                                                              #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.asarray(e["input"], int)
            b = np.asarray(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            out.append((a, b))
    return out


def _fixed_square(prs):
    """Return S if every input and output is the same SxS square (1<=S<=30), else None."""
    shapes = {a.shape for a, _ in prs} | {b.shape for _, b in prs}
    if len(shapes) != 1:
        return None
    (h, w), = shapes
    if h != w or not (1 <= h <= 30):
        return None
    return h


def _match(prs, ref):
    """True iff ref(a)==b for every pair (same shape) and at least one pair changes."""
    changed = False
    for a, b in prs:
        try:
            o = ref(a)
        except Exception:
            return False
        if o is None:
            return False
        o = np.asarray(o)
        if o.shape != b.shape or not np.array_equal(o, b):
            return False
        if not np.array_equal(a, b):
            changed = True
    return changed


def _crop_wrap(model, S):
    """Slice input->[1,10,S,S], run the (unchanged) SxS graph, Pad result->30x30."""
    m = copy.deepcopy(model)
    g = m.graph
    for nd in g.node:
        nd.input[:] = ["inp_s" if x == "input" else x for x in nd.input]
        nd.output[:] = ["out_s" if x == "output" else x for x in nd.output]
    g.initializer.extend([
        oh.make_tensor("cwS", TP.INT64, [2], [0, 0]),
        oh.make_tensor("cwE", TP.INT64, [2], [S, S]),
        oh.make_tensor("cwA", TP.INT64, [2], [2, 3]),
    ])
    g.node.insert(0, oh.make_node("Slice", ["input", "cwS", "cwE", "cwA"], ["inp_s"], name="cw_s"))
    g.node.append(oh.make_node("Pad", ["out_s"], ["output"], mode="constant", value=0.0,
                               pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S], name="cw_p"))
    return m


# --------------------------------------------------------------------------- #
# per-task cropped builders (monkeypatch the incumbent's size globals -> S)     #
# --------------------------------------------------------------------------- #
def _build_99(S):
    old = c60.G
    c60.G = S
    try:
        mdl = c60.build99()
    finally:
        c60.G = old
    return _crop_wrap(mdl, S)


def _build_62(S):
    oh_, ow_ = c24.H, c24.W
    c24.H = c24.W = S
    try:
        g = c24._G()
        mdl = c24.build_62(g)
    finally:
        c24.H, c24.W = oh_, ow_
    return _crop_wrap(mdl, S)


def _build_165(S):
    old = c83.G
    c83.G = S
    try:
        mdl = c83.build_lines165()
    finally:
        c83.G = old
    return _crop_wrap(mdl, S)


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    S = _fixed_square(prs)
    if S is None:
        return []
    out = []

    # task 99 : container_fill
    if _match(prs, c60.solve99):
        try:
            m = _build_99(S)
            onnx.checker.check_model(m, full_check=True)
            out.append(("container_fill_crop", m))
        except Exception:
            pass

    # task 62 : reflection stamp
    if _match(prs, c24.r62):
        try:
            m = _build_62(S)
            onnx.checker.check_model(m, full_check=True)
            out.append(("crk2_4_t62_crop", m))
        except Exception:
            pass

    # task 165 : column rays from shape through dots
    if _match(prs, c83._ref165):
        try:
            m = _build_165(S)
            onnx.checker.check_model(m, full_check=True)
            out.append(("col_rays165_crop", m))
        except Exception:
            pass

    return out
