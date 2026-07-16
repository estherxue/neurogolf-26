"""PROJECTION / SHADOW / BROADCAST along an axis (origin-anchored, opset 10).

Every rule here REDUCES along the channel and/or a spatial axis and then
BROADCASTS the result back over the (real, top-left-anchored) grid region.  The
one-hot tensor is zero-padded to 30x30 with the grid at (0,0), so the real-cell
mask is simply  R = ReduceSum(input, axis=channel)  (==1 on real cells, ==0 on
padding).  All broadcasts are gated by R, so content never leaks into the pad and
the rules generalise to grids of any size.

Rules (the matching one is inferred from train/test/arc-gen pairs)
-----------------------------------------------------------------
  global_fill        Fill the WHOLE real region with one globally-selected
                     colour: the most frequent colour (per-channel ReduceSum ->
                     ArgMax over channels, first-index tie-break, optionally
                     counting background).  -> a pure reduce+broadcast.

  mark_mono(axis,M)  Detect rows (or columns) that are entirely a single non-bg
                     colour (a "monochrome line"); paint every real cell of such
                     a line with a fixed mark colour M, and make every other real
                     cell background.  A line i is monochrome iff
                     max_{c!=0} count_c(line_i) == (#real cells in line_i) >= 1.

  posmap(axis,map)   Each row (or column) holds exactly one non-bg cell; the
                     output broadcasts a colour over that whole line, chosen by
                     the ABSOLUTE position of the marker (a fixed, origin-anchored
                     position->colour table).  Realised as a single Conv whose
                     1xW (or Hx1) kernel contracts the marker plane along the
                     line, emitting a per-line one-hot colour that is then
                     broadcast back over the real region.

Realisation is opset-10 only (ReduceSum/ReduceMax/ArgMax with axes as
ATTRIBUTES, Equal/Greater/Cast/Mul/Sub/Concat/Conv/Slice) and materialises only
small [1,*,30,1]/[1,1,30,30] intermediates, so the cost stays low.  Detection
reproduces the ONNX semantics exactly (including ArgMax first-index tie-break)
and only emits a candidate when it matches EVERY available pair.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
_BIG = 1.0e6


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals):
        nm = self.name("c")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(dims),
                                         np.asarray(vals, np.float32).ravel().tolist()))
        return nm

    def i64(self, dims, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, list(dims),
                                         [int(v) for v in np.asarray(vals).ravel().tolist()]))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _model(g):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _nbg():
    return [0.0] + [1.0] * (CHANNELS - 1)


def _onehot_ch(k):
    return [1.0 if c == k else 0.0 for c in range(CHANNELS)]


# --------------------------------------------------------------------------- #
# ONNX builders                                                               #
# --------------------------------------------------------------------------- #
def build_global_fill(include_bg):
    """Fill the whole real region with the most-frequent colour (ArgMax over
    per-channel counts; first-index tie-break; bg counted iff include_bg)."""
    g = _G()
    counts = g.node("ReduceSum", ["input"], axes=[2, 3], keepdims=1)      # [1,10,1,1]
    sel = counts
    if not include_bg:
        bgneg = g.f([1, CHANNELS, 1, 1], [-_BIG] + [0.0] * (CHANNELS - 1))
        sel = g.node("Add", [counts, bgneg])
    amax = g.node("ArgMax", [sel], axis=1, keepdims=1)                    # int64 [1,1,1,1]
    idx = g.i64([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    eq = g.node("Equal", [amax, idx])                                     # bool [1,10,1,1]
    gate = g.node("Cast", [eq], to=DATA_TYPE)                             # [1,10,1,1]
    R = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)             # [1,1,30,30]
    g.node("Mul", [gate, R], "output")                                   # [1,10,30,30]
    return _model(g)


def _is_mono_line(g, axis):
    """Return a [1,1,30,1] (rows) or [1,1,1,30] (cols) 0/1 tensor flagging
    monochrome non-bg lines, plus the real-cell mask R [1,1,30,30]."""
    red_axis = 3 if axis == 0 else 2          # reduce away the in-line dimension
    rc = g.node("ReduceSum", ["input"], axes=[red_axis], keepdims=1)      # [1,10,30,1]/[1,10,1,30]
    real_line = g.node("ReduceSum", [rc], axes=[1], keepdims=1)           # cells per line
    nbg = g.f([1, CHANNELS, 1, 1], _nbg())
    rc_nbg = g.node("Mul", [rc, nbg])
    maxnbg = g.node("ReduceMax", [rc_nbg], axes=[1], keepdims=1)          # top non-bg count
    half = g.f([1, 1, 1, 1], [0.5])
    ge_real = g.node("Greater", [maxnbg, g.node("Sub", [real_line, half])])  # maxnbg>=real
    ge_one = g.node("Greater", [maxnbg, half])                              # maxnbg>=1
    is_mono = g.node("Mul", [g.node("Cast", [ge_real], to=DATA_TYPE),
                             g.node("Cast", [ge_one], to=DATA_TYPE)])
    R = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)              # [1,1,30,30]
    return is_mono, R


def build_mark_mono(axis, M):
    """Monochrome lines -> mark colour M; every other real cell -> background."""
    g = _G()
    is_mono, R = _is_mono_line(g, axis)
    mark = g.node("Mul", [is_mono, R])          # [1,1,30,30] 1 on marked real cells
    bgp = g.node("Sub", [R, mark])              # unmarked real cells (-> channel 0)
    zeros = g.node("Sub", [R, R])               # [1,1,30,30] zeros
    chans = [zeros] * CHANNELS
    chans[0] = bgp
    chans[M] = mark
    g.node("Concat", chans, "output", axis=1)
    return _model(g)


def build_posmap(axis, colormap):
    """Each line has one non-bg marker; broadcast colormap[pos] over the line.
    colormap: dict {position -> output colour}."""
    g = _G()
    R = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)              # [1,1,30,30]
    s = g.i64([1], [0]); e = g.i64([1], [1]); ax = g.i64([1], [1])
    ch0 = g.node("Slice", ["input", s, e, ax])                           # [1,1,30,30] bg plane
    marker = g.node("Sub", [R, ch0])                                     # non-bg presence
    if axis == 0:                                                        # contract columns
        W = np.zeros((CHANNELS, 1, 1, WIDTH), np.float32)
        for j, c in colormap.items():
            if 0 <= j < WIDTH:
                W[c, 0, 0, j] = 1.0
        wt = g.f([CHANNELS, 1, 1, WIDTH], W)
        colsel = g.node("Conv", [marker, wt], kernel_shape=[1, WIDTH], pads=[0, 0, 0, 0])
    else:                                                                # contract rows
        W = np.zeros((CHANNELS, 1, HEIGHT, 1), np.float32)
        for i, c in colormap.items():
            if 0 <= i < HEIGHT:
                W[c, 0, i, 0] = 1.0
        wt = g.f([CHANNELS, 1, HEIGHT, 1], W)
        colsel = g.node("Conv", [marker, wt], kernel_shape=[HEIGHT, 1], pads=[0, 0, 0, 0])
    g.node("Mul", [colsel, R], "output")                                 # broadcast over region
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy references (mirror ONNX semantics for detection)                      #
# --------------------------------------------------------------------------- #
def _argmax_color(a, include_bg):
    cnt = np.array([(a == c).sum() for c in range(CHANNELS)], np.int64)
    if not include_bg:
        cnt = cnt.copy()
        cnt[0] = -1
    return int(cnt.argmax())          # numpy argmax == ONNX ArgMax (first index)


def _ref_global_fill(a, include_bg):
    return np.full_like(a, _argmax_color(a, include_bg))


def _ref_mark_mono(a, axis, M):
    h, w = a.shape
    out = np.zeros_like(a)
    if axis == 0:
        for i in range(h):
            row = a[i, :]
            if (row != 0).all() and len(set(row.tolist())) == 1:
                out[i, :] = M
    else:
        for j in range(w):
            col = a[:, j]
            if (col != 0).all() and len(set(col.tolist())) == 1:
                out[:, j] = M
    return out


def _fit_mark_mono(prs, axis):
    """Return mark colour M consistent across all pairs, or None."""
    marks = set()
    for a, b in prs:
        if a.shape != b.shape:
            return None
        flag = np.zeros(a.shape, bool)
        h, w = a.shape
        if axis == 0:
            for i in range(h):
                row = a[i, :]
                if (row != 0).all() and len(set(row.tolist())) == 1:
                    flag[i, :] = True
        else:
            for j in range(w):
                col = a[:, j]
                if (col != 0).all() and len(set(col.tolist())) == 1:
                    flag[:, j] = True
        if not flag.any():
            return None                       # nothing flagged -> not this rule here
        if (b[~flag] != 0).any():
            return None
        marks |= set(b[flag].tolist())
    marks.discard(0)
    if len(marks) != 1:
        return None
    return next(iter(marks))


def _fit_posmap(prs, axis):
    """Return position->colour map, or None."""
    mp = {}
    for a, b in prs:
        if a.shape != b.shape:
            return None
        h, w = a.shape
        if axis == 0:
            for i in range(h):
                nz = np.where(a[i, :] != 0)[0]
                if nz.size == 0:
                    if (b[i, :] != 0).any():
                        return None
                    continue
                if nz.size != 1:
                    return None
                line = set(b[i, :].tolist())
                if len(line) != 1:
                    return None
                y = next(iter(line))
                if y == 0:
                    return None
                pos = int(nz[0])
                if pos in mp and mp[pos] != y:
                    return None
                mp[pos] = y
        else:
            for j in range(w):
                nz = np.where(a[:, j] != 0)[0]
                if nz.size == 0:
                    if (b[:, j] != 0).any():
                        return None
                    continue
                if nz.size != 1:
                    return None
                line = set(b[:, j].tolist())
                if len(line) != 1:
                    return None
                y = next(iter(line))
                if y == 0:
                    return None
                pos = int(nz[0])
                if pos in mp and mp[pos] != y:
                    return None
                mp[pos] = y
    return mp or None


def _ref_posmap(a, axis, mp):
    h, w = a.shape
    out = np.zeros_like(a)
    if axis == 0:
        for i in range(h):
            nz = np.where(a[i, :] != 0)[0]
            if nz.size == 1 and int(nz[0]) in mp:
                out[i, :] = mp[int(nz[0])]
            elif nz.size != 0:
                return None
    else:
        for j in range(w):
            nz = np.where(a[:, j] != 0)[0]
            if nz.size == 1 and int(nz[0]) in mp:
                out[:, j] = mp[int(nz[0])]
            elif nz.size != 0:
                return None
    return out


# --------------------------------------------------------------------------- #
# entry point                                                                 #
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


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if all(np.array_equal(a, b) for a, b in prs):     # identity -> not our family
        return []

    out = []

    # ---- global fill (most-frequent colour broadcast over the region) ------- #
    if all(a.shape == b.shape for a, b in prs):
        for inc in (True, False):
            if all(np.array_equal(_ref_global_fill(a, inc), b) for a, b in prs):
                try:
                    out.append((f"globalfill_{'bg' if inc else 'nbg'}",
                                build_global_fill(inc)))
                except Exception:
                    pass
                break

    # ---- mark monochrome rows / columns ------------------------------------ #
    for axis, nm in ((0, "row"), (1, "col")):
        M = _fit_mark_mono(prs, axis)
        if M is None:
            continue
        if all(np.array_equal(_ref_mark_mono(a, axis, M), b) for a, b in prs):
            try:
                out.append((f"markmono_{nm}_M{M}", build_mark_mono(axis, M)))
            except Exception:
                pass

    # ---- position -> colour broadcast over each line ----------------------- #
    for axis, nm in ((0, "row"), (1, "col")):
        mp = _fit_posmap(prs, axis)
        if mp is None or len(mp) == 0:
            continue
        ok = True
        for a, b in prs:
            r = _ref_posmap(a, axis, mp)
            if r is None or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            try:
                out.append((f"posmap_{nm}", build_posmap(axis, mp)))
            except Exception:
                pass

    return out
