"""COMPLETE / REGULARISE partial axis-aligned shapes (origin-anchored, opset 10).

Every rule here REPAIRS an incomplete, regular shape by reconstructing it from a
small, structurally-inferred descriptor and OR-ing the reconstruction onto the
(preserved) input.  The one-hot tensor is zero-padded to 30x30 with the grid
anchored at (0,0), so all reductions ignore padding (it is all-zero) and every
shift is a zero-padded Pad+Slice -> the rules stay anchored at the top-left for
grids of any size and generalise structurally.

Rules (the matching one is inferred from train/test/arc-gen pairs)
-----------------------------------------------------------------
  rect_filled / rect_outline / rect_corners
        For every present non-background colour, reconstruct the AXIS-ALIGNED
        BOUNDING BOX of its cells and draw it as a solid block / a 1-cell-thick
        outline / its four corners.  This completes "3 corners -> rectangle",
        "two opposite corners -> filled rectangle", "broken outline -> closed
        rectangle", etc.  The bounding box is built WITHOUT data-dependent index
        maths: per colour we take row/column presence with ``ReduceMax`` over a
        spatial axis, then a doubling-shift running ``Max`` ("cumulative max")
        from both ends turns the two extreme rows/cols into a filled [rmin,rmax]
        x [cmin,cmax] interval mask; ``Mul`` of the row- and column-interval
        masks is the filled box, ``Min``-erosion gives the interior (outline =
        box - interior) and edge masks give the four corners.  Overlap of two
        colours' boxes is rejected by detection (it would break the one-hot).

  sym_ud / sym_lr / sym_rot180 / sym_4fold / sym_transpose
        MIRROR-COMPLETE a half (or quarter) shape into its full symmetric self by
        OR-ing the grid with its reflection(s).  ``sym_transpose`` (reflection
        across the main diagonal) is origin-safe for ANY square grid via a plain
        ``Transpose``; the row/col/point reflections are realised by reverse
        ``Slice`` on the real HxW region (zero-padded back), so they need the
        relevant dimension(s) to be CONSTANT across every split (enforced).

Detection mirrors the ONNX semantics EXACTLY and only emits a candidate that
reproduces EVERY available pair, so wrong hypotheses are dropped before scoring.
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
_NEG = -(1 << 31)            # full-axis reverse Slice sentinel
_OFFS = [1, 2, 4, 8, 16]     # doubling offsets -> running max covers 31 >= 30


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def cf(self, dims, vals):
        n = self.nm("cf")
        self.inits.append(oh.make_tensor(n, F, list(dims),
                                         [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        return n

    def ci(self, vals):
        n = self.nm("ci")
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


def _nbg():
    return [0.0] + [1.0] * (CHANNELS - 1)


def _onehot(k):
    return [1.0 if c == k else 0.0 for c in range(CHANNELS)]


# --------------------------------------------------------------------------- #
# shifts / cumulative max along a single spatial axis                          #
# --------------------------------------------------------------------------- #
def _shift(g, x, axis, delta, length):
    """Move content by +delta (toward higher index) along `axis`, zero-filled,
    keeping `length` elements anchored at 0.  delta<0 moves toward lower index."""
    pads = [0] * 8
    if delta >= 0:
        pads[axis] = delta                       # pad begin side
        p = g.nd("Pad", [x], mode="constant", value=0.0, pads=pads)
        return g.nd("Slice", [p, g.ci([0]), g.ci([length]), g.ci([axis])])
    d = -delta
    pads[4 + axis] = d                            # pad end side
    p = g.nd("Pad", [x], mode="constant", value=0.0, pads=pads)
    return g.nd("Slice", [p, g.ci([d]), g.ci([d + length]), g.ci([axis])])


def _cummax(g, x, axis, length, down):
    """Running max from one end (down=True -> incorporate lower indices)."""
    cur = x
    for d in _OFFS:
        sh = _shift(g, cur, axis, d if down else -d, length)
        cur = g.nd("Max", [cur, sh])
    return cur


def _interval(g, has, axis, length):
    """has: 0/1 presence along `axis`; returns 1 on the [first,last] interval."""
    a = _cummax(g, has, axis, length, True)       # 1 at/after first present
    b = _cummax(g, has, axis, length, False)      # 1 at/before last present
    return g.nd("Mul", [a, b])


def _erode(g, rng, axis, length):
    """1 strictly inside the interval (drop the two boundary cells)."""
    up = _shift(g, rng, axis, 1, length)
    dn = _shift(g, rng, axis, -1, length)
    return g.nd("Min", [rng, up, dn])


# --------------------------------------------------------------------------- #
# rectangle / bounding-box reconstruction                                      #
# --------------------------------------------------------------------------- #
def build_rect(mode):
    g = _G()
    row_has = g.nd("ReduceMax", ["input"], axes=[3], keepdims=1)    # [1,10,30,1]
    col_has = g.nd("ReduceMax", ["input"], axes=[2], keepdims=1)    # [1,10,1,30]
    row_rng = _interval(g, row_has, 2, H)                           # [1,10,30,1]
    col_rng = _interval(g, col_has, 3, W)                           # [1,10,1,30]

    if mode == "filled":
        shape = g.nd("Mul", [row_rng, col_rng])                    # [1,10,30,30]
    elif mode == "outline":
        filled = g.nd("Mul", [row_rng, col_rng])
        inner = g.nd("Mul", [_erode(g, row_rng, 2, H), _erode(g, col_rng, 3, W)])
        shape = g.nd("Clip", [g.nd("Sub", [filled, inner])], min=0.0, max=1.0)
    else:  # corners
        row_edge = g.nd("Sub", [row_rng, _erode(g, row_rng, 2, H)])
        col_edge = g.nd("Sub", [col_rng, _erode(g, col_rng, 3, W)])
        shape = g.nd("Mul", [row_edge, col_edge])                  # [1,10,30,30]

    nbg = g.cf([1, CHANNELS, 1, 1], _nbg())
    shape_nz = g.nd("Mul", [shape, nbg])                           # drop background channel
    covered = g.nd("ReduceSum", [shape_nz], axes=[1], keepdims=1)  # [1,1,30,30]
    R = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)         # real-cell mask
    bg = g.nd("Clip", [g.nd("Sub", [R, covered])], min=0.0, max=1.0)
    e0 = g.cf([1, CHANNELS, 1, 1], _onehot(0))
    g.nd("Add", [shape_nz, g.nd("Mul", [bg, e0])], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# mirror-completion (symmetry closure)                                         #
# --------------------------------------------------------------------------- #
def _rev(g, t, axis, n):
    return g.nd("Slice", [t, g.ci([n - 1]), g.ci([_NEG]), g.ci([axis]), g.ci([-1])])


def _union_bg(g, reg, variants, padh, padw):
    """Background-safe OR of one-hot tensors: union the non-background channels of
    `variants` (each [1,10,h,w]), then rebuild the background channel from the real
    region of `reg`, finally zero-pad the [1,10,h,w] result back to [1,10,30,30]."""
    nbg = g.cf([1, CHANNELS, 1, 1], _nbg())
    nbs = [g.nd("Mul", [v, nbg]) for v in variants]
    union = nbs[0] if len(nbs) == 1 else g.nd("Max", nbs)        # [1,10,h,w], ch0=0
    covered = g.nd("ReduceSum", [union], axes=[1], keepdims=1)   # [1,1,h,w]
    R = g.nd("ReduceSum", [reg], axes=[1], keepdims=1)           # real-cell mask
    bg = g.nd("Clip", [g.nd("Sub", [R, covered])], min=0.0, max=1.0)
    e0 = g.cf([1, CHANNELS, 1, 1], _onehot(0))
    body = g.nd("Add", [union, g.nd("Mul", [bg, e0])])
    if padh == 0 and padw == 0:
        g.nd("Identity", [body], "output")
    else:
        g.nd("Pad", [body], "output", mode="constant", value=0.0,
             pads=[0, 0, 0, 0, 0, 0, padh, padw])


def build_sym(kind, h, w):
    g = _G()
    if kind == "transpose":
        t = g.nd("Transpose", ["input"], perm=[0, 1, 3, 2])
        _union_bg(g, "input", ["input", t], 0, 0)
        return _model(g)
    if kind == "ud":
        sub = g.nd("Slice", ["input", g.ci([0]), g.ci([h]), g.ci([2])])
        _union_bg(g, sub, [sub, _rev(g, sub, 2, h)], H - h, 0)
        return _model(g)
    if kind == "lr":
        sub = g.nd("Slice", ["input", g.ci([0]), g.ci([w]), g.ci([3])])
        _union_bg(g, sub, [sub, _rev(g, sub, 3, w)], 0, W - w)
        return _model(g)
    # rot180 / 4fold operate on the HxW region
    sub = g.nd("Slice", ["input", g.ci([0, 0]), g.ci([h, w]), g.ci([2, 3])])
    ud = _rev(g, sub, 2, h)
    lr = _rev(g, sub, 3, w)
    bo = _rev(g, lr, 2, h)
    variants = [sub, bo] if kind == "rot180" else [sub, ud, lr, bo]
    _union_bg(g, sub, variants, H - h, W - w)
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX semantics exactly)                         #
# --------------------------------------------------------------------------- #
def _colors(a):
    return sorted(set(a[a != 0].tolist()))


def _bbox_mask(a, c, mode):
    rs, cs = np.where(a == c)
    r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
    m = np.zeros(a.shape, bool)
    if mode == "filled":
        m[r0:r1 + 1, c0:c1 + 1] = True
    elif mode == "outline":
        m[r0, c0:c1 + 1] = True
        m[r1, c0:c1 + 1] = True
        m[r0:r1 + 1, c0] = True
        m[r0:r1 + 1, c1] = True
    else:  # corners
        for rr in (r0, r1):
            for cc in (c0, c1):
                m[rr, cc] = True
    return m


def _ref_rect(a, mode):
    cover = np.zeros(a.shape, int)
    col = np.zeros(a.shape, int)
    for c in _colors(a):
        m = _bbox_mask(a, c, mode)
        cover += m.astype(int)
        col[m] = c
    if (cover > 1).any():               # overlapping boxes break the one-hot
        return None
    out = np.zeros_like(a)
    out[cover == 1] = col[cover == 1]
    return out


def _ref_sym(a, kind):
    if kind == "ud":
        f = a[::-1, :]
    elif kind == "lr":
        f = a[:, ::-1]
    elif kind == "rot180":
        f = a[::-1, ::-1]
    elif kind == "transpose":
        if a.shape[0] != a.shape[1]:
            return None
        f = a.T
    else:                               # 4fold
        out = a.copy()
        for k in ("ud", "lr", "rot180"):
            r = _ref_sym(out, k)
            if r is None:
                return None
            out = r
        return out
    both = (a != 0) & (f != 0)
    if (a[both] != f[both]).any():       # conflicting overlap -> not symmetric
        return None
    out = a.copy()
    add = (out == 0) & (f != 0)
    out[add] = f[add]
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


def _emit(out, seen, name, builder):
    if name in seen:
        return
    try:
        m = builder()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return
    seen.add(name)
    out.append((name, m))


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if any(a.shape != b.shape for a, b in prs):       # completion preserves shape
        return []
    if all(np.array_equal(a, b) for a, b in prs):     # identity -> not our family
        return []
    # additive: every non-background input cell is preserved with its colour
    if not all(np.array_equal(b[a != 0], a[a != 0]) for a, b in prs):
        return []

    out, seen = [], set()

    # ---- bounding-box rectangle reconstruction ----------------------------- #
    for mode in ("filled", "outline", "corners"):
        ok = True
        for a, b in prs:
            r = _ref_rect(a, mode)
            if r is None or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            _emit(out, seen, f"rect_{mode}", lambda mode=mode: build_rect(mode))
            break

    # ---- mirror-completion (symmetry closure) ------------------------------ #
    Hs = {a.shape[0] for a, _ in prs}
    Ws = {a.shape[1] for a, _ in prs}
    sq = all(a.shape[0] == a.shape[1] for a, _ in prs)
    h0 = next(iter(Hs)); w0 = next(iter(Ws))
    sym_specs = [
        ("transpose", sq),                       # any square grid (no const size)
        ("ud", len(Hs) == 1),
        ("lr", len(Ws) == 1),
        ("rot180", len(Hs) == 1 and len(Ws) == 1),
        ("4fold", len(Hs) == 1 and len(Ws) == 1),
    ]
    for kind, allowed in sym_specs:
        if not allowed:
            continue
        ok = True
        for a, b in prs:
            r = _ref_sym(a, kind)
            if r is None or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            _emit(out, seen, f"sym_{kind}",
                  lambda kind=kind: build_sym(kind, h0, w0))

    return out
