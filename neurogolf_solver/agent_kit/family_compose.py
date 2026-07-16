"""Composition + origin-anchored concat/pad family.

Three origin-anchored solver shapes, all yielding (name, onnx_model) from
`candidates(examples)`:

1. Two-step compositions of existing safe ops.  Every op here is origin-safe
   (keeps content at the top-left and is size-independent):
     - transpose  (perm [0,1,3,2])
     - upscale k  (Resize nearest + crop)
     - downscale k(strided Slice + Pad)
     - recolor    (per-color map; Gather if bijective else 1x1 Conv)
   Spatial ops and recolor commute, so `recolor(geom(x)) == geom(recolor(x))`;
   we therefore only need geom-then-recolor plus geom2(geom1(x)).  The frags are
   chained into ONE graph (each frag maps [1,10,30,30] -> [1,10,30,30] keeping the
   origin, so chaining is safe).

2. Origin-anchored concat: output = input stacked with a transformed copy where
   the FIRST block is the original input at the top-left (origin-safe).  Only the
   transpose / identity copy is origin-safe (flips would un-anchor the copy).
   Built as Add(input, translate(T(input))) at fixed offsets -> requires a fixed
   input size across examples (the harness rejects it otherwise).

3. Origin-anchored constant-color border ADDED on the bottom/right only (top/left
   would shift content -> not origin-safe).  Built as Add(input, const) where the
   constant carries the border color in the L-shaped region.  Fixed size only.

All non-generalizing guesses are rejected by the harness (train+test+arc-gen).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64


# --------------------------------------------------------------------------- #
# graph fragments: (in_name, out_name, prefix) -> (nodes, initializers)
# each maps a [1,10,30,30] tensor to a [1,10,30,30] tensor, origin-anchored.
# --------------------------------------------------------------------------- #

def _f_identity(i, o, p):
    return [oh.make_node("Identity", [i], [o])], []


def _f_transpose(i, o, p):
    return [oh.make_node("Transpose", [i], [o], perm=[0, 1, 3, 2])], []


def _f_upscale(k):
    def frag(i, o, p):
        scales = oh.make_tensor(p + "sc", DATA_TYPE, [4], [1.0, 1.0, float(k), float(k)])
        rz = oh.make_node("Resize", [i, p + "sc"], [p + "up"], mode="nearest")
        s = oh.make_tensor(p + "s", INT64, [2], [0, 0])
        e = oh.make_tensor(p + "e", INT64, [2], [HEIGHT, WIDTH])
        a = oh.make_tensor(p + "a", INT64, [2], [2, 3])
        cr = oh.make_node("Slice", [p + "up", p + "s", p + "e", p + "a"], [o])
        return [rz, cr], [scales, s, e, a]
    return frag


def _f_downscale(k):
    def frag(i, o, p):
        sz_h = len(range(0, HEIGHT, k))
        sz_w = len(range(0, WIDTH, k))
        s = oh.make_tensor(p + "s", INT64, [2], [0, 0])
        e = oh.make_tensor(p + "e", INT64, [2], [HEIGHT, WIDTH])
        a = oh.make_tensor(p + "a", INT64, [2], [2, 3])
        st = oh.make_tensor(p + "st", INT64, [2], [k, k])
        sl = oh.make_node("Slice", [i, p + "s", p + "e", p + "a", p + "st"], [p + "sm"])
        pad = oh.make_node("Pad", [p + "sm"], [o], mode="constant", value=0.0,
                           pads=[0, 0, 0, 0, 0, 0, HEIGHT - sz_h, WIDTH - sz_w])
        return [sl, pad], [s, e, a, st]
    return frag


def _is_bijection(cmap):
    return sorted(cmap) == list(range(CHANNELS))


def _f_recolor(cmap):
    def frag(i, o, p):
        if _is_bijection(cmap):
            # Gather: output[:, j] = input[:, src[j]]; src is the inverse map.
            src = [0] * CHANNELS
            for inp, outp in enumerate(cmap):
                src[outp] = inp
            idx = oh.make_tensor(p + "idx", INT64, [CHANNELS], src)
            return [oh.make_node("Gather", [i, p + "idx"], [o], axis=1)], [idx]
        weights = [0.0] * (CHANNELS * CHANNELS)  # [O, I, 1, 1]
        for inp, outp in enumerate(cmap):
            weights[outp * CHANNELS + inp] = 1.0
        w = oh.make_tensor(p + "W", DATA_TYPE, [CHANNELS, CHANNELS, 1, 1], weights)
        return [oh.make_node("Conv", [i, p + "W"], [o], kernel_shape=[1, 1],
                             pads=[0, 0, 0, 0])], [w]
    return frag


def _f_translate(dy, dx):
    """Shift content down/right by (dy, dx) with zero fill (dy, dx >= 0)."""
    def frag(i, o, p):
        pad = oh.make_node("Pad", [i], [p + "pd"], mode="constant", value=0.0,
                           pads=[0, 0, dy, dx, 0, 0, 0, 0])
        s = oh.make_tensor(p + "s", INT64, [2], [0, 0])
        e = oh.make_tensor(p + "e", INT64, [2], [HEIGHT, WIDTH])
        a = oh.make_tensor(p + "a", INT64, [2], [2, 3])
        cr = oh.make_node("Slice", [p + "pd", p + "s", p + "e", p + "a"], [o])
        return [pad, cr], [s, e, a]
    return frag


def _build_chain(frags):
    """Chain frags input -> ... -> output."""
    nodes, inits, cur = [], [], "input"
    n = len(frags)
    for idx, fr in enumerate(frags):
        nxt = "output" if idx == n - 1 else f"t{idx}"
        nn, ii = fr(cur, nxt, f"p{idx}_")
        nodes += nn
        inits += ii
        cur = nxt
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# numpy reference transforms (operate on raw, variable-size grids)
# --------------------------------------------------------------------------- #

def _np_upscale(k):
    return lambda a: np.kron(a, np.ones((k, k), dtype=a.dtype))


def _np_downscale(k):
    return lambda a: a[::k, ::k]


# (name, np_fn, frag) for every origin-safe spatial op (incl. identity).
GEOMS = [
    ("id", lambda a: a, _f_identity),
    ("T", lambda a: a.T, _f_transpose),
]
for _k in range(2, 6):
    GEOMS.append((f"up{_k}", _np_upscale(_k), _f_upscale(_k)))
    GEOMS.append((f"dn{_k}", _np_downscale(_k), _f_downscale(_k)))


def _pairs(ex):
    return [(np.array(e["input"]), np.array(e["output"]))
            for e in ex.get("train", []) + ex.get("test", [])]


def _infer_map(mids):
    """Consistent cellwise color map over (X, Y) pairs of equal shape."""
    m = {}
    for X, Y in mids:
        if X.shape != Y.shape:
            return None
        for xv, yv in zip(X.ravel().tolist(), Y.ravel().tolist()):
            xv, yv = int(xv), int(yv)
            if xv in m:
                if m[xv] != yv:
                    return None
            else:
                m[xv] = yv
    return [m.get(i, i) for i in range(CHANNELS)]


def _identity_map(cmap):
    return cmap == list(range(CHANNELS))


# --------------------------------------------------------------------------- #
# concat (origin-anchored): input stacked with id/transpose copy
# --------------------------------------------------------------------------- #

_CONCAT_TFS = [("id", lambda a: a, _f_identity),
               ("T", lambda a: a.T, _f_transpose)]


def _detect_concat(prs):
    out = []
    shapes = {a.shape for a, _ in prs}
    if len(shapes) != 1:
        return out  # need a fixed input size for a static build
    H, W = next(iter(shapes))
    for tname, tf, tfrag in _CONCAT_TFS:
        for axis in (0, 1):  # 0 = stacked below, 1 = stacked to the right
            ok = True
            for a, b in prs:
                t = tf(a)
                try:
                    cat = np.concatenate([a, t], axis=axis)
                except ValueError:
                    ok = False
                    break
                if cat.shape != b.shape or not np.array_equal(cat, b):
                    ok = False
                    break
            if not ok:
                continue
            # transformed block dims to size the shift
            tb = tf(prs[0][0])
            if axis == 0:
                dy, dx = H, 0
                if H + tb.shape[0] > HEIGHT or W > WIDTH:
                    continue
            else:
                dy, dx = 0, W
                if H > HEIGHT or W + tb.shape[1] > WIDTH:
                    continue
            nodes, inits = [], []
            nn, ii = tfrag("input", "tt", "tf_")
            nodes += nn
            inits += ii
            nn, ii = _f_translate(dy, dx)("tt", "shifted", "tr_")
            nodes += nn
            inits += ii
            nodes.append(oh.make_node("Add", ["input", "shifted"], ["output"]))
            d = "below" if axis == 0 else "right"
            out.append((f"concat_{tname}_{d}", _model(nodes, inits)))
    return out


# --------------------------------------------------------------------------- #
# bottom/right constant-color border (origin-anchored)
# --------------------------------------------------------------------------- #

def _detect_border(prs):
    params = set()
    for a, b in prs:
        H, W = a.shape
        bH, bW = b.shape
        pb, pr = bH - H, bW - W
        if pb < 0 or pr < 0 or (pb == 0 and pr == 0):
            return None
        if not np.array_equal(b[:H, :W], a):
            return None
        regions = []
        if pb > 0:
            regions.append(b[H:H + pb, 0:W + pr].ravel())
        if pr > 0:
            regions.append(b[0:H, W:W + pr].ravel())
        vals = np.unique(np.concatenate(regions))
        if vals.size != 1:
            return None
        params.add((H, W, pb, pr, int(vals[0])))
    if len(params) != 1:
        return None
    H, W, pb, pr, c = next(iter(params))
    if H + pb > HEIGHT or W + pr > WIDTH:
        return None
    return (H, W, pb, pr, c)


def _build_border(H, W, pb, pr, c):
    const = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    if pb > 0:
        const[0, c, H:H + pb, 0:W + pr] = 1.0
    if pr > 0:
        const[0, c, 0:H, W:W + pr] = 1.0
    t = oh.make_tensor("BC", DATA_TYPE, list(GRID_SHAPE), const.ravel().tolist())
    cn = oh.make_node("Constant", [], ["bc"], value=t)
    add = oh.make_node("Add", ["input", "bc"], ["output"])
    return _model([cn, add])


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    # (1a) geom-then-recolor (covers pure recolor via id, and pure geom via id-map)
    for gname, gf, gfrag in GEOMS:
        mids, ok = [], True
        for a, b in prs:
            try:
                m = gf(a)
            except Exception:
                ok = False
                break
            if m.shape != b.shape:
                ok = False
                break
            mids.append((m, b))
        if not ok:
            continue
        cmap = _infer_map(mids)
        if cmap is None:
            continue
        ident_geom = gname == "id"
        ident_cmap = _identity_map(cmap)
        if ident_geom and ident_cmap:
            continue  # pure identity -> not this family
        frags = []
        if not ident_geom:
            frags.append(gfrag)
        if not ident_cmap:
            frags.append(_f_recolor(cmap))
        if not frags:
            continue
        name = (gname if not ident_geom else "") + ("_recolor" if not ident_cmap else "")
        out.append((name.strip("_"), _build_chain(frags)))

    # (1b) geom2(geom1(x)) for two distinct spatial ops (genuine 2-step)
    for n1, f1, fr1 in GEOMS:
        if n1 == "id":
            continue
        for n2, f2, fr2 in GEOMS:
            if n2 == "id":
                continue
            ok = True
            for a, b in prs:
                try:
                    m = f2(f1(a))
                except Exception:
                    ok = False
                    break
                if m.shape != b.shape or not np.array_equal(m, b):
                    ok = False
                    break
            if ok:
                out.append((f"{n1}_then_{n2}", _build_chain([fr1, fr2])))

    # (2) origin-anchored concat
    out += _detect_concat(prs)

    # (3) bottom/right constant-color border
    bd = _detect_border(prs)
    if bd is not None:
        out.append(("border_br", _build_border(*bd)))

    # de-dup by name (keep first)
    seen, uniq = set(), []
    for name, model in out:
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, model))
    return uniq
