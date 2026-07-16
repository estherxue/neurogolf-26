"""Two-step compositions of origin-safe primitives, chained end-to-end.

Family members (recolor is a per-color map; the spatial op keeps content anchored
at the top-left and is grid-size independent):

    recolor o upscale_k     upscale_k o recolor
    recolor o downscale_k   downscale_k o recolor
    recolor o transpose     transpose o recolor

recolor commutes with every origin-safe spatial op here (recolor is a per-pixel
color relabel; upscale = block replicate, downscale = block subsample, transpose =
axis swap all merely move/copy whole pixels).  So `recolor o S == S o recolor` and
both orders are realised by the SAME cheapest graph.  The key cost trick: place the
recolor (and the Resize) on the SMALLEST tensor in the chain --

  * upscale_k:  Slice the input to the ceil(30/k) x ceil(30/k) top-left region that
    actually feeds the 30x30 output, recolor THAT, then Resize(k).  This avoids the
    huge [1,10,30k,30k] Resize intermediate that a naive recolor->upscale produces.
  * downscale_k: strided Slice -> recolor the small tensor -> Pad back to 30x30.

All three spatial primitives are origin-anchored (verified by builders.py); the
recolor is a no-bias linear channel op, so zero-padding cells stay all-zero (one-hot
padding is all-zero, not channel-0) -> padding-safe.

Detection infers the spatial op (transpose / upscale_k / downscale_k) and a single
consistent per-color map from ALL available pairs (train+test+arc-gen).  Every
candidate is re-validated for EXACT equality on train+test+arc-gen (and the private
set) by the harness, so coincidental stride/shape matches are rejected.
"""
from __future__ import annotations

import math

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64


# --------------------------------------------------------------------------- #
# tiny node / initializer accumulator with auto-unique tensor names
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def init(self, dtype, dims, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, dtype, list(dims), list(vals)))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _is_bijection(cmap):
    return sorted(cmap) == list(range(CHANNELS))


def _recolor(g, src, cmap, out=None):
    """Append a per-color recolor (Gather if bijective else 1x1 Conv) on `src`."""
    if _is_bijection(cmap):
        inv = [0] * CHANNELS              # output[:, j] = input[:, inv[j]]
        for i, o in enumerate(cmap):
            inv[o] = i
        idx = g.init(INT64, [CHANNELS], inv)
        return g.node("Gather", [src, idx], out, axis=1)
    weights = [0.0] * (CHANNELS * CHANNELS)   # [O, I, 1, 1]
    for i, o in enumerate(cmap):
        weights[o * CHANNELS + i] = 1.0
    w = g.init(DATA_TYPE, [CHANNELS, CHANNELS, 1, 1], weights)
    return g.node("Conv", [src, w], out, kernel_shape=[1, 1], pads=[0, 0, 0, 0])


def _identity_map(cmap):
    return cmap == list(range(CHANNELS))


# --------------------------------------------------------------------------- #
# graph builders (each returns a full ModelProto, lowest cost for the combo)
# --------------------------------------------------------------------------- #
def _build_transpose(cmap):
    g = _G()
    if _identity_map(cmap):
        g.node("Transpose", ["input"], "output", perm=[0, 1, 3, 2])
    else:
        t = g.node("Transpose", ["input"], perm=[0, 1, 3, 2])
        _recolor(g, t, cmap, "output")
    return _model(g.nodes, g.inits)


def _build_upscale(k, cmap):
    g = _G()
    nh = math.ceil(HEIGHT / k)
    nw = math.ceil(WIDTH / k)
    s0 = g.init(INT64, [2], [0, 0])
    se = g.init(INT64, [2], [nh, nw])
    sa = g.init(INT64, [2], [2, 3])
    s = g.node("Slice", ["input", s0, se, sa])          # [1,10,nh,nw] (small)
    mid = s if _identity_map(cmap) else _recolor(g, s, cmap)
    sc = g.init(DATA_TYPE, [4], [1.0, 1.0, float(k), float(k)])
    if nh * k == HEIGHT and nw * k == WIDTH:
        g.node("Resize", [mid, sc], "output", mode="nearest")
    else:
        u = g.node("Resize", [mid, sc], mode="nearest")  # [1,10,nh*k,nw*k]
        c0 = g.init(INT64, [2], [0, 0])
        ce = g.init(INT64, [2], [HEIGHT, WIDTH])
        ca = g.init(INT64, [2], [2, 3])
        g.node("Slice", [u, c0, ce, ca], "output")
    return _model(g.nodes, g.inits)


def _build_downscale(k, cmap):
    g = _G()
    sz_h = len(range(0, HEIGHT, k))
    sz_w = len(range(0, WIDTH, k))
    s0 = g.init(INT64, [2], [0, 0])
    se = g.init(INT64, [2], [HEIGHT, WIDTH])
    sa = g.init(INT64, [2], [2, 3])
    st = g.init(INT64, [2], [k, k])
    small = g.node("Slice", ["input", s0, se, sa, st])   # [1,10,sz_h,sz_w]
    mid = small if _identity_map(cmap) else _recolor(g, small, cmap)
    g.node("Pad", [mid], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - sz_h, WIDTH - sz_w])
    return _model(g.nodes, g.inits)


def _build_recolor(cmap):
    g = _G()
    _recolor(g, "input", cmap, "output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"])
            b = np.array(e["output"])
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > HEIGHT or max(b.shape) > HEIGHT:
                continue
            out.append((a, b))
    return out


def _consistent_map(grids):
    """Single per-color map over (X, Y) pairs of equal shape, or None."""
    m = {}
    for X, Y in grids:
        if X.shape != Y.shape:
            return None
        for xv, yv in zip(X.ravel().tolist(), Y.ravel().tolist()):
            xv, yv = int(xv), int(yv)
            if not (0 <= xv < CHANNELS) or not (0 <= yv < CHANNELS):
                return None
            if xv in m:
                if m[xv] != yv:
                    return None
            else:
                m[xv] = yv
    return [m.get(i, i) for i in range(CHANNELS)]


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    # Pure identity is not this family.
    if all(a.shape == b.shape and np.array_equal(a, b) for a, b in prs):
        return []

    out = []

    # ---- transpose o recolor (covers pure transpose when map is identity) ----
    if all(b.shape == (a.shape[1], a.shape[0]) for a, b in prs):
        cmap = _consistent_map([(a.T, b) for a, b in prs])
        if cmap is not None:
            # skip if it collapses to identity (symmetric grids + id map)
            if not all(np.array_equal(b, a) for a, b in prs):
                nm = "recolor_transpose" if not _identity_map(cmap) else "transpose"
                out.append((nm, _build_transpose(cmap)))

    # ---- upscale_k o recolor (covers pure upscale when map is identity) ------
    for k in range(2, 6):
        if not all(b.shape == (a.shape[0] * k, a.shape[1] * k) for a, b in prs):
            continue
        grids, ok = [], True
        for a, b in prs:
            rep = b[::k, ::k]
            if not np.array_equal(b, np.kron(rep, np.ones((k, k), dtype=int))):
                ok = False
                break
            grids.append((a, rep))
        if not ok:
            continue
        cmap = _consistent_map(grids)
        if cmap is None:
            continue
        nm = (f"recolor_upscale{k}" if not _identity_map(cmap) else f"upscale{k}")
        out.append((nm, _build_upscale(k, cmap)))
        break

    # ---- downscale_k o recolor (covers pure downscale when map is identity) --
    for k in range(2, 6):
        if not all(a.shape[0] >= 1 and a.shape[1] >= 1
                   and b.shape == a[::k, ::k].shape for a, b in prs):
            continue
        # must genuinely subsample (some pair larger than its k-subsample)
        if all(a.shape == a[::k, ::k].shape for a, b in prs):
            continue
        cmap = _consistent_map([(a[::k, ::k], b) for a, b in prs])
        if cmap is None:
            continue
        nm = (f"recolor_downscale{k}" if not _identity_map(cmap) else f"downscale{k}")
        out.append((nm, _build_downscale(k, cmap)))
        break

    # ---- pure recolor (recolor o identity) -----------------------------------
    if all(a.shape == b.shape for a, b in prs):
        cmap = _consistent_map(prs)
        if cmap is not None and not _identity_map(cmap):
            out.append(("recolor", _build_recolor(cmap)))

    # de-dup by name (keep first)
    seen, uniq = set(), []
    for name, model in out:
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, model))
    return uniq
