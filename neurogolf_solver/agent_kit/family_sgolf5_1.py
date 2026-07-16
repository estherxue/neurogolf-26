"""family_sgolf5_1 -- ultra-cheap EXACT solvers for "stamp a fixed template on
every marker" tasks, expressed as ONE origin-anchored Conv that writes straight
into the FREE `output` tensor (zero intermediate memory -> params-only cost).

Rule (marker-stamp): the grid holds a small fixed multi-colour template plus a
number of single-cell markers of one colour `m`.  Each marker is replaced by a
copy of the template, aligned so the template's anchor cell lands on the marker
(the original template / legend and all other cells are preserved).

Everything is a translation-equivariant local rewrite, so it is exactly a Conv
with SAME zero-padding (pads=[R,R,R,R]); padding cells stay all-zero, so the
top-left anchoring contract is honoured for ANY grid size.  We fold four things
into a single [10,10,K,K] kernel:
  * identity            W[c,c,0,0]+=1              (keep the input one-hot)
  * marker erase        W[m,m,0,0] :=0             (drop the marker colour)
  * stamp paint         W[v,m,dr,dc]+=1            (write template colour v)
  * background clear     W[0,m,dr,dc]-=1 (v!=0)     (un-set bg where a colour lands)
The grader only checks sign (output>0), so no Clip is needed: cleared cells land
at 0 (off) and painted cells at >=1 (on).  The single Conv output IS `output`,
so the model has NO intermediate tensors -> cost = params only.

Detection infers `m` and the template offset map from train+test+arc-gen and
rebuilds every pair in numpy first; the harness then re-checks EXACTNESS.  No
cropping, no work-area shrink, no dtype change: the canvas is 30x30 end to end.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                return None
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _infer_offmap(prs, m):
    """Infer a single template offset->colour map for marker colour m, or None.

    Offsets are measured from a marker cell; every marker in every pair must be
    reproduced by the SAME map, and no other cell may change."""
    offmap = None
    for a, b in prs:
        mk = list(zip(*np.where(a == m)))
        if not mk:
            return None
        diffs = [(r, c, int(b[r, c])) for r, c in zip(*np.where(a != b))]
        # assign each diff cell to its nearest marker (Chebyshev)
        local = {}
        for r, c, v in diffs:
            p = min(mk, key=lambda q: max(abs(q[0] - r), abs(q[1] - c)))
            local.setdefault(p, {})[(r - p[0], c - p[1])] = v
        # every marker must carry an identical stamp
        for p in mk:
            om = local.get(p, {})
            # marker cell itself must be rewritten (offset (0,0) present)
            if (0, 0) not in om:
                return None
            if offmap is None:
                offmap = om
            elif om != offmap:
                return None
    if not offmap:
        return None
    # template must not repaint with the marker colour (keeps kernel unambiguous)
    if any(v == m for v in offmap.values()):
        return None
    return offmap


def _apply(a, m, offmap):
    b = a.copy()
    for r, c in zip(*np.where(a == m)):
        for (dr, dc), v in offmap.items():
            rr, cc = r + dr, c + dc
            if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]:
                b[rr, cc] = v
            elif v != 0:
                return None  # stamp would spill past the real grid edge
    return b


def _build(m, offmap):
    R = max(max(abs(dr), abs(dc)) for dr, dc in offmap)
    K = 2 * R + 1
    W = np.zeros((CHANNELS, CHANNELS, K, K), np.float32)
    for c in range(CHANNELS):
        W[c, c, R, R] = 1.0            # identity: keep the input one-hot
    W[m, m, R, R] = 0.0                # erase the marker colour
    for (dr, dc), v in offmap.items():
        W[v, m, R + dr, R + dc] += 1.0     # paint template colour v
        if v != 0:
            W[0, m, R + dr, R + dc] += -1.0  # clear background where a colour lands
    wt = oh.make_tensor("W", DATA_TYPE, [CHANNELS, CHANNELS, K, K], W.ravel().tolist())
    node = oh.make_node("Conv", ["input", "W"], ["output"],
                        kernel_shape=[K, K], pads=[R, R, R, R])
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph([node], "g", [x], [y], [wt])
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs or len(prs) < 1:
        return []
    if not all(a.shape == b.shape for a, b in prs):
        return []
    if not any((a != b).any() for a, b in prs):
        return []

    colors = sorted({int(v) for a, _ in prs for v in np.unique(a)} - {0})
    out = []
    for m in colors:
        # marker colour must vanish from every output
        if any((b == m).any() for _, b in prs):
            continue
        try:
            offmap = _infer_offmap(prs, m)
        except Exception:
            offmap = None
        if offmap is None:
            continue
        ok = True
        for a, b in prs:
            pred = _apply(a, m, offmap)
            if pred is None or not np.array_equal(pred, b):
                ok = False
                break
        if not ok:
            continue
        try:
            out.append((f"stamp_m{m}", _build(m, offmap)))
        except Exception:
            pass
        break
    return out
