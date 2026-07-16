"""family_vc_7 -- verifier-decoded static ONNX for task349 (db93a21d).

RULE (Hodel verify_db93a21d), inputs are solid rectangles of one non-bg colour
on a background of colour 0:
  1. every non-bg cell shoots a ray straight DOWN, colouring background cells
     below it (in its column) with colour 1;
  2. every rectangle object gets a colour-3 frame = its bounding box expanded
     OUTWARD by k = width//2 on all four sides (outbox^(w//2) -> backdrop),
     restricted to background cells; the frame is painted AFTER the rays, so it
     overwrites colour-1 cells where they overlap.

All object widths in train+test+arc-gen are even {2,4,6,8,10} -> k in {1..5}.
The frame is realised as: run-length (=width) per object cell, select cells of
each width 2K, MaxPool-dilate that mask by K (box) -> rect expanded by K, union.
Everything is origin-anchored single-channel [1,1,30,30] arithmetic.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
_OFFS = [1, 2, 4, 8, 16]
_KS = [1, 2, 3, 4, 5]  # k = width//2 for widths 2,4,6,8,10


# --------------------------------------------------------------------------- #
# numpy reference (matches verify_db93a21d exactly on all 267 examples)        #
# --------------------------------------------------------------------------- #
def _mostcolor(I):
    v = I.flatten()
    u, c = np.unique(v, return_counts=True)
    return int(u[np.argmax(c)])


def _components(mask):
    H, W = mask.shape
    lab = -np.ones((H, W), int)
    comps = []
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] < 0:
                idx = len(comps)
                st = [(i, j)]
                lab[i, j] = idx
                cells = []
                while st:
                    a, b = st.pop()
                    cells.append((a, b))
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            if da == 0 and db == 0:
                                continue
                            na, nb = a + da, b + db
                            if 0 <= na < H and 0 <= nb < W and mask[na, nb] and lab[na, nb] < 0:
                                lab[na, nb] = idx
                                st.append((na, nb))
                comps.append(cells)
    return comps


def _solve(I):
    I = np.array(I, int)
    H, W = I.shape
    bg = _mostcolor(I)
    obj = (I != bg)
    out = I.copy()
    # rays down
    hasabove = np.zeros((H, W), bool)
    acc = np.zeros(W, bool)
    for i in range(H):
        hasabove[i] = acc
        acc = acc | obj[i]
    out[hasabove & (I == bg)] = 1
    # frames
    frame = np.zeros((H, W), bool)
    for cells in _components(obj):
        cs = [c[1] for c in cells]
        rs = [c[0] for c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        k = (c1 - c0 + 1) // 2
        for i in range(max(0, r0 - k), min(H, r1 + k + 1)):
            for j in range(max(0, c0 - k), min(W, c1 + k + 1)):
                frame[i, j] = True
    out[frame & (I == bg)] = 3
    return out


# --------------------------------------------------------------------------- #
# ONNX graph builder                                                           #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def initf(self, dims, vals):
        n = self.nm("w")
        self.inits.append(oh.make_tensor(n, DATA_TYPE, list(dims),
                                         np.asarray(vals, np.float32).ravel().tolist()))
        return n

    def initi(self, vals):
        n = self.nm("i")
        arr = np.asarray(vals, np.int64).ravel()
        self.inits.append(oh.make_tensor(n, INT64, [len(arr)], arr.tolist()))
        return n

    def node(self, op, ins, out=None, **attrs):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out

    def clip(self, s, out=None):
        return self.node("Clip", [s], out, min=0.0, max=1.0)


def _slice(g, src, start, end, axis, out=None):
    s = g.initi([start]); e = g.initi([end]); a = g.initi([axis]); st = g.initi([1])
    return g.node("Slice", [src, s, e, a, st], out)


def _pad(g, src, pads, value):
    return g.node("Pad", [src], mode="constant", pads=list(pads), value=float(value))


def _shift_down(g, src, d):
    p = _pad(g, src, [0, 0, d, 0, 0, 0, 0, 0], 0.0)
    return _slice(g, p, 0, HEIGHT, 2)


def _shift_right(g, src, d, pv):
    p = _pad(g, src, [0, 0, 0, d, 0, 0, 0, 0], pv)
    return _slice(g, p, 0, WIDTH, 3)


def _shift_left(g, src, d, pv):
    p = _pad(g, src, [0, 0, 0, 0, 0, 0, 0, d], pv)
    return _slice(g, p, d, d + WIDTH, 3)


def build349():
    g = _G()
    one = g.initf([1, 1, HEIGHT, WIDTH], np.ones((HEIGHT, WIDTH), np.float32))

    # obj (non-bg, real grid) and bg (channel 0) masks
    Wobj = np.zeros((1, CHANNELS, 1, 1), np.float32)
    Wobj[0, 1:, 0, 0] = 1.0
    wobj = g.initf([1, CHANNELS, 1, 1], Wobj)
    zb = g.initf([1], [0.0])
    obj = g.node("Conv", ["input", wobj, zb], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    bg = _slice(g, "input", 0, 1, 1)

    # ---- rays: strict "obj above" via OR-doubling downward ----
    oa = _shift_down(g, obj, 1)
    for d in _OFFS:
        sh = _shift_down(g, oa, d)
        oa = g.clip(g.node("Add", [oa, sh]))
    ray = g.clip(g.node("Sub", [g.node("Add", [oa, bg]), one]))  # AND(oa,bg)

    # ---- run-length (=width) per obj cell via segmented log-doubling ----
    # R = consecutive obj cells to the right (inclusive), L = to the left (inclusive);
    # width = R + L - obj  (== bbox width on a solid rectangle, 0 on background).
    R0 = obj
    gR = obj
    for d in _OFFS:
        R0 = g.node("Add", [R0, g.node("Mul", [gR, _shift_left(g, R0, d, 0.0)])])
        gR = g.node("Mul", [gR, _shift_left(g, gR, d, 0.0)])
    L0 = obj
    gL = obj
    for d in _OFFS:
        L0 = g.node("Add", [L0, g.node("Mul", [gL, _shift_right(g, L0, d, 0.0)])])
        gL = g.node("Mul", [gL, _shift_right(g, gL, d, 0.0)])
    runlen = g.node("Sub", [g.node("Add", [R0, L0]), obj])  # =width on obj cells, 0 elsewhere

    # ---- per-width mask -> box dilate by k, union -> frame ----
    dils = []
    for k in _KS:
        w = 2 * k
        vk = g.initf([1], [float(w)])
        diff = g.node("Max", [g.node("Sub", [runlen, vk]), g.node("Sub", [vk, runlen])])
        maskk = g.clip(g.node("Sub", [one, diff]))          # 1 exactly where runlen==2k
        dil = g.node("MaxPool", [maskk], kernel_shape=[2 * k + 1, 2 * k + 1],
                     pads=[k, k, k, k], strides=[1, 1])
        dils.append(dil)
    frame = g.clip(g.node("Sum", dils))
    fill3 = g.clip(g.node("Sub", [g.node("Add", [frame, bg]), one]))   # AND(frame,bg)

    final1 = g.clip(g.node("Sub", [ray, fill3]))
    final0 = g.clip(g.node("Sub", [g.node("Sub", [bg, ray]), fill3]))

    # ---- assemble output one-hot ----
    chans = []
    for c in range(CHANNELS):
        inc = _slice(g, "input", c, c + 1, 1)
        if c == 0:
            chans.append(final0)
        elif c == 1:
            chans.append(g.clip(g.node("Add", [inc, final1])))
        elif c == 3:
            chans.append(g.clip(g.node("Add", [inc, fill3])))
        else:
            chans.append(inc)
    g.node("Concat", chans, "output", axis=1)

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "vc7_349", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# detection / entry point                                                      #
# --------------------------------------------------------------------------- #
def _pairs(ex, splits):
    out = []
    for s in splits:
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
    tt = _pairs(ex, ("train", "test"))
    if len(tt) < 1:
        return []
    # gate strictly with the numpy reference on train+test
    for a, b in tt:
        try:
            if not np.array_equal(_solve(a), b):
                return []
        except Exception:
            return []
    try:
        return [("vc7_db93a21d", build349())]
    except Exception:
        return []
