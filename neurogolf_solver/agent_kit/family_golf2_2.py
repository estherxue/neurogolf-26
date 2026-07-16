"""family_golf2_2 -- cheaper exact solvers for a slice of golf targets.

Each candidate re-derives the task rule from train+test+arc-gen pairs, verifies
EXACT equality against a numpy reference on every available pair, and only then
emits a minimal opset-10 ONNX graph.  The integrator keeps whichever correct
solver is cheapest (cost = params + intermediate-tensor bytes; input/output are
free), so these only need to be exact and cheaper than the incumbent.

Cost levers used here:
  * single-channel [1,1,30,30] masks instead of [1,10,30,30] intermediates
  * MatMul with [30,30] triangular matrices for cumulative "fill-between"
    (no log-step shift unrolls)
  * collapse to [1,10,1,1] per-colour statistics as early as possible
  * write the final grid straight to the FREE `output` tensor via Concat/Pad
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL
_NEG = -(1 << 31)


class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def iconst(self, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)], list(vals)))
        return nm

    def fconst(self, vals, shape):
        nm = self.name("f")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(shape),
                                         [float(v) for v in vals]))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _slice(g, src, starts, ends, axes, steps=None):
    ins = [src, g.iconst(starts), g.iconst(ends), g.iconst(axes)]
    if steps is not None:
        ins.append(g.iconst(steps))
    return g.node("Slice", ins)


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
# 346  fragswatch_1x1 : 1x1 output = colour with FEWEST 4-adjacent same pairs
# ===========================================================================
def _ref_fragswatch(a):
    cols = sorted(set(np.unique(a).tolist()) - {0})
    if len(cols) < 1:
        return None

    def edges(k):
        m = (a == k).astype(int)
        return int((m[:, :-1] * m[:, 1:]).sum() + (m[:-1, :] * m[1:, :]).sum())

    best = min(cols, key=lambda k: (edges(k), k))
    return np.array([[best]], int)


def _build_fragswatch():
    g = _G()
    # horizontal adjacency products
    sw0 = _slice(g, "input", [0], [WIDTH - 1], [3])      # [1,10,30,29]
    sw1 = _slice(g, "input", [1], [WIDTH], [3])
    rp = g.node("Mul", [sw0, sw1])
    rs = g.node("ReduceSum", [rp], axes=[2, 3], keepdims=1)  # [1,10,1,1]
    # vertical adjacency products
    sh0 = _slice(g, "input", [0], [HEIGHT - 1], [2])     # [1,10,29,30]
    sh1 = _slice(g, "input", [1], [HEIGHT], [2])
    dp = g.node("Mul", [sh0, sh1])
    ds = g.node("ReduceSum", [dp], axes=[2, 3], keepdims=1)
    edges = g.node("Add", [rs, ds])                       # [1,10,1,1]
    # penalise absent colours and the background channel
    presence = g.node("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    clip = g.node("Clip", [presence], min=0.0, max=1.0)
    one = g.fconst([1.0], [1])
    big = g.fconst([1e6], [1])
    absent = g.node("Sub", [one, clip])
    pen = g.node("Mul", [absent, big])
    bgpen = g.fconst([1e6] + [0.0] * (CHANNELS - 1), [1, CHANNELS, 1, 1])
    score = g.node("Add", [g.node("Add", [edges, pen]), bgpen])  # [1,10,1,1]
    mn = g.node("ReduceMin", [score], axes=[1], keepdims=1)      # [1,1,1,1]
    half = g.fconst([0.5], [1])
    thr = g.node("Add", [mn, half])
    selb = g.node("Less", [score, thr])
    sel = g.node("Cast", [selb], to=DATA_TYPE)                   # [1,10,1,1]
    g.node("Pad", [sel], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - 1, WIDTH - 1])
    return _model(g.nodes, g.inits)


# ===========================================================================
# 41  connect_RLUD : per row, fill between leftmost/rightmost same colour
#                    (each row carries a single colour)
# ===========================================================================
def _ref_fillrows(a):
    o = a.copy()
    H, W = a.shape
    for r in range(H):
        for c in range(1, 10):
            idx = np.where(a[r] == c)[0]
            if len(idx) >= 2:
                o[r, idx[0]:idx[-1] + 1] = c
    return o


def _build_fillrows():
    g = _G()
    inC = _slice(g, "input", [1], [CHANNELS], [1])                # [1,9,30,30]
    UT = g.fconst([1.0 if i <= j else 0.0 for i in range(WIDTH) for j in range(WIDTH)],
                  [WIDTH, WIDTH])   # UT[i,j]=1 if i<=j  -> leftcount
    LT = g.fconst([1.0 if i >= j else 0.0 for i in range(WIDTH) for j in range(WIDTH)],
                  [WIDTH, WIDTH])   # rightcount
    L = g.node("MatMul", [inC, UT])                               # [1,9,30,30]
    R = g.node("MatMul", [inC, LT])
    mn = g.node("Min", [L, R])
    half = g.fconst([0.5], [1])
    between = g.node("Cast", [g.node("Greater", [mn, half])], to=DATA_TYPE)  # [1,9,30,30]
    ingrid = g.node("ReduceMax", ["input"], axes=[1], keepdims=1)  # [1,1,30,30]
    anycolor = g.node("ReduceSum", [between], axes=[1], keepdims=1)  # [1,1,30,30]
    bg = g.node("Sub", [ingrid, anycolor])                        # [1,1,30,30]
    g.node("Concat", [bg, between], "output", axis=1)
    return _model(g.nodes, g.inits)


# ===========================================================================
# 293  crk2_5_swap : full-row bar crosses full-col bar; swap colours at the
#                    intersection cells (each becomes the other bar's colour)
# ===========================================================================
def _ref_swap(a):
    H, W = a.shape
    o = a.copy()
    fr = [r for r in range(H) if np.all(a[r] != 0)]
    fc = [c for c in range(W) if np.all(a[:, c] != 0)]
    for r in fr:
        rowc = [a[r, x] for x in range(W) if x not in fc]
        hc = max(set(rowc), key=rowc.count) if rowc else None
        for c in fc:
            colc = [a[y, c] for y in range(H) if y not in fr]
            vc = max(set(colc), key=colc.count) if colc else None
            if hc is None or vc is None:
                return None
            o[r, c] = vc if a[r, c] == hc else hc
    return o


def _build_swap():
    g = _G()
    rowsum = g.node("ReduceSum", ["input"], axes=[3], keepdims=1)   # [1,10,H,1]
    colsum = g.node("ReduceSum", ["input"], axes=[2], keepdims=1)   # [1,10,1,W]
    half = g.fconst([0.5], [1])
    rowmax = g.node("ReduceMax", [rowsum], axes=[1], keepdims=1)    # [1,1,H,1]
    colmax = g.node("ReduceMax", [colsum], axes=[1], keepdims=1)    # [1,1,1,W]
    hcolor = g.node("Cast", [g.node("Greater", [rowsum, g.node("Sub", [rowmax, half])])],
                    to=DATA_TYPE)                                    # [1,10,H,1]
    vcolor = g.node("Cast", [g.node("Greater", [colsum, g.node("Sub", [colmax, half])])],
                    to=DATA_TYPE)                                    # [1,10,1,W]
    rowbg = _slice(g, rowsum, [0], [1], [1])                        # [1,1,H,1]
    colbg = _slice(g, colsum, [0], [1], [1])                        # [1,1,1,W]
    rowcells = g.node("ReduceSum", [rowsum], axes=[1], keepdims=1)  # [1,1,H,1] in-grid cells
    colcells = g.node("ReduceSum", [colsum], axes=[1], keepdims=1)  # [1,1,1,W]
    fullrow = g.node("Mul", [g.node("Cast", [g.node("Less", [rowbg, half])], to=DATA_TYPE),
                             g.node("Cast", [g.node("Greater", [rowcells, half])], to=DATA_TYPE)])
    fullcol = g.node("Mul", [g.node("Cast", [g.node("Less", [colbg, half])], to=DATA_TYPE),
                             g.node("Cast", [g.node("Greater", [colcells, half])], to=DATA_TYPE)])
    inter = g.node("Cast", [g.node("Mul", [fullrow, fullcol])], to=BOOL)     # [1,1,H,W]
    hv = g.node("Add", [hcolor, vcolor])                            # [1,10,H,W]
    swapped = g.node("Sub", [hv, "input"])                          # [1,10,H,W]
    g.node("Where", [inter, swapped, "input"], "output")
    return _model(g.nodes, g.inits)


# ===========================================================================
# 98  hollow_box3 : clear every colour cell whose 4 orthogonal neighbours are
#                   all non-background (the solid interior of each block)
# ===========================================================================
def _ref_hollow(a):
    H, W = a.shape
    o = a.copy()
    for r in range(H):
        for c in range(W):
            if a[r, c] != 0:
                interior = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W and a[nr, nc] != 0):
                        interior = False
                if interior:
                    o[r, c] = 0
    return o


def _build_hollow():
    g = _G()
    ingrid = g.node("ReduceMax", ["input"], axes=[1], keepdims=1)   # [1,1,30,30]
    bgchan = _slice(g, "input", [0], [1], [1])                     # [1,1,30,30]
    nz = g.node("Sub", [ingrid, bgchan])                          # [1,1,30,30] colour mask
    K = g.fconst([0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 3, 3])      # 4-neighbour cross
    cnt = g.node("Conv", [nz, K], kernel_shape=[3, 3], pads=[1, 1, 1, 1])  # [1,1,30,30]
    half = g.fconst([3.5], [1])
    interior = g.node("Mul", [nz, g.node("Cast", [g.node("Greater", [cnt, half])], to=DATA_TYPE)])
    one = g.fconst([1.0], [1])
    keep = g.node("Sub", [one, interior])                         # [1,1,30,30]
    inC = _slice(g, "input", [1], [CHANNELS], [1])               # [1,9,30,30]
    colors = g.node("Mul", [inC, keep])                          # [1,9,30,30]
    newbg = g.node("Add", [bgchan, interior])                    # [1,1,30,30]
    g.node("Concat", [newbg, colors], "output", axis=1)
    return _model(g.nodes, g.inits)


# ===========================================================================
# 215  per_3x6_C0 : extend a 3-row band periodically (vertical period 3) over
#                   the whole grid; columns are already full in the band.
# ===========================================================================
def _ref_periodic(a, P):
    H, W = a.shape
    o = np.zeros_like(a)
    for r in range(H):
        for c in range(W):
            vals = set()
            for rr in range(r % P, H, P):
                if a[rr, c] != 0:
                    vals.add(a[rr, c])
            if len(vals) > 1:
                return None
            o[r, c] = vals.pop() if vals else 0
    return o


def _build_periodic(P):
    g = _G()
    inC = _slice(g, "input", [1], [CHANNELS], [1])               # [1,9,30,30]
    Pm = g.fconst([1.0 if (r - rr) % P == 0 else 0.0
                   for r in range(HEIGHT) for rr in range(HEIGHT)], [HEIGHT, HEIGHT])
    acc = g.node("MatMul", [Pm, inC])                            # [1,9,30,30] periodic OR
    ingrid = g.node("ReduceMax", ["input"], axes=[1], keepdims=1)  # [1,1,30,30] grid region
    colors = g.node("Mul", [acc, ingrid])                       # [1,9,30,30] (kill out-of-grid)
    anycolor = g.node("ReduceSum", [colors], axes=[1], keepdims=1)
    bg = g.node("Sub", [ingrid, anycolor])                      # [1,1,30,30]
    g.node("Concat", [bg, colors], "output", axis=1)
    return _model(g.nodes, g.inits)


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    def emit(name, fn):
        try:
            out.append((name, fn()))
        except Exception:
            pass

    same_size = all(a.shape == b.shape for a, b in prs)

    # ---- 346 fragswatch_1x1 -------------------------------------------------
    if all(b.shape == (1, 1) for _, b in prs):
        ok = True
        for a, b in prs:
            r = _ref_fragswatch(a)
            if r is None or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            emit("fragswatch", _build_fragswatch)

    # ---- 41 connect_RLUD ----------------------------------------------------
    if same_size and all(np.array_equal(_ref_fillrows(a), b) for a, b in prs):
        emit("connect_RLUD", _build_fillrows)

    # ---- 293 crk2_5_swap ----------------------------------------------------
    if same_size:
        ok = True
        for a, b in prs:
            r = _ref_swap(a)
            if r is None or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            emit("crk2_5_swap", _build_swap)

    # ---- 98 hollow_box3 -----------------------------------------------------
    if same_size and all(np.array_equal(_ref_hollow(a), b) for a, b in prs):
        emit("hollow_box3", _build_hollow)

    # ---- 215 per_3x6_C0 (vertical periodic extension) -----------------------
    if same_size:
        for P in (3,):
            refs = [_ref_periodic(a, P) for a, _ in prs]
            if all(r is not None and np.array_equal(r, b)
                   for r, (_, b) in zip(refs, prs)):
                emit(f"periodic{P}", lambda P=P: _build_periodic(P))
                break

    return out
