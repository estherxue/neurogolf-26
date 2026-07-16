"""family_golf8_5: cheaper EXACT golf reformulations for slice [5::6] targets.

Implemented families
--------------------
* cross_shift (task 362, g25_cross362): a full-row + full-column "plus" of one
  colour C sits on a 10x10 grid, together with k stacked colour-5 markers in the
  last column.  The plus moves DOWN by k and LEFT by k (k in {1,2,3}); the markers
  vanish.  Rebuilt on a 10x10 slice so every intermediate is tiny: extract the
  row/column indicator lines, shift each by the (data-dependent) k via a 3-way
  select, recolour into channel C, and Pad back to 30x30 (the output is free).

* fill_enclosed (task 251, fillenc_keep_e0_n1): background cells fully enclosed by
  a colour-2 outline are recoloured to 1.  Implemented as a maze flood-fill of the
  exterior (seeded from the padding + top/left frame) on a small fixed window, so
  the enclosed set = background AND NOT reachable-from-outside.  Only channels 0/1/2
  change, so the result is packed with a Concat (no [1,10,H,W] scratch needed).

Each family auto-detects with a numpy reference over train+test and is proposed
only when the reference reproduces every pair exactly; the grader then re-checks
arc-gen for EXACTness, so wrong guesses cost nothing.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

INT64 = onnx.TensorProto.INT64
FLOAT = onnx.TensorProto.FLOAT


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


class _G:
    """Tiny node/initializer accumulator with helpers."""

    def __init__(self):
        self.nodes = []
        self.inits = []

    def cst(self, name, arr, dtype=FLOAT):
        arr = np.asarray(arr)
        self.inits.append(oh.make_tensor(name, dtype, list(arr.shape), arr.ravel().tolist()))
        return name

    def node(self, op, ins, outs, **attrs):
        self.nodes.append(oh.make_node(op, list(ins), list(outs), **attrs))
        return outs[0] if len(outs) == 1 else outs

    def trans(self, src, dy, dx, N, tag):
        """Shift a [1,1,N,N] tensor by (dy down, dx right) with zero fill."""
        h0, w0 = max(dy, 0), max(dx, 0)
        h1, w1 = max(-dy, 0), max(-dx, 0)
        self.node("Pad", [src], [f"pad_{tag}"], mode="constant", value=0.0,
                  pads=[0, 0, h0, w0, 0, 0, h1, w1])
        self.cst(f"ts_{tag}", np.array([h1, w1], np.int64), INT64)
        self.cst(f"te_{tag}", np.array([h1 + N, w1 + N], np.int64), INT64)
        self.cst(f"ta_{tag}", np.array([2, 3], np.int64), INT64)
        self.node("Slice", [f"pad_{tag}", f"ts_{tag}", f"te_{tag}", f"ta_{tag}"],
                  [f"tr_{tag}"])
        return f"tr_{tag}"


# ==========================================================================
# cross_shift (task 362)
# ==========================================================================

def _cross_solve_np(a):
    a = np.asarray(a)
    if a.shape != (10, 10):
        return None
    H, W = a.shape
    C = None
    for col in set(a.flatten().tolist()) - {0, 5}:
        rows = [r for r in range(H) if (a[r] == col).all()]
        cols = [c for c in range(W) if (a[:, c] == col).all()]
        if rows and cols:
            if C is not None:
                return None                # more than one cross colour
            C, r, c = col, rows[0], cols[0]
    if C is None:
        return None
    k = int((a == 5).sum())
    if k not in (1, 2, 3):
        return None
    out = np.zeros_like(a)
    rr, cc = r + k, c - k
    if not (0 <= rr < H and 0 <= cc < W):
        return None
    out[rr, :] = C
    out[:, cc] = C
    return out


def _build_cross():
    g = _G()
    # ---- 10x10 slice of all channels
    g.cst("sp0", np.array([0, 0], np.int64), INT64)
    g.cst("sp10", np.array([10, 10], np.int64), INT64)
    g.cst("spax", np.array([2, 3], np.int64), INT64)
    g.node("Slice", ["input", "sp0", "sp10", "spax"], ["S"])            # [1,10,10,10]

    # ---- channel slices
    g.cst("c0", np.array([0], np.int64), INT64)
    g.cst("c1", np.array([1], np.int64), INT64)
    g.cst("c5", np.array([5], np.int64), INT64)
    g.cst("c6", np.array([6], np.int64), INT64)
    g.cst("cax", np.array([1], np.int64), INT64)
    g.node("Slice", ["S", "c0", "c1", "cax"], ["ch0"])                  # [1,1,10,10]
    g.node("Slice", ["S", "c5", "c6", "cax"], ["ch5"])                  # [1,1,10,10]

    g.cst("one", np.array([[[[1.0]]]], np.float32))                     # [1,1,1,1]
    g.node("Sub", ["one", "ch0"], ["m0"])
    g.node("Sub", ["m0", "ch5"], ["M"])                                # cross mask [1,1,10,10]

    g.node("ReduceSum", ["ch5"], ["k"], axes=[2, 3], keepdims=1)       # [1,1,1,1]

    g.node("ReduceMin", ["M"], ["RowM"], axes=[3], keepdims=1)          # [1,1,10,1]
    g.node("ReduceMin", ["M"], ["ColM"], axes=[2], keepdims=1)          # [1,1,1,10]

    # ---- 3 candidate shifts (down d / left d)
    for d in (1, 2, 3):
        g.node("Pad", ["RowM"], [f"Rp{d}"], mode="constant", value=0.0,
               pads=[0, 0, d, 0, 0, 0, 0, 0])                           # [1,1,10+d,1]
        g.cst(f"r0{d}", np.array([0], np.int64), INT64)
        g.cst(f"r1{d}", np.array([10], np.int64), INT64)
        g.cst("rax2", np.array([2], np.int64), INT64) if d == 1 else None
        g.node("Slice", [f"Rp{d}", f"r0{d}", f"r1{d}", "rax2"], [f"R{d}"])  # [1,1,10,1]

        g.node("Pad", ["ColM"], [f"Cp{d}"], mode="constant", value=0.0,
               pads=[0, 0, 0, 0, 0, 0, 0, d])                           # [1,1,1,10+d]
        g.cst(f"k0{d}", np.array([d], np.int64), INT64)
        g.cst(f"k1{d}", np.array([d + 10], np.int64), INT64)
        g.cst("cax3", np.array([3], np.int64), INT64) if d == 1 else None
        g.node("Slice", [f"Cp{d}", f"k0{d}", f"k1{d}", "cax3"], [f"C{d}"])  # [1,1,1,10]

    # ---- selectors s_j = relu(1 - |k - j|)
    for d in (1, 2, 3):
        g.cst(f"j{d}", np.array([[[[float(d)]]]], np.float32))
        g.node("Sub", ["k", f"j{d}"], [f"kd{d}"])
        g.node("Abs", [f"kd{d}"], [f"ka{d}"])
        g.node("Sub", ["one", f"ka{d}"], [f"kt{d}"])
        g.node("Relu", [f"kt{d}"], [f"s{d}"])                           # [1,1,1,1]

    # ---- combine
    g.node("Mul", ["R1", "s1"], ["Rp_a"])
    g.node("Mul", ["R2", "s2"], ["Rp_b"])
    g.node("Mul", ["R3", "s3"], ["Rp_c"])
    g.node("Add", ["Rp_a", "Rp_b"], ["Rp_ab"])
    g.node("Add", ["Rp_ab", "Rp_c"], ["RowSh"])                        # [1,1,10,1]
    g.node("Mul", ["C1", "s1"], ["Cp_a"])
    g.node("Mul", ["C2", "s2"], ["Cp_b"])
    g.node("Mul", ["C3", "s3"], ["Cp_c"])
    g.node("Add", ["Cp_a", "Cp_b"], ["Cp_ab"])
    g.node("Add", ["Cp_ab", "Cp_c"], ["ColSh"])                        # [1,1,1,10]
    g.node("Max", ["RowSh", "ColSh"], ["Ms"])                          # [1,1,10,10]

    # ---- recolour: out10 = e0 + Ms * (colorvecC - e0)
    e0 = np.zeros((1, 10, 1, 1), np.float32); e0[0, 0] = 1.0
    e5 = np.zeros((1, 10, 1, 1), np.float32); e5[0, 5] = 1.0
    g.cst("e0", e0)
    g.cst("e5", e5)
    g.node("ReduceMax", ["S"], ["cvA"], axes=[2, 3], keepdims=1)       # [1,10,1,1]
    g.node("Sub", ["cvA", "e0"], ["cv1"])
    g.node("Sub", ["cv1", "e5"], ["cvC"])                              # 1 only at channel C
    g.node("Sub", ["cvC", "e0"], ["P"])                                # +1 at C, -1 at 0
    g.node("Mul", ["Ms", "P"], ["MsP"])                                # [1,10,10,10]
    g.node("Add", ["MsP", "e0"], ["out10"])                            # [1,10,10,10]
    g.node("Pad", ["out10"], ["output"], mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, 20, 20])                            # [1,10,30,30]
    return _model(g.nodes, g.inits)


def _try_cross(prs):
    saw = False
    for a, b in prs:
        pred = _cross_solve_np(a)
        if pred is None or pred.shape != b.shape or not (pred == b).all():
            return None
        if (pred != a).any():
            saw = True
    return _build_cross() if saw else None


# ==========================================================================
# fill_enclosed (task 251)
# ==========================================================================

_FILL_N = 14          # fixed window (observed grids <= 12)
_FILL_T = 12          # flood steps (observed depth <= 8)


def _flood_outside(free):
    from collections import deque
    H, W = free.shape
    out = np.zeros_like(free, dtype=bool)
    dq = deque()
    for r in range(H):
        for c in (0, W - 1):
            if free[r, c] and not out[r, c]:
                out[r, c] = True; dq.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if free[r, c] and not out[r, c]:
                out[r, c] = True; dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and free[nr, nc] and not out[nr, nc]:
                out[nr, nc] = True; dq.append((nr, nc))
    return out


def _fill_solve_np(a):
    a = np.asarray(a)
    if a.shape[0] > _FILL_N or a.shape[1] > _FILL_N:
        return None
    nz = set(a[a != 0].tolist())
    if nz != {2}:                          # barrier colour must be exactly 2
        return None
    outside = _flood_outside(a == 0)
    enc = (a == 0) & (~outside)
    if not enc.any():
        return None
    pred = a.copy()
    pred[enc] = 1
    return pred


def _build_fill():
    g = _G()
    N = _FILL_N
    g.cst("sp0", np.array([0, 0], np.int64), INT64)
    g.cst("spN", np.array([N, N], np.int64), INT64)
    g.cst("spax", np.array([2, 3], np.int64), INT64)
    g.node("Slice", ["input", "sp0", "spN", "spax"], ["S"])            # [1,10,N,N]

    g.node("ReduceSum", ["S"], ["realmask"], axes=[1], keepdims=1)     # [1,1,N,N]
    g.cst("c0", np.array([0], np.int64), INT64)
    g.cst("c1", np.array([1], np.int64), INT64)
    g.cst("c2", np.array([2], np.int64), INT64)
    g.cst("c3", np.array([3], np.int64), INT64)
    g.cst("cax", np.array([1], np.int64), INT64)
    g.node("Slice", ["S", "c0", "c1", "cax"], ["ch0"])
    g.node("Slice", ["S", "c2", "c3", "cax"], ["ch2"])

    g.cst("one", np.array([[[[1.0]]]], np.float32))
    g.node("Sub", ["one", "ch2"], ["O"])                               # free mask
    g.node("Sub", ["one", "realmask"], ["padmask"])

    E = np.zeros((1, 1, N, N), np.float32); E[:, :, 0, :] = 1.0; E[:, :, :, 0] = 1.0
    g.cst("E", E)
    g.node("Mul", ["O", "E"], ["seedE"])
    g.node("Max", ["padmask", "seedE"], ["R0a"])
    g.node("Min", ["O", "R0a"], ["R0"])

    plus = np.array([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], np.float32)
    g.cst("plusW", plus)
    cur = "R0"
    for t in range(_FILL_T):
        g.node("Conv", [cur, "plusW"], [f"dil{t}"], kernel_shape=[3, 3],
               pads=[1, 1, 1, 1], group=1)
        g.node("Min", ["O", f"dil{t}"], [f"R{t + 1}"])
        cur = f"R{t + 1}"
    outside = cur

    g.node("Sub", ["ch0", outside], ["encS"])
    g.node("Relu", ["encS"], ["enc"])
    g.node("Sub", ["ch0", "enc"], ["ch0out"])
    g.node("Concat", ["ch0out", "enc", "ch2"], ["cat3"], axis=1)       # [1,3,N,N]
    g.node("Pad", ["cat3"], ["output"], mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 7, 30 - N, 30 - N])                    # [1,10,30,30]
    return _model(g.nodes, g.inits)


def _try_fill(prs):
    saw = False
    for a, b in prs:
        pred = _fill_solve_np(a)
        if pred is None or pred.shape != b.shape or not (pred == b).all():
            return None
        saw = True
    return _build_fill() if saw else None


# ==========================================================================
# diag_rays (task 136)
# ==========================================================================

def _diag_solve_np(a):
    a = np.asarray(a)
    if a.shape != (10, 10):
        return None
    nz = set(a[a != 0].tolist())
    if not nz or not nz <= {1, 2}:
        return None
    H, W = a.shape
    pred = a.copy()
    for col in nz:
        M = (a == col)
        for r in range(H):
            for c in range(W):
                if not M[r, c]:
                    continue
                if col == 1 and (r == 0 or not M[r - 1, c]) and (c == 0 or not M[r, c - 1]):
                    rr, cc = r - 1, c - 1
                    while rr >= 0 and cc >= 0:
                        pred[rr, cc] = 1; rr -= 1; cc -= 1
                if col == 2 and (r == H - 1 or not M[r + 1, c]) and (c == W - 1 or not M[r, c + 1]):
                    rr, cc = r + 1, c + 1
                    while rr < H and cc < W:
                        pred[rr, cc] = 2; rr += 1; cc += 1
    return pred


def _build_diag():
    g = _G()
    N = 10
    g.cst("sp0", np.array([0, 0], np.int64), INT64)
    g.cst("spN", np.array([N, N], np.int64), INT64)
    g.cst("spax", np.array([2, 3], np.int64), INT64)
    g.node("Slice", ["input", "sp0", "spN", "spax"], ["S"])            # [1,10,10,10]

    g.cst("c1", np.array([1], np.int64), INT64)
    g.cst("c2", np.array([2], np.int64), INT64)
    g.cst("c3", np.array([3], np.int64), INT64)
    g.cst("cax", np.array([1], np.int64), INT64)
    g.node("Slice", ["S", "c1", "c2", "cax"], ["B1"])
    g.node("Slice", ["S", "c2", "c3", "cax"], ["B2"])

    g.cst("one", np.array([[[[1.0]]]], np.float32))
    # TL corner of B1
    d1 = g.trans("B1", 1, 0, N, "b1d")
    r1 = g.trans("B1", 0, 1, N, "b1r")
    g.node("Sub", ["one", d1], ["nb1d"])
    g.node("Sub", ["one", r1], ["nb1r"])
    g.node("Mul", ["B1", "nb1d"], ["cTLa"])
    g.node("Mul", ["cTLa", "nb1r"], ["cTL"])
    # BR corner of B2
    u2 = g.trans("B2", -1, 0, N, "b2u")
    l2 = g.trans("B2", 0, -1, N, "b2l")
    g.node("Sub", ["one", u2], ["nb2u"])
    g.node("Sub", ["one", l2], ["nb2l"])
    g.node("Mul", ["B2", "nb2u"], ["cBRa"])
    g.node("Mul", ["cBRa", "nb2l"], ["cBR"])

    # diagonal propagation (up-left for ray1, down-right for ray2)
    cur = "cTL"
    for k in (1, 2, 4, 8):
        s = g.trans(cur, -k, -k, N, f"r1s{k}")
        g.node("Max", [cur, s], [f"R1_{k}"]); cur = f"R1_{k}"
    ray1 = cur
    cur = "cBR"
    for k in (1, 2, 4, 8):
        s = g.trans(cur, k, k, N, f"r2s{k}")
        g.node("Max", [cur, s], [f"R2_{k}"]); cur = f"R2_{k}"
    ray2 = cur

    g.node("Max", ["B1", ray1], ["o1"])
    g.node("Max", ["B2", ray2], ["o2"])
    g.node("ReduceSum", ["S"], ["realmask"], axes=[1], keepdims=1)
    g.node("Sub", ["realmask", "o1"], ["o0a"])
    g.node("Sub", ["o0a", "o2"], ["o0"])
    g.node("Concat", ["o0", "o1", "o2"], ["cat3"], axis=1)             # [1,3,10,10]
    g.node("Pad", ["cat3"], ["output"], mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 7, 20, 20])
    return _model(g.nodes, g.inits)


def _try_diag(prs):
    saw = False
    for a, b in prs:
        pred = _diag_solve_np(a)
        if pred is None or pred.shape != b.shape or not (pred == b).all():
            return None
        if (pred != a).any():
            saw = True
    return _build_diag() if saw else None


# ==========================================================================
# pin_stripes (task 200)
# ==========================================================================

def _pin_solve_np(a):
    a = np.asarray(a)
    if a.shape != (10, 10):
        return None
    H, W = a.shape
    nz = np.argwhere(a != 0)
    if len(nz) != 1:
        return None
    dr, dc = nz[0]
    if dr != H - 1:
        return None
    D = int(a[dr, dc])
    if D == 5:
        return None
    pred = np.zeros_like(a)
    for cc in range(dc, W, 2):
        pred[:, cc] = D
    for cc in range(dc + 1, W, 4):
        pred[0, cc] = 5
    for cc in range(dc + 3, W, 4):
        pred[H - 1, cc] = 5
    return pred


def _build_pin():
    g = _G()
    N = 10
    g.cst("sp0", np.array([0, 0], np.int64), INT64)
    g.cst("spN", np.array([N, N], np.int64), INT64)
    g.cst("spax", np.array([2, 3], np.int64), INT64)
    g.node("Slice", ["input", "sp0", "spN", "spax"], ["S"])
    g.cst("c0", np.array([0], np.int64), INT64)
    g.cst("c1", np.array([1], np.int64), INT64)
    g.cst("cax", np.array([1], np.int64), INT64)
    g.node("Slice", ["S", "c0", "c1", "cax"], ["ch0"])
    g.node("ReduceSum", ["S"], ["realmask"], axes=[1], keepdims=1)
    g.node("Sub", ["realmask", "ch0"], ["dot"])                        # single 1 at (9,dc)

    cur = "dot"
    for k in (1, 2, 4, 8):
        s = g.trans(cur, -k, 0, N, f"vup{k}")
        g.node("Max", [cur, s], [f"V{k}"]); cur = f"V{k}"
    vcol = cur
    for k in (2, 4, 8):
        s = g.trans(cur, 0, k, N, f"lrt{k}")
        g.node("Max", [cur, s], [f"L{k}"]); cur = f"L{k}"
    L = cur

    t0 = g.trans("dot", -9, 1, N, "t0")
    cur = t0
    for k in (4, 8):
        s = g.trans(cur, 0, k, N, f"trt{k}")
        g.node("Max", [cur, s], [f"T{k}"]); cur = f"T{k}"
    Tm = cur
    b0 = g.trans("dot", 0, 3, N, "b0")
    cur = b0
    for k in (4, 8):
        s = g.trans(cur, 0, k, N, f"brt{k}")
        g.node("Max", [cur, s], [f"B{k}"]); cur = f"B{k}"
    Bm = cur
    g.node("Max", [Tm, Bm], ["M5"])

    e0 = np.zeros((1, 10, 1, 1), np.float32); e0[0, 0] = 1.0
    e5 = np.zeros((1, 10, 1, 1), np.float32); e5[0, 5] = 1.0
    g.cst("e0", e0)
    g.cst("e5", e5)
    g.node("ReduceMax", ["S"], ["cvA"], axes=[2, 3], keepdims=1)
    g.node("Sub", ["cvA", "e0"], ["cvD"])                              # 1 at ch D
    g.node("Sub", ["cvD", "e0"], ["PD"])                               # +1 at D, -1 at 0
    g.node("Sub", ["e5", "e0"], ["P5"])                                # +1 at 5, -1 at 0
    g.node("Mul", ["e0", "realmask"], ["bg10"])                        # [1,10,10,10]
    g.node("Mul", [L, "PD"], ["lineC"])
    g.node("Mul", ["M5", "P5"], ["markC"])
    g.node("Add", ["bg10", "lineC"], ["o10a"])
    g.node("Add", ["o10a", "markC"], ["out10"])
    g.node("Pad", ["out10"], ["output"], mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, 20, 20])
    return _model(g.nodes, g.inits)


def _try_pin(prs):
    saw = False
    for a, b in prs:
        pred = _pin_solve_np(a)
        if pred is None or pred.shape != b.shape or not (pred == b).all():
            return None
        saw = True
    return _build_pin() if saw else None


# ==========================================================================
# band_cells (task 226)
# ==========================================================================

def _band_solve_np(a):
    a = np.asarray(a)
    if a.shape != (10, 10):
        return None
    if set(a.flatten().tolist()) - {0, 5} != set() or 5 not in a:
        return None
    H, W = a.shape
    ch5 = (a == 5).astype(int)
    HL = ch5.min(axis=1); VL = ch5.min(axis=0)
    cumH = np.cumsum(HL); cumV = np.cumsum(VL)
    totH, totV = cumH[-1], cumV[-1]
    midH, midV = (totH + 1) // 2, (totV + 1) // 2
    rowTL = (cumH == 0) & (HL == 0); rowBR = (cumH == totH) & (HL == 0)
    rowC = (cumH == midH) & (HL == 0)
    colTL = (cumV == 0) & (VL == 0); colBR = (cumV == totV) & (VL == 0)
    colC = (cumV == midV) & (VL == 0)
    out = a.copy()
    out[np.outer(rowTL, colTL)] = 1
    out[np.outer(rowC, colC)] = 2
    out[np.outer(rowBR, colBR)] = 3
    return out


def _build_band():
    g = _G()
    g.cst("s5", np.array([5, 0, 0], np.int64), INT64)
    g.cst("e5", np.array([6, 10, 10], np.int64), INT64)
    g.cst("s4", np.array([4, 0, 0], np.int64), INT64)
    g.cst("e4", np.array([5, 10, 10], np.int64), INT64)
    g.cst("cax", np.array([1, 2, 3], np.int64), INT64)
    g.node("Slice", ["input", "s5", "e5", "cax"], ["ch5"])
    g.node("Slice", ["input", "s4", "e4", "cax"], ["z"])               # zeros

    g.node("ReduceMin", ["ch5"], ["HL"], axes=[3], keepdims=1)         # [1,1,10,1]
    g.node("ReduceMin", ["ch5"], ["VL"], axes=[2], keepdims=1)         # [1,1,1,10]
    g.cst("Ltri", np.tril(np.ones((10, 10), np.float32)).reshape(1, 1, 10, 10))
    g.cst("Utri", np.triu(np.ones((10, 10), np.float32)).reshape(1, 1, 10, 10))
    g.node("MatMul", ["Ltri", "HL"], ["cumH"])                         # [1,1,10,1]
    g.node("MatMul", ["VL", "Utri"], ["cumV"])                         # [1,1,1,10]
    g.node("ReduceSum", ["HL"], ["totH"], axes=[2], keepdims=1)        # [1,1,1,1]
    g.node("ReduceSum", ["VL"], ["totV"], axes=[3], keepdims=1)

    g.cst("one", np.array([[[[1.0]]]], np.float32))
    g.cst("half", np.array([[[[0.5]]]], np.float32))
    g.node("Add", ["totH", "one"], ["hH0"]); g.node("Mul", ["hH0", "half"], ["hH1"])
    g.node("Floor", ["hH1"], ["midH"])
    g.node("Add", ["totV", "one"], ["hV0"]); g.node("Mul", ["hV0", "half"], ["hV1"])
    g.node("Floor", ["hV1"], ["midV"])
    g.node("Sub", ["one", "HL"], ["nlR"])
    g.node("Sub", ["one", "VL"], ["nlV"])
    g.node("Sub", ["totH", "half"], ["totHm"])
    g.node("Sub", ["totV", "half"], ["totVm"])

    def band(cum, tot_m, mid, nl, tag):
        g.node("Less", [cum, "half"], [f"tl_{tag}"]); g.node("Cast", [f"tl_{tag}"], [f"tlf_{tag}"], to=FLOAT)
        g.node("Mul", [nl, f"tlf_{tag}"], [f"TL_{tag}"])
        g.node("Greater", [cum, tot_m], [f"br_{tag}"]); g.node("Cast", [f"br_{tag}"], [f"brf_{tag}"], to=FLOAT)
        g.node("Mul", [nl, f"brf_{tag}"], [f"BR_{tag}"])
        g.node("Sub", [cum, mid], [f"dm_{tag}"]); g.node("Abs", [f"dm_{tag}"], [f"adm_{tag}"])
        g.node("Greater", ["half", f"adm_{tag}"], [f"cc_{tag}"]); g.node("Cast", [f"cc_{tag}"], [f"ccf_{tag}"], to=FLOAT)
        g.node("Mul", [nl, f"ccf_{tag}"], [f"C_{tag}"])
        return f"TL_{tag}", f"C_{tag}", f"BR_{tag}"

    rTL, rC, rBR = band("cumH", "totHm", "midH", "nlR", "r")
    cTL, cC, cBR = band("cumV", "totVm", "midV", "nlV", "c")
    g.node("Mul", [rTL, cTL], ["cell1"])
    g.node("Mul", [rC, cC], ["cell2"])
    g.node("Mul", [rBR, cBR], ["cell3"])
    g.node("Sub", ["one", "ch5"], ["b0"])
    g.node("Sub", ["b0", "cell1"], ["b1"])
    g.node("Sub", ["b1", "cell2"], ["b2"])
    g.node("Sub", ["b2", "cell3"], ["ch0"])
    g.node("Concat",
           ["ch0", "cell1", "cell2", "cell3", "z", "ch5", "z", "z", "z", "z"],
           ["cat10"], axis=1)
    g.node("Pad", ["cat10"], ["output"], mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, 20, 20])
    return _model(g.nodes, g.inits)


def _try_band(prs):
    saw = False
    for a, b in prs:
        pred = _band_solve_np(a)
        if pred is None or pred.shape != b.shape or not (pred == b).all():
            return None
        if (pred != a).any():
            saw = True
    return _build_band() if saw else None


# ==========================================================================
# grav_stack (task 78)
# ==========================================================================

def _grav_solve_np(a):
    a = np.asarray(a)
    if a.shape != (10, 10):
        return None
    if set(a.flatten().tolist()) - {0, 1, 2} != set() or 2 not in a:
        return None
    H, W = a.shape
    RI = np.arange(H).reshape(H, 1)
    ch1 = (a == 1).astype(float)
    ch2 = (a == 2).astype(float)
    lowest1 = (ch1 * RI).max(axis=0)                    # per column
    n2 = ch2.sum(axis=0)
    o2 = ((RI > lowest1) & (RI <= lowest1 + n2))
    pred = a.copy()
    pred[a == 2] = 0
    pred[o2] = 2
    return pred


def _build_grav():
    g = _G()
    g.cst("s1", np.array([1, 0, 0], np.int64), INT64)
    g.cst("e1", np.array([2, 10, 10], np.int64), INT64)
    g.cst("s2", np.array([2, 0, 0], np.int64), INT64)
    g.cst("e2", np.array([3, 10, 10], np.int64), INT64)
    g.cst("cax", np.array([1, 2, 3], np.int64), INT64)
    g.node("Slice", ["input", "s1", "e1", "cax"], ["ch1"])
    g.node("Slice", ["input", "s2", "e2", "cax"], ["ch2"])

    RI = np.arange(10, dtype=np.float32).reshape(1, 1, 10, 1)
    g.cst("RI", RI)
    g.cst("half", np.array([[[[0.5]]]], np.float32))
    g.cst("one", np.array([[[[1.0]]]], np.float32))
    g.node("Mul", ["ch1", "RI"], ["ch1RI"])
    g.node("ReduceMax", ["ch1RI"], ["low1"], axes=[2], keepdims=1)     # [1,1,1,10]
    g.node("ReduceSum", ["ch2"], ["n2"], axes=[2], keepdims=1)         # [1,1,1,10]
    g.node("Add", ["low1", "n2"], ["upper"])
    g.node("Add", ["upper", "half"], ["upp05"])
    g.node("Greater", ["RI", "low1"], ["g1"])
    g.node("Cast", ["g1"], ["g1f"], to=FLOAT)
    g.node("Greater", ["upp05", "RI"], ["g2"])
    g.node("Cast", ["g2"], ["g2f"], to=FLOAT)
    g.node("Mul", ["g1f", "g2f"], ["o2"])                              # [1,1,10,10]
    g.node("Sub", ["one", "ch1"], ["t0"])
    g.node("Sub", ["t0", "o2"], ["o0"])
    g.node("Concat", ["o0", "ch1", "o2"], ["cat3"], axis=1)
    g.node("Pad", ["cat3"], ["output"], mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 7, 20, 20])
    return _model(g.nodes, g.inits)


def _try_grav(prs):
    saw = False
    for a, b in prs:
        pred = _grav_solve_np(a)
        if pred is None or pred.shape != b.shape or not (pred == b).all():
            return None
        if (pred != a).any():
            saw = True
    return _build_grav() if saw else None


# ==========================================================================
# close_loops (task 279)
# ==========================================================================

_LOOP_N = 18
_LOOP_D1 = 16      # exterior bg flood (observed depth <= 14)
_LOOP_D2 = 16      # closed-label spread within fg


def _loop_label(mask):
    from collections import deque
    H, W = mask.shape
    lbl = np.zeros((H, W), int); n = 0
    for r in range(H):
        for c in range(W):
            if mask[r, c] and lbl[r, c] == 0:
                n += 1; dq = deque([(r, c)]); lbl[r, c] = n
                while dq:
                    y, x = dq.popleft()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lbl[ny, nx] == 0:
                            lbl[ny, nx] = n; dq.append((ny, nx))
    return lbl, n


def _loop_solve_np(a):
    a = np.asarray(a)
    if a.shape[0] > _LOOP_N or a.shape[1] > _LOOP_N:
        return None
    if set(a.flatten().tolist()) - {1, 9} != set() or 9 not in a or 1 not in a:
        return None
    H, W = a.shape
    lbl, n = _loop_label(a == 9)
    border = set(lbl[0, :]).union(lbl[-1, :]).union(lbl[:, 0]).union(lbl[:, -1]) - {0}
    hole = np.isin(lbl, [i for i in range(1, n + 1) if i not in border])
    clbl, cn = _loop_label(a == 1)
    pred = a.copy()
    for ci in range(1, cn + 1):
        comp = (clbl == ci)
        d = comp.copy()
        d[1:, :] |= comp[:-1, :]; d[:-1, :] |= comp[1:, :]
        d[:, 1:] |= comp[:, :-1]; d[:, :-1] |= comp[:, 1:]
        if (d & hole).any():
            pred[comp] = 8
    return pred


def _build_loop():
    g = _G()
    N = _LOOP_N
    # slice a single channel + NxN window in one Slice (no [1,10,N,N] scratch)
    g.cst("s0", np.array([0, 0, 0], np.int64), INT64)
    g.cst("s1", np.array([1, 0, 0], np.int64), INT64)
    g.cst("s9", np.array([9, 0, 0], np.int64), INT64)
    g.cst("e0", np.array([1, N, N], np.int64), INT64)
    g.cst("e1", np.array([2, N, N], np.int64), INT64)
    g.cst("e9", np.array([10, N, N], np.int64), INT64)
    g.cst("cax", np.array([1, 2, 3], np.int64), INT64)
    g.node("Slice", ["input", "s0", "e0", "cax"], ["o0"])              # zeros
    g.node("Slice", ["input", "s1", "e1", "cax"], ["ch1"])
    g.node("Slice", ["input", "s9", "e9", "cax"], ["ch9"])

    g.cst("one", np.array([[[[1.0]]]], np.float32))
    g.node("Add", ["ch1", "ch9"], ["realmask"])
    g.node("Sub", ["one", "realmask"], ["padmask"])
    g.node("Max", ["ch9", "padmask"], ["freeP"])
    E = np.zeros((1, 1, N, N), np.float32); E[:, :, 0, :] = 1.0; E[:, :, :, 0] = 1.0
    g.cst("E", E)
    g.node("Mul", ["freeP", "E"], ["seedE"])
    g.node("Max", ["padmask", "seedE"], ["R0a"])
    g.node("Min", ["freeP", "R0a"], ["R0"])

    plus = np.array([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], np.float32)
    g.cst("plusW", plus)
    cur = "R0"
    for t in range(_LOOP_D1):
        g.node("Conv", [cur, "plusW"], [f"hd{t}"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        g.node("Min", ["freeP", f"hd{t}"], [f"HR{t + 1}"]); cur = f"HR{t + 1}"
    g.node("Sub", ["ch9", cur], ["holeS"])
    g.node("Relu", ["holeS"], ["holes"])
    g.node("Conv", ["holes", "plusW"], ["holed"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    g.node("Min", ["ch1", "holed"], ["cseed"])
    cur = "cseed"
    for t in range(_LOOP_D2):
        g.node("Conv", [cur, "plusW"], [f"cd{t}"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        g.node("Min", ["ch1", f"cd{t}"], [f"CR{t + 1}"]); cur = f"CR{t + 1}"
    closed = cur
    g.node("Sub", ["ch1", closed], ["o1"])
    g.node("Concat",
           ["o0", "o1", "o0", "o0", "o0", "o0", "o0", "o0", closed, "ch9"],
           ["cat10"], axis=1)                                          # [1,10,N,N]
    g.node("Pad", ["cat10"], ["output"], mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, 30 - N, 30 - N])
    return _model(g.nodes, g.inits)


def _try_loop(prs):
    saw = False
    for a, b in prs:
        pred = _loop_solve_np(a)
        if pred is None or pred.shape != b.shape or not (pred == b).all():
            return None
        if (pred != a).any():
            saw = True
    return _build_loop() if saw else None


# ==========================================================================
# entry point
# ==========================================================================

def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return []
    out = []
    m = _try_cross(prs)
    if m is not None:
        out.append(("g25_cross362", m))
    m = _try_fill(prs)
    if m is not None:
        out.append(("fillenc_keep_e0_n1", m))
    m = _try_diag(prs)
    if m is not None:
        out.append(("g55_diagrays", m))
    m = _try_pin(prs)
    if m is not None:
        out.append(("pin200", m))
    m = _try_loop(prs)
    if m is not None:
        out.append(("g73_f16_279", m))
    m = _try_grav(prs)
    if m is not None:
        out.append(("grav_up_bg0_k9", m))
    m = _try_band(prs)
    if m is not None:
        out.append(("t226_bandfill", m))
    return out
