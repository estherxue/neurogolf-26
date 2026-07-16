"""family_golf5_3 — cheaper EXACT solvers for selected golf targets (slice [3::6]).

The integrator auto-picks the cheapest exact model per task, so we only need to
BEAT the current cost (cost = params + intermediate_memory; points = 25 - ln cost).

Two solvers here, both designed to keep intermediates small (single-channel
[1,1,30,30] masks; only the unavoidable [1,10,30,30] tensors are full size; the
final output tensor is FREE):

  * xdiag (task 375): draw an X (both diagonals) of background over a solid colour
    block.  out = input + onDiagReal*(e0 - eC), one full-size intermediate.

  * cropswap (task 290): crop the foreground bounding box to the origin and swap
    the two foreground colours.  Recolour with a data-independent algebraic swap
    p*(M-input) (kills background, swaps the two present fg colours), then shift to
    the origin with two index-Gathers (no [30,30] shift-matrix builds).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, HEIGHT, WIDTH, CHANNELS

INT64 = onnx.TensorProto.INT64
FLOAT = DATA_TYPE


def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _pairs(examples):
    out = []
    for s in ("train", "test"):
        for e in examples.get(s, []):
            out.append((np.array(e["input"]), np.array(e["output"])))
    return out


# ===========================================================================
# task 375 : X-diagonal carve over a solid colour block
# ===========================================================================
def _xdiag_model():
    # constants
    K = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    main = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            K[0, 0, i, j] = i + j + 1.0
            if i == j:
                main[0, 0, i, j] = 1.0
    Kt = oh.make_tensor("K", FLOAT, [1, 1, HEIGHT, WIDTH], K.ravel().tolist())
    Mt = oh.make_tensor("MD", FLOAT, [1, 1, HEIGHT, WIDTH], main.ravel().tolist())
    one = oh.make_tensor("one", FLOAT, [1, 1, 1, 1], [1.0])
    half = oh.make_tensor("half", FLOAT, [1, 1, 1, 1], [0.5])
    e0 = oh.make_tensor("e0", FLOAT, [1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))

    nodes = [
        # real-cell mask (1 inside the grid, 0 in padding)
        oh.make_node("ReduceSum", ["input"], ["realMask"], axes=[1], keepdims=1),
        # N = number of occupied rows
        oh.make_node("ReduceSum", ["input"], ["rowCnt"], axes=[1, 3], keepdims=1),
        oh.make_node("Clip", ["rowCnt"], ["rowOcc"], min=0.0, max=1.0),
        oh.make_node("ReduceSum", ["rowOcc"], ["N"], axes=[2], keepdims=1),
        # on-diagonal mask = (i==j) OR (i+j==N-1)
        oh.make_node("Sub", ["K", "N"], ["D"]),
        oh.make_node("Abs", ["D"], ["AD"]),
        oh.make_node("Sub", ["one", "AD"], ["anti"]),       # 1-|K-N|
        oh.make_node("Max", ["MD", "anti"], ["onDiag"]),    # union (>=0, <=1)
        oh.make_node("Mul", ["onDiag", "realMask"], ["onDiagReal"]),
        # eC = one-hot of the (dominant) foreground colour channel
        oh.make_node("ReduceSum", ["input"], ["cnt"], axes=[2, 3], keepdims=1),
        oh.make_node("ReduceMax", ["cnt"], ["mx"], axes=[1], keepdims=1),
        oh.make_node("Sub", ["mx", "half"], ["thr"]),
        oh.make_node("Greater", ["cnt", "thr"], ["gtB"]),
        oh.make_node("Cast", ["gtB"], ["eC"], to=FLOAT),
        oh.make_node("Sub", ["e0", "eC"], ["delta"]),       # +1 at ch0, -1 at chC
        # output = input + onDiagReal*(e0-eC)
        oh.make_node("Mul", ["onDiagReal", "delta"], ["term"]),
        oh.make_node("Add", ["input", "term"], ["output"]),
    ]
    return _model(nodes, [Kt, Mt, one, half, e0])


def _is_xdiag(P):
    if not P:
        return False
    for a, b in P:
        H, W = a.shape
        if H != W or a.shape != b.shape:
            return False
        N = H
        nz = a[a != 0]
        if nz.size == 0:
            return False
        vals, cnts = np.unique(nz, return_counts=True)
        C = vals[cnts.argmax()]
        out = np.full_like(a, C)
        for i in range(N):
            for j in range(N):
                if i == j or i + j == N - 1:
                    out[i, j] = 0
        if not np.array_equal(out, b):
            return False
    return True


# ===========================================================================
# task 290 : crop foreground bbox to origin + swap the two foreground colours
# ===========================================================================
def _cropswap_model():
    RIrow = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    RIcol = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    RIr = oh.make_tensor("RIr", FLOAT, [1, 1, HEIGHT, 1], RIrow.ravel().tolist())
    RIc = oh.make_tensor("RIc", FLOAT, [1, 1, 1, WIDTH], RIcol.ravel().tolist())
    big = oh.make_tensor("big", FLOAT, [1, 1, 1, 1], [100.0])
    one = oh.make_tensor("one", FLOAT, [1, 1, 1, 1], [1.0])
    eNot0 = oh.make_tensor("eNot0", FLOAT, [1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    ar = oh.make_tensor("ar", FLOAT, [HEIGHT], list(range(HEIGHT)))   # 0..29 (H==W)
    ch0s = oh.make_tensor("c0s", INT64, [1], [0])
    ch0e = oh.make_tensor("c0e", INT64, [1], [1])
    ch0a = oh.make_tensor("c0a", INT64, [1], [1])

    nodes = [
        # foreground mask M_in (1 where colour!=0)
        oh.make_node("ReduceSum", ["input"], ["total"], axes=[1], keepdims=1),
        oh.make_node("Slice", ["input", "c0s", "c0e", "c0a"], ["ch0"]),
        oh.make_node("Sub", ["total", "ch0"], ["M"]),           # [1,1,30,30] fg mask
        # present-channel mask p (fg colours only)
        oh.make_node("ReduceSum", ["input"], ["cnt"], axes=[2, 3], keepdims=1),
        oh.make_node("Clip", ["cnt"], ["present"], min=0.0, max=1.0),
        oh.make_node("Mul", ["present", "eNot0"], ["p"]),
        # recolour: swap the two fg colours, kill background
        oh.make_node("Sub", ["M", "input"], ["diff"]),          # [1,10,30,30]
        oh.make_node("Mul", ["p", "diff"], ["recolored"]),      # [1,10,30,30]
        # bbox top row r0 = min occupied row index
        oh.make_node("ReduceSum", ["M"], ["rowHas0"], axes=[3], keepdims=1),
        oh.make_node("Clip", ["rowHas0"], ["rowHas"], min=0.0, max=1.0),
        oh.make_node("Sub", ["one", "rowHas"], ["rowEmpty"]),
        oh.make_node("Mul", ["rowEmpty", "big"], ["rowPen"]),
        oh.make_node("Add", ["RIr", "rowPen"], ["rowScore"]),
        oh.make_node("ReduceMin", ["rowScore"], ["r0"], axes=[2], keepdims=1),
        # bbox left col c0
        oh.make_node("ReduceSum", ["M"], ["colHas0"], axes=[2], keepdims=1),
        oh.make_node("Clip", ["colHas0"], ["colHas"], min=0.0, max=1.0),
        oh.make_node("Sub", ["one", "colHas"], ["colEmpty"]),
        oh.make_node("Mul", ["colEmpty", "big"], ["colPen"]),
        oh.make_node("Add", ["RIc", "colPen"], ["colScore"]),
        oh.make_node("ReduceMin", ["colScore"], ["c0"], axes=[3], keepdims=1),
        # row gather index = clip(arange + r0, 0, H-1)
        oh.make_node("Squeeze", ["r0"], ["r0s"], axes=[0, 1, 2, 3]),
        oh.make_node("Add", ["ar", "r0s"], ["idxRf"]),
        oh.make_node("Clip", ["idxRf"], ["idxRc"], min=0.0, max=float(HEIGHT - 1)),
        oh.make_node("Cast", ["idxRc"], ["idxR"], to=INT64),
        oh.make_node("Gather", ["recolored", "idxR"], ["shiftR"], axis=2),
        # col gather index = clip(arange + c0, 0, W-1)
        oh.make_node("Squeeze", ["c0"], ["c0sq"], axes=[0, 1, 2, 3]),
        oh.make_node("Add", ["ar", "c0sq"], ["idxCf"]),
        oh.make_node("Clip", ["idxCf"], ["idxCc"], min=0.0, max=float(WIDTH - 1)),
        oh.make_node("Cast", ["idxCc"], ["idxC"], to=INT64),
        oh.make_node("Gather", ["shiftR", "idxC"], ["output"], axis=3),
    ]
    inits = [RIr, RIc, big, one, eNot0, ar, ch0s, ch0e, ch0a]
    return _model(nodes, inits)


def _is_cropswap(P):
    if not P:
        return False
    for a, b in P:
        fg = np.argwhere(a != 0)
        if len(fg) == 0:
            return False
        r0, c0 = fg.min(0)
        r1, c1 = fg.max(0)
        crop = a[r0:r1 + 1, c0:c1 + 1]
        cols = sorted(set(crop.ravel()) - {0})
        if len(cols) != 2:
            return False
        if (crop == 0).any():        # solid rectangle, no interior background
            return False
        A, B = cols
        sw = crop.copy()
        sw[crop == A] = B
        sw[crop == B] = A
        if sw.shape != b.shape or not np.array_equal(sw, b):
            return False
    return True


# ===========================================================================
# task 231 : horizontal periodic extension to width 2W (period divides 6)
#   out[r,c] = input[r, c mod 6]   for c < 2W ,  else 0   (rows>=H auto-zero)
# ===========================================================================
_HPER_Q = 6


def _hperiod_model():
    idx = oh.make_tensor("pidx", INT64, [WIDTH], [c % _HPER_Q for c in range(WIDTH)])
    CI = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    CIt = oh.make_tensor("CI", FLOAT, [1, 1, 1, WIDTH], CI.ravel().tolist())
    nodes = [
        # periodic column gather (period 6, fixed indices)
        oh.make_node("Gather", ["input", "pidx"], ["tiled"], axis=3),
        # occupied width W  ->  2W
        oh.make_node("ReduceSum", ["input"], ["colCnt"], axes=[1, 2], keepdims=1),
        oh.make_node("Clip", ["colCnt"], ["colOcc"], min=0.0, max=1.0),
        oh.make_node("ReduceSum", ["colOcc"], ["W"], axes=[3], keepdims=1),
        oh.make_node("Add", ["W", "W"], ["twoW"]),
        # column mask  (c < 2W)
        oh.make_node("Sub", ["twoW", "CI"], ["d"]),
        oh.make_node("Clip", ["d"], ["colMask"], min=0.0, max=1.0),
        oh.make_node("Mul", ["tiled", "colMask"], ["output"]),
    ]
    return _model(nodes, [idx, CIt])


def _is_hperiod(P):
    if not P:
        return False
    for a, b in P:
        H, W = a.shape
        if W < _HPER_Q:
            return False
        if b.shape != (H, 2 * W):
            return False
        out = np.zeros_like(b)
        for r in range(H):
            for c in range(2 * W):
                s = c % _HPER_Q
                out[r, c] = a[r, s] if s < W else 0
        if not np.array_equal(out, b):
            return False
    return True


# ===========================================================================
# task 388 : per-column fill-with-8 (in fg columns) then 2x2 tile -> 2N x 2N
# ===========================================================================
BOOL = onnx.TensorProto.BOOL


def _coltile_model():
    arI = oh.make_tensor("arI", INT64, [HEIGHT], list(range(HEIGHT)))
    RI = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    CI = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    RIt = oh.make_tensor("RI", FLOAT, [1, 1, HEIGHT, 1], RI.ravel().tolist())
    CIt = oh.make_tensor("CI", FLOAT, [1, 1, 1, WIDTH], CI.ravel().tolist())
    e8 = oh.make_tensor("e8", FLOAT, [1, CHANNELS, 1, 1],
                        [1.0 if k == 8 else 0.0 for k in range(CHANNELS)])
    c0s = oh.make_tensor("c0s", INT64, [1], [0])
    c0e = oh.make_tensor("c0e", INT64, [1], [1])
    c0a = oh.make_tensor("c0a", INT64, [1], [1])

    nodes = [
        oh.make_node("Slice", ["input", "c0s", "c0e", "c0a"], ["ch0"]),
        # per-column foreground presence
        oh.make_node("ReduceSum", ["input"], ["colTot"], axes=[1, 2], keepdims=1),
        oh.make_node("ReduceSum", ["ch0"], ["colBg"], axes=[2], keepdims=1),
        oh.make_node("Sub", ["colTot", "colBg"], ["colFg"]),
        oh.make_node("Clip", ["colFg"], ["colHasFg"], min=0.0, max=1.0),
        # cells to recolour: background cell sitting in a foreground column
        oh.make_node("Mul", ["colHasFg", "ch0"], ["fillF"]),
        oh.make_node("Cast", ["fillF"], ["fillB"], to=BOOL),
        oh.make_node("Where", ["fillB", "e8", "input"], ["tile"]),
        # N (square) and dynamic 2x2 tiling indices  i mod N
        oh.make_node("ReduceSum", ["input"], ["rowCnt"], axes=[1, 3], keepdims=1),
        oh.make_node("Clip", ["rowCnt"], ["rowOcc"], min=0.0, max=1.0),
        oh.make_node("ReduceSum", ["rowOcc"], ["Nf"], axes=[2], keepdims=1),
        oh.make_node("Squeeze", ["Nf"], ["Nsq"], axes=[0, 1, 2, 3]),
        oh.make_node("Cast", ["Nsq"], ["Nint"], to=INT64),
        oh.make_node("Mod", ["arI", "Nint"], ["idxMod"]),
        oh.make_node("Gather", ["tile", "idxMod"], ["tiledR"], axis=2),
        oh.make_node("Gather", ["tiledR", "idxMod"], ["tiled"], axis=3),
        # mask away cells with row>=2N or col>=2N
        oh.make_node("Add", ["Nf", "Nf"], ["twoN"]),
        oh.make_node("Sub", ["twoN", "RI"], ["rd"]),
        oh.make_node("Clip", ["rd"], ["rowMask"], min=0.0, max=1.0),
        oh.make_node("Sub", ["twoN", "CI"], ["cd"]),
        oh.make_node("Clip", ["cd"], ["colMask"], min=0.0, max=1.0),
        oh.make_node("Mul", ["rowMask", "colMask"], ["mask2D"]),
        oh.make_node("Mul", ["tiled", "mask2D"], ["output"]),
    ]
    return _model(nodes, [arI, RIt, CIt, e8, c0s, c0e, c0a])


def _is_coltile(P):
    if not P:
        return False
    for a, b in P:
        N = a.shape[0]
        if a.shape != (N, N) or b.shape != (2 * N, 2 * N):
            return False
        tile = a.copy()
        for c in range(N):
            if (a[:, c] != 0).any():
                for r in range(N):
                    if a[r, c] == 0:
                        tile[r, c] = 8
        out = np.zeros((2 * N, 2 * N), int)
        for i in range(2 * N):
            for j in range(2 * N):
                out[i, j] = tile[i % N, j % N]
        if not np.array_equal(out, b):
            return False
    return True


def candidates(examples):
    P = _pairs(examples)
    cands = []
    if _is_xdiag(P):
        cands.append(("xdiag", _xdiag_model()))
    if _is_coltile(P):
        cands.append(("coltile", _coltile_model()))
    if _is_cropswap(P):
        cands.append(("cropswap", _cropswap_model()))
    if _is_hperiod(P):
        cands.append(("hperiod", _hperiod_model()))
    return cands
