"""family_crk9_2 -- hard ARC tasks (slice U[2::7]).

Branches implemented:
  * local_recolor : output == input except some background(0) cells become a single
    colour C, where mark(i,j) is a deterministic function of a fixed KxW local window
    of the background mask.  Detected by fitting a window LUT with zero conflicts over
    all train+test+arc-gen pairs, then realised in ONNX as
        bg   = ch0(input)                                  (1x1 Conv)
        code = Conv(bg, powers-of-two kernel)              (integer per-cell code)
        mark = Gather(table, code) * bg                    (LUT lookup, clamp to bg)
        out  = input + mark*(e_C - e_0)
    (covers task 265: every cell in a 2x2 all-background block -> colour 2, modulo a
     non-overlap tie-break that the local window captures exactly.)
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64


def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "crk9_2", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _pairs(ex, splits=("train", "test", "arc-gen")):
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


# --------------------------------------------------------------------------- #
# local-window recolour                                                       #
# --------------------------------------------------------------------------- #
# window = rows {-1,0,1} x cols {-2,-1,0,1}  (3 tall, 4 wide, cell at (1,2))
_WIN = [(dy, dx) for dy in (-1, 0, 1) for dx in (-2, -1, 0, 1)]
_KH, _KW = 3, 4
_PT, _PL, _PB, _PR = 1, 2, 1, 1
# center cell offset within flattened window (dy=0,dx=0) -> p=1,q=2 -> 1*4+2 = 6
_CENTER_BIT = 6


def _window_code(bg):
    """integer code per cell exactly matching the ONNX power-of-two Conv."""
    h, w = bg.shape
    code = np.zeros((h, w), np.int64)
    for k, (dy, dx) in enumerate(_WIN):
        # value at (i,j) uses bg[i+dy, j+dx]; out-of-range -> 0
        ys0, ys1 = max(0, dy), h - max(0, -dy)
        xs0, xs1 = max(0, dx), w - max(0, -dx)
        yd0, yd1 = max(0, -dy), h - max(0, dy)
        xd0, xd1 = max(0, -dx), w - max(0, dx)
        shifted = np.zeros((h, w), np.int64)
        shifted[yd0:yd1, xd0:xd1] = bg[ys0:ys1, xs0:xs1]
        code += shifted * (1 << k)
    return code


def _fit_lut(prs):
    """Return (C, table[4096]) if a zero-conflict window LUT reproduces every pair
    (only 0->C changes), else None."""
    # all changes must be 0 -> a single colour C
    C = None
    for a, b in prs:
        if a.shape != b.shape:
            return None
        d = a != b
        if not d.any():
            continue
        if (a[d] != 0).any():
            return None
        vals = np.unique(b[d])
        if vals.size != 1:
            return None
        c = int(vals[0])
        if C is None:
            C = c
        elif C != c:
            return None
    if C is None or C == 0:
        return None

    lut = {}
    for a, b in prs:
        bg = (a == 0).astype(np.int64)
        mark = ((b == C) & (a == 0)).astype(np.int64)
        code = _window_code(bg)
        h, w = a.shape
        for i in range(h):
            for j in range(w):
                if a[i, j] != 0:
                    continue
                key = int(code[i, j])
                m = int(mark[i, j])
                if key in lut:
                    if lut[key] != m:
                        return None
                else:
                    lut[key] = m

    size = 1 << len(_WIN)
    table = np.zeros((size,), np.float32)
    # fallback for unseen codes: the "in some 2x2 all-zero block" rule, decoded
    # from the window bits (center must be bg).
    for code in range(size):
        bits = [(code >> k) & 1 for k in range(len(_WIN))]
        if code in lut:
            table[code] = lut[code]
            continue
        # 2x2 fallback: is the center bg-cell part of an all-zero 2x2 window?
        b = {off: bits[k] for k, off in enumerate(_WIN)}
        if b[(0, 0)] == 0:
            table[code] = 0.0
            continue
        m = 0
        for dy in (-1, 0):
            for dx in (-1, 0):
                cells = [(dy, dx), (dy, dx + 1), (dy + 1, dx), (dy + 1, dx + 1)]
                if all(c in b for c in cells) and all(b[c] == 1 for c in cells):
                    m = 1
        table[code] = float(m)
    return C, table


def _sim_lut(a, C, table):
    bg = (a == 0).astype(np.int64)
    code = _window_code(bg)
    mark = table[code]
    out = a.copy()
    out[(mark > 0.5) & (a == 0)] = C
    return out


def _build_lut(C, table):
    nodes, inits = [], []
    # bg = channel 0
    w_ch0 = np.zeros((1, CHANNELS, 1, 1), np.float32)
    w_ch0[0, 0, 0, 0] = 1.0
    inits.append(oh.make_tensor("w_ch0", DATA_TYPE, [1, CHANNELS, 1, 1], w_ch0.ravel().tolist()))
    nodes.append(oh.make_node("Conv", ["input", "w_ch0"], ["bg"],
                              kernel_shape=[1, 1], pads=[0, 0, 0, 0]))

    # code = Conv(bg, powers-of-two kernel 3x4)
    wcode = np.zeros((1, 1, _KH, _KW), np.float32)
    for k, (dy, dx) in enumerate(_WIN):
        p, q = dy + 1, dx + 2
        wcode[0, 0, p, q] = float(1 << k)
    inits.append(oh.make_tensor("wcode", DATA_TYPE, [1, 1, _KH, _KW], wcode.ravel().tolist()))
    nodes.append(oh.make_node("Conv", ["bg", "wcode"], ["codef"],
                              kernel_shape=[_KH, _KW], pads=[_PT, _PL, _PB, _PR]))
    nodes.append(oh.make_node("Cast", ["codef"], ["codei"], to=INT64))

    # table gather -> mark
    inits.append(oh.make_tensor("table", DATA_TYPE, [table.shape[0]], table.tolist()))
    nodes.append(oh.make_node("Gather", ["table", "codei"], ["mark0"], axis=0))
    # clamp to bg cells
    nodes.append(oh.make_node("Mul", ["mark0", "bg"], ["mark"]))

    # addmap = mark * (e_C - e_0) via 1x1 conv on mark
    wadd = np.zeros((CHANNELS, 1, 1, 1), np.float32)
    wadd[0, 0, 0, 0] = -1.0
    wadd[C, 0, 0, 0] = 1.0
    inits.append(oh.make_tensor("wadd", DATA_TYPE, [CHANNELS, 1, 1, 1], wadd.ravel().tolist()))
    nodes.append(oh.make_node("Conv", ["mark", "wadd"], ["addmap"],
                              kernel_shape=[1, 1], pads=[0, 0, 0, 0]))
    nodes.append(oh.make_node("Add", ["input", "addmap"], ["output"]))
    return _model(nodes, inits)


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    # ---- local-window recolour ------------------------------------------------
    if all(a.shape == b.shape for a, b in prs) and not all((a == b).all() for a, b in prs):
        fit = _fit_lut(prs)
        if fit is not None:
            C, table = fit
            if all(np.array_equal(_sim_lut(a, C, table), b) for a, b in prs):
                try:
                    out.append((f"localrecolor_c{C}", _build_lut(C, table)))
                except Exception:
                    pass

    return out
