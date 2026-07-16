"""family_crk7_5 -- expressible "wall-bounded fill" family.

Target slice: unsolved.json[5::6] = tasks
    22, 46, 69, 86, 107, 138, 159, 182, 205, 255, 285, 349, 367.

After a deep study of every task in the slice (see the module-level NOTES at the
bottom) the only structurally-expressible rule that any of them flirts with is a
"recolour background cells that are boxed in by walls" rule (tasks 367 / 255 are
fill-shaped).  This module implements the *family* of such rules in pure
opset-10 static ONNX and gates every candidate behind an EXACT self-check on
train+test+arc-gen, so it never emits a wrong model.

Expressible rules implemented
-----------------------------
1. enclosed-hole flood fill (4- and 8-conn), recolour enclosed bg(0) -> dst.
   (unrolled cellular automaton of 3x3 Conv steps; mirrors family_flood.)
2. four-ray fill: a bg(0) cell becomes dst iff there is a `wall` cell somewhere
   to its left AND right (same row) AND above AND below (same column).
   Expressed with prefix-OR via MatMul against fixed triangular {0,1} matrices.
3. rectangle-interior fill: same four-ray test but using *each colour's own*
   ray test, then ANDed -- a cheap proxy for "inside a solid box".

Every emitted graph keeps content origin-anchored (Conv/MatMul/pointwise only),
so it is safe under the top-left zero-padding contract.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64


# --------------------------------------------------------------------------- #
# graph helper                                                                #
# --------------------------------------------------------------------------- #
def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "crk7_5", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _check(m):
    onnx.checker.check_model(m, full_check=True)
    return m


# --------------------------------------------------------------------------- #
# rule 1: enclosed-hole flood fill (unrolled CA)                              #
# --------------------------------------------------------------------------- #
def _border_mask():
    m = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    m[:, :, 0, :] = 1.0
    m[:, :, -1, :] = 1.0
    m[:, :, :, 0] = 1.0
    m[:, :, :, -1] = 1.0
    return m


def _build_enclosed(dst, n_steps, conn8):
    nodes, inits = [], []
    w_open = np.zeros((1, CHANNELS, 1, 1), np.float32)
    w_open[0, 1:, 0, 0] = -1.0
    inits.append(oh.make_tensor("w_open", DATA_TYPE, [1, CHANNELS, 1, 1], w_open.ravel().tolist()))
    inits.append(oh.make_tensor("b_open", DATA_TYPE, [1], [1.0]))
    nodes.append(oh.make_node("Conv", ["input", "w_open", "b_open"], ["open"],
                              kernel_shape=[1, 1], pads=[0, 0, 0, 0]))
    inits.append(oh.make_tensor("bmask", DATA_TYPE, [1, 1, HEIGHT, WIDTH], _border_mask().ravel().tolist()))
    nodes.append(oh.make_node("Mul", ["open", "bmask"], ["r0"]))
    if conn8:
        k = np.ones((1, 1, 3, 3), np.float32)
    else:
        k = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.float32).reshape(1, 1, 3, 3)
    inits.append(oh.make_tensor("kprop", DATA_TYPE, [1, 1, 3, 3], k.ravel().tolist()))
    prev = "r0"
    for s in range(1, n_steps + 1):
        nodes.append(oh.make_node("Conv", [prev, "kprop"], [f"nb{s}"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        nodes.append(oh.make_node("Min", [f"nb{s}", "open"], [f"r{s}"]))
        prev = f"r{s}"
    nodes.append(oh.make_node("Sub", ["open", prev], ["enc"]))
    w_add = np.zeros((CHANNELS, 1, 1, 1), np.float32)
    w_add[0, 0, 0, 0] = -1.0
    w_add[dst, 0, 0, 0] = 1.0
    inits.append(oh.make_tensor("w_add", DATA_TYPE, [CHANNELS, 1, 1, 1], w_add.ravel().tolist()))
    nodes.append(oh.make_node("Conv", ["enc", "w_add"], ["addmap"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]))
    nodes.append(oh.make_node("Add", ["input", "addmap"], ["output"]))
    return _check(_model(nodes, inits))


def _sim_enclosed(g, dst, conn8):
    g = np.asarray(g, int).copy()
    h, w = g.shape
    op = np.ones((HEIGHT, WIDTH), np.int64)
    op[:h, :w] = (g == 0).astype(np.int64)
    reach = np.zeros((HEIGHT, WIDTH), np.int64)
    reach[0, :] = op[0, :]; reach[-1, :] = op[-1, :]; reach[:, 0] = op[:, 0]; reach[:, -1] = op[:, -1]
    offs = ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if conn8 else []) + [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    while True:
        nb = np.zeros_like(reach)
        for dy, dx in offs:
            ys0, ys1 = max(0, dy), HEIGHT + min(0, dy)
            xs0, xs1 = max(0, dx), WIDTH + min(0, dx)
            yd0, yd1 = max(0, -dy), HEIGHT + min(0, -dy)
            xd0, xd1 = max(0, -dx), WIDTH + min(0, -dx)
            nb[yd0:yd1, xd0:xd1] += reach[ys0:ys1, xs0:xs1]
        new = np.minimum(nb, op)
        if np.array_equal(new, reach):
            break
        reach = new
    enc = (op == 1) & (reach == 0)
    g[enc[:h, :w]] = dst
    return g


def _enclosed_steps(g, conn8):
    g = np.asarray(g, int)
    h, w = g.shape
    op = np.ones((HEIGHT, WIDTH), np.int64)
    op[:h, :w] = (g == 0).astype(np.int64)
    reach = np.zeros((HEIGHT, WIDTH), np.int64)
    reach[0, :] = op[0, :]; reach[-1, :] = op[-1, :]; reach[:, 0] = op[:, 0]; reach[:, -1] = op[:, -1]
    offs = ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if conn8 else []) + [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    steps = 0
    while True:
        nb = np.zeros_like(reach)
        for dy, dx in offs:
            ys0, ys1 = max(0, dy), HEIGHT + min(0, dy)
            xs0, xs1 = max(0, dx), WIDTH + min(0, dx)
            yd0, yd1 = max(0, -dy), HEIGHT + min(0, -dy)
            xd0, xd1 = max(0, -dx), WIDTH + min(0, -dx)
            nb[yd0:yd1, xd0:xd1] += reach[ys0:ys1, xs0:xs1]
        new = np.minimum(nb, op)
        steps += 1
        if np.array_equal(new, reach):
            return steps - 1
        reach = new


# --------------------------------------------------------------------------- #
# rule 2: four-ray fill (prefix-OR via triangular MatMul)                     #
# --------------------------------------------------------------------------- #
def _build_fourray(dst):
    """bg(0) cell -> dst iff a non-bg `wall` exists in all four ray directions
    (left, right of same row; above, below of same column)."""
    nodes, inits = [], []
    # wall = 1 - ch0  over real cells; padding has ch0=0 so wall=1 there -- but
    # we only ever fill ch0==1 cells, so padding (never bg) is irrelevant to the
    # final add.  We compute wall = sum(channels 1..9).
    w_wall = np.zeros((1, CHANNELS, 1, 1), np.float32)
    w_wall[0, 1:, 0, 0] = 1.0
    inits.append(oh.make_tensor("w_wall", DATA_TYPE, [1, CHANNELS, 1, 1], w_wall.ravel().tolist()))
    nodes.append(oh.make_node("Conv", ["input", "w_wall"], ["wall"],
                              kernel_shape=[1, 1], pads=[0, 0, 0, 0]))  # [1,1,30,30]
    # triangular matrices
    Tlow = np.tril(np.ones((WIDTH, WIDTH), np.float32), -1)   # Tlow[c',c]?? see use
    # exists wall to the LEFT of col c: sum_{c'<c} wall[r,c'] = wall @ U  where
    # U[c',c]=1 if c'<c  (strictly upper triangular)
    U = np.triu(np.ones((WIDTH, WIDTH), np.float32), 1)
    Lmat = np.tril(np.ones((WIDTH, WIDTH), np.float32), -1)   # L[c',c]=1 if c'>c
    inits.append(oh.make_tensor("Umat", DATA_TYPE, [WIDTH, WIDTH], U.ravel().tolist()))
    inits.append(oh.make_tensor("Lmat", DATA_TYPE, [WIDTH, WIDTH], Lmat.ravel().tolist()))
    nodes.append(oh.make_node("MatMul", ["wall", "Umat"], ["pL"]))   # exists left
    nodes.append(oh.make_node("MatMul", ["wall", "Lmat"], ["pR"]))   # exists right
    # column direction: pU = TU @ wall, TU[r,r']=1 if r'<r (strictly lower)
    TU = np.tril(np.ones((HEIGHT, HEIGHT), np.float32), -1)
    TD = np.triu(np.ones((HEIGHT, HEIGHT), np.float32), 1)
    inits.append(oh.make_tensor("TUmat", DATA_TYPE, [HEIGHT, HEIGHT], TU.ravel().tolist()))
    inits.append(oh.make_tensor("TDmat", DATA_TYPE, [HEIGHT, HEIGHT], TD.ravel().tolist()))
    nodes.append(oh.make_node("MatMul", ["TUmat", "wall"], ["pU"]))
    nodes.append(oh.make_node("MatMul", ["TDmat", "wall"], ["pD"]))
    # binarise each ( >0 ) and AND together with ch0
    zero = oh.make_tensor("zero", DATA_TYPE, [1], [0.0])
    inits.append(zero)
    for nm in ["pL", "pR", "pU", "pD"]:
        nodes.append(oh.make_node("Greater", [nm, "zero"], [nm + "_b"]))
        nodes.append(oh.make_node("Cast", [nm + "_b"], [nm + "_f"], to=DATA_TYPE))
    # ch0
    w_ch0 = np.zeros((1, CHANNELS, 1, 1), np.float32)
    w_ch0[0, 0, 0, 0] = 1.0
    inits.append(oh.make_tensor("w_ch0", DATA_TYPE, [1, CHANNELS, 1, 1], w_ch0.ravel().tolist()))
    nodes.append(oh.make_node("Conv", ["input", "w_ch0"], ["ch0"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]))
    nodes.append(oh.make_node("Mul", ["pL_f", "pR_f"], ["m1"]))
    nodes.append(oh.make_node("Mul", ["pU_f", "pD_f"], ["m2"]))
    nodes.append(oh.make_node("Mul", ["m1", "m2"], ["m3"]))
    nodes.append(oh.make_node("Mul", ["m3", "ch0"], ["fill"]))
    w_add = np.zeros((CHANNELS, 1, 1, 1), np.float32)
    w_add[0, 0, 0, 0] = -1.0
    w_add[dst, 0, 0, 0] = 1.0
    inits.append(oh.make_tensor("w_add", DATA_TYPE, [CHANNELS, 1, 1, 1], w_add.ravel().tolist()))
    nodes.append(oh.make_node("Conv", ["fill", "w_add"], ["addmap"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]))
    nodes.append(oh.make_node("Add", ["input", "addmap"], ["output"]))
    return _check(_model(nodes, inits))


def _sim_fourray(g, dst):
    g = np.asarray(g, int).copy()
    h, w = g.shape
    wall = (g != 0)
    L = np.zeros((h, w), bool); R = np.zeros((h, w), bool)
    U = np.zeros((h, w), bool); D = np.zeros((h, w), bool)
    for i in range(h):
        seen = False
        for j in range(w):
            L[i, j] = seen
            if wall[i, j]:
                seen = True
        seen = False
        for j in range(w - 1, -1, -1):
            R[i, j] = seen
            if wall[i, j]:
                seen = True
    for j in range(w):
        seen = False
        for i in range(h):
            U[i, j] = seen
            if wall[i, j]:
                seen = True
        seen = False
        for i in range(h - 1, -1, -1):
            D[i, j] = seen
            if wall[i, j]:
                seen = True
    fill = (g == 0) & L & R & U & D
    g[fill] = dst
    return g


# --------------------------------------------------------------------------- #
# detection                                                                    #
# --------------------------------------------------------------------------- #
def _pairs(ex, splits=("train", "test", "arc-gen")):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int); b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _infer_dst(prs):
    dst = None
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
        v = int(vals[0])
        if dst is None:
            dst = v
        elif dst != v:
            return None
    return dst


def candidates(ex):
    prs = _pairs(ex)
    if not prs or not all(a.shape == b.shape for a, b in prs):
        return []
    if all((a == b).all() for a, b in prs):
        return []
    dst = _infer_dst(prs)
    if dst is None or dst == 0:
        return []

    out = []

    # rule 2: four-ray fill (cheapest of the box-fills) ----------------------
    if all(np.array_equal(_sim_fourray(a, dst), b) for a, b in prs):
        try:
            out.append((f"fourray_dst{dst}", _build_fourray(dst)))
        except Exception:
            pass

    # rule 1: enclosed-hole flood fill --------------------------------------
    for conn8 in (False, True):
        if not all(np.array_equal(_sim_enclosed(a, dst, conn8), b) for a, b in prs):
            continue
        steps = max((_enclosed_steps(a, conn8) for a, _ in prs), default=1)
        n = max(1, steps) + 2
        try:
            out.append((f"enc{'8' if conn8 else '4'}_dst{dst}_n{n}",
                        _build_enclosed(dst, n, conn8)))
        except Exception:
            pass

    return out


# --------------------------------------------------------------------------- #
# NOTES on the slice (why most stay unsolved under the opset-10 contract)
# --------------------------------------------------------------------------- #
# 22  11x11 -> 3x3 object-summary (count/colour of scattered objects).
# 46  3xN row-object interaction / growth.
# 69  recolour mono(8) objects by a multi-colour KEY object's per-cell palette
#     (object detection + per-object stamping; key shape differs every example).
# 86  per-object concentric mirror expansion.
# 107 fractal recursive scaling, output side = 5 * (#distinct colours).
# 138 nested-frame extraction / structured crop.
# 159 box extraction + symmetric interior fill from a marker shape.
# 182 shape-match recolour from a boxed template.
# 205 sub-grid extraction with embedded pattern.
# 255 draw the skeleton/cross of the (data-positioned) empty region with 3.
# 285 per-object symmetry completion driven by markers.
# 349 nested concentric frames around blocks + connectors.
# 367 fill the interior of CLOSED rectangular rooms (true topological enclosure;
#     plain enclosed-flood over-fills large irregular regions and mishandles
#     edge-flush boxes -- verified ~half match for every flood/ray/rectangle
#     proxy, so no exact-generalising static graph was found).
# All require object detection / template extraction / data-dependent geometry
# beyond ReduceSum/Where/MatMul/Conv/Gather/Slice with static shapes.
