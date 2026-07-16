"""family_crk11_2 -- slice unsolved.json[2::3] = tasks
    [23, 54, 79, 96, 133, 170, 201, 233, 264, 349, 367]

Every task in this slice was reverse-engineered from all train/test/arc-gen
pairs.  Ten of the eleven are provably NOT expressible as an opset-10 static,
origin-anchored graph; the eleventh (367) is an interior-fill that is *almost*
expressible but is blocked by the top-left-padding contract (see below).  The
only structurally-expressible rule any of them flirts with is an origin-safe
"recolour background cells boxed in by walls" (four-ray / enclosed-flood).  This
module implements that family and gates every candidate behind an EXACT
self-check on train+test+arc-gen, so it never emits a wrong model.

Per-task findings (why they stay unsolved)
-------------------------------------------
23  (same-shape, cols 5->{2,8}).  Each colour-5 blob is decomposed into a
    fixed 2x3 stamp; the 2x2 sub-block recolours to 8 and the trailing column
    to 2.  The stamp's anchor/orientation is read from the blob geometry
    (per-object tiling) -> data-dependent decomposition, not static.  A plain
    "in-a-2x2-block -> 8 else 2" proxy misses 2-6 cells/pair (verified).
54  (30x30 recolour) inconsistent colour maps across pairs (1->2 and 1->3 in
    one pair; 2->3,3->1,4->1 in another) -> per-object routing keyed on object
    identity.  No global pointwise map.  Not static.
79  (14x14 -> 3x3) extract/summarise scattered objects into a 3x3 code.
    Data-dependent tiny output + counting.  Not static.
96  (var -> var) sub-grid / object extraction, output size varies (7x7,11x11).
    Data-dependent output geometry.  Not static.
133 (same-shape) a KEY/template object dictates a per-object expansion stamped
    around every other object; template differs each pair (data-tensor (x)
    data-tensor correlation, banned).  Not static.
170 (var -> 3x3/4x4) most-common-tile / pattern summary.  Data-dependent size.
201 (13x13 -> var) crop of an embedded pattern, output size varies.  Not static.
233 (var -> var) object extraction/crop, output size varies.  Not static.
264 (var -> 9x9) assemble a fixed-size summary from scattered content.  Not
    static (data-dependent gather).
349 (same-shape) nested concentric frames (3/1) grown around each 9-block plus
    cross-connectors spanning the grid; per-object multi-ring construction with
    grid-spanning rays -> data-dependent geometry.  Not static.
367 (same-shape, fill 0->4) fill the interior of hollow-rectangle "rooms" of
    colour 5.  four-ray (wall in all 4 directions among REAL cells) matches the
    genuine rectangle interiors AND correctly excludes open plus-quadrants, but
    UNDER-fills boxes whose wall is the GRID EDGE (e.g. train0 col-18 boxes):
    filling those needs the data-dependent grid width, which is unavailable
    under the top-left zero-padding contract.  enclosed-flood instead OVER-fills
    large enclosed quadrants (train2: 38 wrong cells).  No exact static graph
    -> self-check rejects, module emits nothing for this task.

The four-ray / enclosed-flood family below is retained (exact-gated, origin-safe)
so any interior-fill task whose rule *is* expressible is still captured, but for
this slice candidates() legitimately yields nothing.
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
# graph plumbing                                                              #
# --------------------------------------------------------------------------- #
def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "crk11_2", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _check(m):
    onnx.checker.check_model(m, full_check=True)
    return m


# --------------------------------------------------------------------------- #
# rule A: four-ray interior fill  (bg cell -> dst iff a wall lies in each of    #
#          the four axis directions, among real cells)                         #
# --------------------------------------------------------------------------- #
def _build_fourray(dst):
    nodes, inits = [], []
    w_wall = np.zeros((1, CHANNELS, 1, 1), np.float32)
    w_wall[0, 1:, 0, 0] = 1.0
    inits.append(oh.make_tensor("w_wall", DATA_TYPE, [1, CHANNELS, 1, 1], w_wall.ravel().tolist()))
    nodes.append(oh.make_node("Conv", ["input", "w_wall"], ["wall"],
                              kernel_shape=[1, 1], pads=[0, 0, 0, 0]))
    U = np.triu(np.ones((WIDTH, WIDTH), np.float32), 1)
    Lmat = np.tril(np.ones((WIDTH, WIDTH), np.float32), -1)
    inits.append(oh.make_tensor("Umat", DATA_TYPE, [WIDTH, WIDTH], U.ravel().tolist()))
    inits.append(oh.make_tensor("Lmat", DATA_TYPE, [WIDTH, WIDTH], Lmat.ravel().tolist()))
    nodes.append(oh.make_node("MatMul", ["wall", "Umat"], ["pL"]))
    nodes.append(oh.make_node("MatMul", ["wall", "Lmat"], ["pR"]))
    TU = np.tril(np.ones((HEIGHT, HEIGHT), np.float32), -1)
    TD = np.triu(np.ones((HEIGHT, HEIGHT), np.float32), 1)
    inits.append(oh.make_tensor("TUmat", DATA_TYPE, [HEIGHT, HEIGHT], TU.ravel().tolist()))
    inits.append(oh.make_tensor("TDmat", DATA_TYPE, [HEIGHT, HEIGHT], TD.ravel().tolist()))
    nodes.append(oh.make_node("MatMul", ["TUmat", "wall"], ["pU"]))
    nodes.append(oh.make_node("MatMul", ["TDmat", "wall"], ["pD"]))
    inits.append(oh.make_tensor("zero", DATA_TYPE, [1], [0.0]))
    for nm in ["pL", "pR", "pU", "pD"]:
        nodes.append(oh.make_node("Greater", [nm, "zero"], [nm + "_b"]))
        nodes.append(oh.make_node("Cast", [nm + "_b"], [nm + "_f"], to=DATA_TYPE))
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
    g[(g == 0) & L & R & U & D] = dst
    return g


# --------------------------------------------------------------------------- #
# rule B: enclosed-hole flood fill (unrolled CA)                              #
# --------------------------------------------------------------------------- #
def _border_mask():
    m = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    m[:, :, 0, :] = 1.0; m[:, :, -1, :] = 1.0
    m[:, :, :, 0] = 1.0; m[:, :, :, -1] = 1.0
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
    # rule A: four-ray interior fill (cheapest) -----------------------------
    if all(np.array_equal(_sim_fourray(a, dst), b) for a, b in prs):
        try:
            out.append((f"fourray_dst{dst}", _build_fourray(dst)))
        except Exception:
            pass
    # rule B: enclosed-hole flood fill --------------------------------------
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
