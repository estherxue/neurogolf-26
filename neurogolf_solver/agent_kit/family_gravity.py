"""Gravity family: non-background cells 'fall' toward one edge (up/down/left/right)
until packed against the edge or another cell, preserving their relative order
(stable gravity).

Realized as a FIXED unroll of K identical single-cell CA steps (Conv->Relu->Conv->Add),
one per "fall by one" move.  Each step is a *local* rule over a 3x1 (vertical) or 1x3
(horizontal) neighborhood:

    A_k(p)   = relu( T_k(p) + T_bg(p_toward_edge) - 1 )   # cell colour k can move 1 step
    T'_k     = T_k - A_k + A_k(p_away_from_edge)          # k leaves / k arrives
    T'_bg    = T_bg + sum_k A_k - sum_k A_k(away)         # background complement

`A_k` reads the BACKGROUND channel of the neighbour toward the edge; conv's zero
padding makes the out-of-grid neighbour read 0 == "occupied wall", so cells never
leave the grid -> the edge acts as a wall for free.  All ten channels are conserved
each step, so padding cells (all-zero) stay all-zero automatically.

Origin safety
-------------
* UP   : wall is row 0 (always the tensor top)   -> origin-safe for any grid, full 30x30.
* LEFT : wall is col 0 (always the tensor left)   -> origin-safe for any grid, full 30x30.
* DOWN : wall is row h-1 (data-dependent).  Only expressible when the grid HEIGHT is
         constant across all examples: Slice to [0:H], run the CA (zero-pad below row
         H-1 acts as the wall), Pad back to 30.
* RIGHT: symmetric to DOWN, needs constant WIDTH.

The exact rule + step count is verified by the harness, which rejects any mis-detected
direction / background / step budget.
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
# model scaffold                                                              #
# --------------------------------------------------------------------------- #
def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# numpy reference gravity + CA settle-time                                     #
# --------------------------------------------------------------------------- #
def _gravity(grid, direction, bg):
    """Stable gravity (preserve order) toward `direction`, background colour bg."""
    h, w = grid.shape
    out = np.full((h, w), bg, dtype=grid.dtype)
    if direction in ("up", "down"):
        for c in range(w):
            col = grid[:, c]
            objs = col[col != bg]
            if direction == "up":
                out[:len(objs), c] = objs
            else:
                out[h - len(objs):, c] = objs
    else:
        for r in range(h):
            row = grid[r, :]
            objs = row[row != bg]
            if direction == "left":
                out[r, :len(objs)] = objs
            else:
                out[r, w - len(objs):] = objs
    return out


def _ca_step(grid, direction, bg):
    """One synchronous single-cell move toward `direction` (matches the ONNX step)."""
    new = grid.copy()
    occ = grid != bg
    if direction == "up":
        m = np.zeros_like(occ); m[1:, :] = occ[1:, :] & (grid[:-1, :] == bg)
        new[:-1, :] = np.where(m[1:, :], grid[1:, :], new[:-1, :]); new[m] = bg
    elif direction == "down":
        m = np.zeros_like(occ); m[:-1, :] = occ[:-1, :] & (grid[1:, :] == bg)
        new[1:, :] = np.where(m[:-1, :], grid[:-1, :], new[1:, :]); new[m] = bg
    elif direction == "left":
        m = np.zeros_like(occ); m[:, 1:] = occ[:, 1:] & (grid[:, :-1] == bg)
        new[:, :-1] = np.where(m[:, 1:], grid[:, 1:], new[:, :-1]); new[m] = bg
    else:  # right
        m = np.zeros_like(occ); m[:, :-1] = occ[:, :-1] & (grid[:, 1:] == bg)
        new[:, 1:] = np.where(m[:, :-1], grid[:, :-1], new[:, 1:]); new[m] = bg
    return new


def _settle_steps(grid, direction, bg, cap):
    """Number of CA steps until the configuration stops changing (<= cap)."""
    g = grid.copy()
    for s in range(cap):
        n = _ca_step(g, direction, bg)
        if np.array_equal(n, g):
            return s
        g = n
    return cap


# --------------------------------------------------------------------------- #
# conv weight construction for one gravity step                               #
# --------------------------------------------------------------------------- #
def _step_weights(direction, bg):
    """Return (W1, b1, W2, kh, kw, pads) for a single gravity step.

    Kernel index convention (pad = size//2):
      vertical  kh=3: idx0 = row r-1 (above), idx1 = r (center), idx2 = r+1 (below)
      horizontal kw=3: idx0 = col c-1 (left),  idx1 = c (center), idx2 = c+1 (right)
    """
    nb = [c for c in range(CHANNELS) if c != bg]  # 9 non-background colours
    if direction in ("up", "down"):
        kh, kw, pads = 3, 1, [1, 0, 1, 0]
        center = (1, 0)
        toward = (0, 0) if direction == "up" else (2, 0)   # neighbour the cell falls TOWARD
        away = (2, 0) if direction == "up" else (0, 0)     # neighbour a cell arrives FROM
    else:
        kh, kw, pads = 1, 3, [0, 1, 0, 1]
        center = (0, 1)
        toward = (0, 0) if direction == "left" else (0, 2)
        away = (0, 2) if direction == "left" else (0, 0)

    # conv1: T(10) -> preA(9);  A_j = relu(T_{nb[j]}(center) + T_bg(toward) - 1)
    W1 = np.zeros((9, CHANNELS, kh, kw), np.float32)
    b1 = np.full((9,), -1.0, np.float32)
    for j, k in enumerate(nb):
        W1[j, k, center[0], center[1]] = 1.0
        W1[j, bg, toward[0], toward[1]] = 1.0

    # conv2: A(9) -> delta(10);  d_k = -A_j(center) + A_j(away);  d_bg = +sumA(center) - sumA(away)
    W2 = np.zeros((CHANNELS, 9, kh, kw), np.float32)
    for j, k in enumerate(nb):
        W2[k, j, center[0], center[1]] = -1.0
        W2[k, j, away[0], away[1]] = 1.0
        W2[bg, j, center[0], center[1]] = 1.0
        W2[bg, j, away[0], away[1]] = -1.0
    return W1, b1, W2, kh, kw, pads


def _build(direction, bg, steps, size=None):
    """Build the unrolled-gravity ONNX model.

    `size` = (H, W) when the grid size is constant across all examples.  In that
    case the CA runs on the Sliced [0:H, 0:W] region so (a) the grid edge -- not
    the tensor edge -- is the conv's zero-pad wall (makes down/right realizable),
    and (b) intermediates are HxW instead of 30x30 (far cheaper).  When `size` is
    None the CA runs on the full 30x30 tensor (only valid for up/left).
    """
    W1, b1, W2, kh, kw, pads = _step_weights(direction, bg)
    inits = [
        oh.make_tensor("W1", DATA_TYPE, list(W1.shape), W1.ravel().tolist()),
        oh.make_tensor("b1", DATA_TYPE, [9], b1.tolist()),
        oh.make_tensor("W2", DATA_TYPE, list(W2.shape), W2.ravel().tolist()),
    ]
    nodes = []

    cur = "input"
    if size is not None:
        H, W = size
        inits += [
            oh.make_tensor("cs", INT64, [2], [0, 0]),
            oh.make_tensor("ce", INT64, [2], [H, W]),
            oh.make_tensor("ca", INT64, [2], [2, 3]),
        ]
        nodes.append(oh.make_node("Slice", ["input", "cs", "ce", "ca"], ["g0"]))
        cur = "g0"

    for i in range(steps):
        a_pre = f"apre{i}"
        a = f"a{i}"
        delta = f"d{i}"
        out = f"g_{i}"
        nodes.append(oh.make_node("Conv", [cur, "W1", "b1"], [a_pre],
                                  kernel_shape=[kh, kw], pads=pads))
        nodes.append(oh.make_node("Relu", [a_pre], [a]))
        nodes.append(oh.make_node("Conv", [a, "W2"], [delta],
                                  kernel_shape=[kh, kw], pads=pads))
        nodes.append(oh.make_node("Add", [cur, delta], [out]))
        cur = out

    if size is not None:
        H, W = size
        nodes[-1].output[0] = "settled"
        nodes.append(oh.make_node("Pad", ["settled"], ["output"],
                                  mode="constant", value=0.0,
                                  pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W]))
    else:
        nodes[-1].output[0] = "output"

    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# detection / entry point                                                     #
# --------------------------------------------------------------------------- #
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


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    # gravity preserves grid shape; at least one example must actually move.
    if not all(a.shape == b.shape for a, b in prs):
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []

    colors = sorted({int(v) for a, _ in prs for v in np.unique(a)})
    shapes = {a.shape for a, _ in prs}
    const_size = next(iter(shapes)) if len(shapes) == 1 else None  # (H, W) or None

    # Direction realizability:
    #   * constant size -> Slice/Pad the grid region: all four directions work and
    #     intermediates are the (small) grid size.
    #   * variable size -> only up/left are origin-anchored (full 30x30 tensor).
    directions = ("up", "down", "left", "right") if const_size else ("up", "left")

    out = []
    seen = set()
    for direction in directions:
        for bg in colors:
            if not all(np.array_equal(_gravity(a, direction, bg), b) for a, b in prs):
                continue
            # require the rule to be non-trivial (something actually falls somewhere)
            if all(np.array_equal(_gravity(a, direction, bg), a) for a, b in prs):
                continue
            if const_size is not None:
                H, W = const_size
                cap = (H if direction in ("up", "down") else W) - 1
            else:
                cap = (HEIGHT if direction in ("up", "down") else WIDTH) - 1
            k = max((_settle_steps(a, direction, bg, cap + 1) for a, _ in prs),
                    default=0)
            steps = max(1, min(cap, k + 1))   # +1 margin, clamped to wall distance
            key = (direction, bg)
            if key in seen:
                continue
            seen.add(key)
            try:
                model = _build(direction, bg, steps, const_size)
            except Exception:
                continue
            tag = f"grav_{direction}_bg{bg}_k{steps}"
            out.append((tag, model))

    return out
