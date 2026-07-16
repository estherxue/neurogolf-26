"""family_crk9_0 — a small bundle of exact, generalising ARC->ONNX solvers built
with the advanced arsenal (computed linear maps, per-candidate search, diagonal
tiling).  candidates() tries each sub-family's structural gate and emits the
matching model.

Sub-families
------------
* crk9_concentric  (task 392): the input has two colours {0, X}; the X cells are
  (clipped) concentric square frames.  rho = max(|r-cr|,|c-cc|); a cell is X iff
  (rho-phi) % s == 0.  Output renders ALL frames (bg 0 -> 5).  The graph searches
  every doubled-integer centre, derives phase/spacing per candidate, scores the
  clipped reconstruction against the input and renders the best centre.
* crk9_diag       (task 260): the input has colours {0, 5, X}; X is a single
  "\\" diagonal (constant c-r = m) and the 5 cells form bands adjacent to it.
  Output keeps the marker diagonal and draws a parallel X diagonal just beyond
  each 5 band (at  max5above+2  /  min5below-2 ), dropping the 5s.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
FLOAT = onnx.TensorProto.FLOAT
GRID = HEIGHT
NC = 2 * GRID
BIG = 1.0e7


def _const(name, arr, dtype=FLOAT):
    arr = np.asarray(arr)
    return oh.make_tensor(name, dtype, list(arr.shape), arr.flatten().tolist())


def _finish(nodes, inits, name):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, name, [x], [y], inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# ============================================================ concentric frames
def _solve_concentric(I):
    H, W = I.shape
    nz = [c for c in np.unique(I) if c != 0]
    if len(nz) != 1:
        return None
    X = int(nz[0])
    big = np.zeros((GRID, GRID), float); big[:H, :W] = I
    grid = np.zeros((GRID, GRID), float); grid[:H, :W] = 1.0
    M = ((big == X) & (grid > 0.5)).astype(float)
    if M.sum() == 0:
        return None
    R = 2.0 * np.arange(GRID); C = 2.0 * np.arange(GRID)
    cr2 = np.arange(NC, dtype=float); cc2 = np.arange(NC, dtype=float)
    drow = np.abs(R[None, :] - cr2[:, None])
    dcol = np.abs(C[None, :] - cc2[:, None])
    rho = np.maximum(drow[:, None, :, None], dcol[None, :, None, :])
    Mb = M[None, None] > 0.5
    gridb = grid[None, None] > 0.5
    phi2 = np.where(Mb, rho, BIG).reshape(NC, NC, -1).min(2)[:, :, None, None]
    sel = (rho > phi2 + 0.5) & Mb
    second = np.where(sel, rho, BIG).reshape(NC, NC, -1).min(2)[:, :, None, None]
    s2 = second - phi2
    rmax = np.where(Mb, rho, -1.0).reshape(NC, NC, -1).max(2)[:, :, None, None]
    on = np.abs(np.mod(rho - phi2, s2)) < 0.5
    predclip = on & (rho < rmax + 0.5) & gridb
    mm = (predclip != Mb).sum((2, 3))
    bi = int(np.argmax(-mm.reshape(-1)))
    a, b = bi // NC, bi % NC
    raw = np.abs(np.mod(rho[a, b] - phi2[a, b, 0, 0], s2[a, b, 0, 0])) < 0.5
    out = np.where(raw & (grid > 0.5), X, 5)
    out = np.where(grid > 0.5, out, 0)
    return out[:H, :W].astype(int)


def _build_concentric():
    nodes, inits = [], []

    def C(name, arr, dtype=FLOAT):
        inits.append(_const(name, arr, dtype)); return name

    nodes.append(oh.make_node("ReduceSum", ["input"], ["grid_mask"], axes=[1], keepdims=1))
    C("ax0_s", [0], INT64); C("ax0_e", [1], INT64); C("ax0_a", [1], INT64)
    nodes.append(oh.make_node("Slice", ["input", "ax0_s", "ax0_e", "ax0_a"], ["ch0"]))
    nodes.append(oh.make_node("Sub", ["grid_mask", "ch0"], ["M"]))
    C("half", [0.5])
    nodes.append(oh.make_node("Greater", ["M", "half"], ["Mbool"]))
    nodes.append(oh.make_node("Greater", ["grid_mask", "half"], ["gridbool"]))
    nodes.append(oh.make_node("ReduceSum", ["input"], ["chan_sum"], axes=[2, 3], keepdims=1))
    nodes.append(oh.make_node("Greater", ["chan_sum", "half"], ["chan_pos"]))
    nodes.append(oh.make_node("Cast", ["chan_pos"], ["chan_pos_f"], to=FLOAT))
    C("notch0", np.array([0.] + [1.] * 9).reshape(1, CHANNELS, 1, 1))
    nodes.append(oh.make_node("Mul", ["chan_pos_f", "notch0"], ["e_Xc"]))
    e5 = np.zeros((1, CHANNELS, 1, 1)); e5[0, 5, 0, 0] = 1.0
    C("e5", e5)

    Rcoord = (2.0 * np.arange(GRID)).reshape(1, 1, GRID, 1)
    Ccoord = (2.0 * np.arange(GRID)).reshape(1, 1, 1, GRID)
    cr2c = np.arange(NC, dtype=float).reshape(NC, 1, 1, 1)
    cc2c = np.arange(NC, dtype=float).reshape(1, NC, 1, 1)
    C("Rcoord", Rcoord); C("Ccoord", Ccoord); C("cr2c", cr2c); C("cc2c", cc2c)
    nodes.append(oh.make_node("Sub", ["Rcoord", "cr2c"], ["drow_s"]))
    nodes.append(oh.make_node("Abs", ["drow_s"], ["drow"]))
    nodes.append(oh.make_node("Sub", ["Ccoord", "cc2c"], ["dcol_s"]))
    nodes.append(oh.make_node("Abs", ["dcol_s"], ["dcol"]))
    nodes.append(oh.make_node("Max", ["drow", "dcol"], ["rho"]))

    C("BIG", [BIG]); C("negone", [-1.0])
    nodes.append(oh.make_node("Where", ["Mbool", "rho", "BIG"], ["rad_at_M"]))
    nodes.append(oh.make_node("ReduceMin", ["rad_at_M"], ["phi2"], axes=[2, 3], keepdims=1))
    nodes.append(oh.make_node("Add", ["phi2", "half"], ["phi2_h"]))
    nodes.append(oh.make_node("Greater", ["rho", "phi2_h"], ["gt"]))
    nodes.append(oh.make_node("And", ["gt", "Mbool"], ["sel"]))
    nodes.append(oh.make_node("Where", ["sel", "rho", "BIG"], ["rad_gt"]))
    nodes.append(oh.make_node("ReduceMin", ["rad_gt"], ["second"], axes=[2, 3], keepdims=1))
    nodes.append(oh.make_node("Sub", ["second", "phi2"], ["s2"]))
    nodes.append(oh.make_node("Where", ["Mbool", "rho", "negone"], ["rad_or_neg"]))
    nodes.append(oh.make_node("ReduceMax", ["rad_or_neg"], ["rmax"], axes=[2, 3], keepdims=1))

    nodes.append(oh.make_node("Sub", ["rho", "phi2"], ["g"]))
    nodes.append(oh.make_node("Mod", ["g", "s2"], ["modg"], fmod=1))
    nodes.append(oh.make_node("Abs", ["modg"], ["amodg"]))
    nodes.append(oh.make_node("Less", ["amodg", "half"], ["on"]))
    nodes.append(oh.make_node("Add", ["rmax", "half"], ["rmax_h"]))
    nodes.append(oh.make_node("Less", ["rho", "rmax_h"], ["clip"]))
    nodes.append(oh.make_node("And", ["on", "clip"], ["pc1"]))
    nodes.append(oh.make_node("And", ["pc1", "gridbool"], ["predclip"]))
    nodes.append(oh.make_node("Xor", ["predclip", "Mbool"], ["mm_cell"]))
    nodes.append(oh.make_node("Cast", ["mm_cell"], ["mm_cell_f"], to=FLOAT))
    nodes.append(oh.make_node("ReduceSum", ["mm_cell_f"], ["mm"], axes=[2, 3], keepdims=0))
    nodes.append(oh.make_node("Neg", ["mm"], ["score"]))
    C("flatNN", [NC * NC], INT64)
    nodes.append(oh.make_node("Reshape", ["score", "flatNN"], ["score_flat"]))
    nodes.append(oh.make_node("ArgMax", ["score_flat"], ["best"], axis=0, keepdims=1))

    C("rho_shape", [NC * NC, GRID, GRID], INT64)
    nodes.append(oh.make_node("Reshape", ["rho", "rho_shape"], ["rho_flat"]))
    nodes.append(oh.make_node("Gather", ["rho_flat", "best"], ["rho_best"], axis=0))
    nodes.append(oh.make_node("Reshape", ["phi2", "flatNN"], ["phi2_flat"]))
    nodes.append(oh.make_node("Gather", ["phi2_flat", "best"], ["phi2_b"], axis=0))
    nodes.append(oh.make_node("Reshape", ["s2", "flatNN"], ["s2_flat"]))
    nodes.append(oh.make_node("Gather", ["s2_flat", "best"], ["s2_b"], axis=0))
    C("sc111", [1, 1, 1], INT64)
    nodes.append(oh.make_node("Reshape", ["phi2_b", "sc111"], ["phi2_b3"]))
    nodes.append(oh.make_node("Reshape", ["s2_b", "sc111"], ["s2_b3"]))
    nodes.append(oh.make_node("Sub", ["rho_best", "phi2_b3"], ["gb"]))
    nodes.append(oh.make_node("Mod", ["gb", "s2_b3"], ["modb"], fmod=1))
    nodes.append(oh.make_node("Abs", ["modb"], ["amodb"]))
    nodes.append(oh.make_node("Less", ["amodb", "half"], ["rawon"]))
    nodes.append(oh.make_node("Cast", ["rawon"], ["rawon_f"], to=FLOAT))
    C("sh1130", [1, 1, GRID, GRID], INT64)
    nodes.append(oh.make_node("Reshape", ["rawon_f", "sh1130"], ["rawon4"]))
    nodes.append(oh.make_node("Mul", ["rawon4", "grid_mask"], ["on_f"]))
    nodes.append(oh.make_node("Sub", ["grid_mask", "on_f"], ["off_f"]))
    nodes.append(oh.make_node("Mul", ["on_f", "e_Xc"], ["term1"]))
    nodes.append(oh.make_node("Mul", ["off_f", "e5"], ["term2"]))
    nodes.append(oh.make_node("Add", ["term1", "term2"], ["output"]))
    return _finish(nodes, inits, "crk9_concentric")


# ================================================================ diagonal rays
def _solve_diag(I):
    H, W = I.shape
    cols = [c for c in np.unique(I) if c != 0]
    if 5 not in cols:
        return None
    Xs = [c for c in cols if c != 5]
    if len(Xs) != 1:
        return None
    X = int(Xs[0])
    xr, xc = np.where(I == X)
    if len(xr) == 0:
        return None
    cr = (xc - xr)
    if len(set(cr.tolist())) != 1:        # only "\" orientation handled
        return None
    m = int(cr[0])
    fr, fc = np.where(I == 5)
    d5 = (fc - fr)
    diagI = (np.arange(W)[None, :] - np.arange(H)[:, None])
    S = {m}
    above = [v for v in d5 if v > m]
    below = [v for v in d5 if v < m]
    if above:
        S.add(int(max(above)) + 2)
    if below:
        S.add(int(min(below)) - 2)
    out = np.zeros((H, W), int)
    for v in S:
        out[diagI == v] = X
    return out


def _build_diag():
    nodes, inits = [], []

    def C(name, arr, dtype=FLOAT):
        inits.append(_const(name, arr, dtype)); return name

    B = 1.0e6
    C("half", [0.5]); C("two", [2.0]); C("BIG", [B]); C("nBIG", [-B])
    C("hBIG", [B / 2]); C("nhBIG", [-B / 2])
    nodes.append(oh.make_node("ReduceSum", ["input"], ["grid_mask"], axes=[1], keepdims=1))
    nodes.append(oh.make_node("Greater", ["grid_mask", "half"], ["gridbool"]))
    nodes.append(oh.make_node("ReduceSum", ["input"], ["chan_sum"], axes=[2, 3], keepdims=1))
    nodes.append(oh.make_node("Greater", ["chan_sum", "half"], ["chan_pos"]))
    nodes.append(oh.make_node("Cast", ["chan_pos"], ["chan_pos_f"], to=FLOAT))
    notch = np.ones((1, CHANNELS, 1, 1)); notch[0, 0] = 0; notch[0, 5] = 0
    C("notch05", notch)
    nodes.append(oh.make_node("Mul", ["chan_pos_f", "notch05"], ["e_X"]))
    e0 = np.zeros((1, CHANNELS, 1, 1)); e0[0, 0] = 1.0; C("e0", e0)
    C("c5s", [5], INT64); C("c5e", [6], INT64); C("c1a", [1], INT64)
    nodes.append(oh.make_node("Slice", ["input", "c5s", "c5e", "c1a"], ["M5"]))
    nodes.append(oh.make_node("Greater", ["M5", "half"], ["M5b"]))
    nodes.append(oh.make_node("Mul", ["input", "e_X"], ["xsel"]))
    nodes.append(oh.make_node("ReduceSum", ["xsel"], ["Mx"], axes=[1], keepdims=1))
    nodes.append(oh.make_node("Greater", ["Mx", "half"], ["Mxb"]))

    Dbs = (np.arange(WIDTH)[None, :] - np.arange(HEIGHT)[:, None]).astype(float).reshape(1, 1, HEIGHT, WIDTH)
    C("Dbs", Dbs)
    nodes.append(oh.make_node("Where", ["Mxb", "Dbs", "nBIG"], ["dx"]))
    nodes.append(oh.make_node("ReduceMax", ["dx"], ["m"], axes=[1, 2, 3], keepdims=1))
    nodes.append(oh.make_node("Greater", ["Dbs", "m"], ["gtm"]))
    nodes.append(oh.make_node("And", ["M5b", "gtm"], ["abv"]))
    nodes.append(oh.make_node("Where", ["abv", "Dbs", "nBIG"], ["av"]))
    nodes.append(oh.make_node("ReduceMax", ["av"], ["maxabv"], axes=[1, 2, 3], keepdims=1))
    nodes.append(oh.make_node("Add", ["maxabv", "two"], ["newp"]))
    nodes.append(oh.make_node("Greater", ["maxabv", "nhBIG"], ["has_ab"]))
    nodes.append(oh.make_node("Less", ["Dbs", "m"], ["ltm"]))
    nodes.append(oh.make_node("And", ["M5b", "ltm"], ["blw"]))
    nodes.append(oh.make_node("Where", ["blw", "Dbs", "BIG"], ["bv"]))
    nodes.append(oh.make_node("ReduceMin", ["bv"], ["minblw"], axes=[1, 2, 3], keepdims=1))
    nodes.append(oh.make_node("Sub", ["minblw", "two"], ["newm"]))
    nodes.append(oh.make_node("Less", ["minblw", "hBIG"], ["has_bl"]))

    def eqdiag(dval, outn):
        nodes.append(oh.make_node("Sub", ["Dbs", dval], [outn + "_d"]))
        nodes.append(oh.make_node("Abs", [outn + "_d"], [outn + "_a"]))
        nodes.append(oh.make_node("Less", [outn + "_a", "half"], [outn]))

    eqdiag("m", "on_m")
    eqdiag("newp", "on_p0"); nodes.append(oh.make_node("And", ["on_p0", "has_ab"], ["on_p"]))
    eqdiag("newm", "on_n0"); nodes.append(oh.make_node("And", ["on_n0", "has_bl"], ["on_n"]))
    nodes.append(oh.make_node("Or", ["on_m", "on_p"], ["on_a"]))
    nodes.append(oh.make_node("Or", ["on_a", "on_n"], ["on_all"]))
    nodes.append(oh.make_node("And", ["on_all", "gridbool"], ["on_g"]))
    nodes.append(oh.make_node("Cast", ["on_g"], ["on_f"], to=FLOAT))
    nodes.append(oh.make_node("Sub", ["grid_mask", "on_f"], ["off_f"]))
    nodes.append(oh.make_node("Mul", ["on_f", "e_X"], ["t1"]))
    nodes.append(oh.make_node("Mul", ["off_f", "e0"], ["t2"]))
    nodes.append(oh.make_node("Add", ["t1", "t2"], ["output"]))
    return _finish(nodes, inits, "crk9_diag")


# ===================================================================== registry
_CACHE = {}
_SOLVERS = [
    ("crk9_concentric", _solve_concentric, _build_concentric),
    ("crk9_diag", _solve_diag, _build_diag),
]


def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return []
    out = []
    for name, solve, build in _SOLVERS:
        good = True
        for I, O in prs:
            if I.shape != O.shape:
                good = False; break
            pred = solve(I)
            if pred is None or not np.array_equal(pred, O):
                good = False; break
        if good:
            if name not in _CACHE:
                _CACHE[name] = build()
            out.append((name, _CACHE[name]))
    return out
