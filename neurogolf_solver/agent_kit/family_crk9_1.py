"""family_crk9_1 — hard residual ARC tasks (slice U[1::7]).

Solved:
  * task361: C4 rotational symmetrisation about the centre of the (unique)
    largest fully-coloured square.  The input is one "stamp"; the output is the
    union of the stamp with its 90/180/270 rotations about that centre.
    Centre is found in-graph by detecting all kxk all-coloured squares
    (k=1..4) via Conv, summing to a per-cell "largest-square-with-this-top-left"
    map, taking its argmax (unique), then deriving the doubled centre
    coordinates (cy2,cx2).  Each rotation is realised as transpose+flip of the
    coloured channels followed by a data-dependent integer translation built
    from computed [30,30] shift matrices (MatMul).  Background channel 0 is
    re-filled where no colour landed.

Rule (verified EXACT on train+test+arc-gen, 262/262 arc-gen via numpy mirror).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
_NEG = -(1 << 31)


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# ----------------------------------------------------------------- numpy mirror

def _msc(a):
    """doubled centre coords + size of unique largest all-nonzero square."""
    nz = (a != 0).astype(int)
    HH, WW = a.shape
    dp = np.zeros((HH, WW), int)
    best = 0; bi = bj = 0
    for i in range(HH):
        for j in range(WW):
            if nz[i, j]:
                dp[i, j] = 1 if (i == 0 or j == 0) else 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
                if dp[i, j] > best:
                    best = dp[i, j]; bi = i; bj = j
    return 2 * bi - best + 1, 2 * bj - best + 1, best


def _to_oh(a):
    g = np.zeros((10, HEIGHT, WIDTH), np.float32)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            g[a[r, c], r, c] = 1.0
    return g


def _shift_row(sy):
    M = np.zeros((HEIGHT, WIDTH), np.float32)
    for i in range(HEIGHT):
        k = i - sy
        if 0 <= k < HEIGHT:
            M[i, k] = 1.0
    return M


def _shift_col(sx):
    M = np.zeros((HEIGHT, WIDTH), np.float32)
    for j in range(WIDTH):
        l = j - sx
        if 0 <= l < WIDTH:
            M[l, j] = 1.0
    return M


def _T(src, sy, sx):
    return np.einsum('ik,ckl,lj->cij', _shift_row(sy), src, _shift_col(sx)).astype(np.float32)


def _solve_np(a):
    if a.size == 0 or (a != 0).sum() == 0:
        return None
    g = _to_oh(a)
    cy2, cx2, best = _msc(a)
    if best < 1:
        return None
    ci = g.copy(); ci[0] = 0
    Tp = np.transpose(ci, (0, 2, 1))
    FW = Tp[:, :, ::-1]
    FH = Tp[:, ::-1, :]
    G180 = ci[:, ::-1, ::-1]
    s = (cy2 + cx2) // 2
    dlt = (cy2 - cx2) // 2
    R90 = _T(FW, dlt, s - (WIDTH - 1))
    R270 = _T(FH, s - (HEIGHT - 1), -dlt)
    R180 = _T(G180, cy2 - (HEIGHT - 1), cx2 - (WIDTH - 1))
    U = np.maximum(np.maximum(ci, R90), np.maximum(R180, R270))
    INgrid = g.sum(0)
    colored = U.sum(0)
    bg = np.clip(INgrid - colored, 0, 1)
    out = U.copy(); out[0] = bg
    # decode (mirror grader)
    res = []
    for r in range(HEIGHT):
        row = []
        for c in range(WIDTH):
            ch = [k for k in range(10) if out[k, r, c] > 0]
            row.append(ch[0] if len(ch) == 1 else (11 if ch else 10))
        res.append(row)
    ex = []
    for row in res:
        rr = row[:]
        while rr and rr[-1] == 10:
            rr.pop()
        ex.append(rr)
    while ex and not ex[-1]:
        ex.pop()
    mw = max((len(r) for r in ex), default=0)
    arr = np.zeros((len(ex), mw), int)
    for i, row in enumerate(ex):
        for j, v in enumerate(row):
            arr[i, j] = v
    return arr


def _matches(pairs):
    if not pairs:
        return False
    for a, b in pairs:
        p = _solve_np(a)
        if p is None or p.shape != b.shape or not np.array_equal(p, b):
            return False
    return True


# ----------------------------------------------------------------- ONNX builder

def _const(name, arr):
    arr = np.asarray(arr, np.float32)
    return oh.make_tensor(name, DATA_TYPE, list(arr.shape), arr.ravel().tolist())


def _build_c4(maxk=4):
    nodes = []
    inits = []

    # constant grids ----------------------------------------------------------
    rowidx = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    colidx = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    diff_r = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)   # [i-k] for row shift mat
    diff_c = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)   # [j-l] for col shift mat
    for i in range(HEIGHT):
        for j in range(WIDTH):
            rowidx[0, 0, i, j] = i
            colidx[0, 0, i, j] = j
            diff_r[0, 0, i, j] = i - j
            diff_c[0, 0, i, j] = j - i
    inits.append(_const("ROWIDX", rowidx))
    inits.append(_const("COLIDX", colidx))
    inits.append(_const("DIFFR", diff_r))
    inits.append(_const("DIFFC", diff_c))

    chmask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], np.float32).reshape(1, 10, 1, 1)
    inits.append(_const("CHMASK", chmask))
    sel0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.float32).reshape(1, 10, 1, 1)
    inits.append(_const("SEL0", sel0))

    def scalar(name, v):
        t = oh.make_tensor(name, DATA_TYPE, [1, 1, 1, 1], [float(v)])
        inits.append(t)
        return name

    scalar("HALF", 0.5)
    scalar("ONE", 1.0)
    scalar("TWO", 2.0)
    scalar("WM1", float(WIDTH - 1))
    scalar("HM1", float(HEIGHT - 1))

    # coloured mask M [1,1,30,30] --------------------------------------------
    nodes.append(oh.make_node("ReduceSum", ["input"], ["total"], axes=[1], keepdims=1))
    s0 = oh.make_tensor("c0s", INT64, [1], [0]); e0 = oh.make_tensor("c0e", INT64, [1], [1])
    a0 = oh.make_tensor("c0a", INT64, [1], [1])
    inits += [s0, e0, a0]
    nodes.append(oh.make_node("Slice", ["input", "c0s", "c0e", "c0a"], ["ch0"]))
    nodes.append(oh.make_node("Sub", ["total", "ch0"], ["M"]))    # coloured mask

    # per-k square detection --------------------------------------------------
    size_terms = []
    for k in range(1, maxk + 1):
        wname = f"Wk{k}"
        w = oh.make_tensor(wname, DATA_TYPE, [1, 1, k, k], [1.0] * (k * k))
        inits.append(w)
        nodes.append(oh.make_node("Conv", ["M", wname], [f"ws{k}"],
                                  kernel_shape=[k, k], pads=[0, 0, k - 1, k - 1]))
        thr = oh.make_tensor(f"thr{k}", DATA_TYPE, [1, 1, 1, 1], [float(k * k) - 0.5])
        inits.append(thr)
        nodes.append(oh.make_node("Greater", [f"ws{k}", f"thr{k}"], [f"gt{k}"]))
        nodes.append(oh.make_node("Cast", [f"gt{k}"], [f"sq{k}"], to=1))
        size_terms.append(f"sq{k}")
    # maxsize = sum of sq_k
    cur = size_terms[0]
    for idx, t in enumerate(size_terms[1:], 1):
        nxt = f"acc{idx}"
        nodes.append(oh.make_node("Add", [cur, t], [nxt]))
        cur = nxt
    nodes.append(oh.make_node("Identity", [cur], ["maxsize"]))

    nodes.append(oh.make_node("ReduceMax", ["maxsize"], ["best"], axes=[2, 3], keepdims=1))
    nodes.append(oh.make_node("Sub", ["best", "HALF"], ["bestm"]))
    nodes.append(oh.make_node("Greater", ["maxsize", "bestm"], ["ohb"]))
    nodes.append(oh.make_node("Cast", ["ohb"], ["oh1"], to=1))      # one-hot top-left
    # i0, j0
    nodes.append(oh.make_node("Mul", ["oh1", "ROWIDX"], ["ohr"]))
    nodes.append(oh.make_node("ReduceSum", ["ohr"], ["i0"], axes=[2, 3], keepdims=1))
    nodes.append(oh.make_node("Mul", ["oh1", "COLIDX"], ["ohc"]))
    nodes.append(oh.make_node("ReduceSum", ["ohc"], ["j0"], axes=[2, 3], keepdims=1))
    # cy2 = 2*i0 + best - 1 ; cx2 = 2*j0 + best - 1
    nodes.append(oh.make_node("Mul", ["i0", "TWO"], ["i0t"]))
    nodes.append(oh.make_node("Add", ["i0t", "best"], ["cy2a"]))
    nodes.append(oh.make_node("Sub", ["cy2a", "ONE"], ["cy2"]))
    nodes.append(oh.make_node("Mul", ["j0", "TWO"], ["j0t"]))
    nodes.append(oh.make_node("Add", ["j0t", "best"], ["cx2a"]))
    nodes.append(oh.make_node("Sub", ["cx2a", "ONE"], ["cx2"]))

    # derived shift scalars ---------------------------------------------------
    nodes.append(oh.make_node("Add", ["cy2", "cx2"], ["sumc"]))
    nodes.append(oh.make_node("Mul", ["sumc", "HALF"], ["s"]))          # (cy2+cx2)/2
    nodes.append(oh.make_node("Sub", ["cy2", "cx2"], ["difc"]))
    nodes.append(oh.make_node("Mul", ["difc", "HALF"], ["dlt"]))        # (cy2-cx2)/2
    nodes.append(oh.make_node("Sub", ["s", "WM1"], ["sx90"]))           # s-(W-1)
    nodes.append(oh.make_node("Sub", ["s", "HM1"], ["sy270"]))          # s-(H-1)
    nodes.append(oh.make_node("Sub", ["dlt", "dlt"], ["zero_tmp"]))     # 0
    nodes.append(oh.make_node("Sub", ["zero_tmp", "dlt"], ["ndlt"]))    # -dlt
    nodes.append(oh.make_node("Sub", ["cy2", "HM1"], ["sy180"]))        # cy2-(H-1)
    nodes.append(oh.make_node("Sub", ["cx2", "WM1"], ["sx180"]))        # cx2-(W-1)

    # coloured input + its transpose/flips -----------------------------------
    nodes.append(oh.make_node("Mul", ["input", "CHMASK"], ["ci"]))
    nodes.append(oh.make_node("Transpose", ["ci"], ["Tp"], perm=[0, 1, 3, 2]))

    # flipW(Tp) -> FW (reverse axis 3) ; flipH(Tp)->FH (axis2) ; G180 flip ci 2,3
    def rev(src, axes, tag):
        n = len(axes)
        st = oh.make_tensor(tag + "_s", INT64, [n], [(HEIGHT if ax == 2 else WIDTH) - 1 for ax in axes])
        en = oh.make_tensor(tag + "_e", INT64, [n], [_NEG] * n)
        ax = oh.make_tensor(tag + "_a", INT64, [n], list(axes))
        sp = oh.make_tensor(tag + "_p", INT64, [n], [-1] * n)
        inits.extend([st, en, ax, sp])
        nodes.append(oh.make_node("Slice", [src, tag + "_s", tag + "_e", tag + "_a", tag + "_p"], [tag]))

    rev("Tp", [3], "FW")
    rev("Tp", [2], "FH")
    rev("ci", [2, 3], "G180")

    # shift-matrix helper: builds Mr/Mc from a scalar -------------------------
    def row_mat(syname, tag):
        nodes.append(oh.make_node("Sub", ["DIFFR", syname], [tag + "_d"]))
        nodes.append(oh.make_node("Abs", [tag + "_d"], [tag + "_ad"]))
        nodes.append(oh.make_node("Less", [tag + "_ad", "HALF"], [tag + "_lt"]))
        nodes.append(oh.make_node("Cast", [tag + "_lt"], [tag], to=1))
        return tag

    def col_mat(sxname, tag):
        nodes.append(oh.make_node("Sub", ["DIFFC", sxname], [tag + "_d"]))
        nodes.append(oh.make_node("Abs", [tag + "_d"], [tag + "_ad"]))
        nodes.append(oh.make_node("Less", [tag + "_ad", "HALF"], [tag + "_lt"]))
        nodes.append(oh.make_node("Cast", [tag + "_lt"], [tag], to=1))
        return tag

    def rotate(src, syname, sxname, tag):
        rm = row_mat(syname, tag + "_rm")
        cm = col_mat(sxname, tag + "_cm")
        nodes.append(oh.make_node("MatMul", [rm, src], [tag + "_t1"]))
        nodes.append(oh.make_node("MatMul", [tag + "_t1", cm], [tag]))
        return tag

    R90 = rotate("FW", "dlt", "sx90", "R90")
    R270 = rotate("FH", "sy270", "ndlt", "R270")
    R180 = rotate("G180", "sy180", "sx180", "R180")

    # union of coloured channels ---------------------------------------------
    nodes.append(oh.make_node("Max", ["ci", R90, R180, R270], ["U"]))
    # background re-fill
    nodes.append(oh.make_node("ReduceSum", ["input"], ["INgrid"], axes=[1], keepdims=1))
    nodes.append(oh.make_node("ReduceSum", ["U"], ["coloredany"], axes=[1], keepdims=1))
    nodes.append(oh.make_node("Sub", ["INgrid", "coloredany"], ["bgraw"]))
    nodes.append(oh.make_node("Clip", ["bgraw"], ["bg"], min=0.0, max=1.0))
    nodes.append(oh.make_node("Mul", ["bg", "SEL0"], ["bgch"]))
    nodes.append(oh.make_node("Add", ["U", "bgch"], ["output"]))

    return _model(nodes, inits)


# ----------------------------------------------------------------- dispatch

def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return []
    out = []
    try:
        if _matches(prs):
            out.append(("c4_rot_symm", _build_c4(4)))
    except Exception:
        pass
    return out
