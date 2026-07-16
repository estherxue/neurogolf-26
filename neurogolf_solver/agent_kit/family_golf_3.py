"""family_golf_3 — cheaper EXACT solvers for a slice of golf targets.

Each detector re-derives its rule from the train+test pairs (numpy), and only
emits an ONNX graph when that rule reproduces every pair exactly.  The graphs
are built from cheap origin-anchored ops (per-channel reductions, small
triangular MatMuls, single-channel masks) instead of [900,900] equality
matrices or long CA unrolls, so the integrator auto-picks them when cheaper.

I/O contract: input/output FLOAT[1,10,30,30] one-hot, grid top-left, zero pad,
channel0 = background.  Grader thresholds (output>0) for EXACT equality.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE


# --------------------------------------------------------------------------- #
# graph helpers
# --------------------------------------------------------------------------- #
def _model(nodes, inits=()):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _f(name, arr, shape):
    return oh.make_tensor(name, F, shape, np.asarray(arr, np.float32).ravel().tolist())


def _i(name, vals):
    vals = list(vals)
    return oh.make_tensor(name, INT64, [len(vals)], vals)


_TRIL = np.tril(np.ones((HEIGHT, HEIGHT), np.float32))          # [j,i]=1 if i<=j
_TRIU = np.triu(np.ones((HEIGHT, WIDTH), np.float32))           # [j,i]=1 if i>=j
_SLO = np.tril(np.ones((HEIGHT, HEIGHT), np.float32), -1)       # strict lower
_SUP = np.triu(np.ones((HEIGHT, WIDTH), np.float32), 1)         # strict upper


# --------------------------------------------------------------------------- #
# numpy reference rules (used for detection)
# --------------------------------------------------------------------------- #
def _pairs(examples):
    out = []
    for sec in ("train", "test"):
        for e in examples.get(sec, []):
            out.append((np.array(e["input"], int), np.array(e["output"], int)))
    return out


def _r_connect(g):
    H, W = g.shape
    out = np.zeros_like(g)
    for c in range(1, 10):
        m = (g == c)
        for r in range(H):
            cols = np.where(m[r])[0]
            if len(cols):
                out[r, cols.min():cols.max() + 1] = c
        for cc in range(W):
            rows = np.where(m[:, cc])[0]
            if len(rows):
                out[rows.min():rows.max() + 1, cc] = c
    return out


def _r_lines24(g):
    H, W = g.shape
    out = np.zeros_like(g)
    for cc in range(W):
        if (g[:, cc] == 2).any():
            out[:, cc] = 2
    for r in range(H):
        for c in range(1, 10):
            if c == 2:
                continue
            if (g[r] == c).any():
                out[r, :] = c
    return out


def _r_emptyline(g):
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        if (g[r] == 0).all():
            out[r, :] = 2
    for cc in range(W):
        if (g[:, cc] == 0).all():
            out[:, cc] = 2
    return out


def _r_bbox(g):
    out = g.copy()
    for c in range(1, 10):
        ys, xs = np.where(g == c)
        if len(ys):
            out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = c
    return out


def _r_ell835(g):
    a = np.argwhere(g == 8)
    b = np.argwhere(g == 2)
    if len(a) != 1 or len(b) != 1:
        raise ValueError
    out = g.copy()
    r8, c8 = a[0]
    r2, c2 = b[0]
    out[min(r8, r2):max(r8, r2) + 1, c8] = 4    # vertical along 8's column
    out[r2, min(c8, c2):max(c8, c2) + 1] = 4    # horizontal along 2's row
    out[r8, c8] = 8
    out[r2, c2] = 2
    return out


def _r_cornerfill(g):
    H, W = g.shape
    M = (g == 4).astype(int)
    if M.sum() < 4:
        raise ValueError
    out = g.copy()
    for r in range(H):
        for c in range(W):
            if g[r, c] != 0:
                continue
            if (M[:r, :c].any() and M[:r, c + 1:].any()
                    and M[r + 1:, :c].any() and M[r + 1:, c + 1:].any()):
                out[r, c] = 2
    return out


def _r_spray(g):
    H, W = g.shape
    d = np.argwhere(g != 0)
    if len(d) != 1:
        raise ValueError
    r, c = d[0]
    k = g[r, c]
    if k == 4 or r + 1 >= H:
        raise ValueError
    out = np.zeros_like(g)
    par = c % 2
    for rr in range(r + 1):
        for cc in range(W):
            if cc % 2 == par:
                out[rr, cc] = 4
    out[r + 1, c] = k
    return out


def _match(examples, rule):
    prs = _pairs(examples)
    if not prs:
        return False
    for a, b in prs:
        try:
            p = rule(a)
        except Exception:
            return False
        if p.shape != b.shape or not (p == b).all():
            return False
    return True


def _single_colour(examples):
    """True iff every train/test grid has at most one distinct non-bg colour."""
    for a, _ in _pairs(examples):
        if len(set(a[a != 0].tolist())) > 1:
            return False
    return True


# --------------------------------------------------------------------------- #
# ONNX builders
# --------------------------------------------------------------------------- #
def _build_connect():
    """Connect equal-colour dots along rows and columns (span fill).
    Per colour channel: exists-left & exists-right (row), exists-up & exists-down
    (col) via two triangular MatMuls each; fill = min(L,R); union of H,V.
    All work on the 9 colour channels (channel0 handled as background)."""
    L = _f("L", _TRIL, [HEIGHT, HEIGHT])      # MatMul(L, X): exists row<=r
    U = _f("U", _TRIU, [HEIGHT, WIDTH])       # MatMul(U, X): exists row>=r
    n = [
        oh.make_node("Slice", ["input", "c1", "c10", "ax1"], ["X9"]),  # colours
        # horizontal (right-multiply contracts W)
        oh.make_node("MatMul", ["X9", "U"], ["HL"]),   # exists col<=c
        oh.make_node("MatMul", ["X9", "L"], ["HR"]),   # exists col>=c
        oh.make_node("Min", ["HL", "HR"], ["Hm"]),
        # vertical (left-multiply contracts H)
        oh.make_node("MatMul", ["L", "X9"], ["VL"]),   # exists row<=r
        oh.make_node("MatMul", ["U", "X9"], ["VR"]),   # exists row>=r
        oh.make_node("Min", ["VL", "VR"], ["Vm"]),
        oh.make_node("Add", ["Hm", "Vm"], ["tail"]),   # >0 where filled
        oh.make_node("ReduceMax", ["input"], ["real"], axes=[1], keepdims=1),
        oh.make_node("ReduceMax", ["tail"], ["anyl"], axes=[1], keepdims=1),
        oh.make_node("Sub", ["real", "anyl"], ["bg0"]),
        oh.make_node("Concat", ["bg0", "tail"], ["output"], axis=1),
    ]
    inits = [L, U, _i("c1", [1]), _i("c10", [10]), _i("ax1", [1])]
    return _model(n, inits)


def _build_connect1():
    """Single-colour variant of connect: collapse all colours to one presence
    channel, span-fill on [1,1,30,30], then re-attach the (unique) colour."""
    L = _f("L", _TRIL, [HEIGHT, HEIGHT])
    U = _f("U", _TRIU, [HEIGHT, WIDTH])
    n = [
        oh.make_node("ReduceMax", ["input"], ["R"], axes=[1], keepdims=1),     # real mask
        oh.make_node("Slice", ["input", "c0", "c1", "ax1"], ["ch0"]),
        oh.make_node("Sub", ["R", "ch0"], ["pres"]),                           # any colour
        oh.make_node("MatMul", ["pres", "U"], ["HL"]),
        oh.make_node("MatMul", ["pres", "L"], ["HR"]),
        oh.make_node("Min", ["HL", "HR"], ["Hm"]),
        oh.make_node("MatMul", ["L", "pres"], ["VL"]),
        oh.make_node("MatMul", ["U", "pres"], ["VR"]),
        oh.make_node("Min", ["VL", "VR"], ["Vm"]),
        oh.make_node("Add", ["Hm", "Vm"], ["lines1"]),                         # [1,1,30,30]
        oh.make_node("ReduceMax", ["input"], ["cp"], axes=[2, 3], keepdims=1), # [1,10,1,1]
        oh.make_node("Slice", ["cp", "c1", "c10", "ax1"], ["cp9"]),            # [1,9,1,1]
        oh.make_node("Mul", ["lines1", "cp9"], ["tail"]),                      # [1,9,30,30]
        oh.make_node("Sub", ["R", "lines1"], ["bg0"]),
        oh.make_node("Concat", ["bg0", "tail"], ["output"], axis=1),
    ]
    inits = [L, U, _i("c0", [0]), _i("c1", [1]), _i("c10", [10]), _i("ax1", [1])]
    return _model(n, inits)


def _build_lines24():
    """colour 2 -> fill its columns (vertical); every other colour -> fill its
    rows (horizontal); horizontal overwrites vertical.  Work on 9 colour
    channels: index j in 0..8 == colour j+1, so colour 2 is index 1."""
    hm9 = [1, 0, 1, 1, 1, 1, 1, 1, 1]   # zero colour 2
    vm9 = [0, 1, 0, 0, 0, 0, 0, 0, 0]   # only colour 2
    n = [
        oh.make_node("ReduceMax", ["input"], ["ROWf"], axes=[3], keepdims=1),  # [1,10,30,1]
        oh.make_node("ReduceMax", ["input"], ["COLf"], axes=[2], keepdims=1),  # [1,10,1,30]
        oh.make_node("ReduceMax", ["input"], ["R"], axes=[1], keepdims=1),     # [1,1,30,30]
        oh.make_node("Slice", ["ROWf", "c1", "c10", "ax1"], ["ROW"]),          # [1,9,30,1]
        oh.make_node("Slice", ["COLf", "c1", "c10", "ax1"], ["COL"]),          # [1,9,1,30]
        oh.make_node("Mul", ["ROW", "hm"], ["ROWm"]),
        oh.make_node("Mul", ["ROWm", "R"], ["Hfull"]),                        # [1,9,30,30]
        oh.make_node("ReduceMax", ["Hfull"], ["Hany"], axes=[1], keepdims=1), # [1,1,30,30]
        oh.make_node("Sub", ["one", "Hany"], ["notH"]),
        oh.make_node("Mul", ["R", "notH"], ["Rmask"]),
        oh.make_node("Mul", ["COL", "vm"], ["COLm"]),
        oh.make_node("Mul", ["COLm", "Rmask"], ["Vmask"]),                    # [1,9,30,30]
        oh.make_node("Add", ["Hfull", "Vmask"], ["tail"]),
        oh.make_node("ReduceMax", ["tail"], ["cov"], axes=[1], keepdims=1),
        oh.make_node("Sub", ["R", "cov"], ["bg0"]),
        oh.make_node("Concat", ["bg0", "tail"], ["output"], axis=1),
    ]
    inits = [
        _f("hm", hm9, [1, CHANNELS - 1, 1, 1]),
        _f("vm", vm9, [1, CHANNELS - 1, 1, 1]),
        _f("one", [1.0], [1, 1, 1, 1]),
        _i("c1", [1]), _i("c10", [10]), _i("ax1", [1]),
    ]
    return _model(n, inits)


def _build_emptyline():
    """Rows/columns that are entirely background(0) become colour 2; everything
    else is preserved.  delta = fill2 * (e2 - e0)."""
    vec = [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0]   # e2 - e0
    n = [
        oh.make_node("ReduceMax", ["input"], ["R"], axes=[1], keepdims=1),    # real mask
        oh.make_node("Slice", ["input", "c0", "c1", "ax1"], ["ch0"]),         # background ch
        oh.make_node("Sub", ["R", "ch0"], ["col_"]),                          # 1 at coloured cells
        oh.make_node("ReduceMax", ["col_"], ["rownz"], axes=[3], keepdims=1),
        oh.make_node("ReduceMax", ["col_"], ["colnz"], axes=[2], keepdims=1),
        oh.make_node("ReduceMax", ["R"], ["rowr"], axes=[3], keepdims=1),
        oh.make_node("ReduceMax", ["R"], ["colr"], axes=[2], keepdims=1),
        oh.make_node("Sub", ["rowr", "rownz"], ["erow"]),                     # empty row
        oh.make_node("Sub", ["colr", "colnz"], ["ecol"]),                     # empty col
        oh.make_node("Max", ["erow", "ecol"], ["ec"]),                        # broadcast
        oh.make_node("Mul", ["ec", "R"], ["fill2"]),
        oh.make_node("Mul", ["fill2", "vec"], ["delta"]),
        oh.make_node("Add", ["input", "delta"], ["output"]),
    ]
    inits = [
        _f("vec", vec, [1, CHANNELS, 1, 1]),
        _i("c0", [0]), _i("c1", [1]), _i("ax1", [1]),
    ]
    return _model(n, inits)


def _build_bbox():
    """Per colour: fill its bounding rectangle.  row/col presence projected then
    bounded by triangular MatMuls (tiny), single [1,10,30,30] product."""
    L = _f("L", _TRIL, [HEIGHT, HEIGHT])
    U = _f("U", _TRIU, [HEIGHT, WIDTH])
    n = [
        oh.make_node("ReduceMax", ["input"], ["rowf"], axes=[3], keepdims=1),  # [1,10,30,1]
        oh.make_node("ReduceMax", ["input"], ["colf"], axes=[2], keepdims=1),  # [1,10,1,30]
        oh.make_node("Slice", ["rowf", "c1", "c10", "ax1"], ["rowh"]),         # [1,9,30,1]
        oh.make_node("Slice", ["colf", "c1", "c10", "ax1"], ["colh"]),         # [1,9,1,30]
        oh.make_node("MatMul", ["L", "rowh"], ["rPD"]),
        oh.make_node("MatMul", ["U", "rowh"], ["rPU"]),
        oh.make_node("Min", ["rPD", "rPU"], ["rBox"]),
        oh.make_node("MatMul", ["colh", "U"], ["cPL"]),
        oh.make_node("MatMul", ["colh", "L"], ["cPR"]),
        oh.make_node("Min", ["cPL", "cPR"], ["cBox"]),
        oh.make_node("Mul", ["rBox", "cBox"], ["tail"]),                       # [1,9,30,30]
        oh.make_node("ReduceMax", ["input"], ["real"], axes=[1], keepdims=1),
        oh.make_node("ReduceMax", ["tail"], ["anyb"], axes=[1], keepdims=1),
        oh.make_node("Sub", ["real", "anyb"], ["bg0"]),
        oh.make_node("Concat", ["bg0", "tail"], ["output"], axis=1),
    ]
    inits = [L, U, _i("c1", [1]), _i("c10", [10]), _i("ax1", [1])]
    return _model(n, inits)


def _build_ell835():
    """L-path from the single colour-8 dot down its column to the colour-2 dot's
    row, then across to the colour-2 dot; path colour 4, endpoints preserved."""
    L = _f("L", _TRIL, [HEIGHT, HEIGHT])
    U = _f("U", _TRIU, [HEIGHT, WIDTH])
    vec = [-1, 0, 0, 0, 1, 0, 0, 0, 0, 0]   # e4 - e0
    n = [
        oh.make_node("Slice", ["input", "c8", "c9", "ax1"], ["A8"]),   # channel 8
        oh.make_node("Slice", ["input", "c2", "c3", "ax1"], ["B2"]),   # channel 2
        oh.make_node("ReduceMax", ["A8"], ["col8"], axes=[2], keepdims=1),
        oh.make_node("ReduceMax", ["A8"], ["row8"], axes=[3], keepdims=1),
        oh.make_node("ReduceMax", ["B2"], ["col2"], axes=[2], keepdims=1),
        oh.make_node("ReduceMax", ["B2"], ["row2"], axes=[3], keepdims=1),
        oh.make_node("Max", ["row8", "row2"], ["rmk"]),
        oh.make_node("Max", ["col8", "col2"], ["cmk"]),
        oh.make_node("MatMul", ["L", "rmk"], ["rPD"]),
        oh.make_node("MatMul", ["U", "rmk"], ["rPU"]),
        oh.make_node("Min", ["rPD", "rPU"], ["rrng"]),
        oh.make_node("MatMul", ["cmk", "U"], ["cPL"]),
        oh.make_node("MatMul", ["cmk", "L"], ["cPR"]),
        oh.make_node("Min", ["cPL", "cPR"], ["crng"]),
        oh.make_node("Mul", ["col8", "rrng"], ["vert"]),       # [1,1,30,30]
        oh.make_node("Mul", ["row2", "crng"], ["horz"]),       # [1,1,30,30]
        oh.make_node("Max", ["vert", "horz"], ["pmask"]),
        oh.make_node("Add", ["A8", "B2"], ["dots"]),
        oh.make_node("Sub", ["pmask", "dots"], ["path"]),
        oh.make_node("Mul", ["path", "vec"], ["delta"]),
        oh.make_node("Add", ["input", "delta"], ["output"]),
    ]
    inits = [
        L, U, _f("vec", vec, [1, CHANNELS, 1, 1]),
        _i("c2", [2]), _i("c3", [3]), _i("c8", [8]), _i("c9", [9]), _i("ax1", [1]),
    ]
    return _model(n, inits)


def _build_cornerfill():
    """Fill the interior of every colour-4 rectangle with colour 2.  A background
    cell is interior iff a 4 exists in each of its 4 strict quadrants (2D strict
    prefix counts via strict-triangular MatMuls)."""
    Su = _f("Su", _SUP, [HEIGHT, WIDTH])   # strict upper
    Sl = _f("Sl", _SLO, [HEIGHT, HEIGHT])  # strict lower
    vec = [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # e2 - e0
    n = [
        oh.make_node("Slice", ["input", "c4", "c5", "ax1"], ["M4"]),
        oh.make_node("Slice", ["input", "c0", "c1", "ax1"], ["ch0"]),
        oh.make_node("MatMul", ["M4", "Su"], ["Mc"]),    # sum c'<c
        oh.make_node("MatMul", ["M4", "Sl"], ["Mcr"]),   # sum c'>c
        oh.make_node("MatMul", ["Sl", "Mc"], ["UL"]),    # r'<r, c'<c
        oh.make_node("MatMul", ["Sl", "Mcr"], ["UR"]),   # r'<r, c'>c
        oh.make_node("MatMul", ["Su", "Mc"], ["DL"]),    # r'>r, c'<c
        oh.make_node("MatMul", ["Su", "Mcr"], ["DR"]),   # r'>r, c'>c
        oh.make_node("Min", ["UL", "UR"], ["q1"]),
        oh.make_node("Min", ["DL", "DR"], ["q2"]),
        oh.make_node("Min", ["q1", "q2"], ["q"]),
        oh.make_node("Min", ["q", "ch0"], ["fill"]),     # interior & background
        oh.make_node("Mul", ["fill", "vec"], ["delta"]),
        oh.make_node("Add", ["input", "delta"], ["output"]),
    ]
    inits = [
        Su, Sl, _f("vec", vec, [1, CHANNELS, 1, 1]),
        _i("c0", [0]), _i("c1", [1]), _i("c4", [4]), _i("c5", [5]), _i("ax1", [1]),
    ]
    return _model(n, inits)


def _build_spray():
    """Single dot (r,c,k): rows 0..r get colour 4 on columns of c's parity; the
    dot drops to (r+1,c) keeping colour k."""
    U = _f("U", _TRIU, [HEIGHT, WIDTH])
    even = [1.0 if i % 2 == 0 else 0.0 for i in range(WIDTH)]
    odd = [0.0 if i % 2 == 0 else 1.0 for i in range(WIDTH)]
    e4 = [1.0 if j == 3 else 0.0 for j in range(CHANNELS - 1)]   # channel 4 in 1..9
    n = [
        oh.make_node("ReduceMax", ["input"], ["R"], axes=[1], keepdims=1),
        oh.make_node("Slice", ["input", "c0", "c1", "ax1"], ["ch0"]),
        oh.make_node("Sub", ["R", "ch0"], ["pres"]),
        oh.make_node("ReduceMax", ["pres"], ["drow"], axes=[3], keepdims=1),  # [1,1,30,1]
        oh.make_node("ReduceMax", ["pres"], ["dcol"], axes=[2], keepdims=1),  # [1,1,1,30]
        oh.make_node("MatMul", ["U", "drow"], ["rowsAA"]),                    # rows<=r
        oh.make_node("Mul", ["dcol", "evn"], ["peM"]),
        oh.make_node("ReduceSum", ["peM"], ["pe"], axes=[3], keepdims=1),     # [1,1,1,1]
        oh.make_node("Sub", ["one", "pe"], ["po"]),
        oh.make_node("Mul", ["pe", "evn"], ["peS"]),
        oh.make_node("Mul", ["po", "odd"], ["poS"]),
        oh.make_node("Add", ["peS", "poS"], ["pcols"]),                       # [1,1,1,30]
        oh.make_node("Mul", ["rowsAA", "pcols"], ["f4m"]),
        oh.make_node("Mul", ["f4m", "R"], ["fill4"]),                         # [1,1,30,30]
        oh.make_node("Pad", ["pres"], ["padM"], mode="constant", value=0.0,
                     pads=[0, 0, 1, 0, 0, 0, 0, 0]),
        oh.make_node("Slice", ["padM", "c0", "c30", "ax2"], ["movedM"]),      # shift down 1
        oh.make_node("ReduceMax", ["input"], ["cp"], axes=[2, 3], keepdims=1),
        oh.make_node("Slice", ["cp", "c1", "c10", "ax1"], ["cp9"]),
        oh.make_node("Mul", ["movedM", "cp9"], ["movedtail"]),                # [1,9,30,30]
        oh.make_node("Mul", ["fill4", "e4"], ["fill4bc"]),                    # [1,9,30,30]
        oh.make_node("Add", ["movedtail", "fill4bc"], ["tail"]),
        oh.make_node("Add", ["fill4", "movedM"], ["cover"]),
        oh.make_node("Sub", ["R", "cover"], ["bg0"]),
        oh.make_node("Concat", ["bg0", "tail"], ["output"], axis=1),
    ]
    inits = [
        U,
        _f("evn", even, [1, 1, 1, WIDTH]), _f("odd", odd, [1, 1, 1, WIDTH]),
        _f("one", [1.0], [1, 1, 1, 1]), _f("e4", e4, [1, CHANNELS - 1, 1, 1]),
        _i("c0", [0]), _i("c1", [1]), _i("c10", [10]), _i("c30", [HEIGHT]),
        _i("ax1", [1]), _i("ax2", [2]),
    ]
    return _model(n, inits)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
_DETECTORS = [
    ("connect_RLUD", _r_connect, _build_connect),
    ("lines_v2_h",   _r_lines24, _build_lines24),
    ("emptyline2",   _r_emptyline, _build_emptyline),
    ("bboxfill",     _r_bbox,     _build_bbox),
    ("ell835",       _r_ell835,   _build_ell835),
    ("cornerfill2",  _r_cornerfill, _build_cornerfill),
    ("spray4",       _r_spray,    _build_spray),
]


def candidates(examples):
    out = []
    for name, rule, build in _DETECTORS:
        if _match(examples, rule):
            try:
                out.append((name, build()))
            except Exception:
                pass
    # cheap single-colour connect specialisation
    if _match(examples, _r_connect) and _single_colour(examples):
        try:
            out.append(("connect1", _build_connect1()))
        except Exception:
            pass
    return out
