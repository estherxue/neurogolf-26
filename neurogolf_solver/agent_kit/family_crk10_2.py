"""family_crk10_2 -- hardest remaining tasks, slice U[2::8] = [22,79,118,158,201,264,382].

Solved:
  * task 382 -- "stepping light ray".  The grid contains a straight line of 8s on one
    edge (the source) and a few 2 markers on a perpendicular edge.  Each 8 shoots a ray
    perpendicular to its edge, into the grid; every time the ray crosses a marker's
    line it steps one cell sideways, away from the marker edge.  2 markers stay.

    Rule (validated EXACT via numpy mirror on all 266 train+test+arc-gen pairs):
        out8[r,c] = u_col[c - sign * s(r)]           (vertical travel; sign=+1 if
                                                       markers on left else -1)
        s(r) = prefix/suffix count of marker rows     (prefix if source is top edge,
                                                       suffix if source is bottom)
      with the transposed analogue for horizontal travel (source on left/right edge).

    ONNX realisation (single fixed input-independent graph):
        - u_col / u_row  : ReduceMax of channel-8 over rows / cols       (source line)
        - m_row / m_col  : ReduceMax of channel-2 over cols / rows       (marker lines)
        - s(.)           : MatMul with a fixed 30x30 lower/upper-tri ones matrix
        - band_k(s)      : Greater(s,k-.5)*Greater(k+.5,s)               (k = 0..4)
        - shift_k(u)     : Pad (attr) + Slice                           (lateral step)
        - variant_out    : sum_k band_k (x) shift_k(u)                  (outer product)
        - 8 orientation variants each multiplied by a scalar gate that is 1 only for
          the (source-edge, marker-edge) combination actually present, then summed.
        - out2 = ch2*(1-out8); bg = grid_mask*(1-out8)*(1-out2); assemble one-hot.

  Tasks 22/79/118/158/201/264 remain unsolved (object-composition / plus-symmetrisation
  / template-stamping tasks with no clean static-graph realisation found).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL
KMAX = 4  # max lateral steps (== max #markers, which is 4 across all pairs)


# --------------------------------------------------------------------------- #
# task detection + numpy mirror (must match the ONNX graph op-for-op)          #
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


def _onehot(g):
    a = np.zeros((10, HEIGHT, WIDTH), np.float32)
    gh, gw = g.shape
    for r in range(gh):
        for c in range(gw):
            a[g[r, c], r, c] = 1.0
    return a


def _GT(x, t):
    return (x > t).astype(np.float32)


def _band(s, k):
    return _GT(s, k - 0.5) * _GT(k + 0.5, s)


def _shift_vec(u, k, sign):
    out = np.zeros_like(u)
    n = len(u)
    for c in range(n):
        src = c - sign * k
        if 0 <= src < n:
            out[c] = u[src]
    return out


def _solve_mirror(a):
    ch8 = a[8]; ch2 = a[2]
    G = a.sum(0)
    rowmask8 = _GT(ch8.max(1), 0.5); colmask8 = _GT(ch8.max(0), 0.5)
    rowmask2 = _GT(ch2.max(1), 0.5); colmask2 = _GT(ch2.max(0), 0.5)
    n_rows8 = rowmask8.sum(); n_cols8 = colmask8.sum()
    has2 = _GT(ch2.sum(), 0.5)
    vertical = _GT(n_rows8, 0.5) * _GT(1.5, n_rows8)
    horizontal = _GT(n_cols8, 0.5) * _GT(1.5, n_cols8)
    src_top = vertical * _GT(rowmask8[0], 0.5)
    src_bot = vertical * _GT(0.5, rowmask8[0])
    src_left = horizontal * _GT(colmask8[0], 0.5)
    src_right = horizontal * _GT(0.5, colmask8[0])
    mk_left = _GT(colmask2[0], 0.5)
    mk_right = has2 * _GT(0.5, colmask2[0])
    mk_top = _GT(rowmask2[0], 0.5)
    mk_bot = has2 * _GT(0.5, rowmask2[0])

    u_col = colmask8; u_row = rowmask8
    m_row = rowmask2; m_col = colmask2
    Lh = np.tril(np.ones((HEIGHT, HEIGHT), np.float32))
    Uh = np.triu(np.ones((HEIGHT, HEIGHT), np.float32))
    Lw = np.tril(np.ones((WIDTH, WIDTH), np.float32))
    Uw = np.triu(np.ones((WIDTH, WIDTH), np.float32))
    s_down = Lh @ m_row; s_up = Uh @ m_row
    s_right = m_col @ Uw; s_left = m_col @ Lw

    out8 = np.zeros((HEIGHT, WIDTH), np.float32)

    def vert(s_arr, sign, g):
        acc = np.zeros((HEIGHT, WIDTH), np.float32)
        for k in range(KMAX + 1):
            acc += np.outer(_band(s_arr, k), _shift_vec(u_col, k, sign))
        return g * acc

    def horiz(s_arr, sign, g):
        acc = np.zeros((HEIGHT, WIDTH), np.float32)
        for k in range(KMAX + 1):
            acc += np.outer(_shift_vec(u_row, k, sign), _band(s_arr, k))
        return g * acc

    out8 += vert(s_down, +1, src_top * mk_left)
    out8 += vert(s_down, -1, src_top * mk_right)
    out8 += vert(s_up, +1, src_bot * mk_left)
    out8 += vert(s_up, -1, src_bot * mk_right)
    out8 += horiz(s_right, +1, src_left * mk_top)
    out8 += horiz(s_right, -1, src_left * mk_bot)
    out8 += horiz(s_left, +1, src_right * mk_top)
    out8 += horiz(s_left, -1, src_right * mk_bot)
    out8 = _GT(out8, 0.5) * G
    out2 = ch2 * (1.0 - out8)
    bg = G * (1.0 - out8) * (1.0 - out2)
    res = np.zeros((10, HEIGHT, WIDTH), np.float32)
    res[0] = bg; res[2] = out2; res[8] = out8
    return res


def _matches(prs):
    for a, b in prs:
        if set(np.unique(a)) - {0, 2, 8} or set(np.unique(b)) - {0, 2, 8}:
            return False
    for a, b in prs:
        res = _solve_mirror(_onehot(a))
        # decode compare on b's shape region and padding
        pred = np.zeros_like(b)
        gh, gw = b.shape
        ok_pad = True
        for r in range(HEIGHT):
            for c in range(WIDTH):
                col = [k for k in range(10) if res[k, r, c] > 0]
                v = col[0] if col else 0
                if r < gh and c < gw:
                    pred[r, c] = v
                elif v != 0:
                    ok_pad = False
        if not ok_pad or not np.array_equal(pred, b):
            return False
    return True


# --------------------------------------------------------------------------- #
# ONNX graph builder (fixed, input-independent)                                #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0

    def name(self, p="t"):
        self.n += 1
        return f"{p}{self.n}"

    def initf(self, arr, shape):
        nm = self.name("cf")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, shape, np.asarray(arr, np.float32).ravel().tolist()))
        return nm

    def initi(self, arr, shape):
        nm = self.name("ci")
        self.inits.append(oh.make_tensor(nm, INT64, shape, [int(v) for v in np.asarray(arr).ravel()]))
        return nm

    def node(self, op, ins, out=None, **attr):
        out = out or self.name(op.lower())
        self.nodes.append(oh.make_node(op, ins, [out], **attr))
        return out

    def slice(self, x, starts, ends, axes, steps=None):
        s = self.initi(starts, [len(starts)])
        e = self.initi(ends, [len(ends)])
        ax = self.initi(axes, [len(axes)])
        ins = [x, s, e, ax]
        if steps is not None:
            ins.append(self.initi(steps, [len(steps)]))
        return self.node("Slice", ins)

    def greater(self, a, b):
        g = self.node("Greater", [a, b])
        return self.node("Cast", [g], to=DATA_TYPE)


def _build():
    g = _G()
    one = g.initf([1.0], [1])
    c05 = g.initf([0.5], [1])
    c15 = g.initf([1.5], [1])

    ch8 = g.slice("input", [8], [9], [1])
    ch2 = g.slice("input", [2], [3], [1])
    G = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)

    rowmax8 = g.node("ReduceMax", [ch8], axes=[3], keepdims=1)   # [1,1,30,1]
    colmax8 = g.node("ReduceMax", [ch8], axes=[2], keepdims=1)   # [1,1,1,30]
    rowmax2 = g.node("ReduceMax", [ch2], axes=[3], keepdims=1)
    colmax2 = g.node("ReduceMax", [ch2], axes=[2], keepdims=1)

    n_rows8 = g.node("ReduceSum", [rowmax8], axes=[2, 3], keepdims=1)
    n_cols8 = g.node("ReduceSum", [colmax8], axes=[2, 3], keepdims=1)
    has2sum = g.node("ReduceSum", [ch2], axes=[2, 3], keepdims=1)
    has2 = g.greater(has2sum, c05)

    vertical = g.node("Mul", [g.greater(n_rows8, c05), g.greater(c15, n_rows8)])
    horizontal = g.node("Mul", [g.greater(n_cols8, c05), g.greater(c15, n_cols8)])

    rm8_0 = g.slice(rowmax8, [0], [1], [2])   # rowmask8[0]
    cm8_0 = g.slice(colmax8, [0], [1], [3])
    rm2_0 = g.slice(rowmax2, [0], [1], [2])
    cm2_0 = g.slice(colmax2, [0], [1], [3])

    src_top = g.node("Mul", [vertical, g.greater(rm8_0, c05)])
    src_bot = g.node("Mul", [vertical, g.greater(c05, rm8_0)])
    src_left = g.node("Mul", [horizontal, g.greater(cm8_0, c05)])
    src_right = g.node("Mul", [horizontal, g.greater(c05, cm8_0)])
    mk_left = g.greater(cm2_0, c05)
    mk_right = g.node("Mul", [has2, g.greater(c05, cm2_0)])
    mk_top = g.greater(rm2_0, c05)
    mk_bot = g.node("Mul", [has2, g.greater(c05, rm2_0)])

    # prefix/suffix counts via triangular matmul
    LT = np.tril(np.ones((HEIGHT, HEIGHT), np.float32))
    UT = np.triu(np.ones((HEIGHT, HEIGHT), np.float32))
    LTn = g.initf(LT, [1, 1, HEIGHT, HEIGHT])
    UTn = g.initf(UT, [1, 1, HEIGHT, WIDTH])
    LTw = g.initf(np.tril(np.ones((WIDTH, WIDTH), np.float32)), [1, 1, WIDTH, WIDTH])
    UTw = g.initf(np.triu(np.ones((WIDTH, WIDTH), np.float32)), [1, 1, WIDTH, WIDTH])

    s_down = g.node("MatMul", [LTn, rowmax2])    # [1,1,30,1]
    s_up = g.node("MatMul", [UTn, rowmax2])
    s_right = g.node("MatMul", [colmax2, UTw])   # [1,1,1,30]
    s_left = g.node("MatMul", [colmax2, LTw])

    u_col = colmax8   # [1,1,1,30]
    u_row = rowmax8   # [1,1,30,1]

    def shift_col(sign):  # returns list over k of shifted u_col [1,1,1,30]
        res = []
        for k in range(KMAX + 1):
            if sign > 0:
                pads = [0, 0, 0, k, 0, 0, 0, 0]
                padded = g.node("Pad", [u_col], mode="constant", value=0.0, pads=pads)
                res.append(g.slice(padded, [0], [WIDTH], [3]))
            else:
                pads = [0, 0, 0, 0, 0, 0, 0, k]
                padded = g.node("Pad", [u_col], mode="constant", value=0.0, pads=pads)
                res.append(g.slice(padded, [k], [k + WIDTH], [3]))
        return res

    def shift_row(sign):  # shifted u_row [1,1,30,1]
        res = []
        for k in range(KMAX + 1):
            if sign > 0:
                pads = [0, 0, k, 0, 0, 0, 0, 0]
                padded = g.node("Pad", [u_row], mode="constant", value=0.0, pads=pads)
                res.append(g.slice(padded, [0], [HEIGHT], [2]))
            else:
                pads = [0, 0, 0, 0, 0, 0, k, 0]
                padded = g.node("Pad", [u_row], mode="constant", value=0.0, pads=pads)
                res.append(g.slice(padded, [k], [k + HEIGHT], [2]))
        return res

    ucp = shift_col(+1); ucm = shift_col(-1)
    urp = shift_row(+1); urm = shift_row(-1)

    def bands(s):
        res = []
        for k in range(KMAX + 1):
            lo = g.initf([k - 0.5], [1]); hi = g.initf([k + 0.5], [1])
            res.append(g.node("Mul", [g.greater(s, lo), g.greater(hi, s)]))
        return res

    b_down = bands(s_down); b_up = bands(s_up)
    b_right = bands(s_right); b_left = bands(s_left)

    def accumulate(band_list, shift_list):
        acc = None
        for k in range(KMAX + 1):
            m = g.node("Mul", [band_list[k], shift_list[k]])
            acc = m if acc is None else g.node("Add", [acc, m])
        return acc

    variants = [
        (accumulate(b_down, ucp), g.node("Mul", [src_top, mk_left])),
        (accumulate(b_down, ucm), g.node("Mul", [src_top, mk_right])),
        (accumulate(b_up, ucp), g.node("Mul", [src_bot, mk_left])),
        (accumulate(b_up, ucm), g.node("Mul", [src_bot, mk_right])),
        (accumulate(urp, b_right), g.node("Mul", [src_left, mk_top])),
        (accumulate(urm, b_right), g.node("Mul", [src_left, mk_bot])),
        (accumulate(urp, b_left), g.node("Mul", [src_right, mk_top])),
        (accumulate(urm, b_left), g.node("Mul", [src_right, mk_bot])),
    ]
    total = None
    for acc, gate in variants:
        gv = g.node("Mul", [acc, gate])
        total = gv if total is None else g.node("Add", [total, gv])

    out8 = g.node("Mul", [g.greater(total, c05), G])
    inv8 = g.node("Sub", [one, out8])
    out2 = g.node("Mul", [ch2, inv8])
    inv2 = g.node("Sub", [one, out2])
    bg = g.node("Mul", [g.node("Mul", [G, inv8]), inv2])
    zeros = g.node("Sub", [G, G])

    chans = [bg, zeros, out2, zeros, zeros, zeros, zeros, zeros, out8, zeros]
    g.node("Concat", chans, "output", axis=1)

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "crk10_2_ray", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []
    # task 382: stepping ray.  Require the color signature and exact mirror match.
    colset = set()
    for a, b in prs:
        colset |= set(np.unique(a)) | set(np.unique(b))
    if colset <= {0, 2, 8}:
        try:
            if _matches(prs):
                out.append(("ray_step", _build()))
        except Exception:
            pass
    return out
