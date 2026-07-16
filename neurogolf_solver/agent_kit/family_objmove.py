"""OBJECT TRANSLATION / COPY-TEMPLATE-TO-MARKER family (opset-10, origin-anchored).

Two structurally-detected sub-rules, both realised so the grid content stays
TOP-LEFT anchored and never bleeds into the zero-padding (so they generalise to
grids of ANY size):

  translate(dy,dx)   The whole coloured content (channels 1..9) is shifted by a
                     CONSTANT displacement and the vacated cells become real
                     background.  Built with one Pad (attribute pads) + Slice on
                     the [1,9,30,30] colour stack, masked by the real-cell mask
                     R = ReduceSum(input, channel) so nothing escapes the grid;
                     the background channel is rebuilt as R - sum(colours).
                     Handles multi-cell / multi-colour rigid moves.

  stamp(m,T,base)    A FIXED template T (a small {offset -> colour} pattern) is
                     stamped at every cell of marker colour m.  The marker plane
                     M = input[:, m] is fed through ONE Conv whose kernel encodes
                     the template (W[c, 0, K-dr, K-dc] = 1 for each template entry
                     (dr,dc)->c), so every marker simultaneously broadcasts the
                     template into the colour channels -- a single cheap conv,
                     no per-offset shifting.  `base` selects what survives under
                     the stamps: 'zero' (only the stamps), 'input' (keep the input
                     and overlay the stamps) or 'input_nomark' (keep the input but
                     erase the markers first -> rigid MOVE / MOVE+RECOLOUR of a
                     single-colour object, and stamp-with-kept-centre tasks).

Both reconstruct the background channel exactly (the grader thresholds output>0
and demands a real background cell to be channel-0 > 0), mask by R so padding
stays all-zero, and are only emitted when a numpy mirror of the EXACT ONNX
semantics reproduces EVERY available train+test+arc-gen pair (the grader's gate).
Detection is structural (a marker colour + a bounded fixed template, or a single
constant displacement), so wrong hypotheses are dropped before scoring.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
_KMAX = 6          # bound on template offset radius (local stamps only)


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals):
        nm = self.name("c")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(dims),
                                         np.asarray(vals, np.float32).ravel().tolist()))
        return nm

    def i(self, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)],
                                         [int(v) for v in vals]))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _model(g):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _slice_ch(g, x, lo, hi):
    """Slice channels [lo, hi) on axis 1."""
    return g.node("Slice", [x, g.i([lo]), g.i([hi]), g.i([1])])


def _shift(g, x, dy, dx, n_ch):
    """Shift content of a [1,n_ch,30,30] tensor by (dy,dx) with zero fill, kept
    in a 30x30 window anchored so cell (i,j) maps to (i+dy, j+dx)."""
    h0, w0 = max(dy, 0), max(dx, 0)
    h1, w1 = max(-dy, 0), max(-dx, 0)
    p = g.node("Pad", [x], mode="constant", value=0.0,
               pads=[0, 0, h0, w0, 0, 0, h1, w1])
    return g.node("Slice", [p, g.i([h1, w1]), g.i([h1 + HEIGHT, w1 + WIDTH]),
                            g.i([2, 3])])


# --------------------------------------------------------------------------- #
# ONNX builders                                                               #
# --------------------------------------------------------------------------- #
def build_translate(dy, dx):
    """Rigid constant-displacement move of every coloured cell (vacated -> bg)."""
    g = _G()
    colored_in = _slice_ch(g, "input", 1, CHANNELS)        # [1,9,30,30]
    shifted = _shift(g, colored_in, dy, dx, CHANNELS - 1)   # [1,9,30,30]
    R = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)  # [1,1,30,30]
    masked = g.node("Mul", [shifted, R])                    # drop padding spill
    csum = g.node("ReduceSum", [masked], axes=[1], keepdims=1)
    ch0 = g.node("Sub", [R, csum])                          # real background
    g.node("Concat", [ch0, masked], "output", axis=1)
    return _model(g)


def build_stamp(m, templ, base_mode):
    """Stamp the fixed template `templ` at every marker (colour m) cell."""
    g = _G()
    K = max(max(abs(dr), abs(dc)) for (dr, dc) in templ)
    kh = 2 * K + 1
    Wt = np.zeros((CHANNELS, 1, kh, kh), np.float32)
    for (dr, dc), col in templ.items():
        Wt[col, 0, K - dr, K - dc] = 1.0
    w = g.f([CHANNELS, 1, kh, kh], Wt)

    Mpl = _slice_ch(g, "input", m, m + 1)                  # [1,1,30,30] marker plane
    SH = g.node("Conv", [Mpl, w], kernel_shape=[kh, kh], pads=[K, K, K, K])  # [1,10,30,30]
    R = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)   # [1,1,30,30]

    if base_mode == "zero":
        Pfull = SH
    else:
        covered = g.node("ReduceSum", [SH], axes=[1], keepdims=1)
        one = g.f([1, 1, 1, 1], [1.0])
        notcov = g.node("Sub", [one, covered])
        if base_mode == "input":
            Kf = "input"
        else:  # input_nomark: zero out the marker channel before keeping
            vec = g.f([1, CHANNELS, 1, 1],
                      [0.0 if c == m else 1.0 for c in range(CHANNELS)])
            Kf = g.node("Mul", ["input", vec])
        Pfull = g.node("Add", [SH, g.node("Mul", [Kf, notcov])])

    colored = _slice_ch(g, Pfull, 1, CHANNELS)             # [1,9,30,30]
    csum = g.node("ReduceSum", [colored], axes=[1], keepdims=1)
    ch0 = g.node("Sub", [R, csum])
    out = g.node("Concat", [ch0, colored], axis=1)         # [1,10,30,30]
    g.node("Mul", [out, R], "output")                      # mask padding
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX semantics EXACTLY for detection)          #
# --------------------------------------------------------------------------- #
def _onehot(a):
    h, w = a.shape
    o = np.zeros((CHANNELS, h, w), bool)
    for c in range(CHANNELS):
        o[c] = (a == c)
    return o


def _shift_grid_allcolor(a, dy, dx):
    """Move every coloured cell by (dy,dx), vacated -> background (0)."""
    h, w = a.shape
    out = np.zeros_like(a)
    for i in range(h):
        for j in range(w):
            ni, nj = i + dy, j + dx
            if a[i, j] != 0 and 0 <= ni < h and 0 <= nj < w:
                out[ni, nj] = a[i, j]
    return out


def _base_grid(a, base_mode, m):
    if base_mode == "zero":
        return np.zeros_like(a)
    if base_mode == "input":
        return a.copy()
    g = a.copy()
    g[a == m] = 0
    return g


def _ref_stamp(a, m, templ, base_mode):
    """Predicted one-hot bool [10,h,w] under the exact ONNX float arithmetic."""
    h, w = a.shape
    M = (a == m).astype(np.float64)
    Sc = np.zeros((CHANNELS, h, w))
    covered = np.zeros((h, w))
    for (dr, dc), col in templ.items():
        s = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                ni, nj = i + dr, j + dc
                if 0 <= ni < h and 0 <= nj < w:
                    s[ni, nj] += M[i, j]
        Sc[col] += s
        covered += s
    oin = _onehot(a).astype(np.float64)
    if base_mode == "input_nomark":
        oin[m] = 0.0
    pc = np.zeros((CHANNELS, h, w))
    if base_mode == "zero":
        for c in range(1, CHANNELS):
            pc[c] = Sc[c]
    else:
        notcov = 1.0 - covered
        for c in range(1, CHANNELS):
            pc[c] = Sc[c] + oin[c] * notcov
    ch0 = 1.0 - pc[1:].sum(axis=0)
    pred = np.zeros((CHANNELS, h, w), bool)
    pred[0] = ch0 > 0
    for c in range(1, CHANNELS):
        pred[c] = pc[c] > 0
    return pred


def _infer_templ(prs, m, base_mode):
    """Union of {offset -> colour} read from every single-marker pair."""
    templ = {}
    seen = False
    for a, b in prs:
        if a.shape != b.shape:
            return None
        mk = np.argwhere(a == m)
        if len(mk) != 1:
            continue
        seen = True
        pi, pj = int(mk[0][0]), int(mk[0][1])
        base = _base_grid(a, base_mode, m)
        h, w = a.shape
        diff = np.argwhere(b != base)
        for i, j in diff:
            dr, dc = int(i) - pi, int(j) - pj
            col = int(b[i, j])
            if col == 0:
                return None
            if abs(dr) > _KMAX or abs(dc) > _KMAX:
                return None
            if (dr, dc) in templ and templ[(dr, dc)] != col:
                return None
            templ[(dr, dc)] = col
    if not seen or not templ:
        return None
    if all(off == (0, 0) for off in templ):   # pure recolor / no move -> not us
        return None
    return templ


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _detect_translate(prs):
    a0, b0 = prs[0]
    ina = np.argwhere(a0 != 0)
    outb = np.argwhere(b0 != 0)
    if ina.size == 0 or outb.size == 0:
        return None
    dy = int(outb[:, 0].min() - ina[:, 0].min())
    dx = int(outb[:, 1].min() - ina[:, 1].min())
    if dy == 0 and dx == 0:
        return None
    if abs(dy) >= HEIGHT or abs(dx) >= WIDTH:
        return None
    for a, b in prs:
        if not np.array_equal(_shift_grid_allcolor(a, dy, dx), b):
            return None
    return (dy, dx)


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if any(a.shape != b.shape for a, b in prs):   # both rules preserve shape
        return []
    if all(np.array_equal(a, b) for a, b in prs):  # identity -> not us
        return []

    out = []

    # ---- rigid constant-displacement move ---------------------------------- #
    mv = _detect_translate(prs)
    if mv is not None:
        try:
            out.append((f"translate_{mv[0]}_{mv[1]}", build_translate(*mv)))
        except Exception:
            pass

    # ---- fixed template stamped at marker cells ---------------------------- #
    colors = set()
    for a, _ in prs:
        colors |= set(np.unique(a[a != 0]).tolist())
    for m in sorted(colors):
        hit = False
        for base_mode in ("zero", "input_nomark", "input"):
            templ = _infer_templ(prs, m, base_mode)
            if templ is None:
                continue
            ok = True
            for a, b in prs:
                if not np.array_equal(_ref_stamp(a, m, templ, base_mode), _onehot(b)):
                    ok = False
                    break
            if ok:
                try:
                    out.append((f"stamp_m{m}_{base_mode}_k{len(templ)}",
                                build_stamp(m, templ, base_mode)))
                    hit = True
                except Exception:
                    pass
                break
        if hit:
            break   # one marker colour suffices per task

    return out
