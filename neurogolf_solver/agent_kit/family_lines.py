"""LINE / RAY drawing family via UNROLLED directional propagation (Conv + Clip).

Two rules, both origin-anchored and exact under the top-left zero-pad contract:

  * RAY  -- from every non-background seed cell, shoot a ray in a fixed set of
            directions, colouring background cells with the seed's colour until
            the (real) grid edge or another non-background cell (obstacle).
  * LINE -- connect pairs of *consecutive, same-colour* seeds that lie on the
            same row / column / diagonal with a straight segment of that colour.

Both are bounded propagations, so we UNROLL them as a short chain of identical
Conv->Clip steps.  Each step uses a Hillis-Steele *doubling* offset (1,2,4,8,16)
so 5 steps cover any distance <=31 within a 30x30 grid -- only ~9 intermediate
[1,10,30,30] tensors instead of 29 naive single-cell steps.

The single per-step op is a Conv whose kernel reads (a) the current cell and
(b) the cell `d` steps back along the propagation direction, followed by a
Clip(0,1) so every intermediate stays a strict 0/1 one-hot.  Concretely, for a
background cell to be painted colour k it must currently be background and its
source cell must be colour k:

    new_k  = Clip( 2*x_k(cur) + x_bg(cur) + x_k(src) - 1 , 0, 1)   (k != bg)
    new_bg = Clip(    x_bg(cur) - sum_{j!=bg} x_j(src)   , 0, 1)

These thresholds are exact for 0/1 inputs (the grader only checks output>0).
The PADDING GOTCHA is handled automatically: real background cells carry
channel `bg`=1 while padding cells are all-zero, so rays stop at the real edge
and never leak into the pad.

Weights are built analytically (we know the rule); the harness is the final
judge of exactness over train+test+arc-gen.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64

DIRS = {  # name -> (row step, col step) of *propagation* (content travels this way)
    'R': (0, 1), 'L': (0, -1), 'D': (1, 0), 'U': (-1, 0),
    'DR': (1, 1), 'DL': (1, -1), 'UR': (-1, 1), 'UL': (-1, -1),
}
NSTEPS = 5  # doubling offsets 1,2,4,8,16 -> covers distance 31 >= 29 (max in 30)
_OFFS = [1 << k for k in range(NSTEPS)]


# --------------------------------------------------------------------------- #
# numpy reference (used for detection; mirrors the ONNX arithmetic exactly)    #
# --------------------------------------------------------------------------- #
def _onehot(g):
    t = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    h, w = g.shape
    for c in range(CHANNELS):
        t[c, :h, :w] = (g == c)
    return t


def _shift(t, sr, sc, d):
    """S[:,r,c] = t[:, r-sr*d, c-sc*d] with zero fill (content moved by sr*d,sc*d)."""
    S = np.zeros_like(t)
    dr, dc = sr * d, sc * d
    rs0, rs1 = max(dr, 0), HEIGHT + min(dr, 0)
    cs0, cs1 = max(dc, 0), WIDTH + min(dc, 0)
    if rs0 < rs1 and cs0 < cs1:
        S[:, rs0:rs1, cs0:cs1] = t[:, rs0 - dr:rs1 - dr, cs0 - dc:cs1 - dc]
    return S


def _dfill_np(x, dname, bg):
    sr, sc = DIRS[dname]
    cur = x.copy()
    for d in _OFFS:
        s = _shift(cur, sr, sc, d)
        x_bg = cur[bg]
        col_src = s.sum(0) - s[bg]
        new = np.zeros_like(cur)
        for ch in range(CHANNELS):
            if ch == bg:
                new[ch] = np.clip(x_bg - col_src, 0, 1)
            else:
                new[ch] = np.clip(2 * cur[ch] + x_bg + s[ch] - 1, 0, 1)
        cur = new
    return cur


def _ray_np(x, dirs, bg):
    fills = [_dfill_np(x, dn, bg) for dn in dirs]
    if len(fills) == 1:
        return fills[0]
    tot = sum(fills)
    nd = len(fills)
    out = np.zeros_like(x)
    for ch in range(CHANNELS):
        if ch == bg:
            out[ch] = np.clip(tot[ch] - (nd - 1), 0, 1)
        else:
            out[ch] = np.clip(tot[ch], 0, 1)
    return out


def _conn_np(x, axes, bg, fill=None):
    need = {}
    for p, q in axes:
        for dn in (p, q):
            need.setdefault(dn, _dfill_np(x, dn, bg))
    if fill is not None:
        # fixed-colour segments: paint every (currently-background) cell that lies
        # between a same-colour consecutive pair with the single colour `fill`.
        mask = np.zeros((HEIGHT, WIDTH), np.float32)
        for p, q in axes:
            for ch in range(CHANNELS):
                if ch != bg:
                    mask = np.clip(mask + np.clip(need[p][ch] + need[q][ch] - 1, 0, 1), 0, 1)
        mask = np.clip(mask + x[bg] - 1, 0, 1)  # only real background cells
        out = x.copy()
        out[bg] = np.clip(x[bg] - mask, 0, 1)
        out[fill] = np.clip(x[fill] + mask, 0, 1)
        return out
    A = np.zeros_like(x)
    first = True
    for p, q in axes:
        m = np.zeros_like(x)
        for ch in range(CHANNELS):
            if ch != bg:
                m[ch] = np.clip(need[p][ch] + need[q][ch] - 1, 0, 1)
        if first:
            A = m
            first = False
        else:
            filled = m.sum(0) - m[bg]
            newA = np.zeros_like(A)
            for ch in range(CHANNELS):
                if ch != bg:
                    keep = np.clip(A[ch] - filled, 0, 1)
                    newA[ch] = np.clip(keep + m[ch], 0, 1)
            A = newA
    Acol = A.sum(0) - A[bg]
    out = np.zeros_like(x)
    for ch in range(CHANNELS):
        if ch == bg:
            out[ch] = np.clip(x[ch] - Acol, 0, 1)
        else:
            out[ch] = np.clip(x[ch] + A[ch], 0, 1)
    return out


def _from_oh(t, h, w):
    """Threshold >0 and decode; returns int grid or None if any real cell is not
    exactly one-hot (padding region ignored, must be all <=0 within h..,w..)."""
    b = (t > 0)
    out = np.zeros((h, w), int)
    for i in range(h):
        for j in range(w):
            ch = np.where(b[:, i, j])[0]
            if len(ch) != 1:
                return None
            out[i, j] = ch[0]
    # padding must be empty
    if b[:, h:, :].any() or b[:, :, w:].any():
        return None
    return out


def _match(prs, fn, bg):
    for a, b in prs:
        y = fn(_onehot(a), bg)
        dec = _from_oh(y, *b.shape)
        if dec is None or not np.array_equal(dec, b):
            return False
    return True


# --------------------------------------------------------------------------- #
# ONNX graph construction                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def init_f(self, dims, vals):
        nm = self.name("w")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(dims),
                                         np.asarray(vals, np.float32).ravel().tolist()))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out

    def clip(self, src, out=None):
        return self.node("Clip", [src], out, min=0.0, max=1.0)


def _model(g):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _step_weights(sr, sc, d, bg):
    KH = abs(sr) * d + 1
    KW = abs(sc) * d + 1
    pt = d if sr == 1 else 0
    pl = d if sc == 1 else 0
    kr_cur, kc_cur = pt, pl
    kr_src, kc_src = pt - sr * d, pl - sc * d
    pb, pr = KH - 1 - pt, KW - 1 - pl
    W = np.zeros((CHANNELS, CHANNELS, KH, KW), np.float32)
    B = np.zeros((CHANNELS,), np.float32)
    for ch in range(CHANNELS):
        if ch == bg:
            W[bg, bg, kr_cur, kc_cur] += 1.0
            for j in range(CHANNELS):
                if j != bg:
                    W[bg, j, kr_src, kc_src] += -1.0
            B[bg] = 0.0
        else:
            W[ch, ch, kr_cur, kc_cur] += 2.0
            W[ch, bg, kr_cur, kc_cur] += 1.0
            W[ch, ch, kr_src, kc_src] += 1.0
            B[ch] = -1.0
    return W, B, KH, KW, [pt, pl, pb, pr]


def _dfill_chain(g, src, dname, bg):
    sr, sc = DIRS[dname]
    cur = src
    for d in _OFFS:
        W, B, KH, KW, pads = _step_weights(sr, sc, d, bg)
        wt = g.init_f([CHANNELS, CHANNELS, KH, KW], W)
        bt = g.init_f([CHANNELS], B)
        conv = g.node("Conv", [cur, wt, bt], kernel_shape=[KH, KW], pads=pads)
        cur = g.clip(conv)
    return cur


def _conv1x1(g, src, W, B, out=None):
    cin = W.shape[1]
    wt = g.init_f([CHANNELS, cin, 1, 1], W)
    bt = g.init_f([CHANNELS], B)
    return g.node("Conv", [src, wt, bt], out, kernel_shape=[1, 1], pads=[0, 0, 0, 0])


def build_ray(dirs, bg):
    g = _G()
    fills = [_dfill_chain(g, "input", dn, bg) for dn in dirs]
    if len(fills) == 1:
        # last clip of the single chain must be the output
        last = g.nodes[-1]
        last.output[0] = "output"
        # also rename any node consuming it (none) -- it's terminal
        return _model(g)
    nd = len(fills)
    tot = g.node("Sum", fills)
    W = np.zeros((CHANNELS, CHANNELS, 1, 1), np.float32)
    B = np.zeros((CHANNELS,), np.float32)
    for ch in range(CHANNELS):
        W[ch, ch, 0, 0] = 1.0
    B[bg] = -(nd - 1)
    conv = _conv1x1(g, tot, W, B)
    g.clip(conv, "output")
    return _model(g)


def _and_axis(g, fp, fq, bg):
    cat = g.node("Concat", [fp, fq], axis=1)
    W = np.zeros((CHANNELS, 2 * CHANNELS, 1, 1), np.float32)
    B = np.zeros((CHANNELS,), np.float32)
    for ch in range(CHANNELS):
        if ch != bg:
            W[ch, ch, 0, 0] = 1.0
            W[ch, CHANNELS + ch, 0, 0] = 1.0
            B[ch] = -1.0
    conv = _conv1x1(g, cat, W, B)
    return g.clip(conv)


def build_conn(axes, bg, fill=None):
    g = _G()
    need = {}
    for p, q in axes:
        for dn in (p, q):
            if dn not in need:
                need[dn] = _dfill_chain(g, "input", dn, bg)
    if fill is not None:
        # OR of per-axis "between same-colour pair" masks -> single fill colour
        ms = [_and_axis(g, need[p], need[q], bg) for p, q in axes]
        M = ms[0] if len(ms) == 1 else g.node("Sum", ms)
        # between = Clip(sum_colour(M)) -> 1 channel, capped to 0/1 (a cell may be
        # between pairs on several axes -> raw sum can exceed 1)
        Wb = np.zeros((1, CHANNELS, 1, 1), np.float32)
        for j in range(CHANNELS):
            if j != bg:
                Wb[0, j, 0, 0] = 1.0
        wtb = g.init_f([1, CHANNELS, 1, 1], Wb)
        btb = g.init_f([1], np.zeros((1,), np.float32))
        between = g.clip(g.node("Conv", [M, wtb, btb], kernel_shape=[1, 1], pads=[0, 0, 0, 0]))
        # mask = Clip( between + x_bg - 1 )  (gate to real background only)
        cat = g.node("Concat", ["input", between], axis=1)  # 11 channels
        Wm = np.zeros((1, CHANNELS + 1, 1, 1), np.float32)
        Wm[0, bg, 0, 0] = 1.0
        Wm[0, CHANNELS, 0, 0] = 1.0
        wt = g.init_f([1, CHANNELS + 1, 1, 1], Wm)
        bt = g.init_f([1], np.array([-1.0], np.float32))
        maskpre = g.node("Conv", [cat, wt, bt], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
        mask = g.clip(maskpre)
        # output: out_fill = x_fill + mask ; out_bg = x_bg - mask ; others = x
        cat2 = g.node("Concat", ["input", mask], axis=1)  # 11 channels
        Wf = np.zeros((CHANNELS, CHANNELS + 1, 1, 1), np.float32)
        for ch in range(CHANNELS):
            Wf[ch, ch, 0, 0] = 1.0
        Wf[fill, CHANNELS, 0, 0] += 1.0
        Wf[bg, CHANNELS, 0, 0] += -1.0
        wt2 = g.init_f([CHANNELS, CHANNELS + 1, 1, 1], Wf)
        bt2 = g.init_f([CHANNELS], np.zeros((CHANNELS,), np.float32))
        conv = g.node("Conv", [cat2, wt2, bt2], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
        g.clip(conv, "output")
        return _model(g)
    A = None
    for p, q in axes:
        m = _and_axis(g, need[p], need[q], bg)
        if A is None:
            A = m
        else:
            # keep_ch = Clip(A_ch - sum_j m_j) ; A = Clip(keep + m)
            cat = g.node("Concat", [A, m], axis=1)
            Wk = np.zeros((CHANNELS, 2 * CHANNELS, 1, 1), np.float32)
            for ch in range(CHANNELS):
                Wk[ch, ch, 0, 0] = 1.0
                for j in range(CHANNELS):
                    if j != bg:
                        Wk[ch, CHANNELS + j, 0, 0] += -1.0
            keep = g.clip(_conv1x1(g, cat, Wk, np.zeros((CHANNELS,), np.float32)))
            cat2 = g.node("Concat", [keep, m], axis=1)
            Wa = np.zeros((CHANNELS, 2 * CHANNELS, 1, 1), np.float32)
            for ch in range(CHANNELS):
                Wa[ch, ch, 0, 0] = 1.0
                Wa[ch, CHANNELS + ch, 0, 0] = 1.0
            A = g.clip(_conv1x1(g, cat2, Wa, np.zeros((CHANNELS,), np.float32)))
    # final compose with the original input
    catf = g.node("Concat", ["input", A], axis=1)
    Wf = np.zeros((CHANNELS, 2 * CHANNELS, 1, 1), np.float32)
    Bf = np.zeros((CHANNELS,), np.float32)
    for ch in range(CHANNELS):
        if ch == bg:
            Wf[bg, bg, 0, 0] = 1.0
            for j in range(CHANNELS):
                if j != bg:
                    Wf[bg, CHANNELS + j, 0, 0] += -1.0
        else:
            Wf[ch, ch, 0, 0] = 1.0
            Wf[ch, CHANNELS + ch, 0, 0] = 1.0
    conv = _conv1x1(g, catf, Wf, Bf)
    g.clip(conv, "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# detection / entry point                                                      #
# --------------------------------------------------------------------------- #
_RAY_SETS = [
    ['D'], ['U'], ['L'], ['R'], ['DR'], ['DL'], ['UR'], ['UL'],
    ['U', 'D'], ['L', 'R'], ['DR', 'UL'], ['UR', 'DL'],
    ['U', 'D', 'L', 'R'], ['UL', 'UR', 'DL', 'DR'],
    ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR'],
]
_CONN_SETS = [
    [('R', 'L')], [('U', 'D')], [('DR', 'UL')], [('UR', 'DL')],
    [('R', 'L'), ('U', 'D')], [('U', 'D'), ('R', 'L')],
    [('DR', 'UL'), ('UR', 'DL')], [('UR', 'DL'), ('DR', 'UL')],
    [('R', 'L'), ('U', 'D'), ('DR', 'UL'), ('UR', 'DL')],
]


def _pairs(ex, splits):
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


def _bg(prs):
    bgs = set()
    for a, _ in prs:
        vals, cnt = np.unique(a, return_counts=True)
        bgs.add(int(vals[np.argmax(cnt)]))
    return bgs.pop() if len(bgs) == 1 else None


def candidates(ex):
    tt = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if len(tt) < 1:
        return []
    if not all(a.shape == b.shape for a, b in allp):
        return []
    bg = _bg(allp)
    if bg is None or not (0 <= bg < CHANNELS):
        return []
    # additive / input-preserving: every non-background input cell is kept
    for a, b in allp:
        if not np.array_equal(b[a != bg], a[a != bg]):
            return []
    # something must change somewhere (else it's identity, not our family)
    if not any((a != b).any() for a, b in allp):
        return []

    out = []
    for dirs in _RAY_SETS:
        fn = lambda x, g, d=dirs: _ray_np(x, d, g)
        if _match(tt, fn, bg) and _match(allp, fn, bg):
            try:
                out.append(("ray_" + "".join(dirs) + f"_bg{bg}", build_ray(dirs, bg)))
            except Exception:
                pass
            break
    conn_done = False
    for axes in _CONN_SETS:
        fn = lambda x, g, ax=axes: _conn_np(x, ax, g)
        if _match(tt, fn, bg) and _match(allp, fn, bg):
            tag = "_".join(p + q for p, q in axes)
            try:
                out.append((f"line_{tag}_bg{bg}", build_conn(axes, bg)))
                conn_done = True
            except Exception:
                pass
            break
    if not conn_done:
        # fixed-colour segments: candidate fill colours = colours that appear on
        # added (background->non-background) cells.
        fillset = set()
        for a, b in tt:
            fillset |= set(b[(a == bg) & (b != bg)].tolist())
        fillset.discard(bg)
        for fc in sorted(fillset):
            for axes in _CONN_SETS:
                fn = lambda x, g, ax=axes, F=fc: _conn_np(x, ax, g, fill=F)
                if _match(tt, fn, bg) and _match(allp, fn, bg):
                    tag = "_".join(p + q for p, q in axes)
                    try:
                        out.append((f"lineF{fc}_{tag}_bg{bg}", build_conn(axes, bg, fill=fc)))
                        conn_done = True
                    except Exception:
                        pass
                    break
            if conn_done:
                break
    return out
