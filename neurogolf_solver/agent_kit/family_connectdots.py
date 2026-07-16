"""CONNECT-THE-DOTS family: join pairs of same-colour cells with a straight
segment (horizontal, vertical, and both diagonals), origin-anchored & opset-10.

Rule (generalises family_lines' LINE rule to diagonals and arbitrary line colour)
--------------------------------------------------------------------------------
For every colour, look at its cells.  Whenever two same-colour cells lie on the
same row / column / diagonal with only background between them, paint the cells
*between* them.  The painted colour can be

  * the endpoints' own colour                 ("draw line in the marker colour"),
  * a single fixed line colour for all pairs  ("connect with colour F"), or
  * a per-endpoint-colour line colour map     (colour c's pair -> colour cmap[c]).

All three are the same graph parameterised by a colour map ``cmap`` (identity ->
same-colour, constant -> fixed, otherwise -> general).

Realisation (exact & size-independent)
--------------------------------------
"Between a same-colour pair along an axis" = intersection of a directional fill
from each of the two opposing directions.  ``_dfill`` shoots a colour from each
non-bg seed along a direction, stopping at the real grid edge or any obstacle, via
a short Hillis-Steele DOUBLING chain of Conv->Clip steps (offsets 1,2,4,8,16 ->
covers any gap <=31 in a 30x30 grid).  The AND of the forward fill and the
backward fill (per colour channel) marks exactly the cells strictly between a
*consecutive* same-colour pair on that axis.  Multiple axes are merged with a
fixed axis priority (later axis wins at crossings), then routed through ``cmap``
into the output and unioned with the (preserved) input.

Every step keeps a strict 0/1 one-hot (Clip(0,1)); thresholds are exact for 0/1
inputs (the grader only checks output>0).  The PADDING GOTCHA is automatic: real
background carries channel 0 = 1 while padding is all-zero, so fills stop at the
real edge and never leak into the pad -> the rule is origin-anchored.

Detection mirrors the ONNX arithmetic exactly, infers ``cmap`` and the axis set
from the train/test pairs, and only emits a candidate after reproducing EVERY
available pair, so wrong hypotheses are dropped before the grader sees them.
"""
from __future__ import annotations

import itertools

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64

DIRS = {  # name -> (row step, col step) of propagation (content travels this way)
    'R': (0, 1), 'L': (0, -1), 'D': (1, 0), 'U': (-1, 0),
    'DR': (1, 1), 'DL': (1, -1), 'UR': (-1, 1), 'UL': (-1, -1),
}
NSTEPS = 5                       # doubling offsets 1,2,4,8,16 -> covers gap <=31
_OFFS = [1 << k for k in range(NSTEPS)]

# axis = (forward dir, backward dir); "between" = filled from both ends
AXES = [('R', 'L'), ('U', 'D'), ('DR', 'UL'), ('UR', 'DL')]


# --------------------------------------------------------------------------- #
# numpy reference (mirrors the ONNX arithmetic exactly)                        #
# --------------------------------------------------------------------------- #
def _onehot(g):
    t = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    h, w = g.shape
    for c in range(CHANNELS):
        t[c, :h, :w] = (g == c)
    return t


def _shift(t, sr, sc, d):
    """S[:,r,c] = t[:, r-sr*d, c-sc*d] with zero fill."""
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


def _all_fills(x, bg):
    return {dn: _dfill_np(x, dn, bg) for dn in DIRS}


def _A_np(need, axes, bg):
    """Per-endpoint-colour 'between' masks, axis priority = later axis wins."""
    A = None
    for p, q in axes:
        m = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
        for ch in range(CHANNELS):
            if ch != bg:
                m[ch] = np.clip(need[p][ch] + need[q][ch] - 1, 0, 1)
        if A is None:
            A = m
        else:
            filled = m.sum(0) - m[bg]
            newA = np.zeros_like(A)
            for ch in range(CHANNELS):
                if ch != bg:
                    keep = np.clip(A[ch] - filled, 0, 1)
                    newA[ch] = np.clip(keep + m[ch], 0, 1)
            A = newA
    return A


def _apply_np(need, x, axes, bg, cmap):
    A = _A_np(need, axes, bg)
    xbg = x[bg]                      # gate paint to REAL background cells (drop seeds)
    out = x.copy()
    Acol = np.zeros((HEIGHT, WIDTH), np.float32)
    for ch in range(CHANNELS):
        if ch != bg:
            A[ch] = A[ch] * xbg
            Acol = Acol + A[ch]
    out[bg] = np.clip(xbg - Acol, 0, 1)
    for ch in range(CHANNELS):
        if ch != bg:
            tgt = cmap.get(ch, ch)
            out[tgt] = np.clip(out[tgt] + A[ch], 0, 1)
    return out


def _from_oh(t, h, w):
    """Threshold >0 and decode; None if any real cell isn't exactly one-hot or
    padding carries content."""
    b = (t > 0)
    out = np.zeros((h, w), int)
    for i in range(h):
        for j in range(w):
            ch = np.where(b[:, i, j])[0]
            if len(ch) != 1:
                return None
            out[i, j] = ch[0]
    if b[:, h:, :].any() or b[:, :, w:].any():
        return None
    return out


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

    def init_i(self, dims, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, list(dims),
                                         [int(v) for v in np.asarray(vals).ravel().tolist()]))
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


def build_conn(axes, bg, cmap):
    g = _G()
    need = {}
    for p, q in axes:
        for dn in (p, q):
            if dn not in need:
                need[dn] = _dfill_chain(g, "input", dn, bg)

    # merge per-axis 'between' masks with priority (later axis wins at crossings)
    A = None
    for p, q in axes:
        m = _and_axis(g, need[p], need[q], bg)
        if A is None:
            A = m
        else:
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

    # gate the painted region to REAL background cells (drop endpoint seeds) so a
    # line colour different from the endpoint colour never recolours the markers.
    # For the same-colour case (cmap == identity) this is a no-op, so we skip it.
    if any(cmap.get(ch, ch) != ch for ch in range(CHANNELS) if ch != bg):
        s = g.node("Slice", ["input", g.init_i([1], [bg]), g.init_i([1], [bg + 1]),
                             g.init_i([1], [1])])        # [1,1,30,30] bg plane
        A = g.node("Mul", [A, s])                        # broadcast over channels

    # compose with input, routing A[ch] through cmap into the output channels
    catf = g.node("Concat", ["input", A], axis=1)
    Wf = np.zeros((CHANNELS, 2 * CHANNELS, 1, 1), np.float32)
    Bf = np.zeros((CHANNELS,), np.float32)
    for o in range(CHANNELS):
        Wf[o, o, 0, 0] = 1.0                       # carry input channel o
    for ch in range(CHANNELS):
        if ch != bg:
            Wf[bg, CHANNELS + ch, 0, 0] += -1.0    # remove painted cells from bg
            tgt = cmap.get(ch, ch)
            Wf[tgt, CHANNELS + ch, 0, 0] += 1.0    # add A[ch] into target colour
    conv = _conv1x1(g, catf, Wf, Bf)
    g.clip(conv, "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# detection / entry point                                                      #
# --------------------------------------------------------------------------- #
def _axis_combos():
    """Axis subsets, including orderings so the priority at crossings is tried."""
    combos = []
    seen = set()
    for r in range(1, len(AXES) + 1):
        for perm in itertools.permutations(AXES, r):
            key = tuple(perm)
            if key in seen:
                continue
            seen.add(key)
            combos.append(list(perm))
    return combos


_COMBOS = _axis_combos()


def _derive_cmap(prs_fills, bg):
    """Infer endpoint-colour -> line-colour map from how the 'between' cells are
    coloured in the targets.  Returns None on any inconsistency."""
    cmap = {}
    for a, b, need, axes in prs_fills:
        A = _A_np(need, axes, bg)
        h, w = a.shape
        for ch in range(CHANNELS):
            if ch == bg:
                continue
            sub = (A[ch][:h, :w] > 0) & (a == bg)   # real background cells only
            if not sub.any():
                continue
            vals = set(int(v) for v in b[sub].tolist())
            if len(vals) != 1:
                return None
            v = vals.pop()
            if ch in cmap and cmap[ch] != v:
                return None
            cmap[ch] = v
    return cmap if cmap else None


def _reproduces(fills, axes, bg, cmap):
    for a, b, need in fills:
        y = _apply_np(need, _onehot(a), axes, bg, cmap)
        dec = _from_oh(y, *b.shape)
        if dec is None or not np.array_equal(dec, b):
            return False
    return True


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


def _tag(cmap):
    if all(c == t for c, t in cmap.items()):
        return "same"
    tgts = set(cmap.values())
    if len(tgts) == 1:
        return "fix%d" % next(iter(tgts))
    return "map"


def candidates(ex):
    tt = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if len(tt) < 1:
        return []
    if not all(a.shape == b.shape for a, b in allp):
        return []

    bg = 0  # contract: background is channel 0, padding all-zero
    # input-preserving (markers/objects kept) and something added on bg cells
    for a, b in allp:
        if not np.array_equal(b[a != bg], a[a != bg]):
            return []
    if not any(((a == bg) & (b != bg)).any() for a, b in allp):
        return []

    # precompute the 8 directional fills per pair (reused across axis combos)
    tt_need = [(a, b, _all_fills(_onehot(a), bg)) for a, b in tt]

    for axes in _COMBOS:
        prs_fills = [(a, b, need, axes) for a, b, need in tt_need]
        cmap = _derive_cmap(prs_fills, bg)
        if cmap is None:
            continue
        if not _reproduces(tt_need, axes, bg, cmap):
            continue
        # validate on every available pair (the grader's exactness gate)
        all_need = [(a, b, _all_fills(_onehot(a), bg)) for a, b in allp]
        all_pf = [(a, b, need, axes) for a, b, need in all_need]
        cmap2 = _derive_cmap(all_pf, bg)
        if cmap2 is None:
            continue
        # merge maps (train+test colours take precedence; arc-gen may add colours)
        merged = dict(cmap2)
        merged.update(cmap)
        for c, t in cmap2.items():
            if c in cmap and cmap[c] != t:
                merged = None
                break
        if merged is None:
            continue
        if not _reproduces(all_need, axes, bg, merged):
            continue
        # "fixed line colour" generalisation: if every observed pair is joined
        # with ONE colour F, connect markers of ANY colour with F (covers held-out
        # marker colours never seen in the 70% arc-gen split).  Validation-safe:
        # unseen colours don't occur in the observed pairs, so this cannot regress.
        tgts = set(merged.values())
        identity = all(c == t for c, t in merged.items())
        if len(tgts) == 1 and not identity:
            F_col = next(iter(tgts))
            merged = {ch: F_col for ch in range(CHANNELS) if ch != bg}
            if not _reproduces(all_need, axes, bg, merged):
                continue
        try:
            model = build_conn(axes, bg, merged)
            onnx.checker.check_model(model, full_check=True)
        except Exception:
            return []
        tag = "_".join(p + q for p, q in axes)
        return [(f"connect_{_tag(merged)}_{tag}", model)]

    return []
