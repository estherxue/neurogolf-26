"""family_sgolf3_5 -- cheap EXACT solvers for MARKER-BOUNDED region tasks, built
from single-channel directional projections (Hillis-Steele doubling with Pad/Slice
shifts + Max/Mul), so every intermediate is a [1,1,30,30] mask (3600 B) rather than
a full [1,10,30,30] tensor.

Rules covered
-------------
* RECTFILL (task 273): the marker colour sits on the four corners of one or more
  axis-aligned rectangles; fill the *strict interior* of each rectangle with a fill
  colour. A cell is interior iff, in its column, a marker lies strictly above AND
  strictly below (call that VB), in its row a marker lies strictly left AND right
  (HB), and then a VB-cell lies strictly left AND right in its row AND an HB-cell
  lies strictly above AND below in its column. All of these are prefix-OR
  projections, expressible as origin-anchored shift+max doublings.

* CROSSPROJ (task 47): every seed shoots a full-row and full-column ray in its own
  colour; where a row-ray of one seed meets a column-ray of a *different*-coloured
  seed the cell becomes the conflict colour.

Everything is validated EXACTLY (numpy mirror == rule) on train+test+arc-gen before
emit, and the ONNX arithmetic mirrors the numpy reference cell-for-cell.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
_OFFS = [1, 2, 4, 8, 16]  # doubling covers distance 31 >= 29 (max within 30)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX arithmetic exactly)                        #
# --------------------------------------------------------------------------- #
def _excl_up(M):
    r = np.zeros_like(M); acc = np.zeros(M.shape[1], bool)
    for i in range(M.shape[0]):
        r[i] = acc; acc = acc | M[i]
    return r


def _excl_dn(M):
    r = np.zeros_like(M); acc = np.zeros(M.shape[1], bool)
    for i in range(M.shape[0] - 1, -1, -1):
        r[i] = acc; acc = acc | M[i]
    return r


def _excl_lf(M):
    return _excl_up(M.T).T


def _excl_rt(M):
    return _excl_dn(M.T).T


def rule_rectfill(a, mk, fc):
    M = (a == mk)
    Up = _excl_up(M); Dn = _excl_dn(M); Lf = _excl_lf(M); Rt = _excl_rt(M)
    VB = Up & Dn; HB = Lf & Rt
    VBl = _excl_lf(VB); VBr = _excl_rt(VB)
    HBu = _excl_up(HB); HBd = _excl_dn(HB)
    FILL = VBl & VBr & HBu & HBd & (a == 0)
    out = a.copy(); out[FILL] = fc
    return out


# --------------------------------------------------------------------------- #
# graph builder                                                                #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self, S=HEIGHT):
        self.nodes = []
        self.inits = []
        self._k = 0
        self.S = S
        self.offs = [o for o in _OFFS if o < S] or [1]

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def node(self, op, ins, out=None, **attrs):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out

    def const_i(self, vals):
        nm = self.nm("c")
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)], list(vals)))
        return nm

    def slice(self, src, starts, ends, axes, out=None):
        s = self.const_i(starts); e = self.const_i(ends); a = self.const_i(axes)
        return self.node("Slice", [src, s, e, a], out)

    def shift(self, src, sr, sc):
        """out[i,j] = src[i-sr, j-sc] zero-filled, on [1,1,S,S]."""
        if sr == 0 and sc == 0:
            return src
        pt, pb = max(sr, 0), max(-sr, 0)
        pl, pr = max(sc, 0), max(-sc, 0)
        padded = self.node("Pad", [src], mode="constant", value=0.0,
                           pads=[0, 0, pt, pl, 0, 0, pb, pr])
        return self.slice(padded, [pb, pr], [pb + self.S, pr + self.S], [2, 3])

    def fill(self, src, dr, dc):
        """inclusive prefix-OR along (dr,dc): a mark propagates in that direction."""
        cur = src
        for s in self.offs:
            sh = self.shift(cur, dr * s, dc * s)
            cur = self.node("Max", [cur, sh])
        return cur

    def excl(self, src, dr, dc):
        """strict projection: any mark strictly in the -(dr,dc) direction.
        e.g. excl_up = mark strictly above = fill downward then shift down 1."""
        f = self.fill(src, dr, dc)
        return self.shift(f, dr, dc)


def _model(g):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _chan(g, src, c):
    return g.slice(src, [c], [c + 1], [1])


def build_rectfill(mk, fc, S=HEIGHT):
    g = _G(S)
    if S < HEIGHT:
        src = g.slice("input", [0, 0], [S, S], [2, 3])
    else:
        src = "input"
    M = _chan(g, src, mk)
    BG = _chan(g, src, 0)
    Up = g.excl(M, 1, 0)      # mark strictly above
    Dn = g.excl(M, -1, 0)     # mark strictly below
    Lf = g.excl(M, 0, 1)      # mark strictly left
    Rt = g.excl(M, 0, -1)     # mark strictly right
    VB = g.node("Mul", [Up, Dn])
    HB = g.node("Mul", [Lf, Rt])
    VBl = g.excl(VB, 0, 1)
    VBr = g.excl(VB, 0, -1)
    HBu = g.excl(HB, 1, 0)
    HBd = g.excl(HB, -1, 0)
    F1 = g.node("Mul", [VBl, VBr])
    F2 = g.node("Mul", [HBu, HBd])
    F3 = g.node("Mul", [F1, F2])
    FILL = g.node("Mul", [F3, BG])          # only real background cells
    negFILL = g.node("Sub", [BG, BG])       # start a zero tensor
    negFILL = g.node("Sub", [negFILL, FILL])
    Z = g.node("Sub", [BG, BG])
    # assemble additive delta A over 10 channels: -FILL at 0, +FILL at fc
    pieces = []
    for c in range(CHANNELS):
        if c == 0:
            pieces.append(negFILL)
        elif c == fc:
            pieces.append(FILL)
        else:
            pieces.append(Z)
    A = g.node("Concat", pieces, axis=1)
    if S < HEIGHT:
        outs = g.node("Add", [src, A])
        g.node("Pad", [outs], "output", mode="constant", value=0.0,
               pads=[0, 0, 0, 0, 0, 0, HEIGHT - S, WIDTH - S])
    else:
        g.node("Add", ["input", A], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# detection / entry point                                                      #
# --------------------------------------------------------------------------- #
def rule_crossproj(a, conflict):
    h, w = a.shape
    out = np.zeros_like(a)
    rc = np.zeros(h, int); cc = np.zeros(w, int)
    rp = np.zeros(h, bool); cp = np.zeros(w, bool)
    for i in range(h):
        nz = a[i][a[i] != 0]
        if nz.size:
            rp[i] = True; rc[i] = nz[0]
    for j in range(w):
        nz = a[:, j][a[:, j] != 0]
        if nz.size:
            cp[j] = True; cc[j] = nz[0]
    for i in range(h):
        for j in range(w):
            if rp[i] and cp[j]:
                out[i, j] = rc[i] if rc[i] == cc[j] else conflict
            elif rp[i]:
                out[i, j] = rc[i]
            elif cp[j]:
                out[i, j] = cc[j]
    return out


def build_crossproj(seed_colors, conflict, S):
    """Requires a FIXED square grid cropped to SxS (no padding within the work area).
    Every seed shoots a full row/col ray in its colour; crossing of two different
    seed colours -> conflict colour."""
    g = _G(S)
    src = g.slice("input", [0, 0], [S, S], [2, 3]) if S < HEIGHT else "input"
    ones = g.node("ReduceSum", [src], axes=[1], keepdims=1)   # [1,1,S,S] all ones
    NB = g.node("Sub", [ones, _chan(g, src, 0)])              # non-background mask

    def rowproj(m):
        return g.node("Max", [g.fill(m, 0, 1), g.fill(m, 0, -1)])

    def colproj(m):
        return g.node("Max", [g.fill(m, 1, 0), g.fill(m, -1, 0)])

    RowPres = rowproj(NB)
    ColPres = colproj(NB)
    notRP = g.node("Sub", [ones, RowPres])
    notCP = g.node("Sub", [ones, ColPres])

    RowHas = {}; ColHas = {}
    for c in seed_colors:
        ch = _chan(g, src, c)
        RowHas[c] = rowproj(ch)
        ColHas[c] = colproj(ch)

    # same = OR_c (RowHas_c AND ColHas_c)
    same = None
    for c in seed_colors:
        s = g.node("Mul", [RowHas[c], ColHas[c]])
        same = s if same is None else g.node("Max", [same, s])
    notsame = g.node("Sub", [ones, same]) if same is not None else ones
    out_conf = g.node("Mul", [g.node("Mul", [RowPres, ColPres]), notsame])

    out_bg = g.node("Mul", [notRP, notCP])

    pieces = [None] * CHANNELS
    pieces[0] = out_bg
    for c in seed_colors:
        colHasOrNoC = g.node("Max", [ColHas[c], notCP])
        rowHasOrNoR = g.node("Max", [RowHas[c], notRP])
        t1 = g.node("Mul", [RowHas[c], colHasOrNoC])
        t2 = g.node("Mul", [ColHas[c], rowHasOrNoR])
        pieces[c] = g.node("Max", [t1, t2])
    # merge conflict colour
    if pieces[conflict] is None:
        pieces[conflict] = out_conf
    else:
        pieces[conflict] = g.node("Max", [pieces[conflict], out_conf])
    Z = g.node("Sub", [ones, ones])
    for c in range(CHANNELS):
        if pieces[c] is None:
            pieces[c] = Z
    cat = g.node("Concat", pieces, axis=1)
    if S < HEIGHT:
        g.node("Pad", [cat], "output", mode="constant", value=0.0,
               pads=[0, 0, 0, 0, 0, 0, HEIGHT - S, WIDTH - S])
    else:
        g.node("Identity", [cat], "output")
    return _model(g)


def _pairs(ex, splits):
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


def _try_rectfill(allp):
    # bg must be 0; input = {0, mk} single marker colour; output adds one fill colour
    in_colors = set()
    for a, _ in allp:
        in_colors |= set(np.unique(a).tolist())
    in_colors.discard(0)
    if len(in_colors) != 1:
        return None
    mk = in_colors.pop()
    fillset = set()
    for a, b in allp:
        fillset |= set(b[(a == 0) & (b != 0)].tolist())
    fillset.discard(mk)
    for fc in sorted(fillset):
        if all(np.array_equal(rule_rectfill(a, mk, fc), b) for a, b in allp):
            return mk, fc
    return None


def candidates(ex):
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    tt = _pairs(ex, ("train", "test"))
    if len(tt) < 1 or not allp:
        return []
    if not all(a.shape == b.shape for a, b in allp):
        return []
    if not any((a != b).any() for a, b in allp):
        return []

    # fixed square GxG across every input & output -> crop the work area to GxG
    sizes = {a.shape for a, _ in allp} | {b.shape for _, b in allp}
    Gsq = None
    if len(sizes) == 1:
        (h, w), = sizes
        if h == w and 1 <= h <= 27:
            Gsq = h

    out = []
    rf = _try_rectfill(allp)
    if rf is not None:
        mk, fc = rf
        for S in ([Gsq, HEIGHT] if Gsq else [HEIGHT]):
            try:
                out.append((f"rectfill_m{mk}_f{fc}_S{S}", build_rectfill(mk, fc, S)))
            except Exception:
                pass

    # cross-projection: only on a fixed square (crop to GxG so no padding leaks bg)
    if Gsq is not None:
        seed_colors = sorted(set().union(*[set(np.unique(a).tolist()) for a, _ in allp]) - {0})
        confset = sorted(set().union(*[set(np.unique(b).tolist()) for _, b in allp]))
        for conflict in confset:
            if all(np.array_equal(rule_crossproj(a, conflict), b) for a, b in allp):
                try:
                    out.append((f"crossproj_c{conflict}_S{Gsq}",
                                build_crossproj(seed_colors, conflict, Gsq)))
                except Exception:
                    pass
                break
    return out
