"""COUNT / HISTOGRAM OUTPUT family (origin-anchored, opset 10).

Every rule here turns the input into a single bounded INTEGER COUNT and then
renders that count as the output grid.  The count is always a plain reduction of
the zero-padded one-hot tensor (padding is all-zero so it never contributes), and
every output is anchored at the top-left, so the rules generalise to grids of any
(variable) size.

Count sources (all computed with ReduceSum/Conv on the one-hot tensor)
---------------------------------------------------------------------
  total          number of real non-background cells       (ReduceSum ch1..9)
  color:k        number of cells of a fixed colour k       (ReduceSum ch k)
  distinct       number of distinct present non-bg colours  (Greater>0 then sum)
  squares:k:s    number of solid sxs squares of colour k    (sxs ones Conv == s*s)
  ncomp          number of 4-connected non-bg components    (label-propagation CA
                 then count component "roots", cell where label == its own id)

Output renderings (the matching one is inferred from the train/test/arc-gen pairs)
---------------------------------------------------------------------------------
  tightbar       output grid is a SOLID monochrome block whose extent encodes the
                 count: a 1xN row, an Nx1 column, or an NxN square (N = count).
                 The block colour is either fixed or the (unique) present colour.
                 Built from a position-index `Less(pos, count)` mask -> generalises
                 to ANY count (no per-count template).
  fieldbar       output grid is a FIXED-size field; the first N cells of row0 (or
                 col0) are a fixed colour, the remaining field cells background.
                 (A "bar in a box" -- e.g. count the 2x2 squares.)
  swatchmap      output grid is a FIXED-size field flood-filled with the colour
                 given by a small inferred  count -> colour  table (used when the
                 count is a connected-component count, a bounded set).
  fragswatch     output grid is a FIXED-size field flood-filled with the most
                 FRAGMENTED present colour: argmax over colours of
                 (#components / #cells), realised with per-colour component counts
                 and a cross-multiplied ratio comparison (no Div).

Detection mirrors the ONNX semantics exactly and only emits a candidate when it
reproduces EVERY available pair, so wrong hypotheses are dropped before scoring.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
INT32 = onnx.TensorProto.INT32
F = DATA_TYPE
H, W = HEIGHT, WIDTH


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0
        self._cache = {}

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def cf(self, dims, vals):
        n = self.nm("cf")
        self.inits.append(oh.make_tensor(n, F, list(dims),
                                         [float(v) for v in np.asarray(vals).ravel()]))
        return n

    def ci(self, dims, vals, dt=INT64):
        n = self.nm("ci")
        self.inits.append(oh.make_tensor(n, dt, list(dims),
                                         [int(v) for v in np.asarray(vals).ravel()]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _nbg():
    return [0.0] + [1.0] * (CHANNELS - 1)


def _onehot(k):
    return [1.0 if c == k else 0.0 for c in range(CHANNELS)]


# --------------------------------------------------------------------------- #
# shared graph fragments                                                       #
# --------------------------------------------------------------------------- #
def _pos_col(g):
    if "pc" not in g._cache:
        g._cache["pc"] = g.cf([1, 1, 1, W], np.arange(W).reshape(1, 1, 1, W))
    return g._cache["pc"]


def _pos_row(g):
    if "pr" not in g._cache:
        g._cache["pr"] = g.cf([1, 1, H, 1], np.arange(H).reshape(1, 1, H, 1))
    return g._cache["pr"]


def _half(g):
    if "half" not in g._cache:
        g._cache["half"] = g.cf([1, 1, 1, 1], [0.5])
    return g._cache["half"]


def _cnt_ch(g):
    """per-channel cell count [1,10,1,1] (channel0 = real bg cells)."""
    if "cntch" not in g._cache:
        g._cache["cntch"] = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    return g._cache["cntch"]


def _col_lt(g, C):
    """[1,1,1,W] float mask, 1 where column index < C."""
    return g.nd("Cast", [g.nd("Less", [_pos_col(g), C])], to=F)


def _row_lt(g, C):
    """[1,1,H,1] float mask, 1 where row index < C."""
    return g.nd("Cast", [g.nd("Less", [_pos_row(g), C])], to=F)


def _fieldmask(g, fh, fw):
    """[1,1,H,W] float mask, 1 over the top-left fh x fw region."""
    fhc = g.cf([1, 1, 1, 1], [fh - 0.5])
    fwc = g.cf([1, 1, 1, 1], [fw - 0.5])
    rm = g.nd("Cast", [g.nd("Less", [_pos_row(g), fhc])], to=F)   # [1,1,H,1]
    cm = g.nd("Cast", [g.nd("Less", [_pos_col(g), fwc])], to=F)   # [1,1,1,W]
    return g.nd("Mul", [rm, cm])                                  # [1,1,H,W]


# ----- count-source ONNX (-> scalar [1,1,1,1]) ----------------------------- #
def _src_total(g):
    nbg = g.cf([1, CHANNELS, 1, 1], _nbg())
    return g.nd("ReduceSum", [g.nd("Mul", [_cnt_ch(g), nbg])], axes=[1], keepdims=1)


def _src_color(g, k):
    e = g.cf([1, CHANNELS, 1, 1], _onehot(k))
    return g.nd("ReduceSum", [g.nd("Mul", [_cnt_ch(g), e])], axes=[1], keepdims=1)


def _src_distinct(g):
    nbg = g.cf([1, CHANNELS, 1, 1], _nbg())
    pres = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [_cnt_ch(g), _half(g)])], to=F), nbg])
    return g.nd("ReduceSum", [pres], axes=[1], keepdims=1)


def _src_squares(g, k, s):
    Wt = np.zeros((1, CHANNELS, s, s), np.float32)
    Wt[0, k, :, :] = 1.0
    wt = g.cf([1, CHANNELS, s, s], Wt)
    conv = g.nd("Conv", ["input", wt], kernel_shape=[s, s], pads=[0, 0, 0, 0])
    thr = g.cf([1, 1, 1, 1], [s * s - 0.5])
    sq = g.nd("Cast", [g.nd("Greater", [conv, thr])], to=F)
    return g.nd("ReduceSum", [sq], axes=[2, 3], keepdims=1)


def _shift_fns(g):
    """return a shift(x,dr,dc) closure (zero-filled translation) over axes 2,3."""
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    sc = {}
    for dr, dc in dirs:
        sh, sw = max(-dr, 0), max(-dc, 0)
        sc[(dr, dc)] = (g.ci([2], [sh, sw]), g.ci([2], [sh + H, sw + W]), g.ci([2], [2, 3]))

    def shift(x, dr, dc):
        pt, pb = max(dr, 0), max(-dr, 0)
        pl, pr = max(dc, 0), max(-dc, 0)
        p = g.nd("Pad", [x], mode="constant", value=0.0, pads=[0, 0, pt, pl, 0, 0, pb, pr])
        st, en, ax = sc[(dr, dc)]
        return g.nd("Slice", [p, st, en, ax])

    return shift, dirs


def _src_ncomp(g, T):
    """number of 4-connected non-bg components (single-channel CA)."""
    shift, dirs = _shift_fns(g)
    nbg = g.cf([1, CHANNELS, 1, 1], _nbg())
    M = g.nd("ReduceSum", [g.nd("Mul", ["input", nbg])], axes=[1], keepdims=1)  # [1,1,H,W]
    P = g.cf([1, 1, H, W], np.arange(1, H * W + 1).reshape(1, 1, H, W))
    L = g.nd("Mul", [M, P])
    for _ in range(T):
        mx = g.nd("Max", [L] + [shift(L, dr, dc) for dr, dc in dirs])
        L = g.nd("Mul", [mx, M])
    root = g.nd("Cast", [g.nd("Equal", [g.nd("Cast", [L], to=INT32),
                                        g.nd("Cast", [P], to=INT32)])], to=F)
    root = g.nd("Mul", [root, M])
    return g.nd("ReduceSum", [root], axes=[1, 2, 3], keepdims=1)


_SRC = {
    "total": lambda g, p: _src_total(g),
    "distinct": lambda g, p: _src_distinct(g),
    "color": lambda g, p: _src_color(g, p[0]),
    "squares": lambda g, p: _src_squares(g, p[0], p[1]),
    "ncomp": lambda g, p: _src_ncomp(g, p[0]),
}


# ----- colour-vector ONNX (-> [1,10,1,1]) ---------------------------------- #
def _colorvec_present(g):
    nbg = g.cf([1, CHANNELS, 1, 1], _nbg())
    return g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [_cnt_ch(g), _half(g)])], to=F), nbg])


def _colorvec_fixed(g, k):
    return g.cf([1, CHANNELS, 1, 1], _onehot(k))


# --------------------------------------------------------------------------- #
# ONNX builders                                                                #
# --------------------------------------------------------------------------- #
def _count_scalar(g, src):
    return _SRC[src[0]](g, src[1:])


def build_tightbar(orient, src, color_mode, fixed_color):
    g = _G()
    C = _count_scalar(g, src)
    if orient == "row":
        mask = g.nd("Mul", [_row_lt(g, _half(g)), _col_lt(g, C)])      # row0 & col<C
    elif orient == "col":
        mask = g.nd("Mul", [_col_lt(g, _half(g)), _row_lt(g, C)])      # col0 & row<C
    else:                                                              # square
        mask = g.nd("Mul", [_row_lt(g, C), _col_lt(g, C)])
    cv = _colorvec_present(g) if color_mode == "present" else _colorvec_fixed(g, fixed_color)
    g.nd("Mul", [cv, mask], "output")
    return _model(g)


def build_fieldbar(axis, fh, fw, k, src):
    g = _G()
    C = _count_scalar(g, src)
    field = _fieldmask(g, fh, fw)
    if axis == "row":
        bar = g.nd("Mul", [g.nd("Mul", [_row_lt(g, _half(g)), _col_lt(g, C)]), field])
    else:
        bar = g.nd("Mul", [g.nd("Mul", [_col_lt(g, _half(g)), _row_lt(g, C)]), field])
    bg = g.nd("Sub", [field, bar])
    ek = g.cf([1, CHANNELS, 1, 1], _onehot(k))
    e0 = g.cf([1, CHANNELS, 1, 1], _onehot(0))
    out = g.nd("Add", [g.nd("Mul", [bar, ek]), g.nd("Mul", [bg, e0])])
    g.nodes[-1].output[0] = "output"
    return _model(g)


def build_swatchmap(fh, fw, src, table):
    """table: dict {count_value -> colour}. Flood-fill fh x fw with table[count]."""
    g = _G()
    C = _count_scalar(g, src)
    field = _fieldmask(g, fh, fw)
    cv = None
    for v, col in table.items():
        lo = g.cf([1, 1, 1, 1], [v - 0.5])
        hi = g.cf([1, 1, 1, 1], [v + 0.5])
        ind = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [C, lo])], to=F),
                           g.nd("Cast", [g.nd("Less", [C, hi])], to=F)])      # [1,1,1,1]
        e = g.cf([1, CHANNELS, 1, 1], _onehot(col))
        term = g.nd("Mul", [ind, e])                                          # [1,10,1,1]
        cv = term if cv is None else g.nd("Add", [cv, term])
    g.nd("Mul", [cv, field], "output")
    return _model(g)


def _percolor_compcount(g, T):
    """per-colour 4-connected component count -> comp [1,10,1,1]; cnt [1,10,1,1]."""
    shift, dirs = _shift_fns(g)
    nbg = g.cf([1, CHANNELS, 1, 1], _nbg())
    Mc = g.nd("Mul", ["input", nbg])                          # [1,10,H,W] (ch0 zeroed)
    P = g.cf([1, 1, H, W], np.arange(1, H * W + 1).reshape(1, 1, H, W))
    L = g.nd("Mul", [Mc, P])
    for _ in range(T):
        mx = g.nd("Max", [L] + [shift(L, dr, dc) for dr, dc in dirs])
        L = g.nd("Mul", [mx, Mc])
    root = g.nd("Cast", [g.nd("Equal", [g.nd("Cast", [L], to=INT32),
                                        g.nd("Cast", [P], to=INT32)])], to=F)  # [1,10,H,W]
    root = g.nd("Mul", [root, Mc])
    comp = g.nd("ReduceSum", [root], axes=[2, 3], keepdims=1)  # [1,10,1,1]
    cnt = g.nd("ReduceSum", [Mc], axes=[2, 3], keepdims=1)     # [1,10,1,1]
    return comp, cnt


def build_fragswatch(fh, fw, T):
    """Flood-fill fh x fw with argmax_c (comp_c / cnt_c) over present non-bg c
    (low-index tie-break).  comp_a/cnt_a > comp_b/cnt_b  <=>  comp_a*cnt_b > comp_b*cnt_a."""
    g = _G()
    comp, cnt = _percolor_compcount(g, T)                     # [1,10,1,1] (axis1 = colour c)
    nbg = g.cf([1, CHANNELS, 1, 1], _nbg())
    half = _half(g)
    present = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [cnt, half])], to=F), nbg])
    presT = g.nd("Transpose", [present], perm=[1, 0, 2, 3])    # [10,1,1,1] (axis0 = d)
    compT = g.nd("Transpose", [comp], perm=[1, 0, 2, 3])
    cntT = g.nd("Transpose", [cnt], perm=[1, 0, 2, 3])
    # A[d,c] = comp_c*cnt_d - comp_d*cnt_c  -> [10,10,1,1]
    # A>0 <=> ratio_c>ratio_d (c beats d);  A<0 <=> ratio_d>ratio_c (d beats c)
    A = g.nd("Sub", [g.nd("Mul", [comp, cntT]), g.nd("Mul", [compT, cnt])])
    neg_half = g.cf([1, 1, 1, 1], [-0.5])
    gt = g.nd("Cast", [g.nd("Less", [A, neg_half])], to=F)                     # ratio_d>ratio_c
    eq = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [A]), half])], to=F)          # ratio_d==ratio_c
    ltidx = np.zeros((CHANNELS, CHANNELS, 1, 1), np.float32)                   # lt[d,c]=1 if d<c
    for d in range(CHANNELS):
        for c in range(CHANNELS):
            if d < c:
                ltidx[d, c, 0, 0] = 1.0
    lt = g.cf([CHANNELS, CHANNELS, 1, 1], ltidx)
    beats = g.nd("Max", [gt, g.nd("Mul", [eq, lt])])                           # d beats c
    lose = g.nd("ReduceSum", [g.nd("Mul", [presT, beats])], axes=[0], keepdims=1)  # [1,10,1,1]
    win = g.nd("Mul", [present, g.nd("Cast", [g.nd("Less", [lose, half])], to=F)])
    field = _fieldmask(g, fh, fw)
    g.nd("Mul", [win, field], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX semantics for detection)                  #
# --------------------------------------------------------------------------- #
def _comps_nonbg(a):
    h, w = a.shape
    seen = np.zeros((h, w), bool)
    n = 0
    for i in range(h):
        for j in range(w):
            if seen[i, j] or a[i, j] == 0:
                continue
            n += 1
            q = deque([(i, j)])
            seen[i, j] = True
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and a[nr, nc] != 0:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return n


def _comps_color(a, col):
    h, w = a.shape
    seen = np.zeros((h, w), bool)
    n = 0
    for i in range(h):
        for j in range(w):
            if seen[i, j] or a[i, j] != col:
                continue
            n += 1
            q = deque([(i, j)])
            seen[i, j] = True
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and a[nr, nc] == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return n


def _count_squares(a, k, s):
    h, w = a.shape
    m = (a == k).astype(int)
    n = 0
    for i in range(h - s + 1):
        for j in range(w - s + 1):
            if m[i:i + s, j:j + s].sum() == s * s:
                n += 1
    return n


def _diam_nonbg(a):
    h, w = a.shape
    seen = np.zeros((h, w), bool)
    worst = 0
    for i in range(h):
        for j in range(w):
            if seen[i, j] or a[i, j] == 0:
                continue
            cells = []
            q = deque([(i, j)])
            seen[i, j] = True
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and a[nr, nc] != 0:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            root = max(cells, key=lambda rc: rc[0] * W + rc[1])
            cset = set(cells)
            dist = {root: 0}
            q = deque([root])
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p = (r + dr, c + dc)
                    if p in cset and p not in dist:
                        dist[p] = dist[(r, c)] + 1
                        q.append(p)
            worst = max(worst, max(dist.values()))
    return worst


def _diam_percolor(a):
    h, w = a.shape
    seen = np.zeros((h, w), bool)
    worst = 0
    for i in range(h):
        for j in range(w):
            if seen[i, j] or a[i, j] == 0:
                continue
            col = a[i, j]
            cells = []
            q = deque([(i, j)])
            seen[i, j] = True
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and a[nr, nc] == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            root = max(cells, key=lambda rc: rc[0] * W + rc[1])
            cset = set(cells)
            dist = {root: 0}
            q = deque([root])
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p = (r + dr, c + dc)
                    if p in cset and p not in dist:
                        dist[p] = dist[(r, c)] + 1
                        q.append(p)
            worst = max(worst, max(dist.values()))
    return worst


def _present(a):
    return [c for c in range(1, CHANNELS) if (a == c).any()]


def _src_value(a, src):
    kind = src[0]
    if kind == "total":
        return int((a != 0).sum())
    if kind == "distinct":
        return len(_present(a))
    if kind == "color":
        return int((a == src[1]).sum())
    if kind == "squares":
        return _count_squares(a, src[1], src[2])
    if kind == "ncomp":
        return _comps_nonbg(a)
    raise ValueError(kind)


def _solid_rect(b):
    """(colour, height, width) if b's non-bg cells form a full monochrome rect
    anchored at the top-left, else None."""
    nz = b != 0
    if not nz.any():
        return None
    cols = set(b[nz].tolist())
    if len(cols) != 1:
        return None
    rows = np.where(nz.any(axis=1))[0]
    colsc = np.where(nz.any(axis=0))[0]
    if rows.min() != 0 or colsc.min() != 0:
        return None
    r1, c1 = rows.max(), colsc.max()
    if not nz[:r1 + 1, :c1 + 1].all() or nz.sum() != (r1 + 1) * (c1 + 1):
        return None
    return int(next(iter(cols))), int(r1 + 1), int(c1 + 1)


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


def _emit(out, name, builder):
    try:
        m = builder()
        onnx.checker.check_model(m, full_check=True)
        out.append((name, m))
    except Exception:
        pass


def _candidate_sources(prs):
    """count sources to try for the bar-style renderings (cheap ones only)."""
    srcs = [("total",), ("distinct",)]
    present_all = set(range(1, CHANNELS))
    for a, _ in prs:
        present_all &= set(_present(a))
    for k in sorted(present_all):
        srcs.append(("color", k))
    for k in sorted(present_all):
        for s in (2, 3):
            srcs.append(("squares", k, s))
    return srcs


def _try_tightbar(prs, out):
    recs = [_solid_rect(b) for _, b in prs]
    if any(r is None for r in recs):
        return
    Hs = [r[1] for r in recs]
    Ws = [r[2] for r in recs]
    Cs = [r[0] for r in recs]
    if all(h == 1 for h in Hs):
        orient, L = "row", Ws
    elif all(w == 1 for w in Ws):
        orient, L = "col", Hs
    elif all(h == w for h, w in zip(Hs, Ws)):
        orient, L = "sq", Hs
    else:
        return
    if len(set(L)) < 2:          # count never varies -> not a count bar
        return
    # colour mode
    color_modes = []
    if len(set(Cs)) == 1:
        color_modes.append(("fixed", Cs[0]))
    if all(_present(a) == [c] for (a, _), c in zip(prs, Cs)):
        color_modes.append(("present", 0))
    if not color_modes:
        return
    for src in _candidate_sources(prs):
        try:
            if not all(_src_value(a, src) == l for (a, _), l in zip(prs, L)):
                continue
        except Exception:
            continue
        for cm, fc in color_modes:
            nm = f"tightbar_{orient}_{src[0]}{'_'.join(str(x) for x in src[1:])}_{cm}{fc if cm=='fixed' else ''}"
            _emit(out, nm, lambda src=src, cm=cm, fc=fc:
                  build_tightbar(orient, src, cm, fc))
        return


def _try_fieldbar(prs, out):
    osh = {b.shape for _, b in prs}
    if len(osh) != 1:
        return
    fh, fw = next(iter(osh))
    if fh * fw == 1 or fh > 6 or fw > 6:
        return
    # the colour painted in the bar must be a single fixed non-bg colour
    paint = set()
    for _, b in prs:
        paint |= set(b[b != 0].tolist())
    if len(paint) != 1:
        return
    k = next(iter(paint))
    for axis in ("row", "col"):
        Ls = []
        ok = True
        for _, b in prs:
            if axis == "row":
                line = b[0, :fw]
                rest_ok = (b[1:fh, :fw] == 0).all() if fh > 1 else True
            else:
                line = b[:fh, 0]
                rest_ok = (b[:fh, 1:fw] == 0).all() if fw > 1 else True
            nz = np.where(line != 0)[0]
            n = nz.size
            if n and (nz[0] != 0 or not (np.diff(nz) == 1).all()):
                ok = False
                break
            if n and set(line[line != 0].tolist()) != {k}:
                ok = False
                break
            if not rest_ok:
                ok = False
                break
            Ls.append(int(n))
        if not ok or len(set(Ls)) < 2:
            continue
        for src in _candidate_sources(prs):
            try:
                if not all(_src_value(a, src) == l for (a, _), l in zip(prs, Ls)):
                    continue
            except Exception:
                continue
            nm = f"fieldbar_{axis}_{fh}x{fw}_{src[0]}{'_'.join(str(x) for x in src[1:])}_k{k}"
            _emit(out, nm, lambda axis=axis, src=src: build_fieldbar(axis, fh, fw, k, src))
            return


def _try_swatchmap(prs, out):
    recs = [_solid_rect(b) for _, b in prs]
    if any(r is None for r in recs):
        return
    osh = {(r[1], r[2]) for r in recs}
    if len(osh) != 1:                 # fixed field, monochrome flood fill
        return
    fh, fw = next(iter(osh))
    if fh > 3 or fw > 3:
        return
    Cs = [r[0] for r in recs]
    if len(set(Cs)) < 2:              # constant colour -> not a count map
        return
    # count source = connected-component count (and a couple of cheap fallbacks)
    for src in (("ncomp",), ("total",), ("distinct",)):
        table = {}
        ok = True
        for (a, _), col in zip(prs, Cs):
            v = _src_value(a, src)
            if v in table and table[v] != col:
                ok = False
                break
            table[v] = col
        if not ok or len(set(table.values())) < 2:
            continue
        if src[0] == "ncomp":
            need = max(_diam_nonbg(a) for a, _ in prs)
            T = min(60, max(12, need + 6))
            _emit(out, f"swatchmap_{fh}x{fw}_ncomp", lambda T=T, tb=dict(table):
                  build_swatchmap(fh, fw, ("ncomp", T), tb))
        else:
            _emit(out, f"swatchmap_{fh}x{fw}_{src[0]}", lambda src=src, tb=dict(table):
                  build_swatchmap(fh, fw, src, tb))
        return


def _try_fragswatch(prs, out):
    recs = [_solid_rect(b) for _, b in prs]
    if any(r is None for r in recs):
        return
    osh = {(r[1], r[2]) for r in recs}
    if len(osh) != 1:
        return
    fh, fw = next(iter(osh))
    if fh > 3 or fw > 3:
        return
    # need >= 2 present colours, output colour = argmax comp/cnt (low-index tie)
    for a, b in prs:
        pres = _present(a)
        if len(pres) < 2:
            return
        oc = int(b[b != 0][0])
        best, bestnum, bestden = None, None, None
        for c in pres:
            cnt = int((a == c).sum())
            comp = _comps_color(a, c)
            if best is None or comp * bestden > bestnum * cnt or \
               (comp * bestden == bestnum * cnt and c < best):
                best, bestnum, bestden = c, comp, cnt
        if best != oc:
            return
    need = max(_diam_percolor(a) for a, _ in prs)
    T = min(28, max(18, need + 8))
    _emit(out, f"fragswatch_{fh}x{fw}", lambda T=T: build_fragswatch(fh, fw, T))


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []

    out = []
    _try_tightbar(prs, out)
    _try_fieldbar(prs, out)
    _try_swatchmap(prs, out)
    _try_fragswatch(prs, out)
    return out
