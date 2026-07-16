"""LOCAL MAJORITY / MODE FILTER and CLEANUP (origin-anchored, opset 10).

Every rule here is a LOCAL neighbourhood statistic computed with a single box
``Conv`` (a per-channel box / cross kernel applied with ``group=10``) followed by
an ``ArgMax`` / ``Greater`` / ``Less`` decision that is rebuilt into a one-hot
tensor with ``Equal`` / ``Where`` / ``Mul``.  The one-hot input is zero-padded to
30x30 with the grid anchored at (0,0), so:

  * a box ``Conv`` (pads = K//2) over channel ``c`` yields, at every cell, the
    number of colour-``c`` cells inside the KxK window that lie INSIDE the real
    grid -- padding cells are all-zero and contribute nothing, so out-of-grid
    neighbours are simply "not counted" (the standard ARC edge convention), and
    the result is correct at the top-left origin AND at the (variable) bottom /
    right grid boundary for ANY grid size -> the rules generalise structurally;
  * the real-cell mask is ``R = ReduceSum(input, axis=channel)`` (1 on real cells,
    0 on padding); gating every output by ``R`` keeps padding all-zero.

Rules (the matching one is inferred from train/test/arc-gen pairs)
-----------------------------------------------------------------
  mode_KxK         output[r,c] = the most common colour in the KxK box window
                   (background counted), first-index (lowest colour) tie-break --
                   the classic mode / smoothing filter.  Realised as a grouped box
                   ``Conv`` (per-channel counts) -> ``ArgMax`` over channels ->
                   one-hot via ``Equal`` -> ``Mul`` by the real-cell mask.

  denoise_*        Remove small / thin noise: a non-bg colour-c cell is KEPT iff
                   its KxK same-colour count (incl. itself) is >= k, else it is
                   recoloured (to background, or a fixed mark colour).  ``k=2`` box
                   strips fully isolated single pixels; ``k=3`` cross strips cells
                   with fewer than two orthogonal same-colour neighbours.

  hollow_*         The mirror: a non-bg colour-c cell is ERASED iff its KxK
                   same-colour count reaches k (e.g. k=9 box = the cell plus all 8
                   neighbours share its colour -> a fully-interior cell), turning
                   solid blobs into their outlines.

All cleanup rules are a grouped box/cross ``Conv`` -> ``Greater``/``Less``
threshold -> ``Where`` keep/erase, with the (background or fixed) replacement
colour rebuilt from the real-cell mask so padding stays all-zero.

Detection reproduces the ONNX semantics EXACTLY (the ArgMax first-index tie-break
and the in-grid edge counting) and only emits a candidate when it matches EVERY
available pair, so wrong hypotheses are dropped before scoring.
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
F = DATA_TYPE


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def cf(self, dims, vals):
        n = self.nm("cf")
        self.inits.append(oh.make_tensor(n, F, list(dims),
                                         [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        return n

    def ci(self, dims, vals):
        n = self.nm("ci")
        self.inits.append(oh.make_tensor(n, INT64, list(dims),
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


def _kernel(K, conn):
    """[K,K] {0,1} box (full) or cross (plus-shaped) neighbourhood kernel."""
    if conn == "cross":
        ker = np.zeros((K, K), np.float32)
        r = K // 2
        ker[r, :] = 1.0
        ker[:, r] = 1.0
        return ker
    return np.ones((K, K), np.float32)


def _onehot(c):
    return [1.0 if i == c else 0.0 for i in range(CHANNELS)]


# --------------------------------------------------------------------------- #
# ONNX builders                                                               #
# --------------------------------------------------------------------------- #
def build_mode(K, conn="box"):
    """output[r,c] = most common colour (bg counted) in the KxK window.
    Grouped box Conv -> ArgMax over channels -> one-hot (Equal) -> gate by R."""
    g = _G()
    p = K // 2
    base = _kernel(K, conn)
    W = np.zeros((CHANNELS, 1, K, K), np.float32)
    for c in range(CHANNELS):                       # every channel counted (incl bg)
        W[c, 0] = base
    wt = g.cf([CHANNELS, 1, K, K], W)
    counts = g.nd("Conv", ["input", wt], group=CHANNELS, kernel_shape=[K, K],
                  pads=[p, p, p, p])                # [1,10,30,30] per-channel counts
    amax = g.nd("ArgMax", [counts], axis=1, keepdims=1)          # int64 [1,1,30,30]
    idx = g.ci([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    eq = g.nd("Equal", [amax, idx])                             # bool [1,10,30,30]
    gate = g.nd("Cast", [eq], to=F)
    R = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)      # [1,1,30,30] real mask
    g.nd("Mul", [gate, R], "output")                           # zero out padding
    return _model(g)


def build_clean(k, K, conn, direction, C):
    """Local cleanup with a per-channel box/cross count threshold.

    direction 'lt'  : keep colour-c cell iff count >= k  (erase the SMALL/thin ones)
    direction 'ge'  : keep colour-c cell iff count <  k  (erase the SOLID interior)
    Erased non-bg cells are recoloured to colour ``C`` (``C`` may be background 0
    or a fixed mark colour); background and padding are rebuilt from the masks.
    """
    g = _G()
    p = K // 2
    base = _kernel(K, conn)
    W = np.zeros((CHANNELS, 1, K, K), np.float32)
    for c in range(1, CHANNELS):                    # channel 0 kernel stays all-zero
        W[c, 0] = base
    wt = g.cf([CHANNELS, 1, K, K], W)
    counts = g.nd("Conv", ["input", wt], group=CHANNELS, kernel_shape=[K, K],
                  pads=[p, p, p, p])                # [1,10,30,30], channel0 == 0
    thr = g.cf([1, 1, 1, 1], [k - 0.5])
    if direction == "lt":                           # keep iff count >= k
        keep = g.nd("Greater", [counts, thr])
    else:                                           # 'ge': keep iff count < k
        keep = g.nd("Less", [counts, thr])
    zero = g.cf([1, 1, 1, 1], [0.0])
    survive = g.nd("Where", [keep, "input", zero])  # kept cells keep colour, else 0
    R = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)        # real-cell mask

    if C == 0:
        # Route every not-survived real cell to background.  For 'lt' background
        # never survives (counts0==0) so `repl` = bg + erased; for 'ge' background
        # survives via Where (counts0==0 < thr) so `repl` = erased only.  Either
        # way the channel-0 rebuild is exact.
        tot = g.nd("ReduceSum", [survive], axes=[1], keepdims=1)
        repl = g.nd("Sub", [R, tot])
        e0 = g.cf([1, CHANNELS, 1, 1], _onehot(0))
        g.nd("Add", [survive, g.nd("Mul", [repl, e0])], "output")
        return _model(g)

    # Fixed non-bg mark colour: keep genuine background separate from erased cells.
    nbg = g.cf([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    survive_nb = g.nd("Mul", [survive, nbg])        # only kept NON-bg cells
    tot = g.nd("ReduceSum", [survive_nb], axes=[1], keepdims=1)
    bgp = g.nd("Slice", ["input", g.ci([1], [0]), g.ci([1], [1]), g.ci([1], [1])])
    erased = g.nd("Sub", [g.nd("Sub", [R, tot]), bgp])           # erased non-bg cells
    e0 = g.cf([1, CHANNELS, 1, 1], _onehot(0))
    eC = g.cf([1, CHANNELS, 1, 1], _onehot(C))
    body = g.nd("Add", [survive_nb, g.nd("Mul", [bgp, e0])])
    g.nd("Add", [body, g.nd("Mul", [erased, eC])], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX semantics exactly)                        #
# --------------------------------------------------------------------------- #
def _counts_field(a, color, K, conn):
    """KxK in-grid count of `color` (incl. center) at every cell."""
    h, w = a.shape
    r = K // 2
    mask = (a == color).astype(np.int64)
    pad = np.zeros((h + 2 * r, w + 2 * r), np.int64)
    pad[r:r + h, r:r + w] = mask
    ker = _kernel(K, conn).astype(np.int64)
    out = np.zeros((h, w), np.int64)
    for di in range(K):
        for dj in range(K):
            if ker[di, dj]:
                out += pad[di:di + h, dj:dj + w]
    return out


def _ref_mode(a, K, conn):
    """ArgMax over per-channel box counts (bg counted, first-index tie-break)."""
    cnt = np.stack([_counts_field(a, c, K, conn) for c in range(CHANNELS)], axis=0)
    return cnt.argmax(axis=0).astype(int)           # numpy argmax == ONNX first index


def _ref_clean(a, k, K, conn, direction, C):
    out = a.copy()
    for c in range(1, CHANNELS):
        if not (a == c).any():
            continue
        cnt = _counts_field(a, c, K, conn)
        if direction == "lt":
            sel = (a == c) & (cnt < k)
        else:
            sel = (a == c) & (cnt >= k)
        out[sel] = C
    return out


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


def _emit(out, seen, name, builder):
    if name in seen:
        return
    try:
        m = builder()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return
    seen.add(name)
    out.append((name, m))


def _target_colors(prs):
    """Colours that erased cells could become (changed cells in the outputs) + bg."""
    cols = {0}
    for a, b in prs:
        d = a != b
        if d.any():
            cols |= set(np.unique(b[d]).tolist())
    return sorted(cols)


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if any(a.shape != b.shape for a, b in prs):       # local filters preserve shape
        return []
    if all(np.array_equal(a, b) for a, b in prs):     # identity -> not our family
        return []

    out, seen = [], set()
    Cs = _target_colors(prs)

    for K in (3, 5):
        for conn in ("box", "cross"):
            ksize = (2 * K - 1) if conn == "cross" else K * K
            for C in Cs:
                tag = "B" if C == 0 else f"M{C}"
                # remove-small (lt): smallest structural k = most conservative
                for k in range(2, ksize + 1):
                    if all(np.array_equal(_ref_clean(a, k, K, conn, "lt", C), b)
                           for a, b in prs):
                        nm = ("denoise" if C == 0 else "mark") + f"_{conn}{K}_lt{k}_{tag}"
                        _emit(out, seen, nm,
                              lambda k=k, K=K, conn=conn, C=C: build_clean(k, K, conn, "lt", C))
                        break
                # remove-interior (ge): largest structural k = only fully-surrounded
                for k in range(ksize, 1, -1):
                    if all(np.array_equal(_ref_clean(a, k, K, conn, "ge", C), b)
                           for a, b in prs):
                        nm = ("hollow" if C == 0 else "markge") + f"_{conn}{K}_ge{k}_{tag}"
                        _emit(out, seen, nm,
                              lambda k=k, K=K, conn=conn, C=C: build_clean(k, K, conn, "ge", C))
                        break

    # ---- local mode / majority (box smoothing) ----------------------------- #
    for K in (3, 5, 7):
        if all(np.array_equal(_ref_mode(a, K, "box"), b) for a, b in prs):
            _emit(out, seen, f"mode_box{K}", lambda K=K: build_mode(K, "box"))
            break                                     # smallest K

    return out
