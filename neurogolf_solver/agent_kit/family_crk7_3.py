"""family_crk7_3 -- a grab-bag of cracked "hard" tasks (opset-10, origin-anchored).

Currently implemented sub-families:

* recolor_by_shape  (task 364 et al.)
    Every connected stroke is drawn in a single colour `src`.  Each stroke is
    re-coloured by its *topology*:
       - contains a junction cell (4-degree >= 3)         -> colour cJ
       - simple path whose 2 endpoints exit the SAME way  -> colour c1 ("U"/"C")
       - simple path whose 2 endpoints exit PERPENDICULAR -> colour c2 ("L"/"7")
    All three quantities are LOCAL per-pixel features that are then flooded
    (1-step dilation masked by the stroke) so the per-component verdict is painted
    on every cell -- no Loop/Scatter needed.

* periodic completion (task 61 et al.) is delegated to family_dynperiod, whose
    variable-period autocorrelation + doubling-OR fill already nails it.

Detection mirrors the exact ONNX semantics in numpy and only emits a candidate
when it reproduces EVERY train+test+arc-gen pair, so wrong guesses never score.
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
G = HEIGHT  # 30


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

    def f(self, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, F, list(dims),
                                         [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        return n

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
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


# --------------------------------------------------------------------------- #
# pixel shift  out[r,c] = t[r-dr, c-dc]  (zero filled), dr/dc small integers    #
# --------------------------------------------------------------------------- #
def _shift(g, t, dr, dc):
    pt, pb = max(dr, 0), max(-dr, 0)
    pl, pr = max(dc, 0), max(-dc, 0)
    pad = g.nd("Pad", [t], mode="constant", value=0.0,
               pads=[0, 0, pt, pl, 0, 0, pb, pr])
    sr, sc = max(-dr, 0), max(-dc, 0)
    return g.nd("Slice", [pad, g.i64([sr, sc]), g.i64([sr + G, sc + G]), g.i64([2, 3])])


# --------------------------------------------------------------------------- #
# ONNX builder for recolor-by-shape                                            #
# --------------------------------------------------------------------------- #
def build_recolor(src, cJ, c1, c2, K=14):
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    onec = g.f([1, 1, 1, 1], [1.0])

    P = g.nd("Slice", ["input", g.i64([src]), g.i64([src + 1]), g.i64([1])])   # [1,1,30,30]

    Nu = _shift(g, P, 1, 0)
    Nd = _shift(g, P, -1, 0)
    Nl = _shift(g, P, 0, 1)
    Nr = _shift(g, P, 0, -1)
    deg = g.nd("Add", [g.nd("Add", [Nu, Nd]), g.nd("Add", [Nl, Nr])])

    J = g.nd("Mul", [P, g.nd("Cast", [g.nd("Greater", [deg, g.f([1, 1, 1, 1], [2.5])])], to=F)])
    e_lo = g.nd("Cast", [g.nd("Greater", [deg, half])], to=F)
    e_hi = g.nd("Cast", [g.nd("Less", [deg, g.f([1, 1, 1, 1], [1.5])])], to=F)
    E = g.nd("Mul", [P, g.nd("Mul", [e_lo, e_hi])])
    ed0 = g.nd("Mul", [E, Nd])
    ed1 = g.nd("Mul", [E, Nu])
    ed2 = g.nd("Mul", [E, Nr])
    ed3 = g.nd("Mul", [E, Nl])

    S = g.nd("Concat", [J, ed0, ed1, ed2, ed3], axis=1)                        # [1,5,30,30]
    for _ in range(K):
        sU = _shift(g, S, 1, 0)
        sD = _shift(g, S, -1, 0)
        sL = _shift(g, S, 0, 1)
        sR = _shift(g, S, 0, -1)
        m = g.nd("Max", [S, sU, sD, sL, sR])
        S = g.nd("Mul", [m, P])

    hasJ = g.nd("Slice", [S, g.i64([0]), g.i64([1]), g.i64([1])])              # [1,1,30,30]
    eds = g.nd("Slice", [S, g.i64([1]), g.i64([5]), g.i64([1])])              # [1,4,30,30]
    dirsum = g.nd("ReduceSum", [eds], axes=[1], keepdims=1)                    # [1,1,30,30]

    hasJb = g.nd("Cast", [g.nd("Greater", [hasJ, half])], to=F)
    notJ = g.nd("Mul", [P, g.nd("Sub", [onec, hasJb])])
    m_j = g.nd("Mul", [P, hasJb])
    m_1 = g.nd("Mul", [notJ, g.nd("Cast", [g.nd("Less", [dirsum, g.f([1, 1, 1, 1], [1.5])])], to=F)])
    m_2 = g.nd("Mul", [notJ, g.nd("Cast", [g.nd("Greater", [dirsum, g.f([1, 1, 1, 1], [1.5])])], to=F)])

    ch0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])

    def basis(c):
        v = [1.0 if k == c else 0.0 for k in range(CHANNELS)]
        return g.f([1, CHANNELS, 1, 1], v)

    t_j = g.nd("Mul", [basis(cJ), m_j])
    t_1 = g.nd("Mul", [basis(c1), m_1])
    t_2 = g.nd("Mul", [basis(c2), m_2])
    t_0 = g.nd("Mul", [basis(0), ch0])
    g.nd("Add", [g.nd("Add", [g.nd("Add", [t_j, t_1]), t_2]), t_0], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy reference (mirrors ONNX EXACTLY)                                        #
# --------------------------------------------------------------------------- #
def _np_shift(P, dr, dc):
    out = np.zeros_like(P)
    h, w = P.shape[-2], P.shape[-1]
    rs, re = max(0, dr), min(h, h + dr)
    cs, ce = max(0, dc), min(w, w + dc)
    out[..., rs:re, cs:ce] = P[..., rs - dr:re - dr, cs - dc:ce - dc]
    return out


def _features(a, src):
    h, w = a.shape
    P = (a == src).astype(np.float32)
    Nu = _np_shift(P, 1, 0); Nd = _np_shift(P, -1, 0)
    Nl = _np_shift(P, 0, 1); Nr = _np_shift(P, 0, -1)
    deg = Nu + Nd + Nl + Nr
    J = P * (deg >= 2.5)
    E = P * ((deg > 0.5) & (deg < 1.5))
    ed = np.stack([E * Nd, E * Nu, E * Nr, E * Nl], 0)
    return P, J, ed


def _ref(a, src, cJ, c1, c2, K=14):
    P, J, ed = _features(a, src)
    S = np.concatenate([J[None], ed], 0)             # [5,h,w]
    for _ in range(K):
        m = S.copy()
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            sh = np.stack([_np_shift(S[ch], dr, dc) for ch in range(5)], 0)
            m = np.maximum(m, sh)
        S = m * P[None]
    hasJ = S[0] > 0.5
    dirsum = S[1] + S[2] + S[3] + S[4]
    out = np.zeros_like(a)
    shape = P > 0.5
    out[shape & hasJ] = cJ
    nj = shape & (~hasJ)
    out[nj & (dirsum < 1.5)] = c1
    out[nj & (dirsum > 1.5)] = c2
    return out


# --------------------------------------------------------------------------- #
# detection                                                                    #
# --------------------------------------------------------------------------- #
def _pairs(ex, splits):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > G or max(b.shape) > G:
                continue
            out.append((a, b))
    return out


def _components4(a, src):
    h, w = a.shape
    lab = -np.ones((h, w), int)
    out = []
    cid = 0
    for i in range(h):
        for j in range(w):
            if a[i, j] != src or lab[i, j] >= 0:
                continue
            stack = [(i, j)]; lab[i, j] = cid; cells = []
            while stack:
                r, c = stack.pop(); cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and lab[nr, nc] < 0 and a[nr, nc] == src:
                        lab[nr, nc] = cid; stack.append((nr, nc))
            out.append(cells); cid += 1
    return out


def _detect_recolor(det):
    # single non-bg input colour shared by all examples
    srcs = set()
    for a, _ in det:
        nz = set(int(v) for v in np.unique(a).tolist()) - {0}
        srcs |= nz
    if len(srcs) != 1:
        return None
    src = next(iter(srcs))
    cmap = {}        # key in {'J','1','2'} -> colour
    for a, b in det:
        if a.shape != b.shape:
            return None
        # background must be preserved
        bg = (a == 0)
        if not np.array_equal(b[bg], a[bg]):
            return None
        P, J, ed = _features(a, src)
        for cells in _components4(a, src):
            outc = set(int(b[r, c]) for r, c in cells)
            if len(outc) != 1:
                return None
            oc = next(iter(outc))
            hasJ = any(J[r, c] > 0.5 for r, c in cells)
            dirs = set()
            for k in range(4):
                if any(ed[k, r, c] > 0.5 for r, c in cells):
                    dirs.add(k)
            if hasJ:
                key = 'J'
            elif len(dirs) <= 1:
                key = '1'
            else:
                key = '2'
            if key in cmap and cmap[key] != oc:
                return None
            cmap[key] = oc
    # need at least the three behaviours we will branch on to be well defined;
    # default unused branches to background-safe values that never fire
    cJ = cmap.get('J'); c1 = cmap.get('1'); c2 = cmap.get('2')
    if cJ is None and c1 is None and c2 is None:
        return None
    return src, (cJ if cJ is not None else 0), (c1 if c1 is not None else 0), (c2 if c2 is not None else 0)


def _recolor_ok(plist, src, cJ, c1, c2):
    for a, b in plist:
        o = _ref(a, src, cJ, c1, c2)
        if o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def candidates(ex):
    out = []

    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))

    # ---- recolor-by-shape ----
    if det and allp and not any(a.shape != b.shape for a, b in allp) \
            and not all(np.array_equal(a, b) for a, b in allp):
        info = _detect_recolor(det)
        if info is not None:
            src, cJ, c1, c2 = info
            if _recolor_ok(det, src, cJ, c1, c2) and _recolor_ok(allp, src, cJ, c1, c2):
                try:
                    m = build_recolor(src, cJ, c1, c2)
                    onnx.checker.check_model(m, full_check=True)
                    out.append((f"recolor_s{src}_{cJ}{c1}{c2}", m))
                except Exception:
                    pass

    # ---- periodic completion: delegate to dynperiod ----
    try:
        import family_dynperiod as _dp
        for name, model in _dp.candidates(ex):
            out.append((f"dp_{name}", model))
    except Exception:
        pass

    return out
