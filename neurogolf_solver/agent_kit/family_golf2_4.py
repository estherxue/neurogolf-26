"""family_golf2_4 -- cheaper exact solvers for a slice of golf targets.

Each candidate re-derives the task rule from the train+test+arc-gen pairs,
verifies EXACT equality against a numpy reference on EVERY available pair, and
only then emits a minimal opset-10 ONNX graph.  The integrator auto-picks the
cheapest correct solver, so these just need to be exact + cheaper than the
incumbent.

Targets golfed here (all rules verified exact on 100% of provided pairs):
  * 92  connect_same_RL_UD -> span-fill collinear same-colour pairs (vertical
        wins crossings).  Done with two triangular MatMuls (prefix/suffix OR)
        per axis instead of a 25-step Conv/Clip CA unroll.
  * 70  boxrecolor_b8_1to3 -> recolour 1->3 inside the bounding box of colour 8.
        Bounding box via tiny [1,1,30,1]/[1,1,1,30] prefix/suffix MatMuls;
        single [1,10,30,30] delta added straight onto the free output tensor.
  * 24  lines_v2_h -> colours {1,3} paint their whole grid row, colour 2 paints
        its whole grid column, horizontal overwrites vertical.  Pure
        ReduceMax broadcasts on [1,1,*,*] reductions, no full intermediate.
  * 136 diag_rays -> colour-1 2x2 block shoots a 1px ray up-left, colour-2 block
        shoots one down-right (Hillis-Steele diagonal doubling, log steps).

Cost levers: single-channel [1,1,*,*] reductions, [1,9,*,*] colour-only windows
(skip background channel 0), two shared triangular matrices for ALL prefix/
suffix scans, broadcasting instead of Tile, and writing the assembled result
straight into the FREE `output` tensor via Concat/Add.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
_NEG = -(1 << 31)


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

    def iconst(self, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)], list(vals)))
        return nm

    def fconst(self, vals, shape):
        nm = self.name("f")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(shape),
                                         [float(v) for v in vals]))
        return nm

    def tri(self, upper):
        """[30,30] triangular ones (incl. diagonal).  upper: M[i,j]=1 iff j>=i."""
        vals = []
        for i in range(HEIGHT):
            for j in range(HEIGHT):
                vals.append(1.0 if (j >= i if upper else j <= i) else 0.0)
        return self.fconst(vals, [HEIGHT, HEIGHT])

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _slice(g, src, starts, ends, axes, steps=None):
    ins = [src, g.iconst(starts), g.iconst(ends), g.iconst(axes)]
    if steps is not None:
        ins.append(g.iconst(steps))
    return g.node("Slice", ins)


def _translate(g, src, dy, dx):
    """Shift content by (dy,dx) with zero fill, keeping a 30x30 window anchored
    so the origin mapping matches builders.translate.  One Pad + one Slice."""
    h0, w0 = max(dy, 0), max(dx, 0)
    h1, w1 = max(-dy, 0), max(-dx, 0)
    pad = g.node("Pad", [src], mode="constant", value=0.0,
                 pads=[0, 0, h0, w0, 0, 0, h1, w1])
    return _slice(g, pad, [h1, w1], [h1 + HEIGHT, w1 + WIDTH], [2, 3])


# --------------------------------------------------------------------------- #
# pairs                                                                        #
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


# ===========================================================================
# 92  connect_same_RL_UD
# ===========================================================================
def _ref_connect(a):
    H, W = a.shape
    x = np.zeros((CHANNELS, H, W))
    for r in range(H):
        for c in range(W):
            x[a[r, c], r, c] = 1.0
    x[0] = 0.0                                   # ignore background channel
    HL = np.cumsum(x, axis=2)
    HR = np.cumsum(x[:, :, ::-1], axis=2)[:, :, ::-1]
    Hspan = (np.minimum(HL, HR) > 0).astype(float)
    VU = np.cumsum(x, axis=1)
    VD = np.cumsum(x[:, ::-1, :], axis=1)[:, ::-1, :]
    Vspan = (np.minimum(VU, VD) > 0).astype(float)
    Vany = Vspan.max(axis=0)
    fg = np.maximum(Vspan, Hspan * (1 - Vany))   # channel0 stays 0
    fgany = fg.max(axis=0)
    out = np.zeros((H, W), int)
    cnt = (fg > 0).sum(axis=0) + (fgany == 0)    # bg cells -> 1 channel (ch0)
    if (cnt != 1).any():
        return None                              # ambiguous (two fg channels)
    for r in range(H):
        for c in range(W):
            ch = np.where(fg[:, r, c] > 0)[0]
            out[r, c] = ch[0] if len(ch) else 0
    return out


def _build_connect():
    g = _G()
    U = g.tri(upper=True)
    L = g.tri(upper=False)
    one = g.fconst([1.0], [1, 1, 1, 1])
    xfg = _slice(g, "input", [1], [CHANNELS], [1])      # [1,9,30,30]
    HL = g.node("MatMul", [xfg, U])                     # prefix-left  (over W)
    HR = g.node("MatMul", [xfg, L])                     # suffix-right
    Hspan = g.node("Min", [HL, HR, one])               # in {0,1}
    VU = g.node("MatMul", [L, xfg])                     # prefix-up    (over H)
    VD = g.node("MatMul", [U, xfg])                     # suffix-down
    Vspan = g.node("Min", [VU, VD, one])
    Vany = g.node("ReduceMax", [Vspan], axes=[1], keepdims=1)   # [1,1,30,30]
    notV = g.node("Sub", [one, Vany])
    Hsupp = g.node("Mul", [Hspan, notV])
    fg = g.node("Max", [Vspan, Hsupp])                 # [1,9,30,30]
    fgany = g.node("ReduceMax", [fg], axes=[1], keepdims=1)
    grid = g.node("ReduceMax", ["input"], axes=[1], keepdims=1)
    bg = g.node("Sub", [grid, fgany])
    g.node("Concat", [bg, fg], "output", axis=1)
    return _model(g.nodes, g.inits)


# ===========================================================================
# 70  boxrecolor_b8_1to3:  inside bbox(colour8), recolour 1 -> 3
# ===========================================================================
def _ref_box(a):
    H, W = a.shape
    m8 = (a == 8)
    if not m8.any():
        return a.copy()
    rs = np.where(m8.any(1))[0]
    cs = np.where(m8.any(0))[0]
    box = np.zeros((H, W), bool)
    box[rs.min():rs.max() + 1, cs.min():cs.max() + 1] = True
    out = a.copy()
    out[box & (a == 1)] = 3
    return out


def _build_box():
    g = _G()
    U = g.tri(upper=True)
    L = g.tri(upper=False)
    one = g.fconst([1.0], [1, 1, 1, 1])
    m8 = _slice(g, "input", [8], [9], [1])                 # [1,1,30,30]
    rowhas = g.node("ReduceMax", [m8], axes=[3], keepdims=1)  # [1,1,30,1]
    colhas = g.node("ReduceMax", [m8], axes=[2], keepdims=1)  # [1,1,1,30]
    rowpre = g.node("MatMul", [L, rowhas])
    rowsuf = g.node("MatMul", [U, rowhas])
    rowbox = g.node("Min", [rowpre, rowsuf, one])          # [1,1,30,1]
    colpre = g.node("MatMul", [colhas, U])
    colsuf = g.node("MatMul", [colhas, L])
    colbox = g.node("Min", [colpre, colsuf, one])          # [1,1,1,30]
    inbox = g.node("Mul", [rowbox, colbox])                # [1,1,30,30]
    isone = _slice(g, "input", [1], [2], [1])              # [1,1,30,30]
    to3 = g.node("Mul", [inbox, isone])                    # [1,1,30,30]
    sign = [0.0] * CHANNELS
    sign[1] = -1.0
    sign[3] = 1.0
    sv = g.fconst(sign, [1, CHANNELS, 1, 1])
    delta = g.node("Mul", [to3, sv])                       # [1,10,30,30]
    g.node("Add", ["input", delta], "output")
    return _model(g.nodes, g.inits)


# ===========================================================================
# 24  lines_v2_h:  rows with {1,3} fill the row; cols with 2 fill the column;
#                  horizontal overwrites vertical (3 has priority over 1)
# ===========================================================================
def _ref_lines(a):
    H, W = a.shape
    cols = set(np.unique(a).tolist())
    if not cols <= {0, 1, 2, 3}:
        return None
    out = np.zeros((H, W), int)
    rowH = np.zeros(H, int)
    for r in range(H):
        if (a[r] == 1).any():
            rowH[r] = 1
        if (a[r] == 3).any():
            rowH[r] = 3
    colV = np.array([(a[:, c] == 2).any() for c in range(W)])
    for r in range(H):
        for c in range(W):
            if rowH[r]:
                out[r, c] = rowH[r]
            elif colV[c]:
                out[r, c] = 2
    return out


def _build_lines():
    g = _G()
    one = g.fconst([1.0], [1, 1, 1, 1])
    m1 = _slice(g, "input", [1], [2], [1])
    m2 = _slice(g, "input", [2], [3], [1])
    m3 = _slice(g, "input", [3], [4], [1])
    rh1 = g.node("ReduceMax", [m1], axes=[3], keepdims=1)   # [1,1,30,1]
    rh3 = g.node("ReduceMax", [m3], axes=[3], keepdims=1)
    cv2 = g.node("ReduceMax", [m2], axes=[2], keepdims=1)   # [1,1,1,30]
    grid = g.node("ReduceMax", ["input"], axes=[1], keepdims=1)  # [1,1,30,30]
    not3 = g.node("Sub", [one, rh3])
    rh1e = g.node("Mul", [rh1, not3])                       # 1 only if not 3
    horiz = g.node("Max", [rh1e, rh3])
    ch1 = g.node("Mul", [rh1e, grid])                       # [1,1,30,30]
    ch3 = g.node("Mul", [rh3, grid])
    nothoriz = g.node("Sub", [one, horiz])
    cg = g.node("Mul", [cv2, grid])
    ch2 = g.node("Mul", [cg, nothoriz])
    s1 = g.node("Sub", [grid, ch1])
    s2 = g.node("Sub", [s1, ch3])
    ch0 = g.node("Sub", [s2, ch2])
    zeros6 = _slice(g, "input", [4], [CHANNELS], [1])       # [1,6,30,30] (zeros)
    g.node("Concat", [ch0, ch1, ch2, ch3, zeros6], "output", axis=1)
    return _model(g.nodes, g.inits)


# ===========================================================================
# 136  diag_rays
# ===========================================================================
def _ref_diag(a):
    H, W = a.shape
    cols = set(np.unique(a).tolist())
    if not cols <= {0, 1, 2}:
        return None
    m1 = (a == 1).astype(int)
    m2 = (a == 2).astype(int)
    up = np.zeros_like(m1); up[1:] = m1[:-1]
    left = np.zeros_like(m1); left[:, 1:] = m1[:, :-1]
    c1 = m1 * (1 - up) * (1 - left)
    down = np.zeros_like(m2); down[:-1] = m2[1:]
    right = np.zeros_like(m2); right[:, :-1] = m2[:, 1:]
    c2 = m2 * (1 - down) * (1 - right)

    def ray(seed, dr, dc):
        res = seed.copy(); cur = seed.copy()
        for _ in range(max(H, W)):
            nxt = np.zeros_like(cur)
            if dr < 0:
                nxt[:H + dr] = cur[-dr:]
            elif dr > 0:
                nxt[dr:] = cur[:H - dr]
            else:
                nxt[:] = cur
            tmp = nxt.copy(); nxt = np.zeros_like(tmp)
            if dc < 0:
                nxt[:, :W + dc] = tmp[:, -dc:]
            elif dc > 0:
                nxt[:, dc:] = tmp[:, :W - dc]
            else:
                nxt[:] = tmp
            res = np.maximum(res, nxt); cur = nxt
            if nxt.sum() == 0:
                break
        return res
    r1 = ray(c1, -1, -1)
    r2 = ray(c2, 1, 1)
    out = a.copy()
    out[(r1 > 0) & (a == 0)] = 1
    out[(r2 > 0) & (a == 0)] = 2
    return out


def _build_diag():
    g = _G()
    m1 = _slice(g, "input", [1], [2], [1])
    m2 = _slice(g, "input", [2], [3], [1])
    one = g.fconst([1.0], [1, 1, 1, 1])
    up = _translate(g, m1, 1, 0)
    left = _translate(g, m1, 0, 1)
    nu = g.node("Sub", [one, up])
    nl = g.node("Sub", [one, left])
    c1 = g.node("Mul", [g.node("Mul", [m1, nu]), nl])
    down = _translate(g, m2, -1, 0)
    right = _translate(g, m2, 0, -1)
    nd = g.node("Sub", [one, down])
    nr = g.node("Sub", [one, right])
    c2 = g.node("Mul", [g.node("Mul", [m2, nd]), nr])
    acc = c1
    for d in (1, 2, 4, 8, 16):
        acc = g.node("Max", [acc, _translate(g, acc, -d, -d)])
    ray1 = acc
    acc2 = c2
    for d in (1, 2, 4, 8, 16):
        acc2 = g.node("Max", [acc2, _translate(g, acc2, d, d)])
    grid = g.node("ReduceMax", ["input"], axes=[1], keepdims=1)
    ray2 = g.node("Mul", [acc2, grid])
    col1 = g.node("Max", [m1, ray1])
    col2 = g.node("Max", [m2, ray2])
    bg = g.node("Sub", [g.node("Sub", [grid, col1]), col2])
    rest = _slice(g, "input", [3], [CHANNELS], [1])         # [1,7,30,30]
    g.node("Concat", [bg, col1, col2, rest], "output", axis=1)
    return _model(g.nodes, g.inits)


# ===========================================================================
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    def emit(name, ref, fn):
        try:
            ok = True
            for a, b in prs:
                p = ref(a)
                if p is None or p.shape != b.shape or not np.array_equal(p, b):
                    ok = False
                    break
            if ok:
                out.append((name, fn()))
        except Exception:
            pass

    same_size = all(a.shape == b.shape for a, b in prs)
    changes = any(not np.array_equal(a, b) for a, b in prs)
    if not changes:
        return []

    if same_size:
        emit("connect_RLUD", _ref_connect, _build_connect)
        emit("boxrecolor_1to3", _ref_box, _build_box)
        emit("lines_v2_h", _ref_lines, _build_lines)
        emit("diag_rays", _ref_diag, _build_diag)

    return out
