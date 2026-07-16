"""family_golf3_1 -- cheaper exact solvers for a slice of golf targets.

Each candidate re-derives the task rule from train+test+arc-gen pairs, verifies
EXACT equality against a numpy reference (mirroring the emitted ONNX ops on the
true one-hot representation) and only then emits a minimal opset-10 graph.  The
integrator auto-picks the cheapest correct solver, so we only need to be exact
and cheaper than the incumbent.

Targets golfed here (all rules verified exact on 100% of provided pairs):
  * 204 crk2_5_roomfill -- flood the exterior of closed 1-boxes, fill each
        enclosed cavity with colour 7 if its width is odd else 2.  Incumbent
        ~9.15 pts (cost ~7.6M).  We use a 4-connected flood + run-length parity
        entirely in single-channel [1,1,30,30] tensors.
  * 98  hollow_box3 -- hollow every solid rectangle (zero each cell whose 4
        orthogonal neighbours are all non-background).  Single 3x3 conv over the
        non-background mask + a [1,9,*,*] colour window.
  * 202 crk2_4_t202 -- monochrome bands (horizontal OR vertical) get every line
        perpendicular to the band that contains a hole cleared to background.
        Non-iterative reductions; orientation chosen at runtime via Where.
  * 40  crk2_5_border -- two opposite full-line borders; each interior colour-3
        dot is recoloured to the nearer border's colour.  Non-iterative: midpoint
        compare via a column/row index ramp; orientation chosen at runtime.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH, ng

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL
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

    def scalar(self, v):
        return self.fconst([v], [1])

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _slice(g, src, starts, ends, axes, steps=None):
    ins = [src, g.iconst(starts), g.iconst(ends), g.iconst(axes)]
    if steps is not None:
        ins.append(g.iconst(steps))
    return g.node("Slice", ins)


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
# 204  roomfill: enclosed cavities of 1-boxes -> 7 (odd width) / 2 (even)
# ===========================================================================
_PLUS = [0, 1, 0, 1, 1, 1, 0, 1, 0]   # plus kernel incl. centre


def _ref204(inp, niter):
    """Faithful numpy mirror of the emitted ONNX graph (operates on one-hot)."""
    ch0 = inp[:, 0:1]
    ch1 = inp[:, 1:2]
    floodable = 1.0 - ch1
    border = np.zeros((1, 1, 30, 30), np.float32)
    border[0, 0, 0, :] = 1; border[0, 0, -1, :] = 1
    border[0, 0, :, 0] = 1; border[0, 0, :, -1] = 1
    outside = floodable * border

    def cplus(x):
        p = np.pad(x[0, 0], 1)
        c = (p[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1]
             + p[1:-1, :-2] + p[1:-1, 2:])
        return c[None, None]

    for _ in range(niter):
        outside = np.minimum(floodable, cplus(outside))
    enclosed = ch0 * (1.0 - outside)

    def runlen(mask):
        r = mask.copy(); d = 1
        while d < 32:
            sh = np.zeros_like(r); sh[:, :, :, d:] = r[:, :, :, :-d]
            cond = (r >= d - 0.5).astype(np.float32)
            r = r + cond * sh; d *= 2
        return r

    Lrun = runlen(enclosed)
    Rrun = runlen(enclosed[:, :, :, ::-1])[:, :, :, ::-1]
    Wd = (Lrun + Rrun - 1.0) * enclosed
    odd = np.mod(Wd, 2.0)
    ch0o = ch0 - enclosed
    ch2o = enclosed * (1.0 - odd)
    ch7o = enclosed * odd
    Z = np.zeros((1, 1, 30, 30), np.float32)
    return np.concatenate([ch0o, ch1, ch2o, Z, Z, Z, Z, ch7o, Z, Z], axis=1)


def _runlen(g, mask):
    """ONNX run-length: consecutive 1s ending at each position along width."""
    r = mask
    d = 1
    while d < 32:
        # shift r right by d along width (zero fill on the left)
        padded = g.node("Pad", [r], mode="constant", value=0.0,
                        pads=[0, 0, 0, d, 0, 0, 0, 0])          # [1,1,30,30+d]
        sh = _slice(g, padded, [0], [WIDTH], [3])               # [1,1,30,30]
        cond = g.node("Greater", [r, g.scalar(d - 0.5)])        # r >= d
        condf = g.node("Cast", [cond], to=DATA_TYPE)
        term = g.node("Mul", [condf, sh])
        r = g.node("Add", [r, term])
        d *= 2
    return r


def _build204(niter):
    g = _G()
    ch0 = _slice(g, "input", [0], [1], [1])              # [1,1,30,30]
    ch1 = _slice(g, "input", [1], [2], [1])              # walls
    floodable = g.node("Sub", [g.scalar(1.0), ch1])      # 1 - wall

    border = [1.0 if (r == 0 or r == 29 or c == 0 or c == 29) else 0.0
              for r in range(HEIGHT) for c in range(WIDTH)]
    bord = g.fconst(border, [1, 1, HEIGHT, WIDTH])
    outside = g.node("Mul", [floodable, bord])
    W = g.fconst(_PLUS, [1, 1, 3, 3])
    for _ in range(niter):
        conv = g.node("Conv", [outside, W], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        outside = g.node("Min", [floodable, conv])
    notout = g.node("Sub", [g.scalar(1.0), outside])
    enclosed = g.node("Mul", [ch0, notout])              # [1,1,30,30]

    Lrun = _runlen(g, enclosed)
    enc_rev = _slice(g, enclosed, [WIDTH - 1], [_NEG], [3], [-1])
    Rrun_rev = _runlen(g, enc_rev)
    Rrun = _slice(g, Rrun_rev, [WIDTH - 1], [_NEG], [3], [-1])
    s = g.node("Add", [Lrun, Rrun])
    s1 = g.node("Sub", [s, g.scalar(1.0)])
    Wd = g.node("Mul", [s1, enclosed])
    odd = g.node("Mod", [Wd, g.scalar(2.0)], fmod=1)     # [1,1,30,30]

    ch0o = g.node("Sub", [ch0, enclosed])
    even = g.node("Sub", [g.scalar(1.0), odd])
    ch2o = g.node("Mul", [enclosed, even])
    ch7o = g.node("Mul", [enclosed, odd])
    Z = g.fconst([0.0] * (HEIGHT * WIDTH), [1, 1, HEIGHT, WIDTH])
    g.node("Concat", [ch0o, ch1, ch2o, Z, Z, Z, Z, ch7o, Z, Z], "output", axis=1)
    return _model(g.nodes, g.inits)


# ===========================================================================
# 98  hollow_box3: zero every interior cell (4 ortho neighbours non-background)
# ===========================================================================
_PLUS5 = [0, 1, 0, 1, 1, 1, 0, 1, 0]


def _ref98(inp):
    ch0 = inp[:, 0:1]
    inC = inp[:, 1:10]
    M = inC.sum(axis=1, keepdims=True)
    Mp = np.pad(M[0, 0], 1, constant_values=0.0)[None, None]
    p = Mp[0, 0]
    C = (p[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1]
         + p[1:-1, :-2] + p[1:-1, 2:] - 4.0)[None, None]
    interior = np.clip(C, 0, 1)
    inv = 1.0 - interior
    outC = inC * inv
    out0 = ch0 + interior
    return np.concatenate([out0, outC], axis=1)


def _build98():
    g = _G()
    ch0 = _slice(g, "input", [0], [1], [1])                 # [1,1,30,30]
    inC = _slice(g, "input", [1], [CHANNELS], [1])          # [1,9,30,30]
    M = g.node("ReduceSum", [inC], axes=[1], keepdims=1)    # nonzero mask
    Mp = g.node("Pad", [M], mode="constant", value=0.0,
                pads=[0, 0, 1, 1, 0, 0, 1, 1])              # [1,1,32,32]
    W = g.fconst(_PLUS5, [1, 1, 3, 3])
    bias = g.fconst([-4.0], [1])
    C = g.node("Conv", [Mp, W, bias], kernel_shape=[3, 3])  # valid -> [1,1,30,30]
    interior = g.node("Clip", [C], min=0.0, max=1.0)
    inv = g.node("Sub", [g.scalar(1.0), interior])
    outC = g.node("Mul", [inC, inv])                        # [1,9,30,30]
    out0 = g.node("Add", [ch0, interior])                   # [1,1,30,30]
    g.node("Concat", [out0, outC], "output", axis=1)
    return _model(g.nodes, g.inits)


# ===========================================================================
# 202  band-punch: clear band lines (perpendicular) that contain a hole
# ===========================================================================
def _ref202(inp):
    ch0 = inp[:, 0:1]
    Xc = inp[:, 1:10]
    rowHas = (Xc.sum(axis=3, keepdims=True) > 0).astype(np.float32)   # [1,9,30,1]
    colHas = (Xc.sum(axis=2, keepdims=True) > 0).astype(np.float32)   # [1,9,1,30]
    hibH = ((ch0 * rowHas).sum(axis=2, keepdims=True) > 0).astype(np.float32)
    clearH = rowHas * hibH
    hibV = ((ch0 * colHas).sum(axis=3, keepdims=True) > 0).astype(np.float32)
    clearV = colHas * hibV
    maxR = rowHas.sum(axis=1, keepdims=True).max()
    clear = clearH if maxR <= 1.5 else clearV
    outXc = Xc * (1.0 - clear)
    out0 = ch0 + clear.sum(axis=1, keepdims=True)
    return np.concatenate([out0, outXc], axis=1)


def _build202():
    g = _G()
    half = g.scalar(0.5)
    ch0 = _slice(g, "input", [0], [1], [1])                 # [1,1,30,30]
    Xc = _slice(g, "input", [1], [CHANNELS], [1])           # [1,9,30,30]
    rs = g.node("ReduceSum", [Xc], axes=[3], keepdims=1)    # [1,9,30,1]
    rowHas = g.node("Cast", [g.node("Greater", [rs, half])], to=DATA_TYPE)
    cs = g.node("ReduceSum", [Xc], axes=[2], keepdims=1)    # [1,9,1,30]
    colHas = g.node("Cast", [g.node("Greater", [cs, half])], to=DATA_TYPE)

    tmpH = g.node("Mul", [ch0, rowHas])                     # [1,9,30,30]
    shH = g.node("ReduceSum", [tmpH], axes=[2], keepdims=1)
    hibH = g.node("Cast", [g.node("Greater", [shH, half])], to=DATA_TYPE)
    clearH = g.node("Mul", [rowHas, hibH])                  # [1,9,30,30]

    tmpV = g.node("Mul", [ch0, colHas])
    shV = g.node("ReduceSum", [tmpV], axes=[3], keepdims=1)
    hibV = g.node("Cast", [g.node("Greater", [shV, half])], to=DATA_TYPE)
    clearV = g.node("Mul", [colHas, hibV])

    rcc = g.node("ReduceSum", [rowHas], axes=[1], keepdims=1)   # [1,1,30,1]
    maxR = g.node("ReduceMax", [rcc], axes=[2, 3], keepdims=1)  # [1,1,1,1]
    isH = g.node("Less", [maxR, g.scalar(1.5)])                 # bool [1,1,1,1]
    clear = g.node("Where", [isH, clearH, clearV])             # [1,9,30,30]
    inv = g.node("Sub", [g.scalar(1.0), clear])
    outXc = g.node("Mul", [Xc, inv])                          # [1,9,30,30]
    clearedTot = g.node("ReduceSum", [clear], axes=[1], keepdims=1)
    out0 = g.node("Add", [ch0, clearedTot])
    g.node("Concat", [out0, outXc], "output", axis=1)
    return _model(g.nodes, g.inits)


# ===========================================================================
# 40  border: recolour interior colour-3 dots to the nearer of two borders
# ===========================================================================
def _ref40(inp):
    ch0 = inp[:, 0:1]
    nz = inp[:, 1:].sum(axis=1, keepdims=True)
    colCh0 = ch0.sum(axis=2, keepdims=True)
    colNZ = nz.sum(axis=2, keepdims=True)
    fullColMask = ((colCh0 < 0.5) & (colNZ > 0.5)).astype(np.float32)
    rowCh0 = ch0.sum(axis=3, keepdims=True)
    rowNZ = nz.sum(axis=3, keepdims=True)
    fullRowMask = ((rowCh0 < 0.5) & (rowNZ > 0.5)).astype(np.float32)
    colColor = inp.max(axis=2, keepdims=True)
    borderColC = (colColor * fullColMask).max(axis=3, keepdims=True)
    Lvec = inp[:, :, :, 0:1].max(axis=2, keepdims=True)
    Rvec = borderColC - Lvec
    rowColor = inp.max(axis=3, keepdims=True)
    borderColR = (rowColor * fullRowMask).max(axis=2, keepdims=True)
    Tvec = inp[:, :, 0:1, :].max(axis=3, keepdims=True)
    Bvec = borderColR - Tvec
    cidx = np.arange(30).reshape(1, 1, 1, 30).astype(np.float32)
    ridx = np.arange(30).reshape(1, 1, 30, 1).astype(np.float32)
    rb = (cidx * fullColMask).max()
    rbrow = (ridx * fullRowMask).max()
    leftcloser = (2 * cidx < rb).astype(np.float32)
    topcloser = (2 * ridx < rbrow).astype(np.float32)
    recolorLR = leftcloser * Lvec + (1 - leftcloser) * Rvec
    recolorTB = topcloser * Tvec + (1 - topcloser) * Bvec
    recolor = recolorLR if fullColMask.sum() >= 1.5 else recolorTB
    dot = inp[:, 3:4]
    out = inp.copy(); out[:, 3] = 0
    out = out + dot * (recolor * np.ones((1, CHANNELS, 30, 30), np.float32))
    return out


def _build40():
    g = _G()
    half = g.scalar(0.5)
    one = g.scalar(1.0)
    ch0 = _slice(g, "input", [0], [1], [1])                       # [1,1,30,30]
    realmask = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)
    nz = g.node("Sub", [realmask, ch0])                           # [1,1,30,30]

    colCh0 = g.node("ReduceSum", [ch0], axes=[2], keepdims=1)     # [1,1,1,30]
    colNZ = g.node("ReduceSum", [nz], axes=[2], keepdims=1)
    fullColMask = g.node("Cast", [g.node("And", [
        g.node("Less", [colCh0, half]), g.node("Greater", [colNZ, half])])],
        to=DATA_TYPE)                                             # [1,1,1,30]
    rowCh0 = g.node("ReduceSum", [ch0], axes=[3], keepdims=1)     # [1,1,30,1]
    rowNZ = g.node("ReduceSum", [nz], axes=[3], keepdims=1)
    fullRowMask = g.node("Cast", [g.node("And", [
        g.node("Less", [rowCh0, half]), g.node("Greater", [rowNZ, half])])],
        to=DATA_TYPE)                                             # [1,1,30,1]

    colColor = g.node("ReduceMax", ["input"], axes=[2], keepdims=1)   # [1,10,1,30]
    borderColC = g.node("ReduceMax", [g.node("Mul", [colColor, fullColMask])],
                        axes=[3], keepdims=1)                     # [1,10,1,1]
    col0 = _slice(g, "input", [0], [1], [3])                      # [1,10,30,1]
    Lvec = g.node("ReduceMax", [col0], axes=[2], keepdims=1)      # [1,10,1,1]
    Rvec = g.node("Sub", [borderColC, Lvec])

    rowColor = g.node("ReduceMax", ["input"], axes=[3], keepdims=1)   # [1,10,30,1]
    borderColR = g.node("ReduceMax", [g.node("Mul", [rowColor, fullRowMask])],
                        axes=[2], keepdims=1)                     # [1,10,1,1]
    row0 = _slice(g, "input", [0], [1], [2])                      # [1,10,1,30]
    Tvec = g.node("ReduceMax", [row0], axes=[3], keepdims=1)      # [1,10,1,1]
    Bvec = g.node("Sub", [borderColR, Tvec])

    cidx = g.fconst([float(c) for c in range(WIDTH)], [1, 1, 1, WIDTH])
    ridx = g.fconst([float(r) for r in range(HEIGHT)], [1, 1, HEIGHT, 1])
    twoc = g.fconst([float(2 * c) for c in range(WIDTH)], [1, 1, 1, WIDTH])
    twor = g.fconst([float(2 * r) for r in range(HEIGHT)], [1, 1, HEIGHT, 1])
    rb = g.node("ReduceMax", [g.node("Mul", [cidx, fullColMask])],
                axes=[2, 3], keepdims=1)                          # [1,1,1,1]
    rbrow = g.node("ReduceMax", [g.node("Mul", [ridx, fullRowMask])],
                   axes=[2, 3], keepdims=1)
    leftcloser = g.node("Cast", [g.node("Less", [twoc, rb])], to=DATA_TYPE)
    topcloser = g.node("Cast", [g.node("Less", [twor, rbrow])], to=DATA_TYPE)
    recolorLR = g.node("Add", [g.node("Mul", [leftcloser, Lvec]),
                               g.node("Mul", [g.node("Sub", [one, leftcloser]), Rvec])])
    recolorTB = g.node("Add", [g.node("Mul", [topcloser, Tvec]),
                               g.node("Mul", [g.node("Sub", [one, topcloser]), Bvec])])
    isLR = g.node("Greater", [g.node("ReduceSum", [fullColMask],
                  axes=[2, 3], keepdims=1), g.scalar(1.5)])       # bool [1,1,1,1]
    recolor = g.node("Where", [isLR, recolorLR, recolorTB])       # [1,10,30,30]

    dot = _slice(g, "input", [3], [4], [1])                       # [1,1,30,30]
    e3 = g.fconst([1.0 if i == 3 else 0.0 for i in range(CHANNELS)],
                  [1, CHANNELS, 1, 1])
    diff = g.node("Sub", [recolor, e3])
    term = g.node("Mul", [dot, diff])
    g.node("Add", ["input", term], "output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
def _onehot(a):
    t = np.zeros((1, CHANNELS, HEIGHT, WIDTH), np.float32)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            t[0, a[r, c], r, c] = 1.0
    return t


def _target_onehot(b):
    return _onehot(b)


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    def emit(name, fn):
        try:
            out.append((name, fn()))
        except Exception:
            pass

    # ---- 204 roomfill -------------------------------------------------------
    # input must contain only colours {0,1}; output adds 2/7 fills.
    colors_in = set()
    for a, _ in prs:
        colors_in |= set(np.unique(a).tolist())
    if colors_in <= {0, 1} and all(a.shape == b.shape for a, b in prs):
        NITER = 32
        ok = True
        for a, b in prs:
            pred = (_ref204(_onehot(a), NITER) > 0.0).astype(np.float32)
            if not np.array_equal(pred, _onehot(b)):
                ok = False
                break
        if ok:
            emit("roomfill_parity", lambda: _build204(NITER))

    # ---- 98 hollow_box3 -----------------------------------------------------
    if all(a.shape == b.shape for a, b in prs) and any(
            not np.array_equal(a, b) for a, b in prs):
        ok = True
        for a, b in prs:
            pred = (_ref98(_onehot(a)) > 0.0).astype(np.float32)
            if not np.array_equal(pred, _onehot(b)):
                ok = False
                break
        if ok:
            emit("hollow_box", _build98)

    # ---- 202 band-punch -----------------------------------------------------
    if all(a.shape == b.shape for a, b in prs) and any(
            not np.array_equal(a, b) for a, b in prs):
        ok = True
        for a, b in prs:
            pred = (_ref202(_onehot(a)) > 0.0).astype(np.float32)
            if not np.array_equal(pred, _onehot(b)):
                ok = False
                break
        if ok:
            emit("band_punch", _build202)

    # ---- 40 border recolour -------------------------------------------------
    has3 = any(3 in set(np.unique(a).tolist()) for a, _ in prs)
    if has3 and all(a.shape == b.shape for a, b in prs) and any(
            not np.array_equal(a, b) for a, b in prs):
        ok = True
        for a, b in prs:
            pred = (_ref40(_onehot(a)) > 0.0).astype(np.float32)
            if not np.array_equal(pred, _onehot(b)):
                ok = False
                break
        if ok:
            emit("border_recolor", _build40)

    return out
