"""family_golf3_3 — cheaper EXACT solvers for low-scoring golf targets.

Technique: work on a single-channel SCALAR colour grid (values 0..9) instead of
the 10-channel one-hot, do the geometry with cheap [1,1,30,30] intermediates
(forward-fill / span-fill via doubling shifts), then decode the scalar grid back
to one-hot with  onehot[k] = Relu(1 - |Gf - k|)  masked to the real grid.

Targets handled here:
  * task 37 (connect_same_DRUL_URDL): join equal-colour markers that lie on a
    common diagonal, drawing the segment between them.

Detection re-derives the rule from the train+test pairs in numpy and only emits
an ONNX model when the rule reproduces every pair exactly.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL


def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# ---------------- numpy reference ----------------------------------------------

def _shift(a, dr, dc):
    H, W = a.shape
    out = np.zeros_like(a)
    if abs(dr) >= H or abs(dc) >= W:
        return out
    rs, re = max(dr, 0), H + min(dr, 0)
    cs, ce = max(dc, 0), W + min(dc, 0)
    out[rs:re, cs:ce] = a[rs - dr:re - dr, cs - dc:ce - dc]
    return out


def _ffill(G, dr, dc):
    s = G.copy()
    d = 1
    for _ in range(5):
        s = np.where(s > 0, s, _shift(s, dr * d, dc * d))
        d *= 2
    return s


def _colored(G, dr, dc):
    fU = _ffill(G, dr, dc)
    fD = _ffill(G, -dr, -dc)
    return fU * (np.abs(fU - fD) < 0.5)


def connect_max(a, dirs):
    G = np.array(a, int)
    out = G.copy()
    for d in dirs:
        out = np.maximum(out, _colored(G, *d))
    return out


def connect_pri(a, layers):
    G = np.array(a, int)
    out = G.copy()
    for d in layers:
        c = _colored(G, *d)
        out = np.where(c > 0, c, out)
    return out


def markerstamp(a, mc, kr=1, kc=1):
    """Stamp the 3x3 key (centred at (kr,kc)) onto every cell of colour `mc`."""
    a = np.array(a, int)
    H, W = a.shape
    key = a[kr - 1:kr + 2, kc - 1:kc + 2]
    out = a.copy()
    rs, cs = np.where(a == mc)
    for r, c in zip(rs, cs):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = key[dr + 1, dc + 1]
    return out


# ---------------- ONNX builder ------------------------------------------------

class _B:
    """Small node/initializer accumulator with auto-named tensors."""

    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0

    def name(self, p):
        self.n += 1
        return f"{p}{self.n}"

    def node(self, op, ins, p, **attrs):
        o = self.name(p)
        self.nodes.append(oh.make_node(op, ins, [o], **attrs))
        return o

    def fconst(self, p, dims, vals):
        nm = self.name(p)
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(dims), list(vals)))
        return nm

    def iconst(self, p, dims, vals):
        nm = self.name(p)
        self.inits.append(oh.make_tensor(nm, INT64, list(dims), list(vals)))
        return nm


def _shift_node(b, s, dr, dc):
    """ONNX shift: out[r,c]=s[r-dr,c-dc], zero fill (Pad + Slice)."""
    h0, w0 = max(dr, 0), max(dc, 0)
    h1, w1 = max(-dr, 0), max(-dc, 0)
    pad = b.node("Pad", [s], "pad", mode="constant", value=0.0,
                 pads=[0, 0, h0, w0, 0, 0, h1, w1])
    st = b.iconst("cs", [2], [h1, w1])
    en = b.iconst("ce", [2], [h1 + HEIGHT, w1 + WIDTH])
    ax = b.iconst("ca", [2], [2, 3])
    return b.node("Slice", [pad, st, en, ax], "shft")


def _ffill_node(b, G, dr, dc, zero):
    s = G
    d = 1
    for _ in range(5):
        sd = _shift_node(b, s, dr * d, dc * d)
        cond = b.node("Greater", [s, zero], "gt")
        s = b.node("Where", [cond, s, sd], "ff")
        d *= 2
    return s


def _colored_node(b, G, dr, dc, zero, half):
    fU = _ffill_node(b, G, dr, dc, zero)
    fD = _ffill_node(b, G, -dr, -dc, zero)
    diff = b.node("Sub", [fU, fD], "df")
    ad = b.node("Abs", [diff], "ab")
    lt = b.node("Less", [ad, half], "lt")
    eqf = b.node("Cast", [lt], "ef", to=DATA_TYPE)
    return b.node("Mul", [fU, eqf], "col")


def _scalar(b):
    """Scalar colour grid G[1,1,30,30] = sum_c c*input[:,c] via 1x1 Conv."""
    w = b.fconst("cw", [1, CHANNELS, 1, 1], [float(c) for c in range(CHANNELS)])
    return b.node("Conv", ["input", w], "G", kernel_shape=[1, 1], pads=[0, 0, 0, 0])


def _decode(b, Gf):
    """scalar Gf -> one-hot 'output', masked to real cells."""
    kvec = b.fconst("kv", [1, CHANNELS, 1, 1], [float(k) for k in range(CHANNELS)])
    one = b.fconst("one", [1, 1, 1, 1], [1.0])
    dk = b.node("Sub", [Gf, kvec], "dk")
    adk = b.node("Abs", [dk], "adk")
    om = b.node("Sub", [one, adk], "om")
    omr = b.node("Relu", [om], "omr")
    R = b.node("ReduceSum", ["input"], "R", axes=[1], keepdims=1)
    b.nodes.append(oh.make_node("Mul", [omr, R], ["output"]))
    return


def build_connect_max(dirs):
    b = _B()
    zero = b.fconst("z", [1, 1, 1, 1], [0.0])
    half = b.fconst("h", [1, 1, 1, 1], [0.5])
    G = _scalar(b)
    Gf = G
    for d in dirs:
        col = _colored_node(b, G, d[0], d[1], zero, half)
        Gf = b.node("Max", [Gf, col], "mx")
    _decode(b, Gf)
    return _model(b.nodes, b.inits)


def build_markerstamp(mc, kr=1, kc=1):
    b = _B()
    zero = b.fconst("z", [1, 1, 1, 1], [0.0])
    G = _scalar(b)
    # marker mask = one-hot channel `mc`
    ms = b.iconst("ms", [1], [mc])
    me = b.iconst("me", [1], [mc + 1])
    ma = b.iconst("ma", [1], [1])
    marker = b.node("Slice", ["input", ms, me, ma], "mk")  # [1,1,30,30]
    stamp = None
    smask = None
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            sh = _shift_node(b, marker, dr, dc)
            # key value at (kr+dr, kc+dc)
            ks = b.iconst("ks", [2], [kr + dr, kc + dc])
            ke = b.iconst("ke", [2], [kr + dr + 1, kc + dc + 1])
            ka = b.iconst("ka", [2], [2, 3])
            kv = b.node("Slice", [G, ks, ke, ka], "kv")  # [1,1,1,1]
            contrib = b.node("Mul", [sh, kv], "ct")
            stamp = contrib if stamp is None else b.node("Add", [stamp, contrib], "st")
            smask = sh if smask is None else b.node("Max", [smask, sh], "sm")
    cond = b.node("Greater", [smask, zero], "sg")
    Gf = b.node("Where", [cond, stamp, G], "gf")
    _decode(b, Gf)
    return _model(b.nodes, b.inits)


def squarefill(a, fill=2, smax=10):
    """Fill every enclosed bg hole whose component is a solid square, with `fill`."""
    a = np.array(a, int)
    H, W = a.shape
    bg = (a == 0).astype(float)
    wall = (a != 0).astype(float)

    def corr(mask, kh, kw):
        # top-left aligned box sum: out[r,c] = sum mask[r:r+kh, c:c+kw]
        P = np.zeros((H + kh, W + kw))
        P[:H, :W] = mask
        S = np.zeros((H + kh + 1, W + kw + 1))
        S[1:, 1:] = np.cumsum(np.cumsum(P, 0), 1)
        return (S[kh:kh + H, kw:kw + W] - S[0:H, kw:kw + W]
                - S[kh:kh + H, 0:W] + S[0:H, 0:W])

    filled = np.zeros((H, W), bool)
    for s in range(1, smax + 1):
        blockbg = corr(bg, s, s) == s * s
        framecnt = corr(wall, s + 2, s + 2)
        fc = np.zeros((H, W))
        fc[1:, 1:] = framecnt[:H - 1, :W - 1]
        ringok = fc == (s + 2) * (s + 2) - s * s
        valid = blockbg & ringok
        rr, cc = np.where(valid)
        for R, C in zip(rr, cc):
            filled[R:R + s, C:C + s] = True
    out = a.copy()
    out[filled] = fill
    return out


def build_squarefill(fill=2, smax=8):
    b = _B()
    total = b.node("ReduceSum", ["input"], "tot", axes=[1], keepdims=1)
    bs = b.iconst("bs", [1], [0]); be = b.iconst("be", [1], [1]); ba = b.iconst("ba", [1], [1])
    bg = b.node("Slice", ["input", bs, be, ba], "bg")
    wall = b.node("Sub", [total, bg], "wall")
    filled = None
    for s in range(1, smax + 1):
        # block all-bg
        wb = b.fconst("wb", [1, 1, s, s], [1.0] * (s * s))
        bc = b.node("Conv", [bg, wb], "bc", kernel_shape=[s, s], pads=[0, 0, 0, 0])
        bcp = b.node("Pad", [bc], "bcp", mode="constant", value=0.0,
                     pads=[0, 0, 0, 0, 0, 0, s - 1, s - 1])
        tb = b.fconst("tb", [1, 1, 1, 1], [s * s - 0.5])
        bok = b.node("Greater", [bcp, tb], "bok")
        # ring all-walls
        wf = b.fconst("wf", [1, 1, s + 2, s + 2], [1.0] * ((s + 2) * (s + 2)))
        fcn = b.node("Conv", [wall, wf], "fcn", kernel_shape=[s + 2, s + 2], pads=[0, 0, 0, 0])
        fcp = b.node("Pad", [fcn], "fcp", mode="constant", value=0.0,
                     pads=[0, 0, 1, 1, 0, 0, s, s])
        ringneed = (s + 2) * (s + 2) - s * s
        tf = b.fconst("tf", [1, 1, 1, 1], [ringneed - 0.5])
        fok = b.node("Greater", [fcp, tf], "fok")
        valid = b.node("And", [bok, fok], "vd")
        vf = b.node("Cast", [valid], "vf", to=DATA_TYPE)
        # cover the s x s block (window up-left)
        wc = b.fconst("wc", [1, 1, s, s], [1.0] * (s * s))
        cv = b.node("Conv", [vf, wc], "cv", kernel_shape=[s, s],
                    pads=[s - 1, s - 1, 0, 0])
        filled = cv if filled is None else b.node("Max", [filled, cv], "fl")
    zero = b.fconst("z", [1, 1, 1, 1], [0.0])
    fmask = b.node("Greater", [filled, zero], "fm")
    fmf = b.node("Cast", [fmask], "fmf", to=DATA_TYPE)
    delta_vec = [0.0] * CHANNELS
    delta_vec[fill] += 1.0
    delta_vec[0] -= 1.0
    e2me0 = b.fconst("e", [1, CHANNELS, 1, 1], delta_vec)
    delta = b.node("Mul", [fmf, e2me0], "dl")
    b.nodes.append(oh.make_node("Add", ["input", delta], ["output"]))
    return _model(b.nodes, b.inits)


def build_connect_pri(layers):
    b = _B()
    zero = b.fconst("z", [1, 1, 1, 1], [0.0])
    half = b.fconst("h", [1, 1, 1, 1], [0.5])
    G = _scalar(b)
    Gf = G
    for d in layers:
        col = _colored_node(b, G, d[0], d[1], zero, half)
        cond = b.node("Greater", [col, zero], "pg")
        Gf = b.node("Where", [cond, col, Gf], "pw")
    _decode(b, Gf)
    return _model(b.nodes, b.inits)


# ---------------- detection / candidates --------------------------------------

_CONFIGS = [
    ("diagMax", lambda a: connect_max(a, [(1, 1), (1, -1)]),
     lambda: build_connect_max([(1, 1), (1, -1)])),
    ("hvPriV", lambda a: connect_pri(a, [(0, 1), (1, 0)]),
     lambda: build_connect_pri([(0, 1), (1, 0)])),
    ("hvPriH", lambda a: connect_pri(a, [(1, 0), (0, 1)]),
     lambda: build_connect_pri([(1, 0), (0, 1)])),
    ("allMax", lambda a: connect_max(a, [(1, 1), (1, -1), (1, 0), (0, 1)]),
     lambda: build_connect_max([(1, 1), (1, -1), (1, 0), (0, 1)])),
]


def candidates(examples):
    pairs = [(np.array(e["input"], int), np.array(e["output"], int))
             for e in examples.get("train", []) + examples.get("test", [])]
    if not pairs:
        return []
    # only same-size square-able tasks (must fit 30x30; connect is size-preserving)
    if any(a.shape != g.shape for a, g in pairs):
        return []
    if any(max(a.shape) > 30 for a, _ in pairs):
        return []
    out = []
    for nm, fn, build in _CONFIGS:
        try:
            if all((fn(a) == g).all() for a, g in pairs):
                out.append((nm, build()))
                break  # first matching config wins
        except Exception:
            continue
    # markerstamp: stamp top-left 3x3 key centred at each marker cell
    for mc in range(1, CHANNELS):
        try:
            if all((markerstamp(a, mc) == g).all() for a, g in pairs):
                out.append((f"stamp{mc}", build_markerstamp(mc)))
                break
        except Exception:
            continue
    # squarefill: fill enclosed solid-square bg holes with a fixed colour.
    # Guard: every changed cell must be a 0 -> C recolour for a single colour C.
    chg_from, chg_to = set(), set()
    for a, g in pairs:
        m = a != g
        chg_from.update(np.unique(a[m]).tolist())
        chg_to.update(np.unique(g[m]).tolist())
    if chg_to and chg_from == {0} and len(chg_to) == 1:
        fc = int(next(iter(chg_to)))
        try:
            if all((squarefill(a, fc) == g).all() for a, g in pairs):
                out.append((f"sqfill{fc}", build_squarefill(fc)))
        except Exception:
            pass
    return out
