"""family_golf_0 -- cheaper EXACT solvers for golf slice IDX=0.

Each rule has a numpy `sim` (used only for detection: fire iff it reproduces every
train+test pair exactly) and a `build` that emits a minimal opset-10 ONNX graph.

Cost philosophy: keep all intermediates single-channel [1,1,30,30] (3600 B) and build
the [1,10,30,30] output with ONE final Concat whose inputs are single channels, reusing
a shared zero tensor (counted once). Avoid any [1,10,30,30] non-output intermediate
(36000 B) wherever possible.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
FLOAT = onnx.TensorProto.FLOAT


def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# ------------------------- tiny node helpers ------------------------------
class G:
    """Accumulates nodes + initializers, hands out unique names."""
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0

    def _u(self, p):
        self.n += 1
        return f"{p}{self.n}"

    def init_i64(self, vals, name=None):
        name = name or self._u("i")
        self.inits.append(oh.make_tensor(name, INT64, [len(vals)], list(vals)))
        return name

    def init_f(self, vals, dims, name=None):
        name = name or self._u("f")
        self.inits.append(oh.make_tensor(name, FLOAT, list(dims), list(vals)))
        return name

    def scalar(self, v):
        name = self._u("s")
        self.inits.append(oh.make_tensor(name, FLOAT, [1], [float(v)]))
        return name

    def node(self, op, ins, out=None, **attrs):
        out = out or self._u("t")
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out

    # channel c of input -> [1,1,30,30]
    def chan(self, src, c):
        s = self.init_i64([c]); e = self.init_i64([c + 1]); a = self.init_i64([1])
        return self.node("Slice", [src, s, e, a])

    def conv(self, src, kernel3x3):
        w = self.init_f(kernel3x3, [1, 1, 3, 3])
        return self.node("Conv", [src, w], kernel_shape=[3, 3], pads=[1, 1, 1, 1])

    def convK(self, src, weights, kh, kw, pads):
        w = self.init_f(weights, [1, 1, kh, kw])
        return self.node("Conv", [src, w], kernel_shape=[kh, kw], pads=pads)

    def matmul(self, a, b):
        return self.node("MatMul", [a, b])

    def colorint(self, src):
        """[1,1,30,30] integer color index per cell via 1x1 conv (10 params)."""
        w = self.init_f([float(c) for c in range(10)], [1, 10, 1, 1])
        return self.node("Conv", [src, w], kernel_shape=[1, 1], pads=[0, 0, 0, 0])

    def abs(self, x):
        return self.node("Abs", [x])

    def less(self, x, thr):
        b = self.node("Less", [x, self.scalar(thr)])
        return self.node("Cast", [b], to=FLOAT)

    def greater(self, x, thr):
        b = self.node("Greater", [x, self.scalar(thr)])
        return self.node("Cast", [b], to=FLOAT)

    def mul(self, a, b):
        return self.node("Mul", [a, b])

    def add(self, a, b):
        return self.node("Add", [a, b])

    def sub(self, a, b):
        return self.node("Sub", [a, b])

    def rsum(self, x, axes, keepdims=1):
        return self.node("ReduceSum", [x], axes=list(axes), keepdims=keepdims)

    def rmax(self, x, axes, keepdims=1):
        return self.node("ReduceMax", [x], axes=list(axes), keepdims=keepdims)

    def where(self, c, a, b, out=None):
        return self.node("Where", [c, a, b], out=out)

    def concat10(self, chans):
        return self.node("Concat", chans, out="output", axis=1)

    def model(self):
        return _model(self.nodes, self.inits)


PLUS = [0, 1, 0, 1, 0, 1, 0, 1, 0]
BOX8 = [1, 1, 1, 1, 0, 1, 1, 1, 1]


# ============================ T369: size_bg5_fg0 ============================
# grid is {0,5}; recolor each 0-component by its size: 1->3, 2->2, 3->1; 5 stays.
def _conv_plus(z):
    o = np.zeros_like(z)
    o[1:, :] += z[:-1, :]; o[:-1, :] += z[1:, :]
    o[:, 1:] += z[:, :-1]; o[:, :-1] += z[:, 1:]
    return o


def sim369(i):
    if not set(np.unique(i)).issubset({0, 5}):
        return None
    z = (i == 0).astype(np.float32)
    deg = _conv_plus(z)
    ndeg = _conv_plus(deg * z)
    v = deg + ndeg
    s1 = z * (v < 1)
    s3 = z * (v > 2.5)
    s2 = z - s1 - s3
    out = np.zeros_like(i)
    out[s3 > 0.5] = 1
    out[s2 > 0.5] = 2
    out[s1 > 0.5] = 3
    out[i == 5] = 5
    return out


def build369():
    g = G()
    z = g.chan("input", 0)
    deg = g.conv(z, PLUS)
    ndeg = g.conv(g.mul(deg, z), PLUS)
    v = g.add(deg, ndeg)
    s1 = g.mul(z, g.less(v, 1.0))
    s3 = g.mul(z, g.greater(v, 2.5))
    s2 = g.sub(z, g.add(s1, s3))
    zeros = g.sub(z, z)
    ch5 = g.chan("input", 5)
    # channels: 0,1,2,3,4,5,6,7,8,9
    g.concat10([zeros, s3, s2, s1, zeros, ch5, zeros, zeros, zeros, zeros])
    return g.model()


# ============================ T344: localadj_3_2_8 ==========================
# colors {0,2,3,5}; a 3 orthogonally next to a 2 becomes 8, a 2 next to a 3 becomes 0.
def sim344(i):
    if not set(np.unique(i)).issubset({0, 2, 3, 5}):
        return None
    c2 = (i == 2).astype(np.float32); c3 = (i == 3).astype(np.float32)
    has2 = (_conv_plus(c2) > 0.5).astype(np.float32)
    has3 = (_conv_plus(c3) > 0.5).astype(np.float32)
    became8 = c3 * has2; removed2 = c2 * has3
    out = i.copy()
    out[became8 > 0.5] = 8
    out[removed2 > 0.5] = 0
    return out


def build344():
    g = G()
    ch0 = g.chan("input", 0)
    c2 = g.chan("input", 2)
    c3 = g.chan("input", 3)
    c5 = g.chan("input", 5)
    has2 = g.greater(g.conv(c2, PLUS), 0.5)
    has3 = g.greater(g.conv(c3, PLUS), 0.5)
    became8 = g.mul(c3, has2)
    removed2 = g.mul(c2, has3)
    ch0o = g.add(ch0, removed2)
    ch2o = g.sub(c2, removed2)
    ch3o = g.sub(c3, became8)
    zeros = g.sub(c5, c5)
    g.concat10([ch0o, zeros, ch2o, ch3o, zeros, c5, zeros, zeros, became8, zeros])
    return g.model()


# ============================ T359: band majority ===========================
# grid = axis-aligned solid color bands + salt/pepper noise. Restore each band to its
# majority color. Orientation (column vs row bands) chosen by whichever yields the
# higher purity (more cells already equal to that band-majority).
def _maj_axis(i, axis):
    out = np.zeros_like(i)
    if axis == 0:  # column bands: majority down each column
        for c in range(i.shape[1]):
            v, n = np.unique(i[:, c], return_counts=True)
            out[:, c] = v[np.argmax(n)]
    else:
        for r in range(i.shape[0]):
            v, n = np.unique(i[r, :], return_counts=True)
            out[r, :] = v[np.argmax(n)]
    return out


def sim359(i):
    cm = _maj_axis(i, 0); rm = _maj_axis(i, 1)
    pc = (cm == i).mean(); pr = (rm == i).mean()
    return cm if pc >= pr else rm


def build359():
    g = G()
    half = g.scalar(0.5)
    # column-band majority one-hot, broadcast over rows -> [1,10,1,30]
    colsum = g.rsum("input", [2])              # [1,10,1,30]
    colmax = g.rmax(colsum, [1])               # [1,1,1,30]
    col_oh = g.node("Cast", [g.node("Greater", [colsum, g.sub(colmax, half)])], to=FLOAT)
    # row-band majority one-hot -> [1,10,30,1]
    rowsum = g.rsum("input", [3])              # [1,10,30,1]
    rowmax = g.rmax(rowsum, [1])               # [1,1,30,1]
    row_oh = g.node("Cast", [g.node("Greater", [rowsum, g.sub(rowmax, half)])], to=FLOAT)
    # pick orientation: use columns iff sum(colmax) >= sum(rowmax)
    sc = g.rsum(colmax, [0, 1, 2, 3]); sr = g.rsum(rowmax, [0, 1, 2, 3])
    cond = g.node("Greater", [g.add(sc, half), sr])
    chosen = g.where(cond, col_oh, row_oh)     # [1,10,30,30]
    occ = g.rsum("input", [1])                 # [1,1,30,30] in-grid mask
    g.node("Mul", [chosen, occ], out="output")
    return g.model()


# ============================ T10: rank bars by height ======================
# grid {0,5}; each column-bar of 5s recolored by rank of its height (tallest -> 1).
def sim10(i):
    if not set(np.unique(i)).issubset({0, 5}):
        return None
    c5 = (i == 5)
    W = i.shape[1]
    h = c5.sum(axis=0)
    out = i.copy()
    for j in range(W):
        if h[j] > 0:
            out[c5[:, j], j] = 1 + int((h > h[j]).sum())
    out[i == 0] = 0
    return out


def build10():
    g = G()
    c5 = g.chan("input", 5)
    ch0 = g.chan("input", 0)
    h = g.rsum(c5, [2])                                   # [1,1,1,30] heights
    hT = g.node("Transpose", [h], perm=[0, 1, 3, 2])      # [1,1,30,1]
    gt = g.node("Cast", [g.node("Greater", [hT, h])], to=FLOAT)  # [1,1,30,30] h_k>h_j
    count = g.rsum(gt, [2])                               # [1,1,1,30]
    rank = g.add(count, g.scalar(1.0))                    # [1,1,1,30]
    vlo = g.init_f([m - 0.5 for m in range(10)], [1, 10, 1, 1])
    vhi = g.init_f([m + 0.5 for m in range(10)], [1, 10, 1, 1])
    oh = g.mul(g.node("Cast", [g.node("Greater", [rank, vlo])], to=FLOAT),
               g.node("Cast", [g.node("Less", [rank, vhi])], to=FLOAT))  # [1,10,1,30]
    chans = [ch0]
    for m in range(1, 10):
        s = g.init_i64([m]); e = g.init_i64([m + 1]); a = g.init_i64([1])
        eqm = g.node("Slice", [oh, s, e, a])             # [1,1,1,30]
        chans.append(g.mul(c5, eqm))                     # [1,1,30,30]
    g.concat10(chans)
    return g.model()


# ============================ T85: stripemid ================================
# In the middle row of each 3-tall solid bar, blank every other cell (offset-odd from
# the bar's left edge -> background), leaving a dashed middle row.
def sim85(i):
    col = i.astype(float)
    above = np.zeros_like(col); below = np.zeros_like(col)
    above[1:] = col[:-1]; below[:-1] = col[1:]
    mid = (col > 0.5) & ((np.abs(col - above) + np.abs(col - below)) < 0.5)
    cs = np.cumsum(mid.astype(int), axis=1)
    out = i.copy()
    out[mid & (cs % 2 == 0)] = 0
    return out


def build85():
    g = G()
    col = g.colorint("input")                              # [1,1,30,30] color index
    above = g.convK(col, [1, 0, 0], 3, 1, [1, 0, 1, 0])
    below = g.convK(col, [0, 0, 1], 3, 1, [1, 0, 1, 0])
    dsum = g.add(g.abs(g.sub(col, above)), g.abs(g.sub(col, below)))
    mid = g.mul(g.greater(col, 0.5), g.less(dsum, 0.5))    # 1 at 3-tall middle row
    U = g.init_f([1.0 if k <= c else 0.0 for k in range(WIDTH) for c in range(WIDTH)],
                 [WIDTH, WIDTH])
    cs = g.matmul(mid, U)                                  # cumulative count along width
    par = g.node("Mod", [cs, g.scalar(2.0)], fmod=1)       # 0/1 parity
    rem = g.node("Greater", [g.mul(mid, g.less(par, 0.5)), g.scalar(0.5)])  # bool blank mask
    bg = g.init_f([1.0] + [0.0] * 9, [1, 10, 1, 1])        # background one-hot
    g.where(rem, bg, "input", out="output")
    return g.model()


# ============================ T193: denoise =================================
# single fg color; keep a fg cell iff it has >=2 orthogonal same-color neighbors.
def sim193(i):
    if len(set(np.unique(i).tolist()) - {0}) > 1:
        return None
    fg = (i > 0).astype(np.float32)
    keep = (fg > 0.5) & (_conv_plus(fg) >= 2)
    return np.where(keep, i, 0)


def build193():
    g = G()
    fg = g.sub(g.rsum("input", [1]), g.chan("input", 0))
    cnt = g.conv(fg, PLUS)
    rem = g.node("Greater", [g.mul(fg, g.less(cnt, 1.5)), g.scalar(0.5)])
    bg = g.init_f([1.0] + [0.0] * 9, [1, 10, 1, 1])
    g.where(rem, bg, "input", out="output")
    return g.model()


# ============================ T312: row-marker fill =========================
# every 5-cell is recolored to the marker color sitting at col0 of its row.
def sim312(i):
    if 5 not in np.unique(i):
        return None
    out = i.copy()
    for r in range(i.shape[0]):
        out[r, i[r] == 5] = i[r, 0]
    return out


def build312():
    g = G()
    c5 = g.node("Greater", [g.chan("input", 5), g.scalar(0.5)])     # [1,1,30,30] bool
    s = g.init_i64([0]); e = g.init_i64([1]); a = g.init_i64([3])
    col0 = g.node("Slice", ["input", s, e, a])                       # [1,10,30,1]
    g.where(c5, col0, "input", out="output")
    return g.model()


# ============================ T389: figure-ground ===========================
# two colors, one is 5; recolor 5-cells to the other color, blank everything else.
def sim389(i):
    cols = set(np.unique(i).tolist()) - {0}
    if 5 not in cols:
        return None
    others = cols - {5}
    if len(others) != 1:
        return None
    return np.where(i == 5, others.pop(), 0)


def build389():
    g = G()
    c5 = g.node("Greater", [g.chan("input", 5), g.scalar(0.5)])    # [1,1,30,30] bool
    occ = g.rsum("input", [1])                                      # [1,1,30,30]
    pres = g.rmax(g.rmax("input", [2]), [3])                        # [1,10,1,1] color presence
    presmask = g.init_f([0.0, 1, 1, 1, 1, 0, 1, 1, 1, 1], [1, 10, 1, 1])
    v = g.mul(pres, presmask)                                       # one-hot of "other" color
    bgvec = g.init_f([1.0] + [0.0] * 9, [1, 10, 1, 1])
    occ_bg = g.mul(occ, bgvec)                                      # [1,10,30,30] ch0=occupied
    g.where(c5, v, occ_bg, out="output")
    return g.model()


# --------------------------- registry / dispatch ---------------------------
RULES = [
    (sim369, build369),
    (sim344, build344),
    (sim359, build359),
    (sim10, build10),
    (sim85, build85),
    (sim193, build193),
    (sim312, build312),
    (sim389, build389),
]


def _grids(examples):
    out = []
    for e in examples.get("train", []) + examples.get("test", []):
        out.append((np.array(e["input"]), np.array(e["output"])))
    return out


def candidates(examples):
    prs = _grids(examples)
    if not prs:
        return []
    res = []
    for sim, build in RULES:
        ok = True
        for i, o in prs:
            if max(i.shape) > 30:
                ok = False; break
            p = sim(i)
            if p is None or p.shape != o.shape or not (p == o).all():
                ok = False; break
        if ok:
            try:
                res.append((build.__name__, build()))
            except Exception:
                pass
    return res
