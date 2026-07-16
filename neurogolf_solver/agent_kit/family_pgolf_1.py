"""family_pgolf_1 — cheaper GENERALIZING solvers for golf slice G[1::4].

Strategy: the *lowest-scoring* targets in this slice are solved today by very
expensive general solvers (cost in the 0.5M-5M range).  A correct, genuinely
grid-agnostic solver that stays under a few hundred KB of intermediate memory
beats them outright — even without extreme golfing — because
points = 25 - ln(cost).

Every builder below encodes the TRUE underlying rule (no per-neighbourhood LUT,
no grid-size crop) and is emitted only after a strict self-check confirms it is
EXACT on train + test + ALL provided arc-gen pairs.  Under the harness'
max-points-per-task selection an emitted-but-not-cheaper candidate is simply
ignored, so correctness is the only hard requirement.

task 364 (recolor_s3_261): recolour every 4-connected component of colour 3 by
  its shape topology —
    * has a branch cell (degree>=3)          -> 2   (T / plus / junction)
    * else >=2 rectilinear corners           -> 6   (U / S / Z / closed loop)
    * else (<=1 corner: straight line or L)  -> 1
  All three are GLOBAL per-component properties, so they are computed locally
  then flooded across each component with a connectivity-respecting
  (H-then-V, masked) max-dilation.  ">=2 corners" is detected without a sum by
  propagating the max and min corner cell-index and testing max!=min.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

INT64 = onnx.TensorProto.INT64


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _exact(pred_fn, pairs):
    for a, b in pairs:
        p = pred_fn(a)
        if p.shape != b.shape or not (p == b).all():
            return False
    return True


# ---------------------------------------------------------------------------
# task 364
# ---------------------------------------------------------------------------
_IDX = np.zeros((30, 30), np.float32)
for _r in range(30):
    for _c in range(30):
        _IDX[_r, _c] = _r * 30 + _c + 1


def _pred364(a: np.ndarray, NIT: int = 12) -> np.ndarray:
    h, w = a.shape
    x = np.zeros((10, 30, 30), np.float32)
    for r in range(h):
        for c in range(w):
            x[a[r, c], r, c] = 1.0
    M = x[3]

    def conv(m, ker):
        out = np.zeros_like(m)
        mp = np.pad(m, 1)
        for dy in range(3):
            for dx in range(3):
                if ker[dy][dx]:
                    out += ker[dy][dx] * mp[dy:dy + 30, dx:dx + 30]
        return out

    N = conv(M, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    branchInd = np.clip(N - 2, 0, 1)
    branchCell = M * branchInd
    hasH = np.clip(conv(M, [[0, 0, 0], [1, 0, 1], [0, 0, 0]]), 0, 1)
    hasV = np.clip(conv(M, [[0, 1, 0], [0, 0, 0], [0, 1, 0]]), 0, 1)
    cornerCell = M * hasH * hasV * (1 - branchInd)
    Cmax = cornerCell * _IDX
    Cmin = cornerCell * (901 - _IDX)

    P = np.stack([branchCell, Cmax, Cmin])[None]
    M3 = np.stack([M, M, M])[None]

    def hmax(p):
        l = np.pad(p, ((0, 0), (0, 0), (0, 0), (1, 0)))[:, :, :, :30]
        r = np.pad(p, ((0, 0), (0, 0), (0, 0), (0, 1)))[:, :, :, 1:]
        return np.maximum(np.maximum(p, l), r)

    def vmax(p):
        u = np.pad(p, ((0, 0), (0, 0), (1, 0), (0, 0)))[:, :, :30, :]
        d = np.pad(p, ((0, 0), (0, 0), (0, 1), (0, 0)))[:, :, 1:, :]
        return np.maximum(np.maximum(p, u), d)

    for _ in range(NIT):
        P = hmax(P) * M3
        P = vmax(P) * M3

    Bp, Cmx, Cmn = P[0, 0], P[0, 1], P[0, 2]
    hasBranch = np.clip(Bp, 0, 1)
    two = np.clip(Cmx + Cmn - 901, 0, 1)
    nb = 1 - hasBranch
    c2 = M * hasBranch
    c6 = M * nb * two
    c1 = M * nb * (1 - two)
    out = x.copy()
    out[3] -= M
    out[1] += c1
    out[2] += c2
    out[6] += c6
    return np.argmax(out, axis=0)[:h, :w]


def _c(name, arr, shape):
    return oh.make_tensor(name, DATA_TYPE, shape, list(np.asarray(arr, np.float32).flatten()))


def _build364(NIT: int = 12):
    inits = []
    # slice initializers to pull channel 3
    inits.append(oh.make_tensor("s3", INT64, [1], [3]))
    inits.append(oh.make_tensor("e4", INT64, [1], [4]))
    inits.append(oh.make_tensor("ax1", INT64, [1], [1]))
    # neighbour convs packed: out0=deg(cross) out1=hasH-count out2=hasV-count
    ker = np.zeros((3, 1, 3, 3), np.float32)
    ker[0, 0] = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]   # degree
    ker[1, 0] = [[0, 0, 0], [1, 0, 1], [0, 0, 0]]   # horizontal
    ker[2, 0] = [[0, 1, 0], [0, 0, 0], [0, 1, 0]]   # vertical
    inits.append(_c("nker", ker, [3, 1, 3, 3]))
    inits.append(_c("c2", 2.0, [1, 1, 1, 1]))
    inits.append(_c("c1", 1.0, [1, 1, 1, 1]))
    inits.append(_c("c901", 901.0, [1, 1, 1, 1]))
    inits.append(_c("idx", _IDX, [1, 1, 30, 30]))
    # final recolour conv weight [10,4,1,1]; inputs order [c1mask,c2mask,c6mask,M]
    W = np.zeros((10, 4, 1, 1), np.float32)
    W[1, 0] = 1.0    # ch1 += c1
    W[2, 1] = 1.0    # ch2 += c2
    W[6, 2] = 1.0    # ch6 += c6
    W[3, 3] = -1.0   # ch3 -= M
    inits.append(_c("Wrec", W, [10, 4, 1, 1]))

    n = []
    n.append(oh.make_node("Slice", ["input", "s3", "e4", "ax1"], ["M"]))
    # neighbour features
    n.append(oh.make_node("Conv", ["M", "nker"], ["NB"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    n.append(oh.make_node("Slice", ["NB", "s0", "s1", "ax1"], ["deg"]))
    n.append(oh.make_node("Slice", ["NB", "s1", "s2", "ax1"], ["hc"]))
    n.append(oh.make_node("Slice", ["NB", "s2", "s3b", "ax1"], ["vc"]))
    inits += [oh.make_tensor("s0", INT64, [1], [0]),
              oh.make_tensor("s1", INT64, [1], [1]),
              oh.make_tensor("s2", INT64, [1], [2]),
              oh.make_tensor("s3b", INT64, [1], [3])]
    n.append(oh.make_node("Sub", ["deg", "c2"], ["degm2"]))
    n.append(oh.make_node("Clip", ["degm2"], ["branchInd"], min=0.0, max=1.0))
    n.append(oh.make_node("Mul", ["M", "branchInd"], ["branchCell"]))
    n.append(oh.make_node("Clip", ["hc"], ["hasH"], min=0.0, max=1.0))
    n.append(oh.make_node("Clip", ["vc"], ["hasV"], min=0.0, max=1.0))
    n.append(oh.make_node("Sub", ["c1", "branchInd"], ["nbi"]))
    n.append(oh.make_node("Mul", ["hasH", "hasV"], ["hv"]))
    n.append(oh.make_node("Mul", ["M", "hv"], ["hvM"]))
    n.append(oh.make_node("Mul", ["hvM", "nbi"], ["cornerCell"]))
    n.append(oh.make_node("Mul", ["cornerCell", "idx"], ["Cmax0"]))
    n.append(oh.make_node("Sub", ["c901", "idx"], ["invidx"]))
    n.append(oh.make_node("Mul", ["cornerCell", "invidx"], ["Cmin0"]))
    # pack P0 = [branchCell, Cmax0, Cmin0]; M3 = [M,M,M]
    n.append(oh.make_node("Concat", ["branchCell", "Cmax0", "Cmin0"], ["P0"], axis=1))
    n.append(oh.make_node("Concat", ["M", "M", "M"], ["M3"], axis=1))
    cur = "P0"
    for i in range(NIT):
        n.append(oh.make_node("MaxPool", [cur], [f"h{i}"], kernel_shape=[1, 3], pads=[0, 1, 0, 1]))
        n.append(oh.make_node("Mul", [f"h{i}", "M3"], [f"hm{i}"]))
        n.append(oh.make_node("MaxPool", [f"hm{i}"], [f"v{i}"], kernel_shape=[3, 1], pads=[1, 0, 1, 0]))
        n.append(oh.make_node("Mul", [f"v{i}", "M3"], [f"PP{i}"]))
        cur = f"PP{i}"
    # unpack
    n.append(oh.make_node("Slice", [cur, "s0", "s1", "ax1"], ["Bp"]))
    n.append(oh.make_node("Slice", [cur, "s1", "s2", "ax1"], ["Cmx"]))
    n.append(oh.make_node("Slice", [cur, "s2", "s3b", "ax1"], ["Cmn"]))
    n.append(oh.make_node("Clip", ["Bp"], ["hasBranch"], min=0.0, max=1.0))
    n.append(oh.make_node("Add", ["Cmx", "Cmn"], ["sumC"]))
    n.append(oh.make_node("Sub", ["sumC", "c901"], ["sumCm"]))
    n.append(oh.make_node("Clip", ["sumCm"], ["two"], min=0.0, max=1.0))
    n.append(oh.make_node("Sub", ["c1", "hasBranch"], ["nb"]))
    n.append(oh.make_node("Mul", ["M", "hasBranch"], ["c2mask"]))
    n.append(oh.make_node("Mul", ["nb", "two"], ["nbtwo"]))
    n.append(oh.make_node("Mul", ["M", "nbtwo"], ["c6mask"]))
    n.append(oh.make_node("Sub", ["c1", "two"], ["nottwo"]))
    n.append(oh.make_node("Mul", ["nb", "nottwo"], ["nbnt"]))
    n.append(oh.make_node("Mul", ["M", "nbnt"], ["c1mask"]))
    n.append(oh.make_node("Concat", ["c1mask", "c2mask", "c6mask", "M"], ["S"], axis=1))
    n.append(oh.make_node("Conv", ["S", "Wrec"], ["delta"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]))
    n.append(oh.make_node("Add", ["input", "delta"], ["output"]))
    return _model(n, inits)


# ---------------------------------------------------------------------------
def candidates(examples):
    train = examples.get("train", [])
    test = examples.get("test", [])
    gen = examples.get("arc-gen", [])
    pairs = [(np.array(e["input"]), np.array(e["output"]))
             for e in train + test + gen]
    if not pairs:
        return []
    out = []

    # task 364 signature: same-shape, colour 3 -> {1,2,6}, no 3 in output
    same = all(a.shape == b.shape for a, b in pairs)
    if same:
        sig = all((a == 3).any() and not (b == 3).any() for a, b in pairs)
        if sig and _exact(lambda a: _pred364(a, 12), pairs):
            out.append(("shape_topology_recolor", _build364(12)))

    return out
