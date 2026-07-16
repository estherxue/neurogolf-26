"""family_crk7_0 — cracking hard unsolved ARC tasks for NeuroGolf.

Each task is detected by a structural signature on the train+test pairs; if it
matches, we emit a hand-built opset-10 ONNX graph realizing the rule.

Tasks attacked (slice U[0::6]):
  048 : two 2x2 blocks of color 2 connected through a maze of nonzero cells ->
        1x1 output, color 8 if the two blocks are connected else color 0.
        Realized with label-propagation flood (8-conn MaxPool) + max/min compare.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE


def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _ct(name, arr, dtype=F):
    arr = np.asarray(arr)
    return oh.make_tensor(name, dtype, list(arr.shape), arr.ravel().tolist())


# ---------------------------------------------------------------- task 048 ----

def _components(mask, conn=4):
    H, W = mask.shape
    lab = -np.ones((H, W), int)
    n = 0
    if conn == 4:
        nb = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        nb = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] < 0:
                st = [(i, j)]; lab[i, j] = n
                while st:
                    y, x = st.pop()
                    for dy, dx in nb:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lab[ny, nx] < 0:
                            lab[ny, nx] = n; st.append((ny, nx))
                n += 1
    return lab


def _is_task048(pairs):
    # every input has exactly color-2 cells forming >=2 blobs; output is 1x1 of
    # color 0 or 8 according to whether all the 2-cells are in one nonzero comp.
    if not pairs:
        return False
    for a, b in pairs:
        if b.shape != (1, 1):
            return False
        if b[0, 0] not in (0, 8):
            return False
        if (a == 2).sum() == 0:
            return False
        lab = _components(a != 0, 4)
        labs = set(lab[a == 2].tolist())
        pred = 8 if len(labs) == 1 else 0
        if pred != b[0, 0]:
            return False
    return True


def _build_task048(conn8=True, iters=20):
    nodes = []
    inits = []
    # ramp constant for unique labels
    ramp = (np.arange(HEIGHT)[:, None] * WIDTH + np.arange(WIDTH)[None, :] + 1).astype(np.float32)
    inits.append(_ct("ramp", ramp.reshape(1, 1, HEIGHT, WIDTH)))
    inits.append(_ct("one", np.array([1.0], np.float32)))
    inits.append(_ct("BIG", np.array([1e6], np.float32)))
    # slot constants: [1,10,1,1] one-hot on channel 0 and channel 8
    s0 = np.zeros((1, CHANNELS, 1, 1), np.float32); s0[0, 0, 0, 0] = 1.0
    s8 = np.zeros((1, CHANNELS, 1, 1), np.float32); s8[0, 8, 0, 0] = 1.0
    inits.append(_ct("slot0", s0)); inits.append(_ct("slot8", s8))
    # slice channels 1..9 (real colours) and channel 2
    for nm, st, en in (("chc", 1, CHANNELS), ("ch2", 2, 3)):
        inits.append(_ct(nm + "_s", np.array([st], np.int64), INT64))
        inits.append(_ct(nm + "_e", np.array([en], np.int64), INT64))
        inits.append(_ct(nm + "_a", np.array([1], np.int64), INT64))
        nodes.append(oh.make_node("Slice", ["input", nm + "_s", nm + "_e", nm + "_a"], [nm]))
    # fg = sum of channels 1..9 (1 on real coloured cells, 0 on bg and padding)
    nodes.append(oh.make_node("ReduceSum", ["chc"], ["fg"], axes=[1], keepdims=1))
    # L0 = fg * ramp
    nodes.append(oh.make_node("Mul", ["fg", "ramp"], ["L0"]))
    cur = "L0"
    for k in range(iters):
        mp = f"mp{k}"; lk = f"L{k+1}"
        if conn8:
            nodes.append(oh.make_node("MaxPool", [cur], [mp], kernel_shape=[3, 3],
                                      pads=[1, 1, 1, 1], strides=[1, 1]))
        else:
            # 4-conn: max over self + 4 orthogonal neighbours via pad/slice shifts
            raise NotImplementedError
        nodes.append(oh.make_node("Mul", [mp, "fg"], [lk]))
        cur = lk
    # restrict to color-2 cells
    nodes.append(oh.make_node("Mul", [cur, "ch2"], ["Lc2"]))
    nodes.append(oh.make_node("ReduceMax", ["Lc2"], ["maxlab"], axes=[2, 3], keepdims=1))
    # min over 2-cells: add BIG where not a 2-cell
    nodes.append(oh.make_node("Sub", ["one", "ch2"], ["notc2"]))
    nodes.append(oh.make_node("Mul", ["notc2", "BIG"], ["bigmask"]))
    nodes.append(oh.make_node("Add", ["Lc2", "bigmask"], ["Lc2b"]))
    nodes.append(oh.make_node("ReduceMin", ["Lc2b"], ["minlab"], axes=[2, 3], keepdims=1))
    # disconnected = maxlab > minlab
    nodes.append(oh.make_node("Greater", ["maxlab", "minlab"], ["gt"]))
    nodes.append(oh.make_node("Cast", ["gt"], ["disc"], to=int(F)))
    nodes.append(oh.make_node("Sub", ["one", "disc"], ["conn"]))
    # out_small[1,10,1,1] = slot0*disc + slot8*conn
    nodes.append(oh.make_node("Mul", ["slot0", "disc"], ["o0"]))
    nodes.append(oh.make_node("Mul", ["slot8", "conn"], ["o8"]))
    nodes.append(oh.make_node("Add", ["o0", "o8"], ["osmall"]))
    nodes.append(oh.make_node("Pad", ["osmall"], ["output"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, HEIGHT - 1, WIDTH - 1]))
    return _model(nodes, inits)


# ---------------------------------------------------------------- task 355 ----

def _predict355(a):
    H, W = a.shape
    counts = np.array([(a == c).sum() for c in range(10)]).astype(float)
    present = counts > 0
    if present.sum() < 2:
        return None
    cc = counts.copy(); cc[~present] = 1e9
    N = int(cc.argmin())
    Nmask = (a == N)
    noise = np.full(10, -1e9)
    for c in range(10):
        if not present[c] or c == N:
            continue
        ys, xs = np.where(a == c)
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        noise[c] = Nmask[r0:r1 + 1, c0:c1 + 1].sum()
    if noise.max() <= 0:
        return None
    return int(noise.argmax())


def _is_task355(pairs):
    if not pairs:
        return False
    for a, b in pairs:
        if b.shape != (1, 1):
            return False
        p = _predict355(a)
        if p is None or p != b[0, 0]:
            return False
    return True


def _build_task355():
    nodes = []; inits = []
    inits.append(_ct("one", np.array([1.0], np.float32)))
    inits.append(_ct("half", np.array([0.5], np.float32)))
    inits.append(_ct("BIG", np.array([1e6], np.float32)))
    rowidx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    colidx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    inits.append(_ct("rowidx", rowidx)); inits.append(_ct("colidx", colidx))

    def cast(src, dst):
        nodes.append(oh.make_node("Cast", [src], [dst], to=int(F)))

    # counts per colour
    nodes.append(oh.make_node("ReduceSum", ["input"], ["counts"], axes=[2, 3], keepdims=1))
    nodes.append(oh.make_node("Greater", ["counts", "half"], ["presB"]))
    cast("presB", "presF")
    nodes.append(oh.make_node("Sub", ["one", "presF"], ["absent"]))
    nodes.append(oh.make_node("Mul", ["absent", "BIG"], ["absBig"]))
    nodes.append(oh.make_node("Add", ["counts", "absBig"], ["countAdj"]))
    nodes.append(oh.make_node("ReduceMin", ["countAdj"], ["minCount"], axes=[1], keepdims=1))
    nodes.append(oh.make_node("Greater", ["countAdj", "minCount"], ["gtMinB"]))
    cast("gtMinB", "gtMin")
    nodes.append(oh.make_node("Sub", ["one", "gtMin"], ["oneHotN"]))      # [1,10,1,1]
    # Nmask
    nodes.append(oh.make_node("Mul", ["input", "oneHotN"], ["inN"]))
    nodes.append(oh.make_node("ReduceSum", ["inN"], ["Nmask"], axes=[1], keepdims=1))  # [1,1,30,30]
    # bbox per colour
    nodes.append(oh.make_node("ReduceMax", ["input"], ["rowHas"], axes=[3], keepdims=1))  # [1,10,30,1]
    nodes.append(oh.make_node("ReduceMax", ["input"], ["colHas"], axes=[2], keepdims=1))  # [1,10,1,30]
    nodes.append(oh.make_node("Mul", ["rowHas", "rowidx"], ["rH_i"]))
    nodes.append(oh.make_node("ReduceMax", ["rH_i"], ["rmax"], axes=[2], keepdims=1))
    nodes.append(oh.make_node("Sub", ["one", "rowHas"], ["rNo"]))
    nodes.append(oh.make_node("Mul", ["rNo", "BIG"], ["rNoBig"]))
    nodes.append(oh.make_node("Add", ["rH_i", "rNoBig"], ["rminT"]))
    nodes.append(oh.make_node("ReduceMin", ["rminT"], ["rmin"], axes=[2], keepdims=1))
    nodes.append(oh.make_node("Mul", ["colHas", "colidx"], ["cH_j"]))
    nodes.append(oh.make_node("ReduceMax", ["cH_j"], ["cmax"], axes=[3], keepdims=1))
    nodes.append(oh.make_node("Sub", ["one", "colHas"], ["cNo"]))
    nodes.append(oh.make_node("Mul", ["cNo", "BIG"], ["cNoBig"]))
    nodes.append(oh.make_node("Add", ["cH_j", "cNoBig"], ["cminT"]))
    nodes.append(oh.make_node("ReduceMin", ["cminT"], ["cmin"], axes=[3], keepdims=1))
    # inBbox
    nodes.append(oh.make_node("Sub", ["rmin", "half"], ["rmin_h"]))
    nodes.append(oh.make_node("Add", ["rmax", "half"], ["rmax_h"]))
    nodes.append(oh.make_node("Greater", ["rowidx", "rmin_h"], ["geR0B"])); cast("geR0B", "geR0")
    nodes.append(oh.make_node("Less", ["rowidx", "rmax_h"], ["leR1B"])); cast("leR1B", "leR1")
    nodes.append(oh.make_node("Mul", ["geR0", "leR1"], ["rowIn"]))        # [1,10,30,1]
    nodes.append(oh.make_node("Sub", ["cmin", "half"], ["cmin_h"]))
    nodes.append(oh.make_node("Add", ["cmax", "half"], ["cmax_h"]))
    nodes.append(oh.make_node("Greater", ["colidx", "cmin_h"], ["geC0B"])); cast("geC0B", "geC0")
    nodes.append(oh.make_node("Less", ["colidx", "cmax_h"], ["leC1B"])); cast("leC1B", "leC1")
    nodes.append(oh.make_node("Mul", ["geC0", "leC1"], ["colIn"]))        # [1,10,1,30]
    nodes.append(oh.make_node("Mul", ["rowIn", "colIn"], ["inBbox"]))     # [1,10,30,30]
    nodes.append(oh.make_node("Mul", ["inBbox", "Nmask"], ["inBN"]))
    nodes.append(oh.make_node("ReduceSum", ["inBN"], ["noise"], axes=[2, 3], keepdims=1))  # [1,10,1,1]
    # exclude N & absent
    nodes.append(oh.make_node("Add", ["oneHotN", "absent"], ["exMask"]))
    nodes.append(oh.make_node("Mul", ["exMask", "BIG"], ["exBig"]))
    nodes.append(oh.make_node("Sub", ["noise", "exBig"], ["noiseAdj"]))
    nodes.append(oh.make_node("ReduceMax", ["noiseAdj"], ["maxNoise"], axes=[1], keepdims=1))
    nodes.append(oh.make_node("Sub", ["maxNoise", "half"], ["maxNoise_h"]))
    nodes.append(oh.make_node("Greater", ["noiseAdj", "maxNoise_h"], ["ansB"])); cast("ansB", "oneHotA")
    nodes.append(oh.make_node("Pad", ["oneHotA"], ["output"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, HEIGHT - 1, WIDTH - 1]))
    return _model(nodes, inits)


# ---------------------------------------------------------------- task 183 ----

def _predict183(a):
    H, W = a.shape
    if H < 5 or W < 5:
        return None
    TL, TR, BL, BR = a[0, 0], a[0, W - 1], a[H - 1, 0], a[H - 1, W - 1]
    out = np.zeros((H - 4, W - 4), int)
    for r in range(2, H - 2):
        for c in range(2, W - 2):
            if a[r, c] == 8:
                top = r < H / 2
                left = c < W / 2
                out[r - 2, c - 2] = (TL if (top and left) else TR if top else
                                     BL if left else BR)
    return out


def _is_task183(pairs):
    if not pairs:
        return False
    for a, b in pairs:
        if b.shape != (a.shape[0] - 4, a.shape[1] - 4):
            return False
        p = _predict183(a)
        if p is None or p.shape != b.shape or not np.array_equal(p, b):
            return False
    return True


def _build_task183():
    nodes = []; inits = []
    inits.append(_ct("one", np.array([1.0], np.float32)))
    inits.append(_ct("half", np.array([0.5], np.float32)))
    inits.append(_ct("two", np.array([2.0], np.float32)))
    rowidx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    colidx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    inits.append(_ct("rowidx", rowidx)); inits.append(_ct("colidx", colidx))
    slot0 = np.zeros((1, CHANNELS, 1, 1), np.float32); slot0[0, 0, 0, 0] = 1.0
    inits.append(_ct("slot0", slot0))

    def cast(s, d):
        nodes.append(oh.make_node("Cast", [s], [d], to=int(F)))

    def sl(name, src, starts, ends, axes):
        inits.append(_ct(name + "_s", np.array(starts, np.int64), INT64))
        inits.append(_ct(name + "_e", np.array(ends, np.int64), INT64))
        inits.append(_ct(name + "_a", np.array(axes, np.int64), INT64))
        nodes.append(oh.make_node("Slice", [src, name + "_s", name + "_e", name + "_a"], [name]))

    # fg = sum channels 1..9
    sl("chc", "input", [1], [CHANNELS], [1])
    nodes.append(oh.make_node("ReduceSum", ["chc"], ["fg"], axes=[1], keepdims=1))   # [1,1,30,30]
    nodes.append(oh.make_node("ReduceMax", ["fg"], ["rowHas"], axes=[3], keepdims=1))  # [1,1,30,1]
    nodes.append(oh.make_node("ReduceMax", ["fg"], ["colHas"], axes=[2], keepdims=1))  # [1,1,1,30]
    nodes.append(oh.make_node("Mul", ["rowHas", "rowidx"], ["rH_i"]))
    nodes.append(oh.make_node("ReduceMax", ["rH_i"], ["maxRow"], axes=[2], keepdims=1))  # H-1
    nodes.append(oh.make_node("Mul", ["colHas", "colidx"], ["cH_j"]))
    nodes.append(oh.make_node("ReduceMax", ["cH_j"], ["maxCol"], axes=[3], keepdims=1))  # W-1

    # half thresholds = (max+1)/2
    nodes.append(oh.make_node("Add", ["maxRow", "one"], ["Hh"]))
    nodes.append(oh.make_node("Mul", ["Hh", "half"], ["halfH"]))
    nodes.append(oh.make_node("Add", ["maxCol", "one"], ["Ww"]))
    nodes.append(oh.make_node("Mul", ["Ww", "half"], ["halfW"]))
    nodes.append(oh.make_node("Less", ["rowidx", "halfH"], ["topB"])); cast("topB", "top")   # [1,1,30,1]
    nodes.append(oh.make_node("Less", ["colidx", "halfW"], ["leftB"])); cast("leftB", "left")  # [1,1,1,30]
    nodes.append(oh.make_node("Sub", ["one", "top"], ["bot"]))
    nodes.append(oh.make_node("Sub", ["one", "left"], ["right"]))

    # isMaxRow / isMaxCol  (== max via two inequalities)
    nodes.append(oh.make_node("Sub", ["maxRow", "half"], ["mr_lo"]))
    nodes.append(oh.make_node("Add", ["maxRow", "half"], ["mr_hi"]))
    nodes.append(oh.make_node("Greater", ["rowidx", "mr_lo"], ["mrgB"])); cast("mrgB", "mrg")
    nodes.append(oh.make_node("Less", ["rowidx", "mr_hi"], ["mrlB"])); cast("mrlB", "mrl")
    nodes.append(oh.make_node("Mul", ["mrg", "mrl"], ["isMaxRow"]))   # [1,1,30,1]
    nodes.append(oh.make_node("Sub", ["maxCol", "half"], ["mc_lo"]))
    nodes.append(oh.make_node("Add", ["maxCol", "half"], ["mc_hi"]))
    nodes.append(oh.make_node("Greater", ["colidx", "mc_lo"], ["mcgB"])); cast("mcgB", "mcg")
    nodes.append(oh.make_node("Less", ["colidx", "mc_hi"], ["mclB"])); cast("mclB", "mcl")
    nodes.append(oh.make_node("Mul", ["mcg", "mcl"], ["isMaxCol"]))   # [1,1,1,30]

    # corner colour one-hots [1,10,1,1]
    sl("TLvec", "input", [0, 0], [1, 1], [2, 3])                       # (0,0)
    sl("row0", "input", [0], [1], [2])                                 # [1,10,1,30]
    nodes.append(oh.make_node("Mul", ["row0", "isMaxCol"], ["row0m"]))
    nodes.append(oh.make_node("ReduceSum", ["row0m"], ["TRvec"], axes=[3], keepdims=1))  # (0,W-1)
    sl("col0", "input", [0], [1], [3])                                 # [1,10,30,1]
    nodes.append(oh.make_node("Mul", ["col0", "isMaxRow"], ["col0m"]))
    nodes.append(oh.make_node("ReduceSum", ["col0m"], ["BLvec"], axes=[2], keepdims=1))  # (H-1,0)
    nodes.append(oh.make_node("Mul", ["isMaxRow", "isMaxCol"], ["brMask"]))               # [1,1,30,30]
    nodes.append(oh.make_node("Mul", ["input", "brMask"], ["brSel"]))
    nodes.append(oh.make_node("ReduceSum", ["brSel"], ["BRvec"], axes=[2, 3], keepdims=1))  # (H-1,W-1)

    # interior mask
    nodes.append(oh.make_node("Greater", ["rowidx", "one"], ["ir_loB"])); cast("ir_loB", "ir_lo")  # row>=2
    nodes.append(oh.make_node("Sub", ["maxRow", "two"], ["mr2"]))
    nodes.append(oh.make_node("Add", ["mr2", "half"], ["mr2h"]))
    nodes.append(oh.make_node("Less", ["rowidx", "mr2h"], ["ir_hiB"])); cast("ir_hiB", "ir_hi")
    nodes.append(oh.make_node("Mul", ["ir_lo", "ir_hi"], ["intRow"]))  # [1,1,30,1]
    nodes.append(oh.make_node("Greater", ["colidx", "one"], ["ic_loB"])); cast("ic_loB", "ic_lo")
    nodes.append(oh.make_node("Sub", ["maxCol", "two"], ["mc2"]))
    nodes.append(oh.make_node("Add", ["mc2", "half"], ["mc2h"]))
    nodes.append(oh.make_node("Less", ["colidx", "mc2h"], ["ic_hiB"])); cast("ic_hiB", "ic_hi")
    nodes.append(oh.make_node("Mul", ["ic_lo", "ic_hi"], ["intCol"]))  # [1,1,1,30]
    nodes.append(oh.make_node("Mul", ["intRow", "intCol"], ["interior"]))  # [1,1,30,30]

    # m8 restricted to interior
    sl("m8", "input", [8], [9], [1])                                   # channel 8 [1,1,30,30]
    nodes.append(oh.make_node("Mul", ["m8", "interior"], ["m8i"]))
    # quadrant masks (interior cells)
    nodes.append(oh.make_node("Mul", ["top", "left"], ["TLq"]))        # [1,1,30,30]
    nodes.append(oh.make_node("Mul", ["top", "right"], ["TRq"]))
    nodes.append(oh.make_node("Mul", ["bot", "left"], ["BLq"]))
    nodes.append(oh.make_node("Mul", ["bot", "right"], ["BRq"]))
    for q, vec in (("TLq", "TLvec"), ("TRq", "TRvec"), ("BLq", "BLvec"), ("BRq", "BRvec")):
        nodes.append(oh.make_node("Mul", ["m8i", q], [q + "8"]))        # [1,1,30,30]
        nodes.append(oh.make_node("Mul", [vec, q + "8"], [vec + "_o"]))  # [1,10,30,30]
    nodes.append(oh.make_node("Add", ["TLvec_o", "TRvec_o"], ["q01"]))
    nodes.append(oh.make_node("Add", ["BLvec_o", "BRvec_o"], ["q23"]))
    nodes.append(oh.make_node("Add", ["q01", "q23"], ["qsum"]))
    # interior background -> channel 0
    nodes.append(oh.make_node("Sub", ["interior", "m8i"], ["bg0"]))    # interior non-8 cells [1,1,30,30]
    nodes.append(oh.make_node("Mul", ["slot0", "bg0"], ["bg0o"]))      # [1,10,30,30]
    nodes.append(oh.make_node("Add", ["qsum", "bg0o"], ["outFull"]))
    # crop: drop first 2 rows/cols, pad back at end
    sl("crop", "outFull", [2, 2], [HEIGHT, WIDTH], [2, 3])             # [1,10,28,28]
    nodes.append(oh.make_node("Pad", ["crop"], ["output"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, 2, 2]))
    return _model(nodes, inits)


# ------------------------------------------------------------------ task 4 ----

def _predict004(a):
    H, W = a.shape
    lab = -np.ones((H, W), int); n = 0
    for i in range(H):
        for j in range(W):
            if a[i, j] != 0 and lab[i, j] < 0:
                col = a[i, j]; st = [(i, j)]; lab[i, j] = n
                while st:
                    y, x = st.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and a[ny, nx] == col and lab[ny, nx] < 0:
                                lab[ny, nx] = n; st.append((ny, nx));
                n += 1
    mr = {}; mc = {}
    for i in range(H):
        for j in range(W):
            l = lab[i, j]
            if l >= 0:
                mr[l] = max(mr.get(l, -1), i); mc[l] = max(mc.get(l, -1), j)
    out = np.zeros((H, W), int)
    for i in range(H):
        for j in range(W):
            l = lab[i, j]
            if l < 0:
                continue
            if (i == mr[l]) or (j == mc[l]) or j + 1 >= W:
                out[i, j] = a[i, j]
            else:
                out[i, j + 1] = a[i, j]
    return out


def _is_task004(pairs):
    if not pairs:
        return False
    for a, b in pairs:
        if a.shape != b.shape:
            return False
        if np.array_equal(a, b):
            return False
        if not np.array_equal(_predict004(a), b):
            return False
    return True


def _build_task004(iters=24):
    nodes = []; inits = []
    inits.append(_ct("one", np.array([1.0], np.float32)))
    inits.append(_ct("half", np.array([0.5], np.float32)))
    rowidx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    colidx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    inits.append(_ct("rowidx", rowidx)); inits.append(_ct("colidx", colidx))
    slot0 = np.zeros((1, CHANNELS, 1, 1), np.float32); slot0[0, 0, 0, 0] = 1.0
    inits.append(_ct("slot0", slot0))
    inits.append(_ct("chc_s", np.array([1], np.int64), INT64))
    inits.append(_ct("chc_e", np.array([CHANNELS], np.int64), INT64))
    inits.append(_ct("chc_a", np.array([1], np.int64), INT64))

    def cast(s, d):
        nodes.append(oh.make_node("Cast", [s], [d], to=int(F)))

    nodes.append(oh.make_node("Slice", ["input", "chc_s", "chc_e", "chc_a"], ["chc"]))
    nodes.append(oh.make_node("ReduceSum", ["chc"], ["fg"], axes=[1], keepdims=1))  # [1,1,30,30]
    nodes.append(oh.make_node("ReduceSum", ["input"], ["inGrid"], axes=[1], keepdims=1))

    def flood(initmul, prefix):
        nodes.append(oh.make_node("Mul", ["fg", initmul], [prefix + "0"]))
        cur = prefix + "0"
        for k in range(iters):
            mp = f"{prefix}mp{k}"; nx = f"{prefix}{k+1}"
            nodes.append(oh.make_node("MaxPool", [cur], [mp], kernel_shape=[3, 3],
                                      pads=[1, 1, 1, 1], strides=[1, 1]))
            nodes.append(oh.make_node("Mul", [mp, "fg"], [nx]))
            cur = nx
        return cur

    maxColF = flood("colidx", "cf")
    maxRowF = flood("rowidx", "rf")
    nodes.append(oh.make_node("Sub", [maxColF, "half"], ["mcf_h"]))
    nodes.append(oh.make_node("Greater", ["colidx", "mcf_h"], ["rightB"])); cast("rightB", "rightC")
    nodes.append(oh.make_node("Sub", [maxRowF, "half"], ["mrf_h"]))
    nodes.append(oh.make_node("Greater", ["rowidx", "mrf_h"], ["botB"])); cast("botB", "botC")
    nodes.append(oh.make_node("Sub", ["one", "rightC"], ["notR"]))
    nodes.append(oh.make_node("Sub", ["one", "botC"], ["notB"]))
    nodes.append(oh.make_node("Mul", ["notR", "notB"], ["notStay"]))     # [1,1,30,30]
    nodes.append(oh.make_node("Sub", ["one", "notStay"], ["stay"]))
    nodes.append(oh.make_node("Mul", ["fg", "stay"], ["stayMask"]))
    nodes.append(oh.make_node("Mul", ["fg", "notStay"], ["moveMask"]))
    nodes.append(oh.make_node("Mul", ["input", "stayMask"], ["stayed"]))    # [1,10,30,30]
    nodes.append(oh.make_node("Mul", ["input", "moveMask"], ["movedSrc"]))
    nodes.append(oh.make_node("Pad", ["movedSrc"], ["movedPad"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 1, 0, 0, 0, 0]))               # [1,10,30,31]
    inits.append(_ct("sh_s", np.array([0], np.int64), INT64))
    inits.append(_ct("sh_e", np.array([WIDTH], np.int64), INT64))
    inits.append(_ct("sh_a", np.array([3], np.int64), INT64))
    nodes.append(oh.make_node("Slice", ["movedPad", "sh_s", "sh_e", "sh_a"], ["movedShift"]))
    nodes.append(oh.make_node("Add", ["stayed", "movedShift"], ["shapeOut"]))
    nodes.append(oh.make_node("ReduceSum", ["shapeOut"], ["covered"], axes=[1], keepdims=1))
    nodes.append(oh.make_node("Greater", ["covered", "half"], ["covB"])); cast("covB", "covBin")
    nodes.append(oh.make_node("Sub", ["one", "covBin"], ["uncov"]))
    nodes.append(oh.make_node("Mul", ["inGrid", "uncov"], ["bg0"]))
    nodes.append(oh.make_node("Mul", ["slot0", "bg0"], ["bg0o"]))
    nodes.append(oh.make_node("Add", ["shapeOut", "bg0o"], ["output"]))
    return _model(nodes, inits)


# --------------------------------------------------------------- dispatch ----

def candidates(examples):
    prs = [(np.array(e["input"], int), np.array(e["output"], int))
           for e in examples.get("train", []) + examples.get("test", [])]
    out = []
    if _is_task048(prs):
        out.append(("conn2_flood", _build_task048(conn8=True, iters=20)))
    if _is_task355(prs):
        out.append(("noisiest_region", _build_task355()))
    if _is_task183(prs):
        out.append(("quadrant_recolor", _build_task183()))
    if _is_task004(prs):
        out.append(("comp_shift_right", _build_task004(iters=24)))
    return out
