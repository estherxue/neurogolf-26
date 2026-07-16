"""SORT / RANK NON-BACKGROUND COLOURS BY FREQUENCY into a small fixed layout.

The output is a tiny bar / column / row that lists the distinct non-background
colours of the input ORDERED BY THEIR CELL COUNT (ascending or descending), one
cell per colour.  Optionally the most-frequent (background-like, grid-filling)
colour(s) are skipped (a fixed `start` rank offset), so e.g. "drop the dominant
colour, then list the rest from most to least frequent".

Why this is expressible at opset 10 (origin-anchored)
-----------------------------------------------------
The one-hot tensor is zero-padded to 30x30 with the grid at (0,0), so a plain
``ReduceSum`` over the spatial axes is the exact per-colour cell count (padding is
all-zero and the background channel 0 is masked out).  From the counts we build a
tie-broken sort KEY per colour and derive each colour's 0-based RANK with pairwise
``Greater`` / ``Less`` comparisons (only 10 colours, so a 10x10 comparison matrix
is tiny).  For each output position ``p`` the colour whose rank equals
``start+p`` is selected with an integer ``Equal`` (rank is an exact integer), and
the resulting [K,10] one-hot selection is transposed / reshaped into the small
[1,10,K,1] (column) or [1,10,1,K] (row) grid and zero-padded back to
[1,10,30,30].  No [1,10,30,30] intermediate is materialised, so the cost stays
tiny (a handful of params + a 10x10 scratch tensor).

Tie-break
---------
Among equal counts the lower colour index ranks first.  This is baked identically
into both the numpy reference and the ONNX key (``count*100 + tiebreak``), so the
detector validates EXACTLY against every available pair (the grader's gate).  The
provided pairs never contain ties, which indicates the task generator guarantees
distinct counts; the chosen tie-break is nonetheless a valid permutation so the
graph degrades gracefully.

Detection infers (orientation, order, start, K) structurally from the train +
test + arc-gen pairs and only emits a candidate when it reproduces EVERY pair, so
wrong hypotheses are dropped before scoring.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
INT32 = onnx.TensorProto.INT32
F16 = onnx.TensorProto.FLOAT16
F = DATA_TYPE
_SCALE = 100.0          # count weight; counts (<=900) integer-differ by >=1 -> key diff >=100
_BIG = 1.0e7            # push absent / background colours to the far end of the order


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
                                         [float(v) for v in np.asarray(vals, np.float64).ravel()]))
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


def _nbg():
    return [0.0] + [1.0] * (CHANNELS - 1)


# --------------------------------------------------------------------------- #
# ONNX builder                                                                #
# --------------------------------------------------------------------------- #
def build_sortfreq(orient, order, start, K):
    """orient: 'col' -> output K x 1 ; 'row' -> output 1 x K.
    order : 'desc' (most frequent first) / 'asc' (least frequent first).
    start : rank offset (number of leading ranks skipped).
    K     : number of colours emitted (one per output cell)."""
    g = _G()

    # ---- per-colour count (channel 0 = real background cells) -------------- #
    count = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)          # [1,10,1,1]
    nbg = g.cf([1, CHANNELS, 1, 1], _nbg())
    one = g.cf([1, 1, 1, 1], [1.0])
    clip = g.nd("Clip", [count], min=0.0, max=1.0)                          # 1 if any cell
    present = g.nd("Mul", [clip, nbg])                                      # 1 for present non-bg
    notpres = g.nd("Sub", [one, present])

    # ---- tie-broken sort key ---------------------------------------------- #
    # desc: lower colour index first among ties -> larger key.  Absent/bg -> -BIG.
    # asc : lower colour index first among ties -> smaller key.  Absent/bg -> +BIG.
    # key = count*SCALE + tb -/+ (1-present)*BIG.  Background (huge count) is pushed
    # out of range by the BIG penalty, so only present non-bg colours are ranked.
    if order == "desc":
        tb = g.cf([1, CHANNELS, 1, 1], [float(CHANNELS - 1 - c) for c in range(CHANNELS)])
        raw = g.nd("Add", [g.nd("Mul", [count, g.cf([1, 1, 1, 1], [_SCALE])]), tb])
        penalty = g.nd("Mul", [notpres, g.cf([1, 1, 1, 1], [_BIG])])
        key = g.nd("Sub", [raw, penalty])                                 # absent/bg -> very low
    else:
        tb = g.cf([1, CHANNELS, 1, 1], [float(c) for c in range(CHANNELS)])
        raw = g.nd("Add", [g.nd("Mul", [count, g.cf([1, 1, 1, 1], [_SCALE])]), tb])
        bonus = g.nd("Mul", [notpres, g.cf([1, 1, 1, 1], [_BIG])])
        key = g.nd("Add", [raw, bonus])                                   # absent/bg -> very high

    keyT = g.nd("Transpose", [key], perm=[1, 0, 2, 3])                      # [10,1,1,1] (axis0=k)

    # ---- 0-based rank among present colours -------------------------------- #
    # The comparison matrix is the largest tensor; keep it (and its reduction) in
    # float16 -- the values are just {0,1} and a sum of <=9 -> exact in fp16.
    if order == "desc":
        cmp = g.nd("Greater", [keyT, key])      # [10,10,1,1] cmp[k,i] = key_k > key_i
    else:
        cmp = g.nd("Less", [keyT, key])         # cmp[k,i] = key_k < key_i
    rank = g.nd("ReduceSum", [g.nd("Cast", [cmp], to=F16)], axes=[0], keepdims=1)  # [1,10,1,1]
    rank_col = g.nd("Reshape", [g.nd("Cast", [rank], to=INT32),
                                g.ci([2], [CHANNELS, 1])])                 # [10,1] int32

    # ---- select the colour whose rank == start+p for each position p ------- #
    # eq[i,p] = (rank_i == start+p).  For a valid pair the #present colours equal
    # start+K, so absent colours (rank == #present) never hit a target -> each
    # column p is an exact one-hot over the 10 colour channels.
    targets = oh.make_tensor(g.nm("ci"), INT32, [1, K], [start + p for p in range(K)])
    g.inits.append(targets)
    eq = g.nd("Equal", [rank_col, targets.name])                           # [10,K] bool
    selKch = g.nd("Cast", [eq], to=F)                                       # [10,K] one-hot cols

    # ---- scatter into the small grid, then zero-pad to 30x30 --------------- #
    if orient == "col":
        small = g.nd("Reshape", [selKch, g.ci([4], [1, CHANNELS, K, 1])])   # [1,10,K,1]
        pads = [0, 0, 0, 0, 0, 0, HEIGHT - K, WIDTH - 1]
    else:
        small = g.nd("Reshape", [selKch, g.ci([4], [1, CHANNELS, 1, K])])   # [1,10,1,K]
        pads = [0, 0, 0, 0, 0, 0, HEIGHT - 1, WIDTH - K]
    g.nd("Pad", [small], "output", mode="constant", value=0.0, pads=pads)
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy reference (mirrors the ONNX semantics exactly)                        #
# --------------------------------------------------------------------------- #
def _counts(a):
    return {c: int((a == c).sum()) for c in range(1, CHANNELS) if (a == c).any()}


def _sorted_colors(a, order):
    cnt = _counts(a)
    cols = list(cnt.keys())
    if order == "desc":
        # higher count first; among ties lower colour index first
        cols.sort(key=lambda c: (cnt[c], -c), reverse=True)
    else:
        cols.sort(key=lambda c: (cnt[c], c))
    return cols, cnt


def _apply(a, orient, order, start, K):
    cols, cnt = _sorted_colors(a, order)
    if len(set(cnt.values())) != len(cnt):      # require distinct counts
        return None
    if start + K > len(cols):
        return None
    seq = cols[start:start + K]
    if orient == "col":
        return np.array(seq, int).reshape(K, 1)
    return np.array(seq, int).reshape(1, K)


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


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if all(np.array_equal(a, b) for a, b in prs):       # identity -> not our family
        return []

    # output must be a constant-size 1-D list (column K x 1 or row 1 x K)
    oshapes = {b.shape for _, b in prs}
    if len(oshapes) != 1:
        return []
    oh_, ow = next(iter(oshapes))
    if oh_ >= HEIGHT and ow >= WIDTH:
        return []
    if ow == 1 and oh_ >= 2:
        orient, K = "col", oh_
    elif oh_ == 1 and ow >= 2:
        orient, K = "row", ow
    else:
        return []
    if K > min(HEIGHT, WIDTH):
        return []

    # output colours must be non-bg colours present in the corresponding input
    for a, b in prs:
        ina = set(np.unique(a).tolist()) - {0}
        outb = set(np.unique(b).tolist())
        if 0 in outb or not outb.issubset(ina):
            return []

    out = []
    seen = set()
    # start offset is (#present non-bg colours) - K; require it constant
    starts = {len(_counts(a)) - K for a, _ in prs}
    cand_starts = sorted(s for s in starts if 0 <= s <= CHANNELS) if len(starts) == 1 else \
        [s for s in range(0, CHANNELS - K + 1)]

    for order in ("desc", "asc"):
        for start in cand_starts:
            ok = True
            for a, b in prs:
                r = _apply(a, orient, order, start, K)
                if r is None or r.shape != b.shape or not np.array_equal(r, b):
                    ok = False
                    break
            if not ok:
                continue
            key = (orient, order, start, K)
            if key in seen:
                continue
            try:
                m = build_sortfreq(orient, order, start, K)
                onnx.checker.check_model(m, full_check=True)
            except Exception:
                continue
            seen.add(key)
            out.append((f"sortfreq_{orient}_{order}_s{start}_K{K}", m))

    return out
