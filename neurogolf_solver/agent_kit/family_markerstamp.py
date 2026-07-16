"""MARKER-STAMP family (origin-anchored, opset 10, DATA-DEPENDENT positions).

Some ARC tasks place a FIXED little template/shape at every occurrence of a
single "marker" colour.  The markers sit at arbitrary, data-dependent positions,
but a *translation-equivariant* stamp can paint them ALL at once with ONE Conv
whose kernel IS the template -- no per-marker indexing, no Loop/NonZero.

Pipeline (size-independent, padding-safe)
----------------------------------------
1.  M = Slice the marker channel `mc`            -> [1,1,30,30]  (1 at each marker).
2.  stamp = Conv(M, K)                            -> [1,10,30,30].
    K is the template, FLIPPED and laid out so that a marker at (mr,mc) writes
    colour `col` at every offset (dy,dx) of the template:  K[col,0,R-dy,R-dx]=1,
    with symmetric pads [R,R,R,R] so the 30x30 shape is preserved and the stamp
    is centred on the marker.  Hence stamp[col] = (# markers placing `col` here).
3.  Clip to the real grid:  stamp *= realmask  (realmask = ReduceSum over the 10
    colour channels == 1 on real cells, 0 on the zero-padding).  This clips any
    template arm that would fall outside the (variable) grid -- exactly the
    boundary-clipping the true task does.
4.  cond = ReduceSum(stamp over channels) > 0    -> [1,1,30,30] bool.
    output = Where(cond, stamp, input):  stamped cells take the template colour
    (a one-hot column, channel 0 == background for "erase" templates), every
    other real cell keeps the input, padding stays zero.

The marker colour and the template {offset -> colour} are inferred from the
train/test/arc-gen pairs by mirroring this exact semantics, and a candidate is
emitted ONLY when it reproduces EVERY available pair, so wrong hypotheses are
dropped before the grader sees them.  Template colour 0 is a legal entry (it
erases the marker, e.g. via channel 0), and an offset whose colour already
matches the input is simply omitted (the base `input` keeps it).
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

MAX_R = 3            # largest template radius (kernel up to (2R+1)^2)


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
# ONNX builder                                                                 #
# --------------------------------------------------------------------------- #
def build_stamp(mc, T, R):
    """mc: marker colour channel (1..9).  T: {(dy,dx): colour}.  R: radius."""
    g = _G()
    K = 2 * R + 1

    # marker plane -> [1,1,30,30]
    M = g.nd("Slice", ["input", g.i64([mc]), g.i64([mc + 1]), g.i64([1])])

    # template kernel (flipped): stamp[col,r,c] = sum_markers T[(r-mr,c-mc)]==col
    Wk = np.zeros((CHANNELS, 1, K, K), np.float32)
    for (dy, dx), col in T.items():
        Wk[col, 0, R - dy, R - dx] = 1.0
    wt = g.f([CHANNELS, 1, K, K], Wk)
    stamp = g.nd("Conv", [M, wt], kernel_shape=[K, K], pads=[R, R, R, R])  # [1,10,30,30]

    # a stamp is "present" where any template channel fired; clip to the real grid
    # (realmask == 1 on real cells, 0 on the zero-padding) so arms that fall into
    # the padding are dropped -- the Where simply keeps `input` (== 0) there.
    csum = g.nd("ReduceSum", [stamp], axes=[1], keepdims=1)                # [1,1,30,30]
    half = g.f([1, 1, 1, 1], [0.5])
    present = g.nd("Greater", [csum, half])                               # bool
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)          # [1,1,30,30]
    real = g.nd("Greater", [realmask, half])                              # bool
    cond = g.nd("And", [present, real])                                  # bool [1,1,30,30]
    g.nd("Where", [cond, stamp, "input"], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy reference: mirrors the ONNX semantics EXACTLY                          #
# --------------------------------------------------------------------------- #
def _ref(a, mc, T):
    h, w = a.shape
    stamp = np.zeros((CHANNELS, h, w), np.int64)
    mrs, mcs = np.where(a == mc)
    for mr, ml in zip(mrs.tolist(), mcs.tolist()):
        for (dy, dx), col in T.items():
            r, c = mr + dy, ml + dx
            if 0 <= r < h and 0 <= c < w:
                stamp[col, r, c] += 1
    cond = stamp.sum(0) > 0
    nch = (stamp > 0).sum(0)
    if (nch[cond] != 1).any():           # not one-hot -> not expressible -> reject
        return None
    out = a.copy()
    if cond.any():
        ch = np.argmax(stamp, axis=0)
        out[cond] = ch[cond]
    return out


def _infer_T(prs, mc, R):
    """Infer {offset -> colour} from every cell that CHANGES within radius R of a
    marker; reject if an offset shows two different colours."""
    obs = {}
    for a, b in prs:
        h, w = a.shape
        mrs, mcs = np.where(a == mc)
        for mr, ml in zip(mrs.tolist(), mcs.tolist()):
            r0, r1 = max(0, mr - R), min(h, mr + R + 1)
            c0, c1 = max(0, ml - R), min(w, ml + R + 1)
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if b[r, c] != a[r, c]:
                        key = (r - mr, c - ml)
                        col = int(b[r, c])
                        if key in obs and obs[key] != col:
                            return None
                        obs[key] = col
    return obs or None


# --------------------------------------------------------------------------- #
# entry point                                                                  #
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


def _matches(prs, mc, T):
    for a, b in prs:
        o = _ref(a, mc, T)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if any(a.shape != b.shape for a, b in prs):       # stamping preserves grid size
        return []
    if all(np.array_equal(a, b) for a, b in prs):     # identity -> not our family
        return []

    # marker-colour candidates: non-background colours present in the inputs
    incolors = set()
    for a, _ in prs:
        incolors |= set(int(v) for v in np.unique(a).tolist())
    incolors.discard(0)

    out, seen = [], set()
    for mc in sorted(incolors):
        # light sparsity gate: a "marker" colour should not flood the grid
        if any((a == mc).sum() * 2 > a.size for a, _ in prs):
            continue
        for R in range(1, MAX_R + 1):
            T = _infer_T(prs, mc, R)
            if not T:
                continue
            if not _matches(prs, mc, T):
                continue
            name = f"stamp_mc{mc}_r{R}"
            if name in seen:
                break
            try:
                m = build_stamp(mc, T, R)
                onnx.checker.check_model(m, full_check=True)
            except Exception:
                break
            seen.add(name)
            out.append((name, m))
            break          # smallest validating radius wins for this marker colour
    return out
