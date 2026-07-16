"""family_sgolf5_4 -- cheaper EXACT solvers for a fat incumbent in this golf
slice, using grid-agnostic single-channel Hillis-Steele DOUBLING propagation.
NO cropping: the canvas stays 30x30 from input to output.

Sub-solver DIAGBEAM (task 34):
  A single 2x2 block sits on empty background.  The block holds colour C in most
  cells and colour-2 marker(s) on one (or both, opposite) corner(s).  Each
  2-marker fires a width-3 diagonal beam of colour C OUTWARD through the corner
  it occupies, running to the real-grid edge; the marker is recoloured to C.

  A direction d fires iff some colour-2 cell has a block cell on its inward
  diagonal neighbour (uniquely identifying the corner it occupies).  The beam is
  a straight, obstacle-free directional cumulative-OR of the block footprint,
  which is EXACT under doubling.  Each doubling step is a SINGLE 2-tap Conv
  (self + shifted), whose running SUM has exactly the OR support (the grader only
  checks output>0).  Every field is single-channel [1,1,30,30]; the 10-channel
  answer is written straight into the free 'output' tensor by one Concat.

  Origin-anchoring: realmask = sum_c input_c is 1 on every in-grid cell and 0 on
  the zero-pad, so the beam is clipped to the real grid with no size crop.  The
  block colour C is routed dynamically via per-channel presence, so one static
  graph handles every example regardless of C, size or beam direction.

Detection re-checks the exact numpy rule on train+test+arc-gen; the harness is
the final EXACT judge (wrong guesses are rejected, never scored negative).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS

INT64 = onnx.TensorProto.INT64
_OFFS = [1, 2, 4, 8, 16]
_DIAG = [("UR", -1, 1), ("UL", -1, -1), ("DR", 1, 1), ("DL", 1, -1)]


# --------------------------------------------------------------------------- #
# numpy reference (mirrors the ONNX arithmetic exactly)                        #
# --------------------------------------------------------------------------- #
def _shift_np(m, dr, dc):
    """content moves by (dr,dc): out[i,j] = m[i-dr, j-dc], zero fill."""
    H, W = m.shape
    o = np.zeros_like(m)
    r0, r1 = max(dr, 0), H + min(dr, 0)
    c0, c1 = max(dc, 0), W + min(dc, 0)
    if r0 < r1 and c0 < c1:
        o[r0:r1, c0:c1] = m[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return o


def _apply_ref(a):
    """Exact numpy mirror of the ONNX graph (thresholded output as an int grid)."""
    H, W = a.shape
    cols = [c for c in np.unique(a) if c not in (0, 2)]
    if len(cols) != 1:
        return None
    C = int(cols[0])
    nb = (a != 0).astype(np.float64)
    m2 = (a == 2).astype(np.float64)
    band = np.zeros((H, W), np.float64)
    for _, dr, dc in _DIAG:
        inward = _shift_np(nb, dr, dc)        # nb[i-dr, j-dc]  (block cell inward of a corner)
        fires = float((m2 * inward).max()) if m2.size else 0.0
        seed = nb * fires
        acc = seed.copy()
        for s in _OFFS:
            acc = acc + _shift_np(acc, s * dr, s * dc)   # add-doubling (OR support)
        band = band + acc
    band = (band > 0).astype(np.int64)        # gate to real grid is implicit (nb in-grid)
    out = np.zeros((H, W), np.int64)
    out[band > 0] = C
    return out


# --------------------------------------------------------------------------- #
# ONNX graph builder                                                           #
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
        nm = self.nm("w")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(dims),
                                         np.asarray(vals, np.float32).ravel().tolist()))
        return nm

    def i64(self, vals):
        nm = self.nm("i")
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)], list(vals)))
        return nm

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _shift_conv(g, x, dy, dx):
    """out[i,j] = x[i-dy, j-dx] via a single 1-channel Conv (zero pad)."""
    R = max(abs(dy), abs(dx))
    K = 2 * R + 1
    W = np.zeros((1, 1, K, K), np.float32)
    W[0, 0, R - dy, R - dx] = 1.0
    wt = g.f([1, 1, K, K], W)
    return g.nd("Conv", [x, wt], kernel_shape=[K, K], pads=[R, R, R, R])


def _double_conv(g, x, dy, dx):
    """out = x + x[i-dy, j-dx] in ONE 2-tap Conv (running-sum doubling step)."""
    R = max(abs(dy), abs(dx))
    K = 2 * R + 1
    W = np.zeros((1, 1, K, K), np.float32)
    W[0, 0, R, R] = 1.0            # self
    W[0, 0, R - dy, R - dx] = 1.0  # shifted
    wt = g.f([1, 1, K, K], W)
    return g.nd("Conv", [x, wt], kernel_shape=[K, K], pads=[R, R, R, R])


def _model(g):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _build_beam():
    g = _G()
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)          # [1,1,30,30]
    ch0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])     # background
    ch2 = g.nd("Slice", ["input", g.i64([2]), g.i64([3]), g.i64([1])])     # markers
    nb = g.nd("Sub", [realmask, ch0])                                      # non-background

    bands = []
    for _, dr, dc in _DIAG:
        inward = _shift_conv(g, nb, dr, dc)         # nb[i-dr,j-dc]  (block cell inward)
        fmap = g.nd("Mul", [ch2, inward])
        fires = g.nd("ReduceMax", [fmap], axes=[2, 3], keepdims=1)         # [1,1,1,1]
        acc = g.nd("Mul", [nb, fires])                                     # gated seed
        for s in _OFFS:
            acc = _double_conv(g, acc, s * dr, s * dc)
        bands.append(acc)
    band = g.nd("Sum", bands)                                              # OR support
    band = g.nd("Mul", [band, realmask])                                   # clip beam to real grid
    ch0out = g.nd("Sub", [realmask, band])                                 # bg not in beam

    # colourise: build 10 single-channel planes, Concat straight into 'output'.
    presence = g.nd("ReduceMax", ["input"], axes=[2, 3], keepdims=1)       # [1,10,1,1]
    zero = g.nd("Sub", [ch0, ch0])                                         # [1,1,30,30] zeros
    planes = [None] * CHANNELS
    planes[0] = ch0out
    planes[2] = zero
    for c in range(1, CHANNELS):
        if c == 2:
            continue
        pc = g.nd("Slice", [presence, g.i64([c]), g.i64([c + 1]), g.i64([1])])  # [1,1,1,1]
        planes[c] = g.nd("Mul", [band, pc])                               # band where colour c present
    g.nd("Concat", planes, "output", axis=1)
    return _model(g)


# --------------------------------------------------------------------------- #
# detection / entry point                                                     #
# --------------------------------------------------------------------------- #
def _pairs(ex, splits):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _beam_ok(plist):
    for a, b in plist:
        o = _apply_ref(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(ex):
    tt = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not tt or not allp:
        return []
    if not all(a.shape == b.shape for a, b in allp):
        return []
    # our family: inputs over {0, 2, C} with a single extra colour C
    for a, b in allp:
        u = set(np.unique(a).tolist())
        if 2 not in u or len(u - {0, 2}) != 1:
            return []
    out = []
    if _beam_ok(tt) and _beam_ok(allp):
        try:
            m = _build_beam()
            onnx.checker.check_model(m, full_check=True)
            out.append(("sgolf5_4_diagbeam", m))
        except Exception:
            pass
    return out
