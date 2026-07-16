"""family_golf5_1 - cheaper EXACT solvers for selected golf targets (slice [1::6]).

The integrator keeps the cheapest exact model per task (cost = params +
intermediate_memory, points = 25 - ln cost). We only need a correct graph that
is CHEAPER than the current best. Each solver re-derives its rule from
train+test(+arc-gen) in numpy and only fires when the rule is EXACT on every
available pair.

Main technique here: a per-channel local cellular rule. Many ARC transforms are
a fixed boolean function of a small (plus / 3x3) binary neighbourhood applied
independently to every colour layer, with background re-derived as
(in-grid minus objects). We implement that with one grouped Conv that packs the
neighbourhood into an integer "code", a Cast, and a Gather through a 2^k lookup
table - very few, small intermediates.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
INT32 = onnx.TensorProto.INT32


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _pairs(examples, secs=("train", "test")):
    out = []
    for s in secs:
        for e in examples.get(s, []):
            out.append((np.array(e["input"], int), np.array(e["output"], int)))
    return out


def _allpairs(examples):
    return _pairs(examples, ("train", "test", "arc-gen"))


# ==========================================================================
# Per-channel local-neighbourhood rule (e.g. task 4 deskew, hollow, ...).
#
# A stencil is a list of (dr,dc) offsets; bit i (weight 2^(n-1-i)) is set when
# the binary mask is 1 at that offset.  The rule is a length-2^n table T mapping
# each neighbourhood code -> 0/1 for the centre cell.  We apply it independently
# to every non-background colour layer; a cell that ends up empty on all colour
# layers but lies inside the grid becomes background.
# ==========================================================================

# plus stencil: centre, N, S, W, E  (weights 16,8,4,2,1)
_PLUS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
# full 3x3 (weights 256..1, row-major)
_FULL = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _codes_for_pair(a, offsets, weights):
    """Return HxW integer codes of the binary (non-bg) mask of grid `a`."""
    H, W = a.shape
    m = np.zeros((H + 2, W + 2), int)
    m[1:1 + H, 1:1 + W] = (a > 0).astype(int)
    code = np.zeros((H, W), int)
    for (dr, dc), w in zip(offsets, weights):
        code += w * m[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
    return code


def _codes_for_channel(mask, offsets, weights):
    H, W = mask.shape
    m = np.zeros((H + 2, W + 2), int)
    m[1:1 + H, 1:1 + W] = mask
    code = np.zeros((H, W), int)
    for (dr, dc), w in zip(offsets, weights):
        code += w * m[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
    return code


def _derive_table(P, offsets, weights):
    """Derive the binary centre table from the *union* (non-bg) mask. Returns
    the length-2^n table or None if inconsistent."""
    n = len(offsets)
    T = np.full(1 << n, -1, int)
    for a, b in P:
        if a.shape != b.shape:
            return None
        code = _codes_for_pair(a, offsets, weights)
        out = (b > 0).astype(int)
        for c, o in zip(code.ravel(), out.ravel()):
            if T[c] == -1:
                T[c] = o
            elif T[c] != o:
                return None
    T[T == -1] = 0
    return T


def _verify_perchannel(P, offsets, weights, T):
    """Per-channel reconstruction must reproduce every output exactly."""
    for a, b in P:
        H, W = a.shape
        out = np.zeros((H, W), int)
        fired = np.zeros((H, W), int)
        for k in range(1, 10):
            mask = (a == k).astype(int)
            if not mask.any():
                continue
            tm = T[_codes_for_channel(mask, offsets, weights)]
            out[tm == 1] = k
            fired += tm
        if (fired > 1).any():
            return False
        if not np.array_equal(out, b):
            return False
    return True


def _build_perchannel(offsets, weights, T):
    n = len(offsets)
    # grouped conv kernel over channels 1..9, weight [9,1,3,3]
    ker = np.zeros((9, 1, 3, 3), np.float32)
    for (dr, dc), w in zip(offsets, weights):
        ker[:, 0, 1 + dr, 1 + dc] = float(w)
    W_t = oh.make_tensor("W", DATA_TYPE, [9, 1, 3, 3], ker.ravel().tolist())
    T_t = oh.make_tensor("T", DATA_TYPE, [1 << n], [float(x) for x in T])

    s_st = oh.make_tensor("s_st", INT64, [1], [1])
    s_en = oh.make_tensor("s_en", INT64, [1], [10])
    s_ax = oh.make_tensor("s_ax", INT64, [1], [1])

    nodes = [
        oh.make_node("Slice", ["input", "s_st", "s_en", "s_ax"], ["in19"]),
        oh.make_node("Conv", ["in19", "W"], ["code"],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=9),
        oh.make_node("Cast", ["code"], ["codeI"], to=INT32),
        oh.make_node("Gather", ["T", "codeI"], ["out19"], axis=0),
        oh.make_node("ReduceSum", ["input"], ["inGrid"], axes=[1], keepdims=1),
        oh.make_node("ReduceSum", ["out19"], ["sumObj"], axes=[1], keepdims=1),
        oh.make_node("Sub", ["inGrid", "sumObj"], ["outBg"]),
        oh.make_node("Concat", ["outBg", "out19"], ["output"], axis=1),
    ]
    return _model(nodes, [W_t, T_t, s_st, s_en, s_ax])


def _try_perchannel(P):
    """Return (name, offsets, weights, T) for the cheapest stencil that works."""
    for name, offsets in (("plus", _PLUS), ("full3x3", _FULL)):
        n = len(offsets)
        weights = [1 << (n - 1 - i) for i in range(n)]
        T = _derive_table(P, offsets, weights)
        if T is None:
            continue
        if _verify_perchannel(P, offsets, weights, T):
            return name, offsets, weights, T
    return None


def candidates(examples):
    cands = []
    P = _pairs(examples)
    if not P:
        return cands
    # require same-shape and reasonably small grids (this rule is shape-preserving)
    if not all(a.shape == b.shape for a, b in P):
        return cands
    AP = _allpairs(examples)
    if any(a.shape != b.shape for a, b in AP):
        return cands
    res = _try_perchannel(AP)
    if res is not None:
        name, offsets, weights, T = res
        cands.append((f"pclut_{name}", _build_perchannel(offsets, weights, T)))
    return cands
