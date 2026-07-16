"""family_golf8_4 -- aggressive re-golf of low-scoring slice [4::6].

Currently targets:
  * task354 (floodrecolor_s5): markers sit in ROW 0; each solid block of color 5
    is recolored to the marker whose column intersects the block. We derive this
    with a CHEAP SCALAR pipeline (every working tensor is [1,1,30,30]):
      colorval = 1x1 Conv over the one-hot input, mapping color c -> its index but
                 dropping colors 0 and 5  (so only the row-0 markers survive).
      beam     = broadcast row 0 of colorval down every row (Tile).
      seed     = beam * mask5   (marker colour planted on the 5-cells it sits over).
      flood    = horizontal MaxPool(1x3) dilation clipped back to mask5, repeated a
                 few times -> spreads the marker colour across each solid block.
      finalval = colorval(markers) + flood(shapes)   (disjoint supports).
    Only the final scalar->one-hot expansion touches [1,10,30,30] tensors.

The rule is verified EXACTLY on train+test+arc-gen before we emit anything.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, ng

INT64 = onnx.TensorProto.INT64


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _pairs(examples):
    out = []
    for sub in ("train", "test", "arc-gen"):
        for e in examples.get(sub, []):
            b = ng.convert_to_numpy(e)
            if b:
                out.append((b["input"][0], b["output"][0]))  # [10,30,30] each
    return out


# --------------------------------------------------------------------------- #
# task354: flood-recolor of solid color-5 blocks by their row-0 column marker  #
# --------------------------------------------------------------------------- #

_IDX_KEEP = [0, 1, 2, 3, 4, 0, 6, 7, 8, 9]  # colour index, but 0 and 5 -> 0


def _t354_reference(inp, steps):
    """Numpy emulation of the exact ONNX pipeline (colour-index domain)."""
    col = np.zeros((30, 30), np.float32)
    for c in range(10):
        col += _IDX_KEEP[c] * inp[c]
    beam = np.tile(col[0:1, :], (30, 1))
    m5 = inp[5]
    cur = beam * m5
    for _ in range(steps):
        # 1x3 max-pool with -inf pad (values are >=0 so 0-pad is equivalent here
        # because we immediately re-multiply by the 0/1 mask)
        pad = np.pad(cur, ((0, 0), (1, 1)), constant_values=0.0)
        pooled = np.maximum(np.maximum(pad[:, :-2], pad[:, 1:-1]), pad[:, 2:])
        cur = pooled * m5
    finalval = col + cur
    gm = inp.sum(0)                          # 1 inside grid, 0 in padding
    idx = (finalval + (1.0 - gm) * 10.0).astype(np.int64)   # padding -> row 10
    eye2 = np.zeros((11, 10), np.float32)
    eye2[:10, :10] = np.eye(10, dtype=np.float32)           # row 10 = all zeros
    gathered = eye2[idx]                     # [30,30,10]
    out = np.transpose(gathered, (2, 0, 1))  # [10,30,30]
    return out


def _t354_matches(pairs, steps):
    for inp, outp in pairs:
        pred = _t354_reference(inp, steps)
        if not np.array_equal((pred > 0), (outp > 0)):
            return False
    return True


def _build_t354(steps):
    nodes = []
    inits = []
    # colorval = Conv1x1(input) with weights = _IDX_KEEP  -> [1,1,30,30]
    wcol = oh.make_tensor("Wcol", DATA_TYPE, [1, 10, 1, 1],
                          np.asarray(_IDX_KEEP, np.float32).tolist())
    inits.append(wcol)
    nodes.append(oh.make_node("Conv", ["input", "Wcol"], ["colorval"],
                              kernel_shape=[1, 1], pads=[0, 0, 0, 0]))
    # row0 slice then tile down
    r0s = oh.make_tensor("r0s", INT64, [1], [0])
    r0e = oh.make_tensor("r0e", INT64, [1], [1])
    r0a = oh.make_tensor("r0a", INT64, [1], [2])
    inits += [r0s, r0e, r0a]
    nodes.append(oh.make_node("Slice", ["colorval", "r0s", "r0e", "r0a"], ["row0"]))
    reps = oh.make_tensor("reps", INT64, [4], [1, 1, 30, 1])
    inits.append(reps)
    nodes.append(oh.make_node("Tile", ["row0", "reps"], ["beam"]))
    # mask5 = channel 5 of input
    m5s = oh.make_tensor("m5s", INT64, [1], [5])
    m5e = oh.make_tensor("m5e", INT64, [1], [6])
    m5a = oh.make_tensor("m5a", INT64, [1], [1])
    inits += [m5s, m5e, m5a]
    nodes.append(oh.make_node("Slice", ["input", "m5s", "m5e", "m5a"], ["mask5"]))
    # seed
    nodes.append(oh.make_node("Mul", ["beam", "mask5"], ["cur0"]))
    # horizontal flood
    cur = "cur0"
    for k in range(steps):
        pooled = f"pool{k}"
        nxt = f"cur{k+1}"
        nodes.append(oh.make_node("MaxPool", [cur], [pooled],
                                  kernel_shape=[1, 3], pads=[0, 1, 0, 1],
                                  strides=[1, 1]))
        nodes.append(oh.make_node("Mul", [pooled, "mask5"], [nxt]))
        cur = nxt
    # finalval = colorval(markers) + flooded(shapes)
    nodes.append(oh.make_node("Add", ["colorval", cur], ["finalval"]))
    # grid mask (1 inside grid, 0 padding); route padding cells to identity row 10
    nodes.append(oh.make_node("ReduceSum", ["input"], ["gm"], axes=[1], keepdims=1))
    one = oh.make_tensor("one", DATA_TYPE, [1], [1.0])
    ten = oh.make_tensor("ten", DATA_TYPE, [1], [10.0])
    inits += [one, ten]
    nodes.append(oh.make_node("Sub", ["one", "gm"], ["notgrid"]))
    nodes.append(oh.make_node("Mul", ["notgrid", "ten"], ["padoff"]))
    nodes.append(oh.make_node("Add", ["finalval", "padoff"], ["idxf"]))
    nodes.append(oh.make_node("Squeeze", ["idxf"], ["fv"], axes=[1]))
    nodes.append(oh.make_node("Cast", ["fv"], ["idx"], to=INT64))
    # scalar index -> one-hot via Gather from an 11x10 identity (row 10 = zeros)
    eye2 = np.zeros((11, 10), np.float32)
    eye2[:10, :10] = np.eye(10, dtype=np.float32)
    eye = oh.make_tensor("eye", DATA_TYPE, [11, 10], eye2.ravel().tolist())
    inits.append(eye)
    nodes.append(oh.make_node("Gather", ["eye", "idx"], ["gathered"], axis=0))
    # gathered: [1,30,30,10] -> output [1,10,30,30]
    nodes.append(oh.make_node("Transpose", ["gathered"], ["output"], perm=[0, 3, 1, 2]))
    return _model(nodes, inits)


def _fingerprint_t354(pairs):
    """Cheap structural gate so we only fire on the intended family."""
    if not pairs:
        return False
    for inp, outp in pairs:
        if inp.shape != outp.shape:
            return False
        # colour 5 present in input, absent in output (fully recoloured)
        if inp[5].sum() == 0:
            return False
        if outp[5].sum() != 0:
            return False
        # markers live only in row 0 (non-0, non-5 colours)
        marker = inp.copy()
        marker[0] = 0
        marker[5] = 0
        if marker[:, 1:, :].sum() != 0:
            return False
    return True


def candidates(examples):
    pairs = _pairs(examples)
    out = []
    if _fingerprint_t354(pairs):
        # find the minimal flood depth that is exact, then keep a safety buffer for
        # the held-out set (extra steps are idempotent once blocks are filled).
        for n in range(1, 13):
            if _t354_matches(pairs, n):
                steps = min(n + 2, 14)
                out.append((f"t354_flood{steps}", _build_t354(steps)))
                break
    return out
