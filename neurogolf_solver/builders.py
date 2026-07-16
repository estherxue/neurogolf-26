"""Opset-10 ONNX graph builders. Each returns an onnx.ModelProto with a single
FLOAT[1,10,30,30] input named "input" and output named "output", matching the
NeuroGolf I/O contract. Builders are intentionally minimal (low params + few/no
intermediate tensors) because cost = params + intermediate_memory.

Cost intuition (verified against neurogolf_utils):
  * single node input->output, no initializers  -> params 0, memory 0 -> 25.0 pts
  * Gather channel-permute (10-int index init)   -> params 10        -> ~22.7 pts
  * 1x1 Conv recolor ([10,10,1,1])               -> params 100       -> ~20.4 pts
  * any extra [1,10,30,30] intermediate (f32)    -> +36000 bytes mem
"""
from __future__ import annotations

import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
_NEG = -(1 << 31)  # large negative sentinel for full-axis reverse Slice


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# ---- pointwise (position-safe under top-left padding) -----------------------

def identity():
    """output == input. 0 params, 0 memory -> 25 pts."""
    return _model([oh.make_node("Identity", ["input"], ["output"])])


def recolor_gather(src_for_out):
    """Channel permutation/selection: output[:, j] = input[:, src_for_out[j]].
    `src_for_out` is a length-10 list. Used for bijective color maps (cheap, 10 params)."""
    assert len(src_for_out) == CHANNELS
    idx = oh.make_tensor("idx", INT64, [CHANNELS], list(src_for_out))
    node = oh.make_node("Gather", ["input", "idx"], ["output"], axis=1)
    return _model([node], [idx])


def recolor_conv(color_map):
    """General per-pixel color map via 1x1 Conv. `color_map[i]` = output color for input
    color i. weight[o,i,0,0] = 1 if color_map[i]==o else 0. 100 params."""
    assert len(color_map) == CHANNELS
    weights = [0.0] * (CHANNELS * CHANNELS)  # layout [O, I, 1, 1]
    for i, o in enumerate(color_map):
        weights[o * CHANNELS + i] = 1.0
    w = oh.make_tensor("W", DATA_TYPE, [CHANNELS, CHANNELS, 1, 1], weights)
    node = oh.make_node("Conv", ["input", "W"], ["output"], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return _model([node], [w])


# ---- geometric (often position-UNSAFE for <30 grids; validator decides) -----

def transpose_hw():
    """Swap H and W. Position-safe (keeps top-left origin). 0 params -> 25 pts."""
    return _model([oh.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])])


def _reverse_slice(axes):
    """Slice that reverses the given spatial axes fully (steps=-1)."""
    n = len(axes)
    starts = oh.make_tensor("s_starts", INT64, [n], [ (HEIGHT if a == 2 else WIDTH) - 1 for a in axes])
    ends = oh.make_tensor("s_ends", INT64, [n], [_NEG] * n)
    ax = oh.make_tensor("s_axes", INT64, [n], list(axes))
    steps = oh.make_tensor("s_steps", INT64, [n], [-1] * n)
    node = oh.make_node("Slice", ["input", "s_starts", "s_ends", "s_axes", "s_steps"], ["output"])
    return _model([node], [starts, ends, ax, steps])


def flip_w():
    return _reverse_slice([3])


def flip_h():
    return _reverse_slice([2])


def rot180():
    return _reverse_slice([2, 3])


def translate(dy, dx):
    """Shift content by (dy, dx) with zero fill, keeping a 30x30 window.
    Pad (opset-10 attribute pads) then Slice back to 30x30. One intermediate tensor."""
    # pads format for Pad-2: [b0,c0,h0,w0, b1,c1,h1,w1] (begin..., end...)
    h0, w0 = max(dy, 0), max(dx, 0)
    h1, w1 = max(-dy, 0), max(-dx, 0)
    pad = oh.make_node("Pad", ["input"], ["padded"], mode="constant", value=0.0,
                       pads=[0, 0, h0, w0, 0, 0, h1, w1])
    # After padding, crop the 30x30 window anchored to keep original top-left mapping.
    starts = oh.make_tensor("c_starts", INT64, [2], [h1, w1])
    ends = oh.make_tensor("c_ends", INT64, [2], [h1 + HEIGHT, w1 + WIDTH])
    ax = oh.make_tensor("c_axes", INT64, [2], [2, 3])
    crop = oh.make_node("Slice", ["padded", "c_starts", "c_ends", "c_axes"], ["output"])
    return _model([pad, crop], [starts, ends, ax])


def upscale(k):
    """Pixel/nearest upscale by integer k (output[r,c]=input[r//k,c//k]), kept anchored at
    the top-left: Resize(nearest, k) then crop to 30x30. Position-safe."""
    scales = oh.make_tensor("scales", DATA_TYPE, [4], [1.0, 1.0, float(k), float(k)])
    rz = oh.make_node("Resize", ["input", "scales"], ["up"], mode="nearest")
    s = oh.make_tensor("us", INT64, [2], [0, 0])
    e = oh.make_tensor("ue", INT64, [2], [HEIGHT, WIDTH])
    a = oh.make_tensor("ua", INT64, [2], [2, 3])
    crop = oh.make_node("Slice", ["up", "us", "ue", "ua"], ["output"])
    return _model([rz, crop], [scales, s, e, a])


def downscale(k):
    """Block-subsample by integer k (output[r,c]=input[k*r,k*c]), anchored top-left:
    strided Slice then Pad back to 30x30. Position-safe."""
    sz_h = len(range(0, HEIGHT, k))
    sz_w = len(range(0, WIDTH, k))
    s = oh.make_tensor("ds", INT64, [2], [0, 0])
    e = oh.make_tensor("de", INT64, [2], [HEIGHT, WIDTH])
    a = oh.make_tensor("da", INT64, [2], [2, 3])
    st = oh.make_tensor("dst", INT64, [2], [k, k])
    sl = oh.make_node("Slice", ["input", "ds", "de", "da", "dst"], ["small"])
    pad = oh.make_node("Pad", ["small"], ["output"], mode="constant", value=0.0,
                       pads=[0, 0, 0, 0, 0, 0, HEIGHT - sz_h, WIDTH - sz_w])
    return _model([sl, pad], [s, e, a, st])


def constant(out_onehot):
    """Emit a fixed output grid (for tasks whose output never varies). out_onehot is a
    [1,10,30,30] float array."""
    flat = out_onehot.astype(float).ravel().tolist()
    c = oh.make_tensor("C", DATA_TYPE, list(GRID_SHAPE), flat)
    node = oh.make_node("Constant", [], ["output"], value=c)
    return _model([node])


def check(model):
    """Run the official ONNX checker; raise on malformed graph."""
    onnx.checker.check_model(model, full_check=True)
    return model
