"""Fixed-position crop / subgrid extraction family (origin-anchored).

Output is a static sub-rectangle of the input anchored at the top-left:
    output = input[0:h_end:sh, 0:w_end:sw]
for fixed (h_end, w_end, sh, sw) inferred from the train/test pairs. Realized as a
static Slice (top-left, so no shift needed) followed by Pad back to 30x30 when the
result is smaller than the frame. Covers contiguous top-left crops, halving
(quadrant/left-half/top-half) when the absolute dims are constant, and
origin-anchored fixed-stride selections. Content-dependent bounding-box crops are
NOT static-expressible and are skipped.
"""
import onnx
from onnx import helper as oh
import numpy as np
from builders import _model
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64


def _pairs(ex):
    return [(np.array(e["input"]), np.array(e["output"]))
            for e in ex.get("train", []) + ex.get("test", [])]


def _crop_model(h_end, w_end, sh, sw):
    """Static Slice input[0:h_end:sh, 0:w_end:sw] then Pad to 30x30 (top-left)."""
    sliced_h = len(range(0, h_end, sh))
    sliced_w = len(range(0, w_end, sw))
    s = oh.make_tensor("c_starts", INT64, [2], [0, 0])
    e = oh.make_tensor("c_ends", INT64, [2], [h_end, w_end])
    a = oh.make_tensor("c_axes", INT64, [2], [2, 3])
    inits = [s, e, a]
    if sh == 1 and sw == 1:
        sl = oh.make_node("Slice", ["input", "c_starts", "c_ends", "c_axes"], ["small"])
    else:
        st = oh.make_tensor("c_steps", INT64, [2], [sh, sw])
        inits.append(st)
        sl = oh.make_node("Slice",
                          ["input", "c_starts", "c_ends", "c_axes", "c_steps"], ["small"])
    pad = oh.make_node("Pad", ["small"], ["output"], mode="constant", value=0.0,
                       pads=[0, 0, 0, 0, 0, 0, HEIGHT - sliced_h, WIDTH - sliced_w])
    return _model([sl, pad], inits)


def _matches(prs, h_end, w_end, sh, sw):
    for a, b in prs:
        if a.shape[0] < h_end or a.shape[1] < w_end:
            return False
        sub = a[0:h_end:sh, 0:w_end:sw]
        if sub.shape != b.shape or not np.array_equal(sub, b):
            return False
    return True


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    outs = [b for _, b in prs]
    # static output shape required: all outputs must share one shape
    oshape = outs[0].shape
    if not all(o.shape == oshape for o in outs):
        return []
    h0, w0 = oshape
    if h0 == 0 or w0 == 0 or h0 > HEIGHT or w0 > WIDTH:
        return []

    out = []

    # 1) contiguous fixed top-left crop: output = input[0:h0, 0:w0]
    if (h0, w0) != (HEIGHT, WIDTH) and _matches(prs, h0, w0, 1, 1):
        # skip pure identity (h0,w0 == input shape for all) -> still a valid crop but
        # equivalent to identity; only meaningful when it actually trims something.
        if any(a.shape != (h0, w0) for a, _ in prs):
            out.append((f"crop_{h0}x{w0}", _crop_model(h0, w0, 1, 1)))

    # 2) origin-anchored fixed-stride crop: output = input[0:h_end:sh, 0:w_end:sw]
    #    output size is (ceil(h_end/sh), ceil(w_end/sw)) == (h0, w0).
    for sh in range(1, 6):
        for sw in range(1, 6):
            if sh == 1 and sw == 1:
                continue
            # need h_end, w_end such that len(range(0,h_end,sh)) == h0
            h_end = (h0 - 1) * sh + 1
            w_end = (w0 - 1) * sw + 1
            if h_end > HEIGHT or w_end > WIDTH:
                continue
            # allow the slice end to be anywhere in (h_end .. h_end+sh-1]; canonical h_end ok
            if _matches(prs, h_end, w_end, sh, sw):
                out.append((f"stride_{sh}x{sw}_{h0}x{w0}",
                            _crop_model(h_end, w_end, sh, sw)))

    return out
