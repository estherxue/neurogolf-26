"""family_pcrk_5 — CRACK slice U[5::6] = tasks [54, 96, 143, 191, 255, 366].

Deep per-task analysis (see report). All six are the residual "hard reasoning"
tasks. Their exact transformation rules were derived, but each requires a
capability that a *static, origin-anchored, opset-10* graph cannot express
exactly while generalizing across the full arc-gen split:

  54  : draw a scaled cross/plus template through every data-dependent marker
        cell, spanning data-dependent rectangles -> variable count/positions.
  96  : collect all scattered shape fragments, overlay them centered and make
        the union dihedrally symmetric -> data-dependent assembly + size.
  143 : one shape is marked by a 5-frame (the template); recolor to 5 the OTHER
        shape that is an exact translate (same cells & orientation) of it
        -> cross-correlation against a *data-dependent* kernel.
  191 : a 5x5 "boxed" template of 1s + interior 4-dots; find the matching
        cluster of scattered 4-dots and draw the 1-box around it.
  255 : fill (with 3) the interior of the large SOLID-zero corridor region
        (a union of a tall trunk rectangle + wide arm rectangles) sitting in a
        field of ~50% random-zero noise; interior == corridor eroded by 1.
        Exact recovery needs GLOBAL maximal-rectangle boundaries (the corridor
        touches noise-zeros, so local erosion / fixed-size opening / component
        labelling all leak). Not expressible as fixed local ops.
  366 : split into canvas + legend panels; translate each legend shape so its
        distinctive coloured cell lands on the matching canvas marker, stamp
        -> per-shape data-dependent translation.

The 255 corridor-fill is the closest to tractable, so a genuine
erode-based ONNX builder is provided.  EVERY candidate is emitted ONLY after a
strict self-check confirms it is EXACT on train + test + ALL provided arc-gen
pairs (the grader's own gate + the held-out-generalization requirement).  None
of the derived static rules clears that bar, so this family currently emits
nothing rather than over-fit.  The scaffolding stays so any structurally
cleaner arc-gen instance would still be picked up automatically.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# ---------------------------------------------------------------------------
# task 255 : fill the eroded solid-zero corridor with colour 3
# ---------------------------------------------------------------------------
def _pred255(a: np.ndarray) -> np.ndarray:
    """erode8 (grid edges protected) of the colour-0 mask -> paint 3."""
    Z = (a == 0)
    Zp = np.pad(Z, 1, constant_values=True)
    fill = Z.copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            fill &= Zp[dy:dy + Z.shape[0], dx:dx + Z.shape[1]]
    out = a.copy()
    out[fill] = 3
    return out


def _build255():
    """output = input with the eroded colour-0 corridor recoloured to 3.

    fill = Z * (1 - maxpool_3x3( 1 - Z ))  with the 1-Z field zero-padded so the
    true grid border is not treated as a wall (edge-protected erosion).
    delta channels: -fill on ch0, +fill on ch3 -> output = input + delta.
    """
    one = oh.make_tensor("one", DATA_TYPE, [1, 1, 1, 1], [1.0])
    zeros8 = oh.make_tensor("z8", DATA_TYPE, [1, 8, 30, 30], [0.0] * (8 * 30 * 30))
    # slice channel 0 -> Z  [1,1,30,30]
    s0 = oh.make_tensor("s0", onnx.TensorProto.INT64, [1], [0])
    e1 = oh.make_tensor("e1", onnx.TensorProto.INT64, [1], [1])
    ax1 = oh.make_tensor("ax1", onnx.TensorProto.INT64, [1], [1])
    nodes = [
        oh.make_node("Slice", ["input", "s0", "e1", "ax1"], ["Z"]),
        oh.make_node("Sub", ["one", "Z"], ["inv"]),                 # 1 - Z (broadcast)
        oh.make_node("Pad", ["inv"], ["invp"], mode="constant", value=0.0,
                     pads=[0, 0, 1, 1, 0, 0, 1, 1]),
        oh.make_node("MaxPool", ["invp"], ["wall"], kernel_shape=[3, 3],
                     strides=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node("Sub", ["one", "wall"], ["notwall"]),          # 1 - wall
        oh.make_node("Mul", ["Z", "notwall"], ["fill"]),            # eroded corridor
        oh.make_node("Neg", ["fill"], ["nfill"]),
        # delta = concat[ -fill (ch0), +fill (ch3 built via zeros then add), zeros ]
        # channel layout: ch0=-fill, ch1=0, ch2=0, ch3=+fill, ch4..9=0
        oh.make_node("Concat", ["nfill", "fill"], ["ch03"], axis=1),  # [1,2,..]  (0,3 markers)
    ]
    # We need ch0=-fill, ch1,2=0, ch3=+fill, ch4..9=0 -> build explicitly by concat.
    z1 = oh.make_tensor("z1", DATA_TYPE, [1, 1, 30, 30], [0.0] * (30 * 30))
    z2 = oh.make_tensor("z2", DATA_TYPE, [1, 2, 30, 30], [0.0] * (2 * 30 * 30))
    z6 = oh.make_tensor("z6", DATA_TYPE, [1, 6, 30, 30], [0.0] * (6 * 30 * 30))
    nodes = nodes[:-1] + [
        oh.make_node("Concat", ["nfill", "z2", "fill", "z6"], ["delta"], axis=1),
        oh.make_node("Add", ["input", "delta"], ["output"]),
    ]
    return _model(nodes, [one, s0, e1, ax1, z1, z2, z6, zeros8])


def _exact(pred_fn, pairs):
    for a, b in pairs:
        p = pred_fn(a)
        if p.shape != b.shape or not (p == b).all():
            return False
    return True


def candidates(examples):
    train = examples.get("train", [])
    test = examples.get("test", [])
    gen = examples.get("arc-gen", [])
    pairs = [(np.array(e["input"]), np.array(e["output"]))
             for e in train + test + gen]
    if not pairs:
        return []

    out = []

    # ---- task 255 signature: 30x30, colour 3 appears only in output ----
    ins = np.array([p[0].shape for p in pairs])
    if (ins == 30).all():
        introduces_3 = all(((b == 3).any() and not (a == 3).any()) for a, b in pairs)
        if introduces_3:
            # STRICT self-check on train+test+all arc-gen before emitting.
            if _exact(_pred255, pairs):
                out.append(("corridor_fill3", _build255()))

    return out
