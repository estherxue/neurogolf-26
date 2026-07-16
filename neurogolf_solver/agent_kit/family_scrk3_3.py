"""family_scrk3_3 -- deep-crack wave for unsolved slice U[3::4].

Task 175 (transpose-symmetric inpaint).  The clean grid is symmetric under the
H<->W transpose; the input is that grid with rectangular blocks corrupted to
colour 0.  Reconstruct every colour-0 cell:

  1. res = where(cell==0, transpose(input), input)          # fill from mirror
  2. diagonal doubling fill (down-right, then up-left) for the residual colour-0
     cells that lie ON the main diagonal (transpose maps them to themselves).
     Those residual cells sit on a locally diagonal-constant spine, so copying
     the nearest filled cell along the main diagonal recovers them.

All ops are origin-safe under top-left padding: Transpose keeps the origin, and
the diagonal shifts are Pad+Slice windows anchored at (0,0).  Real colour-0 cells
carry channel-0==1 while padding is all-zero, so the fill mask (channel-0>0)
never touches padding.

The candidates() detector runs the exact numpy pipeline; it only emits when the
reconstruction is value-exact on every provided pair, so unrelated tasks are
skipped and the harness re-validates on train+test+arc-gen.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
_KS = [1, 2, 4, 8, 16]   # doubling shift radii (covers runs up to 31)


# --------------------------------------------------------------------------- #
# numpy reference (used for detection)                                        #
# --------------------------------------------------------------------------- #
def _shift_dr(g, k):
    o = np.zeros_like(g)
    if k < g.shape[0] and k < g.shape[1]:
        o[k:, k:] = g[:-k, :-k]
    return o


def _shift_ul(g, k):
    o = np.zeros_like(g)
    if k < g.shape[0] and k < g.shape[1]:
        o[:-k, :-k] = g[k:, k:]
    return o


def _reference(a):
    o = np.where(a == 0, a.T, a)
    for k in _KS:
        o = np.where(o == 0, _shift_dr(o, k), o)
    for k in _KS:
        o = np.where(o == 0, _shift_ul(o, k), o)
    return o


# --------------------------------------------------------------------------- #
# ONNX builder                                                                #
# --------------------------------------------------------------------------- #
def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "scrk3_3", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _build():
    nodes, inits = [], []
    zero = oh.make_tensor("zero", DATA_TYPE, [1], [0.0])
    inits.append(zero)

    # channel-0 slice constants (axis=1, [0:1])
    inits += [
        oh.make_tensor("c0_s", INT64, [1], [0]),
        oh.make_tensor("c0_e", INT64, [1], [1]),
        oh.make_tensor("c0_a", INT64, [1], [1]),
    ]

    def mask(src, tag):
        # cond = (src channel0) > 0   -> BOOL [1,1,30,30]
        nodes.append(oh.make_node("Slice",
                                   [src, "c0_s", "c0_e", "c0_a"], [f"ch0_{tag}"]))
        nodes.append(oh.make_node("Greater", [f"ch0_{tag}", "zero"],
                                   [f"cond_{tag}"]))
        return f"cond_{tag}"

    # ---- step 1: transpose fill -------------------------------------------
    nodes.append(oh.make_node("Transpose", ["input"], ["xT"], perm=[0, 1, 3, 2]))
    c = mask("input", "t")
    nodes.append(oh.make_node("Where", [c, "xT", "input"], ["res0"]))
    cur = "res0"

    # slice-to-30 window constants (H and W axes)
    def add_win(name, lo):
        inits.append(oh.make_tensor(f"{name}_s", INT64, [2], [lo, lo]))
        inits.append(oh.make_tensor(f"{name}_e", INT64, [2], [lo + HEIGHT, lo + WIDTH]))
        inits.append(oh.make_tensor(f"{name}_a", INT64, [2], [2, 3]))

    # ---- step 2: diagonal doubling fills ----------------------------------
    idx = 0
    # down-right shifts: pad top/left by k, then crop [0:30]
    add_win("win0", 0)
    for k in _KS:
        idx += 1
        tag = f"dr{idx}"
        nodes.append(oh.make_node(
            "Pad", [cur], [f"pad_{tag}"], mode="constant", value=0.0,
            pads=[0, 0, k, k, 0, 0, 0, 0]))
        nodes.append(oh.make_node(
            "Slice", [f"pad_{tag}", "win0_s", "win0_e", "win0_a"], [f"sh_{tag}"]))
        c = mask(cur, tag)
        nodes.append(oh.make_node("Where", [c, f"sh_{tag}", cur], [f"r_{tag}"]))
        cur = f"r_{tag}"

    # up-left shifts: pad bottom/right by k, then crop [k:k+30]
    for k in _KS:
        idx += 1
        tag = f"ul{idx}"
        add_win(f"win{tag}", k)
        nodes.append(oh.make_node(
            "Pad", [cur], [f"pad_{tag}"], mode="constant", value=0.0,
            pads=[0, 0, 0, 0, 0, 0, k, k]))
        nodes.append(oh.make_node(
            "Slice", [f"pad_{tag}", f"win{tag}_s", f"win{tag}_e", f"win{tag}_a"],
            [f"sh_{tag}"]))
        c = mask(cur, tag)
        nodes.append(oh.make_node("Where", [c, f"sh_{tag}", cur], [f"r_{tag}"]))
        cur = f"r_{tag}"

    nodes.append(oh.make_node("Identity", [cur], ["output"]))
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return
    # detector: full inpaint must be value-exact, and it must be non-trivial
    # (there must actually be colour-0 corruption that the fill repairs).
    changed = False
    for a, b in prs:
        if a.shape != b.shape or a.shape[0] != a.shape[1]:
            return  # transpose fill needs square grids
        if not np.array_equal(_reference(a), b):
            return
        if not np.array_equal(a, b):
            changed = True
    if not changed:
        return
    yield ("transpose_diag_inpaint", _build())
