"""family_golf5_4 — cheaper EXACT solvers for GOLF targets (slice [4::6]).

Implemented rule families (all origin-anchored, opset-10):

  recolorspan(dir): per non-bg color channel, fill the per-row (or per-col)
  span between its leftmost/rightmost (top/bottom) pixel with that same color.
  Realized with two triangular [30,30] MatMuls (cumulative-OR left & right),
  a product (AND), then channel-0 = background minus covered cells.

Detection is structural + EXACT-validated on train+test+arc-gen at the 30x30
one-hot level (mirrors the grader's (output>0) check), so wrong guesses never
get emitted.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh
from onnx import numpy_helper as nph

from builders import _model
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
FP16 = onnx.TensorProto.FLOAT16

# ---------------------------------------------------------------- numpy refs
def _oh(grid):
    g = np.asarray(grid)
    H, W = g.shape
    a = np.zeros((CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            a[g[r, c], r, c] = 1.0
    return a


def _rowspan(mask):
    H, W = mask.shape
    out = np.zeros_like(mask)
    for r in range(H):
        cols = np.where(mask[r])[0]
        if len(cols):
            out[r, cols.min():cols.max() + 1] = True
    return out


def _colspan(mask):
    return _rowspan(mask.T).T


def _ref_recolorspan(oh_in, direction):
    """oh_in [10,30,30] one-hot -> predicted one-hot via per-color span fill."""
    out = np.zeros_like(oh_in)
    # background handled at end
    covered = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for c in range(1, CHANNELS):
        m = oh_in[c] > 0
        if not m.any():
            continue
        if direction == "row":
            s = _rowspan(m)
        elif direction == "col":
            s = _colspan(m)
        else:
            s = _rowspan(m) | _colspan(m)
        out[c][s] = 1.0
        covered |= s
    bg = (oh_in[0] > 0) & (~covered)
    out[0][bg] = 1.0
    return out


# ---------------------------------------------------------------- ONNX build
def _tri(name, kind, dtype=np.float16):
    """[30,30] triangular initializer.
    kind 'le' : T[k,c]=1 if k<=c   ;  kind 'ge' : T[k,c]=1 if k>=c
    """
    n = HEIGHT
    vals = np.zeros((n, n), dtype=dtype)
    for k in range(n):
        for c in range(n):
            if (kind == "le" and k <= c) or (kind == "ge" and k >= c):
                vals[k, c] = 1.0
    return nph.from_array(vals, name=name)


def _slice_inits(prefix, start, end, axis):
    return [
        oh.make_tensor(f"{prefix}_s", INT64, [1], [start]),
        oh.make_tensor(f"{prefix}_e", INT64, [1], [end]),
        oh.make_tensor(f"{prefix}_a", INT64, [1], [axis]),
    ]


def _model_out16(nodes, inits):
    """Like builders._model but the single output is declared FLOAT16.
    (The grader compares (output>0) so the float width is irrelevant to
    correctness, and a fp16 output skips a final Cast + its intermediate.)"""
    from ng_utils_shim import IR_VERSION, OPSET_IMPORTS
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", FP16, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def build_recolorspan(direction, fp16=True, out16=False):
    """Per-color span fill. direction in {'row','col'}.

    fp16=True does the cumulative MatMuls / product in float16 (counts only need
    a >0 test, so the precision loss is harmless), halving the cost of the four
    [1,9,30,30] working tensors.  out16=True additionally emits a float16 output
    (no final Cast) — cheapest.  out16 requires fp16.
    """
    assert not out16 or fp16
    dt = np.float16 if fp16 else np.float32
    inits = []
    nodes = []
    if fp16:
        nodes.append(oh.make_node("Cast", ["input"], ["X16"], to=FP16))
        src = "X16"
    else:
        src = "input"
    # Xc = src[:, 1:10]
    inits += _slice_inits("xc", 1, CHANNELS, 1)
    nodes.append(oh.make_node("Slice", [src, "xc_s", "xc_e", "xc_a"], ["Xc"]))

    if direction == "row":
        L = _tri("Lm", "le", dt)   # Xc @ L : sum_{k<=c}
        R = _tri("Rm", "ge", dt)   # Xc @ R : sum_{k>=c}
        inits.extend([L, R])
        nodes.append(oh.make_node("MatMul", ["Xc", "Lm"], ["left"]))
        nodes.append(oh.make_node("MatMul", ["Xc", "Rm"], ["right"]))
    else:  # col: cumulative over rows -> A @ Xc
        L = _tri("Lm", "ge", dt)   # Lm[r,k]=1 if k<=r  -> sum_{k<=r}
        R = _tri("Rm", "le", dt)   # Rm[r,k]=1 if k>=r  -> sum_{k>=r}
        inits.extend([L, R])
        nodes.append(oh.make_node("MatMul", ["Lm", "Xc"], ["left"]))
        nodes.append(oh.make_node("MatMul", ["Rm", "Xc"], ["right"]))
    nodes.append(oh.make_node("Mul", ["left", "right"], ["S"]))

    nodes.append(oh.make_node("ReduceSum", ["S"], ["covered"], axes=[1], keepdims=1))
    inits += _slice_inits("in0", 0, 1, 1)
    nodes.append(oh.make_node("Slice", [src, "in0_s", "in0_e", "in0_a"], ["in0"]))
    nodes.append(oh.make_node("Sub", ["in0", "covered"], ["out0"]))

    if out16:
        nodes.append(oh.make_node("Concat", ["out0", "S"], ["output"], axis=1))
        return _model_out16(nodes, inits)
    if fp16:
        nodes.append(oh.make_node("Concat", ["out0", "S"], ["pre"], axis=1))
        nodes.append(oh.make_node("Cast", ["pre"], ["output"], to=DATA_TYPE))
        return _model(nodes, inits)
    nodes.append(oh.make_node("Concat", ["out0", "S"], ["output"], axis=1))
    return _model(nodes, inits)


# ---------------------------------------------------------------- driver
TARGETS = {365, 343, 284, 131, 253, 345, 106, 381, 242, 383, 198, 154, 115,
           226, 68, 177, 114, 156, 251, 330, 41, 33, 384, 310, 199, 122, 39, 346}


def _pairs(examples):
    prs = []
    for k in ("train", "test", "arc-gen"):
        for e in examples.get(k, []) or []:
            i = np.asarray(e["input"])
            o = np.asarray(e["output"])
            if max(i.shape) > 30 or max(o.shape) > 30:
                continue
            prs.append((i, o))
    return prs


def _exact(prs, direction):
    if not prs:
        return False
    for i, o in prs:
        if i.shape != o.shape:
            return False
        pred = _ref_recolorspan(_oh(i), direction)
        if not ((pred > 0) == (_oh(o) > 0)).all():
            return False
    return True


def candidates(examples):
    prs = _pairs(examples)
    out = []
    for direction in ("row", "col"):
        if _exact(prs, direction):
            # cheapest first: fp16 math + fp16 output, then fp16 math + f32 output,
            # then pure f32 (all exact; harness keeps the cheapest that validates).
            variants = [
                (f"recolorspan_{direction}_ho", dict(fp16=True, out16=True)),
                (f"recolorspan_{direction}_h", dict(fp16=True, out16=False)),
                (f"recolorspan_{direction}", dict(fp16=False, out16=False)),
            ]
            for tag, kw in variants:
                try:
                    out.append((tag, build_recolorspan(direction, **kw)))
                except Exception:
                    pass
            break  # row preferred; col has identical cost
    return out
