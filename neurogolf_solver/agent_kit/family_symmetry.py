"""Symmetry / overlay family.

Output = input combined (logical OR) with transformed copies of itself, kept
ORIGIN-ANCHORED and SAME SHAPE as the input.

The OR of one-hot tensors is realised with onnx ``Max`` over the colour
channels (a channel is >0 iff some copy carries that colour at the cell). The
background channel (colour 0) needs the *opposite* logic: a cell is background
only where *every* copy is background, so channel 0 is the AND (product) of the
copies' channel-0 planes. Doing Max on all 10 channels naively would light up
channel 0 wherever any single copy is background, which breaks the one-hot
output, hence the explicit channel-0 fix-up.

Origin safety (see CONTEXT.md padding gotcha): only ``id`` (input itself) and
``T`` (Transpose, perm [0,1,3,2]) keep content anchored at the top-left under
30x30 zero padding. Flips / rotations push content to the far edge for grids
smaller than 30, so they are emitted only as *attempts* -- the harness rejects
the ones that do not generalise.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model, transpose_hw
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
_NEG = -(1 << 31)


# --------------------------------------------------------------------------- #
# numpy reference of the one-hot overlay (mirrors the ONNX graph exactly)
# --------------------------------------------------------------------------- #

def _transform(a, key):
    if key == "id":
        return a
    if key == "T":
        return a.T
    if key == "fliplr":
        return a[:, ::-1]
    if key == "flipud":
        return a[::-1, :]
    if key == "rot180":
        return a[::-1, ::-1]
    raise ValueError(key)


def _oh_overlay(a, keys):
    """Return (out_grid, valid) for the one-hot OR of the transformed copies.

    ``valid`` is False if at some cell two copies disagree on a non-zero colour
    (that cell would have >1 channel set -> not a valid one-hot output)."""
    copies = []
    for k in keys:
        c = _transform(a, k)
        if c.shape != a.shape:
            return None, False
        copies.append(np.asarray(c))
    arr = np.stack(copies, axis=0)            # [K, H, W]
    anynz = (arr != 0).any(axis=0)
    mxcol = arr.max(axis=0)
    masked = np.where(arr != 0, arr, 99)
    mncol = masked.min(axis=0)
    valid = bool(((~anynz) | (mxcol == mncol)).all())
    out = np.where(anynz, mxcol, 0)
    return out, valid


def _matches(prs, keys):
    for a, b in prs:
        if a.shape != b.shape:
            return False
        out, valid = _oh_overlay(a, keys)
        if not valid or out is None or not np.array_equal(out, b):
            return False
    return True


# --------------------------------------------------------------------------- #
# ONNX builders
# --------------------------------------------------------------------------- #

def _slice(inp, out, start, end, axis, steps=None, tag=""):
    n = oh.make_tensor(f"{tag}_s", INT64, [1], [start])
    e = oh.make_tensor(f"{tag}_e", INT64, [1], [end])
    ax = oh.make_tensor(f"{tag}_a", INT64, [1], [axis])
    inits = [n, e, ax]
    names = [inp, f"{tag}_s", f"{tag}_e", f"{tag}_a"]
    if steps is not None:
        st = oh.make_tensor(f"{tag}_st", INT64, [1], [steps])
        inits.append(st)
        names.append(f"{tag}_st")
    return oh.make_node("Slice", names, [out]), inits


def _copy_node(inp, key, tag):
    """Emit nodes producing a transformed copy tensor; returns (nodes, inits, name)."""
    if key == "id":
        return [], [], inp
    if key == "T":
        out = f"{tag}_t"
        return [oh.make_node("Transpose", [inp], [out], perm=[0, 1, 3, 2])], [], out
    # reverse-slice based flips (origin-unsafe; emitted as attempts only)
    axes = {"fliplr": [3], "flipud": [2], "rot180": [2, 3]}[key]
    n = len(axes)
    starts = oh.make_tensor(f"{tag}_rs", INT64, [n],
                            [(HEIGHT if ax == 2 else WIDTH) - 1 for ax in axes])
    ends = oh.make_tensor(f"{tag}_re", INT64, [n], [_NEG] * n)
    axt = oh.make_tensor(f"{tag}_ra", INT64, [n], list(axes))
    steps = oh.make_tensor(f"{tag}_rst", INT64, [n], [-1] * n)
    out = f"{tag}_f"
    node = oh.make_node("Slice",
                        [inp, f"{tag}_rs", f"{tag}_re", f"{tag}_ra", f"{tag}_rst"], [out])
    return [node], [starts, ends, axt, steps], out


def overlay(keys):
    """Build output = one-hot OR of the given transformed copies of input.

    Channels 1..9: elementwise Max across copies (logical OR of colours).
    Channel 0    : elementwise product across copies (background only where all
                   copies are background) -> keeps the output a valid one-hot.
    """
    keys = list(keys)
    nodes, inits, copy_names = [], [], []
    for i, k in enumerate(keys):
        cn, ci, name = _copy_node("input", k, f"c{i}")
        nodes += cn
        inits += ci
        copy_names.append(name)

    # colour channels 1..9 of each copy, then Max them together
    col_names = []
    for i, name in enumerate(copy_names):
        node, ci = _slice(name, f"col{i}", 1, CHANNELS, 1, tag=f"col{i}")
        nodes.append(node)
        inits += ci
        col_names.append(f"col{i}")
    if len(col_names) == 1:
        colored = col_names[0]
    else:
        colored = "colmax"
        nodes.append(oh.make_node("Max", col_names, [colored]))

    # channel 0 of each copy, then product (AND of backgrounds)
    bg_names = []
    for i, name in enumerate(copy_names):
        node, ci = _slice(name, f"bg{i}", 0, 1, 1, tag=f"bg{i}")
        nodes.append(node)
        inits += ci
        bg_names.append(f"bg{i}")
    if len(bg_names) == 1:
        bg = bg_names[0]
    else:
        bg = "bgand"
        # Mul is binary in opset-10; chain it
        prev = bg_names[0]
        for j in range(1, len(bg_names)):
            out = bg if j == len(bg_names) - 1 else f"bgand{j}"
            nodes.append(oh.make_node("Mul", [prev, bg_names[j]], [out]))
            prev = out

    nodes.append(oh.make_node("Concat", [bg, colored], ["output"], axis=1))
    return _model(nodes, inits)


def overlay_recolor_transpose(cmap):
    """output[i,j] = input[i,j] if non-zero else cmap[input[j,i]].

    Overlay of input with a recoloured transpose. The recolour is a 1x1 Conv on
    the transpose copy (channel w of the recoloured copy = OR of source channels
    v with cmap[v]==w). cmap[0] is forced to 0 so background stays background.
    """
    nodes = [oh.make_node("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2])]
    weights = [0.0] * (CHANNELS * CHANNELS)  # [O, I, 1, 1]
    weights[0] = 1.0  # background -> background
    for v in range(1, CHANNELS):
        w = cmap.get(v, v)
        weights[w * CHANNELS + v] = 1.0
    W = oh.make_tensor("Wrc", DATA_TYPE, [CHANNELS, CHANNELS, 1, 1], weights)
    nodes.append(oh.make_node("Conv", ["t", "Wrc"], ["tr"], kernel_shape=[1, 1],
                              pads=[0, 0, 0, 0]))
    # overlay input with tr (same channel-0-AND / colour-Max scheme)
    inits = [W]
    n, ci = _slice("input", "col0", 1, CHANNELS, 1, tag="rc_col0"); nodes.append(n); inits += ci
    n, ci = _slice("tr", "col1", 1, CHANNELS, 1, tag="rc_col1"); nodes.append(n); inits += ci
    nodes.append(oh.make_node("Max", ["col0", "col1"], ["colmax"]))
    n, ci = _slice("input", "bg0", 0, 1, 1, tag="rc_bg0"); nodes.append(n); inits += ci
    n, ci = _slice("tr", "bg1", 0, 1, 1, tag="rc_bg1"); nodes.append(n); inits += ci
    nodes.append(oh.make_node("Mul", ["bg0", "bg1"], ["bgand"]))
    nodes.append(oh.make_node("Concat", ["bgand", "colmax"], ["output"], axis=1))
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# detection / candidate generation
# --------------------------------------------------------------------------- #

def _pairs(ex):
    return [(np.array(e["input"]), np.array(e["output"]))
            for e in ex.get("train", []) + ex.get("test", [])]


def _infer_recolor_transpose(prs):
    cmap = {}
    for a, b in prs:
        if a.shape != b.shape or a.shape[0] != a.shape[1]:
            return None
        H = a.shape[0]
        at = a.T
        for i in range(H):
            for j in range(H):
                if a[i, j] != 0:
                    if b[i, j] != a[i, j]:
                        return None
                else:
                    src = at[i, j]
                    want = b[i, j]
                    if src == 0:
                        if want != 0:
                            return None
                    else:
                        if cmap.get(src, want) != want:
                            return None
                        cmap[src] = want
    return cmap if cmap else None


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []
    seen = set()

    def add(name, model):
        if name not in seen:
            seen.add(name)
            out.append((name, model))

    square = all(a.shape[0] == a.shape[1] for a, _ in prs)

    # --- pure single-transform outputs (degenerate overlay) ---------------- #
    if square and all(a.shape == b.shape and np.array_equal(b, a.T) for a, b in prs):
        add("transpose", transpose_hw())

    # --- origin-safe overlay: input OR transpose (diagonal symmetrise) ----- #
    if square and _matches(prs, ["id", "T"]):
        add("overlay_diag", overlay(["id", "T"]))

    # --- recolour-then-overlay across the diagonal ------------------------- #
    if square:
        cmap = _infer_recolor_transpose(prs)
        # only useful when it actually recolours (otherwise == overlay_diag)
        if cmap and any(v != k for k, v in cmap.items()):
            add("overlay_recolor_diag", overlay_recolor_transpose(cmap))

    # --- flip / rotation overlays: ATTEMPTS (origin-unsafe; harness gates) -- #
    attempt_sets = [
        ("overlay_lr", ["id", "fliplr"]),
        ("overlay_ud", ["id", "flipud"]),
        ("overlay_rot180", ["id", "rot180"]),
        ("overlay_quad", ["id", "fliplr", "flipud", "rot180"]),
    ]
    for name, keys in attempt_sets:
        if _matches(prs, keys):
            add(name, overlay(keys))

    return out
