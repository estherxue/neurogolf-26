"""Torch-free local-rule learner with an anti-overfit generalization gate.

If a task's transformation is a LOCAL function (output[r,c] depends only on a KxK
neighborhood of the input), we solve it EXACTLY with no gradient training:

  1. Scan examples and build a lookup  neighborhood([10,K,K] one-hot) -> output color.
  2. If consistent (a FUNCTION), compile to an exact Conv->ReLU->Conv ONNX: each hidden
     unit is a matched filter firing (=1 after ReLU) on exactly one neighborhood pattern;
     the second conv routes each pattern to its color channel.

ANTI-OVERFIT GATE (the user's private-set concern): a big pattern count means the lookup
is memorizing arc-gen, not learning a compact local rule, and would fail the held-out
private set. So we FIT the lookup on part of arc-gen and REQUIRE it to be exactly correct
on the held-out remainder (drawn from the same generator as the private set). Only if it
generalizes do we accept it; we then REFIT on all examples for maximum vocabulary coverage.
"""
from __future__ import annotations

import math

import numpy as np
import onnx
from onnx import helper as oh
from numpy.lib.stride_tricks import sliding_window_view

from ng_utils_shim import (
    ng, DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

MAX_PATTERNS = 3000      # keep the .onnx under the 1.44MB file limit
HOLDOUT_FRAC = 0.30      # fraction of arc-gen reserved for the generalization gate


def _cells(ex, K):
    """Yield (neighborhood_key, mask[10,K,K], color or -1) for every cell of one example."""
    b = ng.convert_to_numpy(ex)
    if not b:
        return
    pad = K // 2
    inp, outp = b["input"][0], b["output"][0]
    padded = np.pad(inp, ((0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(padded, (K, K), axis=(1, 2)).transpose(1, 2, 0, 3, 4)
    colsum = outp.sum(axis=0)
    colarg = outp.argmax(axis=0)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            m = win[r, c].astype(np.float32)
            key = m.astype(np.int8).tobytes()
            color = int(colarg[r, c]) if colsum[r, c] > 0 else -1
            yield key, m, color


def build_lookup(example_list, K):
    """Return (color_of, mask_of) or None if the examples are not a K-local function."""
    color_of, mask_of, none_keys = {}, {}, set()
    for ex in example_list:
        for key, m, color in _cells(ex, K):
            if color < 0:
                if key in color_of:
                    return None
                none_keys.add(key)
            else:
                if key in none_keys or (key in color_of and color_of[key] != color):
                    return None
                color_of.setdefault(key, color)
                mask_of.setdefault(key, m)
    return color_of, mask_of


def _generalizes(fit_examples, holdout_examples, K):
    """True iff a lookup fit on `fit_examples` is exactly correct on `holdout_examples`."""
    built = build_lookup(fit_examples, K)
    if built is None:
        return False
    color_of, _ = built
    none = set()  # keys seen as none during fit are implicitly "no color"
    for ex in holdout_examples:
        for key, _m, color in _cells(ex, K):
            pred = color_of.get(key, -1)   # unseen neighborhood -> predicts "no color"
            if pred != color:
                return False
    return True


def compile_lookup_onnx(color_of, mask_of, K):
    pats = [(mask_of[k], color_of[k]) for k in color_of]
    P = len(pats)
    pad = K // 2
    W1 = np.empty((P, CHANNELS, K, K), np.float32)
    B1 = np.empty((P,), np.float32)
    W2 = np.zeros((CHANNELS, P, 1, 1), np.float32)
    for j, (mask, color) in enumerate(pats):
        W1[j] = 2.0 * mask - 1.0
        B1[j] = -(float(mask.sum()) - 1.0)
        W2[color, j, 0, 0] = 1.0
    w1 = oh.make_tensor("W1", DATA_TYPE, list(W1.shape), W1.ravel().tolist())
    b1 = oh.make_tensor("B1", DATA_TYPE, [P], B1.tolist())
    w2 = oh.make_tensor("W2", DATA_TYPE, list(W2.shape), W2.ravel().tolist())
    n1 = oh.make_node("Conv", ["input", "W1", "B1"], ["h1"],
                      kernel_shape=[K, K], pads=[pad, pad, pad, pad])
    n2 = oh.make_node("Relu", ["h1"], ["h2"])
    n3 = oh.make_node("Conv", ["h2", "W2"], ["output"], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph([n1, n2, n3], "local", [x], [y], [w1, b1, w2])
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS), P


def local_candidates(examples, ks=(3, 5)):
    """Yield (name, model) for the smallest K that is a consistent, GENERALIZING local rule."""
    train_test = list(examples.get("train", [])) + list(examples.get("test", []))
    arc = list(examples.get("arc-gen", []))
    if len(arc) < 4:
        return
    nfit = max(1, math.ceil(len(arc) * (1 - HOLDOUT_FRAC)))
    fit_ex = train_test + arc[:nfit]
    hold_ex = arc[nfit:]
    for K in ks:
        # 1) generalization gate on held-out arc-gen
        if not _generalizes(fit_ex, hold_ex, K):
            continue
        # 2) refit on ALL examples for maximum vocabulary coverage
        built = build_lookup(train_test + arc, K)
        if built is None:
            continue
        color_of, mask_of = built
        if not color_of or len(color_of) > MAX_PATTERNS:
            continue
        model, P = compile_lookup_onnx(color_of, mask_of, K)
        yield (f"local_k{K}_p{P}", model)
        return
