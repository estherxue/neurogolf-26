"""family_fp16_1 -- FP16-rebuild of memory-dominated incumbents (slice T[1::7]).

Strategy
--------
Each incumbent ONNX in ``out_p5/onnx/taskNNN.onnx`` computes the TRUE rule in
float32.  Its dominant cost is intermediate-tensor memory ([1,10,30,30] float32 =
36000 bytes each).  We re-emit the IDENTICAL graph with every float32 intermediate
lowered to float16 (2 bytes instead of 4), which halves the dominant memory term.

The lowering is a hand transform (NOT onnxconverter_common auto-convert):

  1. the graph input stays FLOAT[1,10,30,30]; a single ``Cast`` at the very top
     turns it into float16 and every consumer is redirected to that float16 copy;
  2. every ``Cast`` whose target was FLOAT32 is retargeted to FLOAT16;
  3. every float32 initializer / Constant tensor is converted to float16;
  4. the graph output is declared FLOAT16 (the grader thresholds ``out > 0`` and
     accepts float16 -- the existing fp16 incumbents already ship float16 output);
  5. stale value_info is dropped so the scorer's own shape-inference re-derives
     the float16 types.

Because EVERY float source (input, Cast-to-float, float initializers) is now
float16, type propagation makes every formerly-float32 tensor float16 while
int / bool tensors are untouched -- so every op keeps consistent dtypes (SSA,
single-type per op).  float16 is exact for integers < 2048, and the grader gate
(EXACT on train+test+arc-gen) rejects any graph where a sum/product loses
precision, so only byte-identical rebuilds survive.

Incumbents already shipping float16 output are re-emitted unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as oh
from onnx import numpy_helper as nh

try:
    import onnxruntime as _ort
except Exception:  # pragma: no cover
    _ort = None

from ng_utils_shim import tasks_dir

FLOAT = TP.FLOAT
FLOAT16 = TP.FLOAT16

# Large "sentinel" constants (e.g. 1e9 used as a dominating mask value in
# min/max selection) overflow float16 (max 65504) -> inf/nan.  They only need to
# DOMINATE real magnitudes (grid sums <= 900, coords <= 30), so clamp them to a
# finite value that still dwarfs any real quantity (>> 2048) with headroom before
# the float16 limit.  Real (< 2048) values are untouched, so the rule is exact.
_CLAMP = 30000.0


def _to_f16_array(arr):
    return np.clip(arr, -_CLAMP, _CLAMP).astype(np.float16)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX_DIR = os.path.join(_HERE, "out_p5", "onnx")

# my slice of the FP16 targets  (fp16_targets.json[1::7])
_TARGETS = [2, 11, 25, 38, 48, 75, 88, 111, 123, 142, 152, 165, 180, 192, 204,
            216, 232, 243, 252, 277, 292, 302, 312, 330, 344, 357, 365, 377,
            387, 26, 64, 125, 184, 227, 289, 373, 84, 225, 386, 326, 155, 56,
            334, 306]


# --------------------------------------------------------------------------- #
# fp16 transform                                                               #
# --------------------------------------------------------------------------- #
def _to_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph

    # 1. redirect every use of the float32 input to a float16 copy
    for node in g.node:
        for i, inp in enumerate(node.input):
            if inp == "input":
                node.input[i] = "input_f16"
    cast_in = oh.make_node("Cast", ["input"], ["input_f16"], to=FLOAT16,
                           name="cast_input_f16")
    g.node.insert(0, cast_in)

    # 2. retarget Cast-to-FLOAT32 -> FLOAT16
    for node in g.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == FLOAT:
                    a.i = FLOAT16

    # 3. float32 initializers -> float16
    for init in g.initializer:
        if init.data_type == FLOAT:
            arr = _to_f16_array(nh.to_array(init))
            init.CopyFrom(nh.from_array(arr, init.name))

    # 3b. float32 Constant / ConstantOfShape tensors -> float16
    for node in g.node:
        if node.op_type in ("Constant", "ConstantOfShape"):
            for a in node.attribute:
                if a.name == "value" and a.t.data_type == FLOAT:
                    arr = _to_f16_array(nh.to_array(a.t))
                    a.t.CopyFrom(nh.from_array(arr))

    # 4. output declared FLOAT16 (grader thresholds > 0)
    g.output[0].type.tensor_type.elem_type = FLOAT16

    # 5. drop stale value_info so the scorer re-infers float16 types
    del g.value_info[:]
    return m


# --------------------------------------------------------------------------- #
# task identification                                                          #
# --------------------------------------------------------------------------- #
def _sig(ex) -> str:
    return hashlib.md5(
        json.dumps(ex.get("train", []), sort_keys=True).encode()
    ).hexdigest()


_SIG2TASK = None


def _sig_map():
    global _SIG2TASK
    if _SIG2TASK is None:
        _SIG2TASK = {}
        tdir = tasks_dir()
        for t in _TARGETS:
            p = tdir / f"task{t:03d}.json"
            if p.exists():
                _SIG2TASK[_sig(json.load(open(p)))] = t
    return _SIG2TASK


# --------------------------------------------------------------------------- #
# self-check: run a model and compare (out > 0) against the example targets     #
# --------------------------------------------------------------------------- #
CHANNELS, HEIGHT, WIDTH = 10, 30, 30


def _onehot(grid):
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    x = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            x[0, int(g[r, c]), r, c] = 1.0
    return x


def _pairs(ex):
    data = []
    for e in ex.get("train", []) + ex.get("test", []) + ex.get("arc-gen", []):
        gi = np.asarray(e["input"])
        go = np.asarray(e["output"])
        if gi.ndim != 2 or go.ndim != 2:
            continue
        if max(gi.shape) > HEIGHT or max(go.shape) > HEIGHT:
            continue  # grader ignores > 30x30 grids
        data.append((_onehot(gi), _onehot(go)))
    return data


def _exact(model, data):
    """True iff (out > 0) matches every example target one-hot exactly."""
    if _ort is None:
        return False
    try:
        so = _ort.SessionOptions()
        so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.log_severity_level = 3
        sess = _ort.InferenceSession(model.SerializeToString(), so)
    except Exception:
        return False
    for x, tg in data:
        try:
            y = np.asarray(sess.run(["output"], {"input": x})[0])
        except Exception:
            return False
        yb = y > 0.0
        if yb.shape != tg.shape or not np.array_equal(yb, tg > 0.0):
            return False
    return True


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def candidates(ex):
    t = _sig_map().get(_sig(ex))
    if t is None:
        return []
    path = os.path.join(_ONNX_DIR, f"task{t:03d}.onnx")
    if not os.path.exists(path):
        return []
    inc = onnx.load(path)

    # already float16 output -> re-emit unchanged (already optimal)
    if inc.graph.output[0].type.tensor_type.elem_type == FLOAT16:
        return [(f"fp16_keep_{t}", inc)]

    data = _pairs(ex)
    try:
        m = _to_fp16(inc)
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        m = None

    # Emit the float16 rebuild ONLY when it reproduces every example EXACTLY
    # (byte-identical numerics after the > 0 threshold).  Otherwise the float16
    # lowering lost precision at a comparison boundary for this graph -> fall
    # back to the unchanged incumbent so the task stays solved (no regression).
    #
    # ALSO always yield the unchanged float32 incumbent as a second candidate.
    # The fp16 lowering adds one extra intermediate: a float16 copy of the input
    # ([1,10,30,30] = 18000 bytes).  For incumbents whose dominant intermediate
    # is LARGE (>> 36000 bytes) halving it more than pays for that 18000-byte
    # copy -> fp16 wins.  But for already-cheap incumbents (tiny intermediates)
    # the 18000-byte input copy is pure overhead -> fp16 is WORSE.  The harness
    # keeps whichever candidate scores higher per task, so yielding both makes
    # every task monotonically improve (or stay equal), never regress.
    if m is not None and data and _exact(m, data):
        return [(f"fp16_{t}", m), (f"fp16_orig_{t}", inc)]
    return [(f"fp16_keep32_{t}", inc)]
