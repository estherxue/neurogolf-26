"""family_lsp_6 -- label-space / FP16 golf of memory-dominated incumbents.

Slice = golf_targets.json[6::7].

Investigation summary
---------------------
Cost is intermediate-memory dominated (params are tiny for every task in the
slice except 306).  I inspected each incumbent's node graph, initializer dtypes
and shape-inferred value_info, then grouped intermediate bytes by dtype and by
"is this a 10-channel one-hot [1,10,H,W] tensor?":

  * LABEL-SPACE headroom exists only where genuine one-hot [1,10,H,W] tensors
    dominate memory (e.g. task9: 419K/467K one-hot; task175: 384K/398K).  BUT
    every such incumbent here is an "fp16_*" solver already carrying float16
    one-hots, and each implements a bespoke rule (grid-cell fill / diagonal
    symmetry completion / quadrant mirror / crop).  A generic op-by-op
    relabelling of an arbitrary mixed graph (ReduceSum-over-channel, Where,
    recolor-MatMul, Equal) is neither mechanical nor safe, so those are kept.

  * FP16 is the real remaining headroom in this slice: eleven incumbents still
    carry float32 internally.  Seven of them have genuine float32 intermediate
    memory to halve:
      - 82, 41, 287, 350 do pure geometry (Transpose/Slice/Pad/Concat/Where) on
        the float32 *input* and never cast, so every geometry intermediate is
        float32 -> a single input Cast to float16 halves all of them;
      - 281, 263, 123 mix float32 initializers + Cast-to-float32.
    The other four (19, 58, 237, 279) already compute in float16 internally
    (only the *output* tensor -- which is FREE -- is float32), so fp16 yields
    nothing; the transform is still emitted but the harness keeps the incumbent.

Transform (identical to the proven fp16 family)
  1. graph input stays FLOAT[1,10,30,30]; one Cast at the top makes a float16
     copy and every consumer is redirected to it;
  2. every Cast-to-FLOAT32 is retargeted to FLOAT16;
  3. every float32 initializer / Constant tensor -> float16 (sentinels clamped
     to +/-30000 so they stay finite yet dominate real magnitudes < 2048);
  4. the output is declared FLOAT16 (grader thresholds out>0, accepts f16);
  5. stale value_info dropped so the scorer re-infers f16 types.

Safety / monotonicity
  * The float16 rebuild is emitted ONLY when it reproduces every train+test+
    arc-gen example EXACTLY after the >0 threshold (byte-identical numerics);
    otherwise a comparison-boundary precision loss is assumed and the incumbent
    is kept (task330-style).
  * The unchanged incumbent is ALWAYS emitted as a second candidate, so the
    harness keeps whichever scores higher per task -- never a regression.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX_DIR = os.path.join(_HERE, "out_p6", "onnx")

# my slice  golf_targets.json[6::7]
_TARGETS = [124, 90, 9, 175, 378, 281, 240, 174, 277, 19, 107, 263, 58, 114,
            198, 279, 355, 290, 14, 57, 237, 306, 342, 82, 162, 249, 353, 381,
            125, 41, 287, 123, 350]

_CLAMP = 30000.0  # finite f16 sentinel that still dwarfs real magnitudes (<2048)


def _to_f16_array(arr):
    return np.clip(arr, -_CLAMP, _CLAMP).astype(np.float16)


def _has_f32(g) -> bool:
    if g.output[0].type.tensor_type.elem_type == FLOAT:
        return True
    if any(i.data_type == FLOAT for i in g.initializer):
        return True
    for n in g.node:
        if n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to" and a.i == FLOAT:
                    return True
        if n.op_type in ("Constant", "ConstantOfShape"):
            for a in n.attribute:
                if a.name == "value" and a.t.data_type == FLOAT:
                    return True
    return False


# --------------------------------------------------------------------------- #
# fp16 transform                                                               #
# --------------------------------------------------------------------------- #
def _to_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph

    for node in g.node:
        for i, inp in enumerate(node.input):
            if inp == "input":
                node.input[i] = "input_f16"
    g.node.insert(0, oh.make_node("Cast", ["input"], ["input_f16"],
                                  to=FLOAT16, name="cast_input_f16"))

    for node in g.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == FLOAT:
                    a.i = FLOAT16

    for init in g.initializer:
        if init.data_type == FLOAT:
            arr = _to_f16_array(nh.to_array(init))
            init.CopyFrom(nh.from_array(arr, init.name))

    for node in g.node:
        if node.op_type in ("Constant", "ConstantOfShape"):
            for a in node.attribute:
                if a.name == "value" and a.t.data_type == FLOAT:
                    arr = _to_f16_array(nh.to_array(a.t))
                    a.t.CopyFrom(nh.from_array(arr))

    g.output[0].type.tensor_type.elem_type = FLOAT16
    del g.value_info[:]
    return m


def _to_fp16_after_input(model: onnx.ModelProto):
    """Alternate fp16 boundary that skips the full-size input Cast.

    A blanket input Cast materialises an [1,10,30,30] f16 copy (18000B).  When
    every node that reads "input" reads only the (free) float32 input plus
    non-float initializers -- e.g. Slice(int params) or ReduceSum -- those nodes
    can stay in float32 and the f16 Cast is placed on their (typically much
    smaller, cropped/reduced) outputs instead, so the 18000B copy is never
    materialised.  Returns None when any input-consumer also reads a float
    initializer (Mul/MatMul recolor), because keeping it float32 would then
    conflict with that initializer being lowered to float16; the caller relies
    on the blanket variant there.
    """
    g0 = model.graph
    initf32 = {i.name for i in g0.initializer if i.data_type == FLOAT}
    init_names = {i.name for i in g0.initializer}
    cons_idx = [i for i, n in enumerate(g0.node) if "input" in list(n.input)]
    if not cons_idx:
        return None
    for i in cons_idx:
        n = g0.node[i]
        for inp in n.input:
            if inp == "input":
                continue
            # every other input must be a known non-float initializer, else its
            # dtype is unknown / float and staying float32 is unsafe.
            if inp in initf32 or inp not in init_names:
                return None

    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph

    # cast each input-consumer's outputs to f16, rewiring downstream consumers.
    src_outs = []
    for i in cons_idx:
        src_outs.extend(o for o in g.node[i].output if o and o != "output")
    cons_set = set(cons_idx)
    rename = {o: o + "__f16" for o in src_outs}
    for j, node in enumerate(g.node):
        if j in cons_set:
            continue
        for k, inp in enumerate(node.input):
            if inp in rename:
                node.input[k] = rename[inp]
    # rebuild node list, inserting each consumer's f16 Cast(s) right after it so
    # the graph stays topologically sorted.
    new_nodes = []
    for j, node in enumerate(g.node):
        new_nodes.append(node)
        if j in cons_set:
            for o in (o for o in node.output if o and o != "output"):
                new_nodes.append(oh.make_node("Cast", [o], [rename[o]],
                                              to=FLOAT16, name=f"cast_{o}_f16"))
    del g.node[:]
    g.node.extend(new_nodes)

    for node in g.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == FLOAT:
                    a.i = FLOAT16

    for init in g.initializer:
        if init.data_type == FLOAT:
            arr = _to_f16_array(nh.to_array(init))
            init.CopyFrom(nh.from_array(arr, init.name))

    for node in g.node:
        if node.op_type in ("Constant", "ConstantOfShape"):
            for a in node.attribute:
                if a.name == "value" and a.t.data_type == FLOAT:
                    arr = _to_f16_array(nh.to_array(a.t))
                    a.t.CopyFrom(nh.from_array(arr))

    g.output[0].type.tensor_type.elem_type = FLOAT16
    del g.value_info[:]
    return m


# --------------------------------------------------------------------------- #
# task identification via train-signature                                       #
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
# self-check: (out > 0) must match every example one-hot exactly               #
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
            continue
        data.append((_onehot(gi), _onehot(go)))
    return data


def _exact(model, data):
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

    # no float32 anywhere -> already representation-golfed; keep unchanged.
    if not _has_f32(inc.graph):
        return [(f"lsp_keep_{t}", inc)]

    data = _pairs(ex)
    out = []
    for tag, fn in (("fp16", _to_fp16), ("fp16s", _to_fp16_after_input)):
        try:
            m = fn(inc)
            if m is None:
                continue
            onnx.checker.check_model(m, full_check=True)
        except Exception:
            continue
        if data and _exact(m, data):
            out.append((f"lsp_{tag}_{t}", m))
    out.append((f"lsp_orig_{t}", inc))  # incumbent fallback -> never regress
    return out
