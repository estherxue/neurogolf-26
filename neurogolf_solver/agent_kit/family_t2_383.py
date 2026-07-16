"""family_t2_383 -- tier-2 golf on task383 (f1cefba8, "box + barnacle stripes").

INCUMBENT DISSECTION (out_blend12/onnx/task383.onnx, 54 nodes, opset 13,
input f32 one-hot [1,10,30,30], output BOOL [1,10,30,30]):

    cost = memory + params,  points = 25 - ln(cost)
    memory = Sum bytes over NAMED intermediates (each once, MAX shape; io free)

Per-tensor bytes, total memory 5734, params 96 -> cost 5830 -> 16.329 pts:

     2304  f32   [1,1,24,24]   color_f    <-- DOMINANT (the Conv output)
      900  u8    [1,1,30,30]   padded     (single-ch canvas, free-output floor)
      576  u8    [1,1,24,24]   final0
      576  u8    [1,1,24,24]   final
      576  u8    [1,1,24,24]   color      (Cast(color_f) -> the base index map)
      ~30x u8/bool [1,24] / [1,1,*,24]   detection + paint vectors (24B each)
      rany_r,cany_r [1,24] u8   = 24 + 24  <-- REMOVED here

The Conv (dilations [6,6], kernel 2x2, only the [0,0] weight nonzero) does two
jobs in one op: (a) contracts the 10 one-hot channels to a per-cell colour index
via weights chv=[10,1..9] + bias 11 (background 0 -> sentinel 10, outside-grid ->
11), and (b) crops 30x30 -> 24x24 (generator max grid = wide/tall<=16 + margin =
24) for free. color = Cast(color_f, u8) is the base canvas; ReduceMin along each
axis finds which rows/cols hold a real colour, ArgMax finds the box edges, and two
Where passes paint the row/column barnacle stripes before Pad -> Equal(chv) emits
the one-hot output.

WHY THE DOMINANT color_f IS FLOORED (tried against the enriched arsenal):
  * color_f is f32 [1,1,24,24] = 2304 ONLY because ONNX Conv output dtype follows
    its (f32) input. To get a 2-byte (f16) contraction I would have to Cast the
    input first -> Cast(input,f16) names [1,10,30,30] f16 = 18000, catastrophic.
    Casting to u8 (9000) to feed QLinearConv/ConvInteger is just as bad -- the
    arsenal's runtime-int8 Conv trick (task191) works there because that task
    slices ONE channel to u8 (529 B) before the QLinearConv; here the colour remap
    must read ALL 10 channels, so no single-channel slice exists.
  * The dilated-2x2 Conv already contracts AND crops to the 24x24 floor in one op;
    a 1x1 Conv (10 weights) outputs 30x30 f32 = 3600 and still needs a Slice to
    2304 (names both). ArgMax(axis=1) would give the index directly but as int64
    [1,1,24,24] = 4608 (>2304) and can't crop. So 2304 f32 is the contraction floor.
  * color / final0 / final are already the 24x24 u8 (1-byte) floor; padded 900 is
    the single-channel 30x30 free-output floor; the two orthogonal-axis paint Wheres
    can't be merged without naming a fresh 24x24 combined condition (no net saving).

WHAT THIS CANDIDATE DOES (strict, safe win that DOESN'T touch color_f):
  The incumbent finds the bottom/right box edges by Gather-reversing rany/cany
  (init `rev` = 24 int64) then ArgMax then subtracting from `last`(23). ONNX
  ArgMax has a `select_last_index` attribute (opset 12+) that returns the LAST
  argmax directly -- semantically identical for the binary rany/cany vectors. That
  deletes init `rev` (24 params), init `last` (1 param), the two reverse Gathers,
  their two [1,24] u8 outputs rany_r/cany_r (48 bytes), and the two Sub nodes. The
  four Reshape(->[1,1,24,1]/[1,1,1,24]) become Unsqueeze with 2-element axes inits
  (sr/sc 4 int64 each -> axr/axc 2 int64 each; -4 params). Net -29 params -48 bytes.

    new memory 5686, params 67 -> cost 5753 -> 16.343 pts  (+0.013 vs incumbent)

The transform is byte-for-byte identical to the incumbent on all local
train+test+arc-gen and on fresh generator samples; only redundant reverse tensors
and constant params are removed. Train+test exact-gated below.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as oh

from ng_utils_shim import tasks_dir

try:
    import onnxruntime as _ort
except Exception:  # pragma: no cover
    _ort = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX_DIR = os.path.join(_HERE, "out_blend12", "onnx")
_TASK = 383
CHANNELS, HEIGHT, WIDTH = 10, 30, 30
_CROP = 24  # generator max grid


def _sig(ex) -> str:
    return hashlib.md5(
        json.dumps(ex.get("train", []), sort_keys=True).encode()).hexdigest()


_SIG = None


def _target_sig():
    global _SIG
    if _SIG is None:
        p = tasks_dir() / f"task{_TASK:03d}.json"
        _SIG = _sig(json.load(open(p))) if p.exists() else ""
    return _SIG


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
    for e in ex.get("train", []) + ex.get("test", []):
        gi, go = np.asarray(e["input"]), np.asarray(e["output"])
        if gi.ndim != 2 or go.ndim != 2:
            continue
        if max(gi.shape) > HEIGHT or max(go.shape) > HEIGHT:
            continue
        data.append((_onehot(gi), _onehot(go)))
    return data


def _init(name, dtype, dims, vals):
    return oh.make_tensor(name, dtype, dims, vals)


def _build():
    # channel remap weights: chan 0 -> 10, chan c(1..9) -> c ; via bias 11 + weight
    cw = np.zeros((1, 10, 2, 2), dtype=np.float32)
    for ch in range(10):
        cw[0, ch, 0, 0] = -(11 - (10 if ch == 0 else ch))
    inits = [
        _init("cw", TP.FLOAT, [1, 10, 2, 2], cw.flatten().tolist()),
        _init("cb", TP.FLOAT, [1], [11.0]),
        _init("u10", TP.UINT8, [1], [10]),
        _init("u11", TP.UINT8, [1], [11]),
        _init("one", TP.INT32, [1], [1]),
        _init("two", TP.INT32, [1], [2]),
        _init("axr", TP.INT64, [2], [1, 3]),   # [1,24] -> [1,1,24,1]
        _init("axc", TP.INT64, [2], [1, 2]),   # [1,24] -> [1,1,1,24]
        _init("pad", TP.INT64, [8], [0, 0, 0, 0, 0, 0, 6, 6]),
        _init("chv", TP.UINT8, [1, 10, 1, 1], [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ]
    N = oh.make_node
    nodes = [
        N("Conv", ["input", "cw", "cb"], ["color_f"],
          dilations=[6, 6], kernel_shape=[2, 2]),
        N("Cast", ["color_f"], ["color"], to=TP.UINT8),
        N("ReduceMin", ["color"], ["rmin"], axes=[1, 3], keepdims=0),
        N("ReduceMin", ["color"], ["cmin"], axes=[1, 2], keepdims=0),
        N("Less", ["rmin", "u11"], ["ingr_b"]),
        N("Less", ["cmin", "u11"], ["ingc_b"]),
        N("Less", ["rmin", "u10"], ["rany_b"]),
        N("Less", ["cmin", "u10"], ["cany_b"]),
        N("Cast", ["rany_b"], ["rany"], to=TP.UINT8),
        N("Cast", ["cany_b"], ["cany"], to=TP.UINT8),
        N("ArgMax", ["rany"], ["top"], axis=1, keepdims=0),
        N("ArgMax", ["cany"], ["left"], axis=1, keepdims=0),
        N("ArgMax", ["rany"], ["botr"], axis=1, keepdims=0, select_last_index=1),
        N("ArgMax", ["cany"], ["rigr"], axis=1, keepdims=0, select_last_index=1),
        N("Cast", ["top"], ["topi"], to=TP.INT32),
        N("Cast", ["left"], ["lefti"], to=TP.INT32),
        N("Cast", ["botr"], ["bot"], to=TP.INT32),
        N("Cast", ["rigr"], ["rig"], to=TP.INT32),
        N("Add", ["topi", "one"], ["top1"]),
        N("Sub", ["bot", "one"], ["bot1"]),
        N("Add", ["lefti", "one"], ["left1"]),
        N("Sub", ["rig", "one"], ["rig1"]),
        N("Add", ["topi", "two"], ["top2"]),
        N("Add", ["lefti", "two"], ["left2"]),
        N("Gather", ["color", "top"], ["orow"], axis=2),
        N("Gather", ["orow", "left"], ["outc"], axis=3),
        N("Gather", ["color", "top2"], ["irow"], axis=2),
        N("Gather", ["irow", "left2"], ["innc"], axis=3),
        N("Gather", ["color", "top1"], ["gt"], axis=2),
        N("Gather", ["color", "bot1"], ["gb"], axis=2),
        N("Equal", ["gt", "innc"], ["et"]),
        N("Equal", ["gb", "innc"], ["eb"]),
        N("Or", ["et", "eb"], ["colmark"]),
        N("Gather", ["color", "left1"], ["gl"], axis=3),
        N("Gather", ["color", "rig1"], ["gr"], axis=3),
        N("Equal", ["gl", "innc"], ["el"]),
        N("Equal", ["gr", "innc"], ["er"]),
        N("Or", ["el", "er"], ["rowmark"]),
        N("Unsqueeze", ["rany_b", "axr"], ["rany_col"]),
        N("Unsqueeze", ["ingr_b", "axr"], ["ingr_col"]),
        N("Where", ["ingr_col", "innc", "u11"], ["cv1"]),
        N("Where", ["rany_col", "outc", "cv1"], ["colval"]),
        N("Unsqueeze", ["cany_b", "axc"], ["cany_row"]),
        N("Unsqueeze", ["ingc_b", "axc"], ["ingc_row"]),
        N("Where", ["ingc_row", "innc", "u11"], ["rv1"]),
        N("Where", ["cany_row", "outc", "rv1"], ["rowval"]),
        N("Where", ["colmark", "colval", "color"], ["final0"]),
        N("Where", ["rowmark", "rowval", "final0"], ["final"]),
        N("Pad", ["final", "pad", "u11"], ["padded"]),
        N("Equal", ["padded", "chv"], ["output"]),
    ]
    xi = oh.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])
    yo = oh.make_tensor_value_info("output", TP.BOOL, [1, 10, 30, 30])
    graph = oh.make_graph(nodes, "t2_383", [xi], [yo], inits)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _exact(model, data):
    if _ort is None or not data:
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
        if y.shape != tg.shape or not np.array_equal(y > 0.0, tg > 0.0):
            return False
    return True


def candidates(ex):
    if _sig(ex) != _target_sig():
        return []
    data = _pairs(ex)
    if not data:
        return []
    model = _build()
    if not _exact(model, data):
        return []
    return [("t2_383", model)]
