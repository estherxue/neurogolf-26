"""family_t2_89 -- tier-2 golf attempt on task089 (payload-stamp / template-copy).

Dissection of the incumbent out_blend12/onnx/task089.onnx (66 nodes, opset 18,
input float32 [1,10,30,30] one-hot, output BOOL [1,10,30,30]):

    cost = memory + params,  points = 25 - ln(cost)
    memory = Σ bytes over NAMED intermediates (each once, MAX shape; io free)

Per-tensor bytes (top of the list), total memory 6585, params 154 -> 16.18 pts:

      900  UINT8  [1,1,30,30]  color30_out   <-- DOMINANT
      676  FLOAT  [1,1,13,13]  color13_f     <-- 2nd, only non-floor dtype
      384  INT32  [96]         all_scatter_idx
      289  UINT8  [1,1,17,17]  color17
      289  UINT8  [289]        color17_flat
      15x  169    [1,1,13,13]/[169] uint8/bool working set (color/anchor/scatter)
      ... many 96 / 24 / scalar int32 index tensors

The algorithm is a hand-crafted crop+colorize (dilated 2x2 Conv, 30x30 -> 13x13),
red/green anchor detect, 3x3 MaxPool payload foreground, ArgMax reflection anchors,
5x5 neighbourhood Gather of the payload template, and a fused ScatterElements that
stamps the template onto each marker. It is already uint8/bool everywhere except
the two structurally-forced tensors below.

WHY IT IS FLOORED (what was tried against the enriched arsenal):

  * DOMINANT color30_out (900): this is the single-channel 30x30 color canvas that
    the terminal Equal(color30_out, color_bank[10]) broadcasts into the [1,10,30,30]
    output. Alternatives all name a BIGGER tensor: doing Equal at 13x13 then padding
    the one-hot names onehot13 [1,10,13,13] = 1690 (> 900); ArgMax/one-hot at 30x30
    is 30x30x{int64|bool*10}. 900 = 1 byte * 900 cells is the minimum full-canvas
    single-channel form. Geometry-locked by the fixed [1,10,30,30] output.

  * 2nd color13_f (676, float32): the crop+colorize is a dilated 2x2 Conv over the
    float one-hot input -> Conv MUST emit float. QLinearConv / ConvInteger (arsenal
    #1/#5) would emit uint8/int32 but require a uint8/int8 x input; quantizing the
    [1,10,30,30] float input first names a 9000-byte tensor (>> the 507 saved).
    Casting input->f16 for an f16 Conv names an 18000-byte [1,10,30,30] f16 tensor.
    Slicing single colour channels emits one 676-byte float slice PER channel. The
    Conv is the uniquely-minimal channel-collapse+crop; its float output is forced.

  * memory is a BYTE-SUM, not a tensor COUNT, so batching the duplicated c2/c3
    pipelines into [2,N] tensors saves 0 bytes. The remaining ~169-byte tensors are
    2D<->flat flips inherent to mixing spatial ops (Conv/Pad/MaxPool) with index ops
    (Gather/Scatter/ArgMax); each flip's alternative re-introduces an equal-size copy
    elsewhere. The scatter-index tensors are int32 (index floor) at [96]/[24] and the
    scalar-base + [24]-offset broadcast is already strictly cheaper than any batched
    [96]+[96] Add (which would materialise both 384-byte operands). Building a flat
    padded map via static Gather trades a 289-byte intermediate for a 289-ELEMENT
    static index initializer (params are element-counted) -> net worse.

Halving the cost needs mem+params ~3370; color30_out(900)+color13_f(676)+the uint8
working set already sum to 6585 with almost every tensor at the 1-byte floor. No
enriched-arsenal rewrite lowers the byte-sum, so the incumbent is re-emitted
unchanged as the never-regress candidate (train+test exact-gated).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import onnx
from onnx import TensorProto as TP

try:
    import onnxruntime as _ort
except Exception:  # pragma: no cover
    _ort = None

from ng_utils_shim import tasks_dir

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX_DIR = os.path.join(_HERE, "out_blend12", "onnx")
_TASK = 89

CHANNELS, HEIGHT, WIDTH = 10, 30, 30


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
    path = os.path.join(_ONNX_DIR, f"task{_TASK:03d}.onnx")
    if not os.path.exists(path):
        return []
    inc = onnx.load(path)
    data = _pairs(ex)
    if not _exact(inc, data):
        return []
    return [("t2_89", inc)]
