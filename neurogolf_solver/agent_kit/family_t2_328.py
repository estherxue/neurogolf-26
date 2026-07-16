"""family_t2_328 -- tier-2 golf on task328 (nearest-corner Chebyshev-parity paint).

Dissection of the incumbent out_blend12/onnx/task328.onnx (69 nodes, opset 18,
input float32 [1,10,30,30] one-hot, output BOOL [1,10,30,30]):

    cost = memory + params,  points = 25 - ln(cost)
    memory = Σ bytes over NAMED intermediates (each once, MAX shape; io free)

Per-tensor bytes, total memory 6181, params 123 -> cost 6304 -> claimed 16.25 pts:

      900  u8    [1,1,30,30]  labelPad     <-- DOMINANT single-channel canvas
      324  u8    [1,1,18,18]  dbestT/dbestB  Manhattan dist to best top/bottom corner
      324  *     [1,1,18,18]  x14          Tle/vTie/colorField/colorF2/winRp/winCp/
                                           cheb/parOdd/evenU8/coreLabel/inside/label...
      ...   1-D  [1,1,1,18]   top_*/bot_*  column-side reduction kept as row vectors

The generator (arc-gen d22278a0) is deterministic: markers sit on 2-4 grid corners;
each cell takes the colour of its UNIQUE Manhattan-nearest corner iff that corner's
Chebyshev distance is even, else 0 (ties -> 0). Max grid 18, so the working canvas
is 18x18; the output is geometry-locked to [1,10,30,30].

THE INCUMBENT IS BROKEN IN THIS ENVIRONMENT (ORT 1.23.2, hard gate a):

    onnxruntime.InferenceSession(incumbent) -> NOT_IMPLEMENTED: Min(13) on uint8

ORT 1.23.2 ships NO uint8 kernel for Min or Max (verified: every other u8 op the
graph uses -- Less/LessOrEqual/Greater/Equal/Add/Sub/Mul/BitwiseAnd/Mod/Where/And
-- loads fine). The incumbent uses two Min(u8) (top_mrow/bot_mrow) and one Max(u8)
(cheb), so it fails session-init at EVERY opt level and scores 0.0 here. Its 16.25
was measured on an environment whose ORT had u8 Min/Max.

FIX (surgical, arsenal #5 "stick to op/dtype paths the scoring pool loads"):

  * top_mrow = Min(top_vL, top_vR)  ->  Where(top_cond, top_vL, top_vR)
    bot_mrow = Min(bot_vL, bot_vR)  ->  Where(bot_cond, bot_vL, bot_vR)
    The graph ALREADY computes top_cond = LessOrEqual(top_vL,top_vR) and bot_cond;
    Where(cond,a,b)=min since cond picks a when a<=b. Reuses existing tensors ->
    ZERO new memory. (Nodes are re-topo-sorted so the cond precedes its Where.)

  * cheb = Max(winRp, winCp)  ->  Greater(winRp,winCp)=cheb_gt ; Where(cheb_gt,...)
    No existing max-cond to reuse, so this adds one bool [1,1,18,18] tensor
    (cheb_gt, +324 bytes). Casting to f16 for the supported Max(f16) would name two
    2-byte 18x18 casts (+1296, worse); no bit-trick yields parity(max(a,b)) from the
    operands. +324 is the minimal loadable form.

Result: memory 6505, params 123, cost 6628, points 16.20, 70 nodes. Strictly beats
the incumbent's ACTUAL 0.0 here. Gates verified: loads ORT 1.23.2; exact on
train+test and 3500 fresh generator samples; output invariant across opt levels on
500; node_count 70 <= 600; trace(20 runs) 1.07 MB <= 6 MB; no dim_param.

The DOMINANT labelPad (900) stays geometry-locked to the [1,10,30,30] output (same
proven floor as family_t2_89's color30_out); the 18x18 working set is already at the
u8/bool 1-byte floor with the front-end collapsed to 1-D vectors, so no further byte
reduction is available -- this is the minimal correction that makes the tight
incumbent actually runnable on the grader's runtime.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import onnx
from onnx import helper

from ng_utils_shim import tasks_dir

try:
    import onnxruntime as _ort
except Exception:  # pragma: no cover
    _ort = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX_DIR = os.path.join(_HERE, "out_blend12", "onnx")
_TASK = 328

CHANNELS, HEIGHT, WIDTH = 10, 30, 30

# Min(u8) / Max(u8) have no ORT-1.23.2 kernel; rewrite to Where(+Greater).
_MIN_COND = {"top_mrow": "top_cond", "bot_mrow": "bot_cond"}


def _patch(model):
    """Replace the u8 Min/Max nodes with loadable Where/Greater equivalents and
    re-topologically-sort so every node's inputs precede it."""
    g = model.graph
    new = []
    for n in g.node:
        if n.op_type == "Min" and n.output[0] in _MIN_COND:
            c = _MIN_COND[n.output[0]]
            new.append(helper.make_node(
                "Where", [c, n.input[0], n.input[1]], [n.output[0]], name=n.output[0]))
        elif n.op_type == "Max" and n.output[0] == "cheb":
            gt = n.output[0] + "_gt"
            new.append(helper.make_node(
                "Greater", [n.input[0], n.input[1]], [gt], name=gt))
            new.append(helper.make_node(
                "Where", [gt, n.input[0], n.input[1]], [n.output[0]], name=n.output[0]))
        else:
            new.append(n)

    avail = {i.name for i in g.initializer} | {i.name for i in g.input}
    ordered, pend = [], list(new)
    while pend:
        rest, progressed = [], False
        for n in pend:
            if all(i in avail or i == "" for i in n.input):
                ordered.append(n)
                avail |= set(n.output)
                progressed = True
            else:
                rest.append(n)
        if not progressed:
            raise RuntimeError("unresolved deps: "
                               + str([n.output[0] for n in rest]))
        pend = rest
    del g.node[:]
    g.node.extend(ordered)
    return model


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
    model = _patch(onnx.load(path))
    data = _pairs(ex)
    if not _exact(model, data):
        return []
    return [("t2_328", model)]
