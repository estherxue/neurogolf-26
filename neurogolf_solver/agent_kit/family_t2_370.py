"""family_t2_370 -- tier-2 golf attempt on task370 (diagonal ricochet / sprite-trail).

Dissection of the incumbent out_blend12/onnx/task370.onnx (84 nodes, opset 13,
input float32 [1,10,30,30] one-hot, output float32 [1,10,30,30] one-hot):

    cost = memory + params,  points = 25 - ln(cost)
    memory = Sum bytes over NAMED node-output tensors (each once, MAX of
             static-inferred and profiled-trace shape; input/output free)

Per-tensor bytes (real grader), memory 6207 / params 535 -> 16.18 pts:

      900  BOOL   [1,1,30,30]  mask30                     <-- DOMINANT single tensor
      400  x8     [1,1,20,20]  Bc,Kd,Tc,tcr,T,T_b,m1b,mask20  (uint8/bool)  = 3200
      ~2087        misc [1,30] float marginals, [4,4]/[20]/5x5 index+crop tensors

The algorithm: find the sprite origin (first shape-color row/col via Einsum
marginals + ArgMax), crop the 5x5 sprite, detect the ball hint direction
(GatherND of the 4 ball neighbours -> ArgMax -> diagonal step D), build a
data-dependent 20x20 diagonal conv kernel Kd, stamp the sprite along the diagonal
with a single uint8 QLinearConv (arsenal #1/#2 -- runtime-weight binary conv /
adjoint stamping), orient the trail to one of 4 diagonal directions with two
static-index Gather flips (arsenal #3), threshold to bool, clip to the actual grid
bounds (row/col presence marginals -- essential: mask20 != T_b on 77% of gen
samples because the 20x20 conv canvas overruns grids smaller than 20), Pad to
30x30 and Where the hint colour onto the trail (output written straight, arsenal
#4). Everything is already u8/bool at the 1-byte floor.

BUG IN THE INCUMBENT (why it needed a recompile at all): node 64 is
    Kd = Min(eyeI:uint8, dpos4:uint8)
ONNX Runtime 1.23.2 has NO Min kernel for uint8 (nor Mul for uint8), so the
incumbent .onnx FAILS TO LOAD under the current runtime (hard gate (a)). It is
replaced by the exact-equivalent, load-clean
    Kd = Where(dpos4:bool, eyeI:uint8, 0:uint8)
(dpos4 is retyped bool -- it is a {0,1} parity mask -- so the now-dead
Cast->dpos_u8 node is dropped, shaving 20 bytes: memory 6207 -> 6187).
This is a strict points gain (16.184 -> 16.187) AND the only variant that runs.

WHY IT IS OTHERWISE FLOORED (enriched arsenal, what was tried):

  * DOMINANT mask30 (900): the terminal Where(mask30, hintc, input) needs a full
    [1,1,30,30] condition to broadcast against the [1,10,30,30] output. bool is the
    1-byte floor; 900 = 30*30. Any smaller mask is not broadcastable to 30x30, and
    routing the paint through Scatter/Add names a [1,10,30,30]=36000 tensor instead.

  * The eight 20x20 tensors are SIZE-locked: the generator draws grids of
    width,height in [10,20], so a 20x20 working canvas is the true maximum and
    cannot shrink. They are COUNT-locked too: from the conv output the pipeline is
    {flip_r, flip_c, threshold, clip_r, clip_c} = 5 ops (=5 named tensors) + Tc +
    the conv's Bc/Kd. The 2 flips are the 2 direction bits (4 diagonal
    orientations), the threshold is the one bool cast (QLinearConv must emit uint8),
    the 2 clips are the row/col grid-bounds intersection (both verified live). No
    ternary And / fused-flip / pre-conv-orient rewrite lowers the count; each
    alternative re-materialises an equal-size 20x20 elsewhere.

  * memory is a BYTE-SUM not a COUNT: the [1,30] float marginals (rmarg/cmarg for
    origin, rowsum/colsum for grid bounds) are the uniquely-minimal channel-collapse
    (reducing to [1,30] is cheaper than any [1,1,20,20] 2D grid mask, which would
    need a [1,1,30,30]=900 ReduceMax en route). The [4,4] GatherND index is int32
    with a single trailing Cast->int64 (int64 is required by GatherND); building it
    int64 throughout grows five [4]/[4,1] operands and is net worse.

No enriched-arsenal rewrite lowers the byte-sum below the incumbent's structural
floor; the only realised gain is the load-fix + the 20 bytes it frees. Emitted
train+test exact-gated.
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
_TASK = 370

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


def _fixed_model():
    """Load the incumbent and replace the ORT-1.23.2-unsupported uint8 Min with an
    exact-equivalent Where, retyping dpos4 to bool and dropping the dead Cast."""
    path = os.path.join(_ONNX_DIR, f"task{_TASK:03d}.onnx")
    if not os.path.exists(path):
        return None
    m = onnx.load(path)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}

    r = prod.get("dpos4")
    k = prod.get("Kd")
    if r is None or k is None or r.op_type != "Reshape" or k.op_type != "Min":
        return None
    # dpos4 = Reshape(dpos_b:bool)  (was Reshape(dpos_u8:uint8))
    r.input[0] = "dpos_b"
    # Kd = Where(dpos4:bool, eyeI:uint8, zpu8:uint8)   (was Min(eyeI, dpos4))
    k.op_type = "Where"
    del k.input[:]
    k.input.extend(["dpos4", "eyeI", "zpu8"])

    # drop the now-dead Cast(dpos_b) -> dpos_u8 if nothing else consumes it
    used = {i for n in g.node for i in n.input if i}
    used.update(o.name for o in g.output)
    if "dpos_u8" not in used and "dpos_u8" in prod:
        g.node.remove(prod["dpos_u8"])

    # keep value_info consistent: dpos4 is now bool, dpos_u8 is gone
    keep = []
    for vi in g.value_info:
        if vi.name == "dpos_u8":
            continue
        if vi.name == "dpos4":
            vi.type.tensor_type.elem_type = TP.BOOL
        keep.append(vi)
    del g.value_info[:]
    g.value_info.extend(keep)
    return m


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
    m = _fixed_model()
    if m is None:
        return []
    data = _pairs(ex)
    if not _exact(m, data):
        return []
    return [("t2_370", m)]
