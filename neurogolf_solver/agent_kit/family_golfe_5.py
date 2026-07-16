"""family_golfe_5 -- cheaper rebuilds of low-scoring incumbents (golf slice [5::6]).

Method
------
For each task in my slice I loaded the incumbent ONNX (out_p15/onnx/taskNNN.onnx),
read its Hodel verifier for the minimal rule, and inspected every intermediate
tensor's dtype/size against the exact cost model
    points = max(1, 25 - ln(memory + params))
    memory = sum over named intermediates of dtype_bytes * elements (input/output free).

The dominant, reliably-recoverable waste in this slice was incumbents whose
intermediates were still FLOAT32 (4 bytes) rather than FLOAT16 (2 bytes) -- the
memory term is entirely float-tensor dominated, so lowering every float32 tensor
to float16 nearly halves the cost.  The lowering is the hand transform from
family_fp16_1._to_fp16 (single Cast of the input, retarget Cast-to-float32 ->
float16, convert float initializers/Constants, declare output float16).  It is
emitted ONLY when it reproduces every train+test+arc-gen example EXACTLY after the
grader's (out > 0) threshold, so precision loss can never regress correctness
(e.g. task169 encodes component labels as integers > 2048 which are not fp16-exact
-> rejected, incumbent kept).

Confirmed improvements (float32 -> float16 intermediates):
  * task096  cost 1_566_060 -> 842_052   pts 10.74 -> 11.36  (+0.62)
  * task089  cost   191_771 -> 114_591   pts 12.84 -> 13.35  (+0.51)

Second lever: MaxPool 4-neighbour dilation rewrite.
------------------------------------------------
Several incumbents run an iterated masked 4-connected flood implemented, per step,
as four axis shifts built from Pad+Slice feeding a Max:
    Max(B, Slice(Pad(B,+H)), Slice(Pad(B,-H)), Slice(Pad(B,+W)), Slice(Pad(B,-W)))
Every step therefore materialises 4 Pad + 4 Slice intermediates (~9 tensors).  The
identical result is the separable cross dilation
    Max( MaxPool(B, kernel=3x1, pad=1), MaxPool(B, kernel=1x3, pad=1) )
which materialises only 2 tensors (the centre B is already contained in both pools,
and for non-negative flood masks ONNX's -inf MaxPool padding equals the original
0-constant Pad boundary).  This is an exact algebraic identity -- verified bit-for-
bit against the incumbent on every train+test+arc-gen example -- so it can only
lower cost, never change the output.  Dead Pad/Slice nodes and their index
initializers are then pruned.
  * task170  cost   428_437 -> 295_669   pts 12.03 -> 12.40  (+0.37)  12 blocks
  * task396  cost 3_200_841 -> 2_868_921 pts 10.02 -> 10.13  (+0.11)  30 blocks

Every other task in the slice is already float16-optimal with no rewritable
dilation (its verifier rule is inherently data-dependent -- object detection /
dynamic canvases / gravity / pairwise gather-by-matmul -- so a structurally cheaper
static graph was not available); for those the incumbent is re-emitted unchanged so
the family never regresses.
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
_ONNX_DIR = os.path.join(_HERE, "out_p15", "onnx")

# my golf slice = [t for t,_,_ in golf_targets.json][5::6]
_TARGETS = [396, 96, 110, 153, 234, 170, 364, 228, 101, 387, 182, 358, 89, 370,
            19, 169, 189, 191, 192, 30, 65, 224, 400, 239, 398, 384, 57, 8]

_CLAMP = 30000.0


def _to_f16_array(arr):
    return np.clip(arr, -_CLAMP, _CLAMP).astype(np.float16)


def _to_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Lower every float32 tensor in the graph to float16 (see module docstring)."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph

    for node in g.node:
        for i, inp in enumerate(node.input):
            if inp == "input":
                node.input[i] = "input_f16g5"
    g.node.insert(0, oh.make_node("Cast", ["input"], ["input_f16g5"],
                                  to=FLOAT16, name="cast_input_f16g5"))

    for node in g.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == FLOAT:
                    a.i = FLOAT16

    for init in g.initializer:
        if init.data_type == FLOAT:
            init.CopyFrom(nh.from_array(_to_f16_array(nh.to_array(init)), init.name))

    for node in g.node:
        if node.op_type in ("Constant", "ConstantOfShape"):
            for a in node.attribute:
                if a.name == "value" and a.t.data_type == FLOAT:
                    a.t.CopyFrom(nh.from_array(_to_f16_array(nh.to_array(a.t))))

    g.output[0].type.tensor_type.elem_type = FLOAT16
    del g.value_info[:]
    return m


# --------------------------------------------------------------------------- #
# MaxPool 4-neighbour dilation rewrite (see module docstring)                   #
# --------------------------------------------------------------------------- #
def _rank4_names(model):
    try:
        gi = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    except Exception:
        return {}
    r = {}
    for vi in list(gi.value_info) + list(gi.input):
        r[vi.name] = len(vi.type.tensor_type.shape.dim)
    return r


def _dce(model):
    """Drop nodes/initializers not reachable from the graph output."""
    g = model.graph
    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
    stack = [o.name for o in g.output]
    needed = set()
    while stack:
        t = stack.pop()
        n = prod.get(t)
        if n is None or id(n) in needed:
            continue
        needed.add(id(n))
        stack.extend(i for i in n.input if i)
    newnodes = [n for n in g.node if id(n) in needed]
    used = set()
    for n in newnodes:
        used.update(n.input)
    del g.node[:]
    g.node.extend(newnodes)
    ninit = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(ninit)
    del g.value_info[:]
    return model


def _rewrite_dilations(model):
    """Replace Max(B, 4x Slice(Pad(B))) cross-dilations with separable MaxPool.

    Returns (new_model, n_blocks) or (None, 0) if no block matched.
    """
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    ranks = _rank4_names(m)
    prod = {o: n for n in g.node for o in n.output}

    matches = {}  # id(Max node) -> base tensor name B
    for M in g.node:
        if M.op_type != "Max" or len(M.input) < 3:
            continue
        bases = []
        for inp in M.input:
            sl = prod.get(inp)
            if sl is not None and sl.op_type == "Slice":
                pd = prod.get(sl.input[0])
                if pd is not None and pd.op_type == "Pad":
                    bases.append(pd.input[0])
        if len(bases) == 4 and len(set(bases)) == 1:
            B = bases[0]
            if B in M.input and ranks.get(B) == 4:
                matches[id(M)] = B
    if not matches:
        return None, 0

    newnodes = []
    k = 0
    for n in g.node:
        if id(n) in matches:
            B = matches[id(n)]
            k += 1
            vo, ho = f"_g5dil_v{k}", f"_g5dil_h{k}"
            newnodes.append(oh.make_node("MaxPool", [B], [vo], kernel_shape=[3, 1],
                                         pads=[1, 0, 1, 0], name=f"_g5mpv{k}"))
            newnodes.append(oh.make_node("MaxPool", [B], [ho], kernel_shape=[1, 3],
                                         pads=[0, 1, 0, 1], name=f"_g5mph{k}"))
            nn = onnx.NodeProto()
            nn.CopyFrom(n)
            del nn.input[:]
            nn.input.extend([vo, ho])  # centre already contained in both pools
            newnodes.append(nn)
        else:
            newnodes.append(n)
    del g.node[:]
    g.node.extend(newnodes)
    _dce(m)
    return m, k


# --------------------------------------------------------------------------- #
# task identification by train-signature                                       #
# --------------------------------------------------------------------------- #
def _sig(ex) -> str:
    return hashlib.md5(
        json.dumps(ex.get("train", []), sort_keys=True).encode()).hexdigest()


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
# self-check: (out > 0) must equal every example target one-hot exactly         #
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
        if y.shape != tg.shape or not np.array_equal(y > 0.0, tg > 0.0):
            return False
    return True


def _has_float32_intermediate(model) -> bool:
    try:
        g = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    except Exception:
        return True  # be permissive; the exact-check still gates emission
    for vi in list(g.value_info) + list(g.output):
        if vi.name in ("input", "output"):
            continue
        if vi.type.tensor_type.elem_type == FLOAT:
            return True
    return False


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
    data = _pairs(ex)
    out = []

    # Lever 2: MaxPool dilation rewrite (applies to float16-optimal flood graphs).
    try:
        md, k = _rewrite_dilations(inc)
        if md is not None:
            onnx.checker.check_model(md, full_check=True)
        else:
            md = None
    except Exception:
        md = None
    if md is not None and data and _exact(md, data):
        out.append((f"golfe5_dil_{t}", md))

    # Lever 1: fp16 rebuild, only when float32 intermediates remain.
    if _has_float32_intermediate(inc):
        try:
            m = _to_fp16(inc)
            onnx.checker.check_model(m, full_check=True)
        except Exception:
            m = None
        if m is not None and data and _exact(m, data):
            out.append((f"golfe5_fp16_{t}", m))

    # Incumbent is always emitted last as the never-regress safety net.
    out.append((f"golfe5_keep_{t}", inc))
    return out
