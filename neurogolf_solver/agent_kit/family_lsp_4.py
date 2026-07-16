"""family_lsp_4 — cheaper-representation rebuilds of memory-dominated incumbents.

Slice: golf_targets.json[4::7] (+ flagship task110).

For each of my tasks the incumbent ONNX in ``out_p6/onnx/taskNNN.onnx`` already
encodes the TRUE, numerically-exact rule.  Its dominant cost is intermediate
one-hot [1,10,30,30] tensor memory.  We try to lower every FLOAT (fp32) tensor to
FLOAT16 (2 bytes instead of 4), uniformly, keeping the graph SSA (single dtype per
op).  Int/bool tensors (indices, masks) are left untouched.

  * graph input stays FLOAT[1,10,30,30]; a single Cast->float16 feeds the body;
  * every FLOAT initializer / Constant value -> float16 (exact for 0/1 and small
    ints these solvers use);
  * every ``Cast to=FLOAT`` -> ``Cast to=FLOAT16``;
  * the graph output is declared FLOAT16 (the grader thresholds ``out>0`` and
    already accepts the existing fp16 incumbents), which also folds away any final
    float32 output-cast intermediate.

float16 is exact for integers < 2048 (grid sums <= 900, coords <= 30).  Any residual
inexactness is caught by a self-check that runs the rebuild through onnxruntime and
compares ``(out>0)`` against every train+test+arc-gen target one-hot EXACTLY; if it
is not byte-identical the fp16 rebuild is dropped and only the incumbent is emitted.

We ALWAYS yield the unchanged incumbent as a candidate too, so the harness keeps
whichever is cheaper per task — monotone, never regress.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import onnx
from onnx import helper as oh, numpy_helper

try:
    import onnxruntime as _ort
except Exception:  # pragma: no cover
    _ort = None

from ng_utils_shim import tasks_dir

FLOAT = onnx.TensorProto.FLOAT       # 1
FLOAT16 = onnx.TensorProto.FLOAT16   # 10

_HERE = pathlib.Path(__file__).resolve().parent
_ONNX_DIR = _HERE / "out_p6" / "onnx"

# my slice golf_targets.json[4::7], flagship 110 prepended.
_MINE = [110, 392, 340, 324, 182, 242, 61, 325, 134, 275, 74, 169, 196, 2, 29,
         65, 271, 185, 128, 361, 397, 91, 161, 20, 345, 302, 24, 354, 213, 70,
         231, 102, 35, 248]

_CLAMP = 30000.0  # dominate real magnitudes (<2048) with headroom below fp16 max


# ---------------------------------------------------------------- lowering ----

def _lower_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a copy of `model` with every fp32 tensor lowered to fp16.

    Graph I/O input stays FLOAT[1,10,30,30] via a boundary Cast; output declared
    FLOAT16.  Assumes single input 'input' and single output 'output'."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph

    # 1) redirect every use of the fp32 input through a Cast->fp16 named 'input_f16'
    IN, IN16 = "input", "input_f16"
    for node in g.node:
        for i, nm in enumerate(node.input):
            if nm == IN:
                node.input[i] = IN16

    # 2) declare graph output FLOAT16 (grader thresholds out>0). Folds away any
    #    final float32 output-cast intermediate.
    g.output[0].type.tensor_type.elem_type = FLOAT16

    # 3) lower every FLOAT initializer -> FLOAT16 (clamp big sentinels into range)
    new_inits = []
    for init in g.initializer:
        if init.data_type == FLOAT:
            arr = np.clip(numpy_helper.to_array(init).astype(np.float32),
                          -_CLAMP, _CLAMP).astype(np.float16)
            new_inits.append(numpy_helper.from_array(arr, init.name))
        else:
            new_inits.append(init)
    del g.initializer[:]
    g.initializer.extend(new_inits)

    # 4) lower Cast->FLOAT to Cast->FLOAT16; lower fp32 Constant values
    for node in g.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == FLOAT:
                    a.i = FLOAT16
        elif node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value" and a.t.data_type == FLOAT:
                    arr = np.clip(numpy_helper.to_array(a.t).astype(np.float32),
                                  -_CLAMP, _CLAMP).astype(np.float16)
                    a.t.CopyFrom(numpy_helper.from_array(arr, a.t.name))

    # 5) prepend the single input boundary cast: input(fp32) -> input_f16(fp16)
    cast_in = oh.make_node("Cast", [IN], [IN16], to=FLOAT16, name=IN16)
    nodes = [cast_in] + list(g.node)
    del g.node[:]
    g.node.extend(nodes)

    # drop stale value_info so the scorer re-infers fp16 types
    del g.value_info[:]
    return m


# ------------------------------------------------ label-space (task110) -------

def _rewrite_label110(model: onnx.ModelProto) -> onnx.ModelProto:
    """LABEL-SPACE rewrite of task110's 32 neighbour-comparison blocks.

    The incumbent carries a full one-hot ``__xin`` [1,10,30,30] through 32 blocks
    of the form::

        rsumC = ReduceSum( (__xin * Slice(Pad(__xin))) * c1 )      # [1,1,30,30]
        subF  = Sub( reducesum3 * Slice(Pad(reducesum3)) , rsumC ) # 4 big one-hot
                                                                    # intermediates

    ``rsumC[p] == 1`` iff cell p and its neighbour share the SAME non-zero colour.
    We add a single integer label map ``Lc = ReduceSum(__xin * [0..9])`` [1,1,30,30]
    and replace each block's four [1,10,30,30] one-hot tensors with an all-[1,1,30,30]
    equality on shifted labels::

        rsumC = reducesum3 * ( |Lc - Slice(Pad(Lc))| < 0.5 )

    reusing the block's OWN Pad/Slice params (identical to the one-hot slice, verified)
    so the neighbour offset matches exactly. Labels 0..9 are exact in float16, so the
    <0.5 test is byte-identical to the one-hot AND. 128 [1,10,30,30] intermediates
    (~2.3 MB) collapse to [1,1,30,30] label tensors. The caller self-checks EXACT.
    """
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    nd = {n.output[0]: n for n in g.node}
    if "reducesum3" not in nd or "__xin" not in {i for n in g.node for i in n.input}:
        return None

    w9 = numpy_helper.from_array(
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float16).reshape(1, 10, 1, 1),
        "lsp_w9")
    half = numpy_helper.from_array(np.array([0.5], dtype=np.float16), "lsp_half")
    g.initializer.extend([w9, half])
    new_nodes = [
        oh.make_node("Mul", ["__xin", "lsp_w9"], ["lsp_mullc"]),
        oh.make_node("ReduceSum", ["lsp_mullc"], ["lsp_Lc"], axes=[1], keepdims=1),
    ]

    def _pads(n):
        for a in n.attribute:
            if a.name == "pads":
                return list(a.ints)
        return None

    k = 0
    dead = set()
    for n in list(g.node):
        if n.op_type != "Sub":
            continue
        mulE = nd.get(n.input[0])
        rsumC = nd.get(n.input[1])
        if not mulE or not rsumC or mulE.op_type != "Mul" or rsumC.op_type != "ReduceSum":
            continue
        if "reducesum3" not in mulE.input:
            continue
        sd = [x for x in mulE.input if x != "reducesum3"]
        if len(sd) != 1:
            continue
        sliceD = nd.get(sd[0])
        if not sliceD or sliceD.op_type != "Slice":
            continue
        padD = nd.get(sliceD.input[0])
        if not padD or padD.op_type != "Pad":
            continue
        mulB = nd.get(rsumC.input[0])
        if not mulB or mulB.op_type != "Mul":
            continue
        pads = _pads(padD)
        if pads is None:
            continue
        new_nodes += [
            oh.make_node("Pad", ["lsp_Lc"], [f"lsp_padL{k}"], mode="constant",
                         pads=pads, value=0.0),
            oh.make_node("Slice", [f"lsp_padL{k}"] + list(sliceD.input[1:]),
                         [f"lsp_shiftL{k}"]),
            oh.make_node("Sub", ["lsp_Lc", f"lsp_shiftL{k}"], [f"lsp_df{k}"]),
            oh.make_node("Abs", [f"lsp_df{k}"], [f"lsp_ab{k}"]),
            oh.make_node("Less", [f"lsp_ab{k}", "lsp_half"], [f"lsp_eq{k}"]),
            oh.make_node("Cast", [f"lsp_eq{k}"], [f"lsp_eqf{k}"], to=FLOAT16),
            oh.make_node("Mul", ["reducesum3", f"lsp_eqf{k}"], [f"lsp_newr{k}"]),
        ]
        n.input[1] = f"lsp_newr{k}"
        dead.update([rsumC.output[0], mulB.output[0]])
        mulA = nd.get(mulB.input[0])
        if mulA and mulA.op_type == "Mul":
            dead.add(mulA.output[0])
            sa = [x for x in mulA.input if x != "__xin"]
            if len(sa) == 1:
                sliceA = nd.get(sa[0])
                if sliceA and sliceA.op_type == "Slice":
                    dead.add(sliceA.output[0])
                    padA = nd.get(sliceA.input[0])
                    if padA and padA.op_type == "Pad":
                        dead.add(padA.output[0])
        k += 1

    if k == 0:
        return None

    used = set()
    for n in g.node:
        if n.output[0] in dead:
            continue
        for i in n.input:
            used.add(i)
    for n in new_nodes:
        for i in n.input:
            used.add(i)
    reallydead = dead - used
    keep = [n for n in g.node if n.output[0] not in reallydead]
    final, inserted = [], False
    for n in keep:
        final.append(n)
        if n.output[0] == "reducesum3" and not inserted:
            final += new_nodes
            inserted = True
    if not inserted:
        return None
    del g.node[:]
    g.node.extend(final)
    del g.value_info[:]
    return m


# ------------------------------------------------------------ fingerprint ----

def _fp(ex):
    """Stable fingerprint of a task from its train+test input/output grids."""
    parts = []
    for sub in ("train", "test"):
        for e in ex.get(sub, []):
            parts.append(np.asarray(e["input"], dtype=np.int64).tobytes())
            parts.append(np.asarray(e["output"], dtype=np.int64).tobytes())
    return hash(tuple(parts))


_INDEX = None


def _build_index():
    idx = {}
    try:
        tdir = tasks_dir()
    except Exception:
        return idx
    for t in _MINE:
        fp_onnx = _ONNX_DIR / f"task{t:03d}.onnx"
        fp_json = tdir / f"task{t:03d}.json"
        if not (fp_onnx.exists() and fp_json.exists()):
            continue
        try:
            idx[_fp(json.load(open(fp_json)))] = t
        except Exception:
            continue
    return idx


# ---------------------------------------------------- exact self-check --------

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
    """True iff (out>0) matches every example target one-hot exactly."""
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


# ------------------------------------------------------------- entry ----------

def candidates(examples):
    global _INDEX
    if _INDEX is None:
        _INDEX = _build_index()
    t = _INDEX.get(_fp(examples))
    if t is None:
        return []
    try:
        inc = onnx.load(str(_ONNX_DIR / f"task{t:03d}.onnx"))
    except Exception:
        return []

    out = [(f"lsp_orig_{t}", inc)]  # incumbent — safety net, never regress
    data = _pairs(examples)

    # fp16 rebuild, emitted only if byte-identical on all examples.
    try:
        m = _lower_fp16(inc)
        onnx.checker.check_model(m, full_check=True)
        if _exact(m, data):
            out.append((f"lsp_fp16_{t}", m))
    except Exception:
        pass

    # LABEL-SPACE rewrite (flagship task110), emitted only if byte-identical.
    if t == 110:
        try:
            ls = _rewrite_label110(inc)
            if ls is not None:
                onnx.checker.check_model(ls, full_check=True)
                if _exact(ls, data):
                    out.append((f"lsp_label_{t}", ls))
        except Exception:
            pass
    return out
