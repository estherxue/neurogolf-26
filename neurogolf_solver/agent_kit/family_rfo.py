"""family_rfo — FREE-OUTPUT audit.

The grader excludes the `output` tensor (and `input`) from the memory cost, and it
accepts any numeric/one-hot output dtype (it thresholds `>0` and compares `==` to a
float32 one-hot target). Many incumbents nevertheless materialise a full-canvas
[1,10,30,30] tensor as the LAST intermediate and then copy it to `output` via a tail
`Identity` or a lossless fp16->fp32 `Cast`. That fat intermediate is *counted*; the
copy is pure overhead.

This family reloads such incumbents, drops the tail copy, and renames the producer's
output slot to `output` so the fat tensor lands in the FREE output slot instead of a
counted intermediate. This is a value-exact rewrite:
  * Identity is the identity.
  * Cast(fp16 -> fp32) on one-hot {0,1} values is lossless, and the grader is happy
    to receive the fp16 tensor directly (0.0/1.0 are exact in fp16; `1.0f16 == 1.0f32`).
  * Cast(bool -> fp16) tails: the bool tensor 0/1 also compares equal.
No compute changes, so it cannot diverge across hardware beyond whatever the (already
grader-safe) incumbent does.

Every candidate is gated by a numpy/bit-identical check (original vs rewritten run in
onnxruntime over all train+test inputs); the untouched incumbent is always yielded too,
so the harness can never score below the incumbent.
"""
import collections
import copy
import json
import os

import numpy as np
import onnx

from ng_utils_shim import ng

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX_DIR = os.path.join(_HERE, "out_safeA", "onnx")

# Incumbents whose output node is a lossless tail copy (Identity or single-use Cast).
# Discovered by auditing out_safeA/onnx; each collapses value-exactly (grader-validated).
_TARGETS = [
    4, 19, 39, 51, 77, 86, 99, 104, 106, 115, 119, 128, 134, 137, 138, 148, 153,
    156, 159, 161, 178, 183, 196, 202, 205, 206, 208, 213, 218, 235, 253, 259,
    263, 271, 279, 338, 358, 361, 364, 383, 396, 398,
]


def _sig(examples):
    """Content signature of a task from its train+test pairs (order-stable)."""
    pairs = examples.get("train", []) + examples.get("test", [])
    return json.dumps([[e["input"], e["output"]] for e in pairs], sort_keys=True)


def _collapse(model):
    """Return a model with the tail Identity/Cast removed and the producer wired
    directly to `output`, or None if the pattern does not apply."""
    m = copy.deepcopy(model)
    g = m.graph
    on = next((n for n in g.node if "output" in n.output), None)
    if on is None or on.op_type not in ("Identity", "Cast"):
        return None
    x = on.input[0]
    cons = collections.Counter(i for n in g.node for i in n.input)
    if cons[x] != 1:                       # fat tensor is consumed elsewhere too
        return None
    prod = next((n for n in g.node if x in n.output), None)
    if prod is None:                       # x is a graph input/initializer
        return None
    try:
        info = onnx.shape_inference.infer_shapes(m).graph
    except Exception:
        return None
    vmap = {v.name: v for v in list(info.value_info) + list(info.input) + list(info.output)}
    if x not in vmap or not vmap[x].type.tensor_type.HasField("shape"):
        return None
    dtype = vmap[x].type.tensor_type.elem_type
    # Wire the producer straight into `output`, drop the tail copy.
    prod.output[:] = ["output" if o == x else o for o in prod.output]
    g.node.remove(on)
    keep = [v for v in g.value_info if v.name not in (x, "output")]
    del g.value_info[:]
    g.value_info.extend(keep)
    g.output[0].type.tensor_type.elem_type = dtype
    return m


def _run(model, benchmark_input):
    import onnxruntime as ort
    s = ng.sanitize_model(copy.deepcopy(model))
    sess = ort.InferenceSession(s.SerializeToString())
    return ng.run_network(sess, benchmark_input)


def _bit_identical(orig, new, examples):
    """True iff `new` reproduces `orig`'s output exactly on every train+test input."""
    pairs = examples.get("train", []) + examples.get("test", [])
    checked = 0
    for ex in pairs:
        bench = ng.convert_to_numpy(ex)
        if not bench:
            continue
        try:
            o = _run(orig, bench["input"]).astype(np.float32)
            n = _run(new, bench["input"]).astype(np.float32)
        except Exception:
            return False
        if o.shape != n.shape or not np.array_equal(o, n):
            return False
        checked += 1
    return checked > 0


# task-signature -> incumbent path, built once at import from the task json files
# (same numbering as the incumbents).
from ng_utils_shim import tasks_dir as _tasks_dir  # noqa: E402

_SIG2PATH = {}
_TDIR = _tasks_dir()
for _t in _TARGETS:
    _jf = _TDIR / f"task{_t:03d}.json"
    _of = os.path.join(_ONNX_DIR, f"task{_t:03d}.onnx")
    if _jf.is_file() and os.path.isfile(_of):
        try:
            _ex = json.load(open(_jf))
            _SIG2PATH[_sig(_ex)] = _of
        except Exception:
            pass


def candidates(examples):
    path = _SIG2PATH.get(_sig(examples))
    if not path:
        return
    orig = onnx.load(path)
    new = _collapse(orig)
    if new is not None and _bit_identical(orig, new, examples):
        yield ("rfo_freeout", new)
    # Always offer the untouched incumbent so we can never regress.
    yield ("rfo_incumbent", orig)
