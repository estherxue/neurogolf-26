"""family_pb_2: minimal-cost recompiles for a targeted set of tasks.

Each candidate is gated on an exact numpy `_ref` matching the true rule on every
supplied train+test example, so the family only fires for the intended tasks.

Tasks & approaches:
  135 (5bd6f4ac): out[a,b] = I[a, 6+b] (top-right 3x3 crop moved to origin).
                  Single rank-factored Einsum -> [1,10,30,30]. 180 params, 0 mem.
"""
import numpy as np
import onnx
from onnx import helper as oh
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

FLOAT = onnx.TensorProto.FLOAT
INT64 = onnx.TensorProto.INT64


def _tensor(name, arr, dtype=FLOAT):
    a = np.asarray(arr)
    a = a.astype(np.int64) if dtype == INT64 else a.astype(np.float32)
    return oh.make_tensor(name, dtype, list(a.shape) if a.shape else [1], a.flatten().tolist())


def _model(nodes, inits, name, opset=13):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, name, [x], [y], inits)
    return oh.make_model(g, ir_version=IR_VERSION,
                         opset_imports=[oh.make_opsetid("", opset)])


# ---------------------------------------------------------------- 135
def _ref_135(I):
    if I.shape[0] < 3 or I.shape[1] < 9:
        return None
    return I[0:3, 6:9].copy()


def _build_135():
    R = np.zeros((30, 3), np.float32)
    for p in range(3):
        R[p, p] = 1.0            # identity-embed: rows 0,1,2 -> out rows 0,1,2
    Sc = np.zeros((30, 3), np.float32)
    for q in range(3):
        Sc[6 + q, q] = 1.0       # col-shift: in cols 6,7,8 -> out cols 0,1,2
    inits = [_tensor("R", R), _tensor("Sc", Sc)]
    node = oh.make_node("Einsum", ["input", "R", "R", "Sc", "R"], ["output"],
                        equation="ncij,ip,ap,jq,bq->ncab")
    return _model([node], inits, "pb2_135")


_TASKS = [
    (_ref_135, _build_135, "pb2_135_einsum"),
]


def _exact(ref, prs):
    try:
        for a, b in prs:
            o = ref(a)
            if o is None or o.shape != b.shape or not np.array_equal(o, b):
                return False
    except Exception:
        return False
    return True


def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return
    for ref, build, name in _TASKS:
        if _exact(ref, prs):
            try:
                yield (name, build())
            except Exception:
                continue
