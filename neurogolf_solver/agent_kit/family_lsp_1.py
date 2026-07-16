"""family_lsp_1: cheaper-representation rebuilds of memory-dominated incumbents.

Slice mine = golf_targets[1::7] (34 tasks, prioritising lowest-points = biggest
memory). For each I loaded the out_p6 incumbent and inspected its intermediates:

  * Most were ALREADY carried as float16 [1,10,30,30] one-hot (an earlier fp16
    pass), so no mechanical dtype win remained.
  * The genuinely one-hot / label-heavy geometry tasks in my slice (216, 218,
    36, 49, 188 ...) are data-dependent object-SELECTION + variable CROP graphs
    (100-400 nodes of Slice/Pad/Max implementing "pick the object with the most
    marks and crop it"), not pure origin-anchored geometry, so a hand label-space
    rewrite is not byte-exact expressible with a static opset-10 graph.
  * The remaining float32 incumbents are small-tensor graphs (reduced masks /
    bool comparisons); a full fp16 downcast there ADDS two [1,10,30,30] boundary
    casts (~36000 B) that exceed the halving saving, so it regresses -> discarded.

The one clean win is task099 (`container_fill_crop`, sameHW): its float32 graph
carries 216 intermediate tensors totalling ~108 KB, so halving every float32
intermediate to float16 (a leading Cast input->f16, trailing Cast f16->output,
all interior ops/initializers retyped, opset-10 preserved) beats the boundary
overhead: cost 109131 -> 90931 B, points 13.40 -> 13.58. Numerics are byte-
identical (one-hot 0/1 and sums <=900 are exact in float16); validated EXACT on
the untouched arc-gen split (261/0) plus train+test (4/0).

Detection is behavioural + self-contained: each embedded model is run under
onnxruntime on the task's train+test pairs and proposed only when it reproduces
every pair EXACTLY after the (>0) threshold, so it fires solely on its own task.
The unchanged incumbent is emitted alongside the rewrite as a fallback so the
family is monotone and never regresses; the grader re-validates arc-gen.
"""
from __future__ import annotations

import base64
import gzip

import numpy as np
import onnx

try:
    import onnxruntime as _ort
except Exception:  # pragma: no cover
    _ort = None

# name -> gzip+base64 ONNX model. For task099 we ship the float16 rewrite AND the
# unchanged float32 incumbent; the harness keeps whichever scores higher.
_MODELS = {
    "fp16_99": "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",
    "inc_99": "H4sIAA+fSGoC/+2azW4cRRSFZzxOMA2I4PyaBaBZoZGQuv6rskkcyDISinewiJzxCFkkjiXPEJZ5BF4AlA3vwaPwKNQ9VdV9qztRbAFSFh6nK1P33K66X1dPzXHLO83dP79rFs2V45PTzbqZLV8eUPOQmv1dij45+3x7+fLJ2fzKwbPj5aqZNynazI6FoEZSo3Zna6HflGOosdQ4yvElx+QcCob5h49XR5vl6mDzfPFRs3346+rs/vT19IPFp83Oz6vV6dHx87M7MbDV3CxDby1lPFW289mjzbNGNfSeAuL8Y92Kg+iGCqcT5fzqo8M1DXadBpNRFBRXcYbjk5hMiTFoKKjrZF2SzSDZUtDWybYku5R8nfpdGX4+O9g8bW5QpqeGLq0MCRPRQA1Vodo+qlqK0iVWoo8KwlM0mZLz2f7RESpQhU11bFtLRwLNpThbqkxRZcr0lSm63go12DQbDUHlKiJWrhsC2TS28CR4VjFGRjSw2kKuTbepNiBTqqbl1aKeTiO1X7pCkuKKxSkWxUDxnvB2is+WAqObqm6Naek200R5+GuKWmroSmnHopiVrpP2LIoa6TrpwKKh4Ji2xjEIihFOissKx7QZx6gKJ8YzjtEVDglxcBJMX4wx1BCkYZCGIA1BGgZpUAtBGgZpfIcTBjhUnm3HOIiLGidkHCtrnFBwrKpx6CpaCLovxtIdbwnSMkhLkJYgLYO0BGkJ0jJI6wqO9TWOpZvVhhEO4q6tcKzPOE5UODGecZyscCxmwBmqL8bRB80hnUE6gnQE6RikI0hHkI5BOltwnKtxHO0Wzo9wUjxUOM5lHN9WODGecbyocBxGogvjZV+MlxTFOAzSE6THIAzSE6QnSM8gvSk4frDzeNp5vBvhpLivcLwtOKHCifGME9oKh4Q4OAmiLybQBuEJMjDIQJCeRg8MMhBkwNAMMuiCE0yNE2jXCHaEk+Kuwgkm4wRf4cR4wek5b2YhDr67vRZtm6q51aBDcYe44HECDR5xyeOEGgLiisdVwqK3OnHdARcCCPdb7R2Q9YplCqKAo3c99V5WgEdvOXiSMAm0vPUmIaAFumiZIFoIYBeCCwIC4AWHJ+eTIYUaQEZDRK0eQ2bF1JBCFUhha8ioFEjhasgoYRJonpecagC94PQC9AL0ktNL0AvQS77y5auX3soBpJQIqzFkVnQNCU8FSGlqyKgUSOaWbmcJk0BzvOQ0E+glp5epONBLTi9BL0Gv+G0P85Qgi3vqIKN7olaOIbOiakglCiQzUntZKZDK1JAK95jCVVOWlawsWtArTq9SCaBXnF6lqkGvAocMHaRuB5A5LMaQWZE1pG4LJPNYe1kpkJXNShImgWZYydqgBb3m9Br0GvSa0+tUG+iL3wKk9j1kGEKiYNO+ATIpYgAZCiRzXntZKZCV+UoSJoGmWckGO5IBveH0BvQG9IbTG9Ab0BcXBkjjOkjjB5AG+5QJY8ik2LaGNL5AMj+2l5UCWVmyJGESaIqVbLEj2XQSp7egt6C3nN6C3oK+eDNAWttBWjeAtNinrB9DZiXUkNYVSObS9rJSICujliRMAk2ykh12JJvG4/QO9C6Nxukd6B3oi2MDpDMdpLMDSId9yrkxZFZ8DelsBxlqSGc7yMq+JQmTQOPfeh47kgO95/Qe9A4TeU7vQe/TNJpBet1BejOA9NinvB1DZsXVkN4USObo9rLSQYYa0uMe87hqgX/rBexIHvSB0wfQe9AHTh9A71FB4I4n9I4nDB1PwD4V3uB4sjJwPKFzPGHgeELveMLA8QTcYwFXLfBvvYAdCWZPtow+diA4CIILoIfdk9zuybZzPLIdOJ4YQHjseIpSO54YzZCyrR0PKRlStrXjIQmTQPO85FSDgRC4ENCCnvs9Cb8n4fekYI5His7xSDFwPDGA8NjxFKV2PDFaIEXteEgpkKJ2PCRhEmiOl5xmAj33e1Kk4kDP/Z6E35Pwe1IyxxM7HaTMjucm4uReMDO5PXrc8xnCoKNHYftPz2ImPRZCAGGdMncRQnXR320/Xj3bNAYxFBaN3QWezOGE9FCM3rr0hCjV6PCIiN759IyoLwj3LJk+KghP4XAFJNagPC/7pkEnPR9L9V15sVk/OZtf/fbFyfJwnao7zsX82CR192r873SzxvPQ0/ns+8OjxfVm+/mLo9V8Z/ni5Gx9eLJ+PZ0t4gKfHh6d3Z+wnxv3byTSK78cPtusbk7i6/V0ujv9aXFjZ5p+rk3n25PJq3sPIs3Cx0iTo19P+tf9+C8er+LxOh5/xePveEz2J5Nr+/FMuWjjWTv5zK/OcYZa/D6LUzUx/bdZmuTVvXcf5XXevHflDl/nzXtb7tte580b5r7rdd68knu+V1wffbk+7/X6mPH6/JdzXJTvItfsIutwkbW9yP1ykXvwIvd1yo3rY//f9bnM/Te5cX3cm/e3y2v0PuTG9fGXn5/3+vMTFn9cfn7e288P/Za3+Dia8Q/uTqcP6I8mSm+LerLSVOlNqGcqzVY9t/hkZyv2tjDFy4PSbRrqPizdrRl193/4Mv+Fx+6tJv66sXut2dqZxqOJxxd0PP2qyb/dvC3jwXYzudb8A4z5HQkwIgAA",
}

_SESS = {}


def _raw(name):
    return gzip.decompress(base64.b64decode(_MODELS[name]))


def _session(name):
    s = _SESS.get(name)
    if s is None:
        so = _ort.SessionOptions()
        so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.log_severity_level = 3
        s = _ort.InferenceSession(_raw(name), so)
        _SESS[name] = s
    return s


def _onehot(grid):
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    x = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r in range(h):
        row = g[r]
        for c in range(w):
            x[0, int(row[c]), r, c] = 1.0
    return x


def candidates(examples):
    if _ort is None:
        return []
    pairs = list(examples.get("train", [])) + list(examples.get("test", []))
    data = []
    for e in pairs:
        gi = np.asarray(e["input"]); go = np.asarray(e["output"])
        if gi.ndim != 2 or go.ndim != 2:
            continue
        if max(gi.shape) > 30 or max(go.shape) > 30:
            continue  # grader ignores >30x30 grids
        data.append((_onehot(gi), _onehot(go)))
    if not data:
        return []
    out = []
    for name in _MODELS:
        try:
            sess = _session(name)
        except Exception:
            continue
        ok = True
        for x, tg in data:
            try:
                y = sess.run(["output"], {"input": x})[0]
            except Exception:
                ok = False
                break
            yb = (np.asarray(y) > 0.0)
            if yb.shape != tg.shape or not np.array_equal(yb, tg > 0.0):
                ok = False
                break
        if ok:
            try:
                out.append((name, onnx.load_from_string(_raw(name))))
            except Exception:
                pass
    return out
