"""family_golf7_0 -- GOLF cheaper EXACT solvers via a SINGLE local Conv.

Many same-size (and some anchored) ARC tasks are a per-pixel function of a fixed
KxK neighborhood of the one-hot input. Such a map can be realised by ONE Conv node
input->output: cost = (10*10*K*K + 10) params, with ZERO intermediate memory
(input and output tensors are FREE in the cost model). For K=3 that is 910 params
-> ~18.2 pts, beating the existing (memory-heavy) solvers for these tasks.

The conv emulates 10 independent integer linear classifiers over the KxK one-hot
patch (channel c fires >0 exactly where the target color is c, <=0 elsewhere incl.
padding). We fit them with an integer batch-perceptron; integer weights keep the
float32 conv output exact, so the grader's (output>0) decode is exact.

We only fire on a fixed fingerprinted set of target tasks (golf slice).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, ng

INT64 = onnx.TensorProto.INT64


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def build_conv(W, B, K):
    """W: [10,10,K,K] float, B: [10] float. Single 'same' Conv input->output."""
    p = K // 2
    w = oh.make_tensor("W", DATA_TYPE, [10, 10, K, K], np.asarray(W, np.float32).ravel().tolist())
    b = oh.make_tensor("B", DATA_TYPE, [10], np.asarray(B, np.float32).ravel().tolist())
    node = oh.make_node("Conv", ["input", "W", "B"], ["output"],
                        kernel_shape=[K, K], pads=[p, p, p, p])
    return _model([node], [w, b])


# ---- fitting ---------------------------------------------------------------

def _onehots(examples):
    res = []
    for sub in ("train", "test", "arc-gen"):
        for e in examples.get(sub, []):
            b = ng.convert_to_numpy(e)
            if b:
                res.append((b["input"][0], b["output"][0]))
    return res


def _labelmap(o):
    s = o.sum(0)
    lab = o.argmax(0)
    lab[s == 0] = -1
    return lab


def _collect(pairs, K):
    p = K // 2
    feats, labs = [], []
    for inp, outp in pairs:
        ip = np.pad(inp, ((0, 0), (p, p), (p, p)))
        lab = _labelmap(outp)
        for r in range(30):
            for c in range(30):
                feats.append(ip[:, r:r + K, c:c + K].reshape(-1))
                labs.append(int(lab[r, c]))
    F = np.asarray(feats, np.int32)
    L = np.asarray(labs)
    # determinism + dedupe
    d = {}
    for f, l in zip(F, L):
        k = f.tobytes()
        if k in d:
            if d[k] != l:
                return None, None
        else:
            d[k] = l
    UF = np.frombuffer(b"".join(d.keys()), dtype=np.int32).reshape(len(d), F.shape[1])
    UL = np.asarray(list(d.values()))
    return UF, UL


def _batch_perceptron(F, y, iters):
    Ff = F.astype(np.int64)
    yf = y.astype(np.int64)
    w = np.zeros(F.shape[1], dtype=np.int64)
    b = np.int64(0)
    for _ in range(iters):
        s = Ff @ w + b
        viol = (yf * s) < 1
        if not viol.any():
            return w, b
        upd = yf[viol]
        w = w + (Ff[viol] * upd[:, None]).sum(0)
        b = b + upd.sum()
    s = Ff @ w + b
    if np.all((yf * s) >= 1):
        return w, b
    return None, None


def fit_conv(examples, K, iters=4000):
    pairs = _onehots(examples)
    if not pairs:
        return None
    UF, UL = _collect(pairs, K)
    if UF is None:
        return None
    Ws, Bs = [], []
    for c in range(10):
        y = np.where(UL == c, 1, -1)
        if np.all(y < 0):
            Ws.append(np.zeros(UF.shape[1], np.int64)); Bs.append(np.int64(-1)); continue
        if np.all(y > 0):
            Ws.append(np.zeros(UF.shape[1], np.int64)); Bs.append(np.int64(1)); continue
        w, b = _batch_perceptron(UF, y, iters)
        if w is None:
            return None
        Ws.append(w); Bs.append(b)
    W = np.asarray(Ws).reshape(10, 10, K, K)  # [out, (ch*K*K)] -> [out,ch,K,K]
    B = np.asarray(Bs)
    return W, B, K


# ---- target gating ---------------------------------------------------------

def _fingerprint(examples):
    tr = examples.get("train", [])
    if not tr:
        return None
    g = tr[0]["input"]
    flat = tuple(int(v) for row in g for v in row)
    return (len(tr), len(g), len(g[0]), flat[:24], sum(flat))


# Fingerprint -> kernel size K for the golf-slice targets this module GOLFS via a
# single local Conv. Only tasks whose single-Conv fit GENERALIZES are listed: each
# is validated with a held-out gate (fit on train+test+60% arc-gen, must reproduce
# the held-out 40% EXACTLY) so we never emit a memorizing conv that would fail the
# private set (drawn from the same generator).
#   t98  (hollow rectangles)  K=3  -> 18.19 pts  (was 13.62); gate: held-out 262/262 OK.
#
# REJECTED (fit exactly on all provided pairs but FAIL the held-out gate -> overfit;
# a single linear conv cannot express these, so emitting them would lose the safe
# baseline on the private set):
#   t192 (denoise rectangles): "local majority" = argmax-of-counts is piecewise-,
#         not single-linear; held-out error 58/60/52/29% at 60/80/90/95% coverage.
#   t293 (crossswap): linearly separable only at K7 by MEMORIZING (21111 patches in
#         490-dim); held-out 77/79 wrong. t222/t243: global rules (rectangle-pick /
#         flood-fill) that only look K-local due to noise-induced patch uniqueness.
_TARGET_K = {
    (3, 18, 16, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0), 474): 3,   # t98
}


def candidates(examples):
    K = _TARGET_K.get(_fingerprint(examples))
    if K is None:
        return []
    try:
        r = fit_conv(examples, K)
    except Exception:
        r = None
    if r is None:
        return []
    W, B, K = r
    return [(f"conv{K}", build_conv(W, B, K))]
