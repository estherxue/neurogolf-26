"""family_golfe_2 -- semantics-preserving graph golf of memory-dominated incumbents.

Slice of golf_targets.json (mine = [g[0] for g in G][2::6], ~29 tasks).

Every incumbent in ``out_p15/onnx/taskNNN.onnx`` already computes the TRUE rule
(it is the current cheapest-valid solution in the portfolio, already fp16-lowered).
The rules themselves are hard, global, connected-component ARC transforms, so a
cheaper *reformulation* of the algorithm is not tractable here.  What IS a safe,
guaranteed win is squeezing the graph without changing a single output bit:

  1. initializer DEDUP  -- two identical initializer tensors are merged into one,
     redirecting references.  ``params`` counts initializer ELEMENTS, so dropping
     a duplicate directly lowers params.
  2. Constant/node CSE  -- common-subexpression elimination.  Two nodes with the
     same op_type + (rewritten) inputs + attributes produce byte-identical
     tensors; keep one, redirect the other.  Each removed node removes one named
     intermediate tensor from the memory sum.
  3. no-op elimination  -- Identity, Cast-to-same-dtype, and shape-preserving
     Reshape/Squeeze/Unsqueeze/Flatten are dropped (shape/type checked via ONNX
     shape inference, so only provable no-ops are removed).
  4. dead-code elimination + unused-initializer pruning.

Plus two cost-reducing structural identities:

  5. Conv(all-ones kernel, no bias) -> Clip(min<=0, max==1)  ==  MaxPool.
     On a 0/1 mask both compute "any in-window neighbour set"; MaxPool needs no
     kernel (drops params) and fuses the Clip, removing one full-grid tensor per
     flood step.  (task 066: 120 steps rewritten, 11.09 -> 11.31.)
  6. Slice(X, channel axis) feeding ONLY spatial Reduce ops == Reduce(X) then
     Slice: reducing first eliminates the full-grid channel-slice, leaving only
     tiny reduced tensors.  (task 128 etc.)

Every rewrite is an exact algebraic identity, so the (output > 0) threshold is
bit-identical to the incumbent.  We additionally re-run the incumbent and the
rewrite through onnxruntime and require byte-identical thresholded outputs on
every train/test/arc-gen example before emitting -- if anything ever fails to
match we emit the incumbent unchanged (never a regression).  The integrator
keeps cheapest-valid, so this can only help.

Measured local gain over the 29-task slice: total points 368.46 -> 368.89.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import onnx
from onnx import helper as oh
from onnx import numpy_helper as nh
from onnx import shape_inference

try:
    import onnxruntime as _ort
except Exception:  # pragma: no cover
    _ort = None

from ng_utils_shim import tasks_dir

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX_DIR = os.path.join(_HERE, "out_p15", "onnx")

_TARGETS = [46, 219, 66, 90, 201, 286, 117, 379, 17, 183, 238, 218, 382, 275,
            92, 112, 107, 202, 206, 48, 148, 198, 137, 263, 128, 42, 49, 105, 222]


# --------------------------------------------------------------------------- #
# semantics-preserving optimizer                                              #
# --------------------------------------------------------------------------- #
def _akey(a):
    return a.SerializeToString()


def _attrs(n):
    return tuple(sorted(_akey(a) for a in n.attribute))


def _const_key(n):
    for a in n.attribute:
        if a.name == "value":
            return ("C", a.t.data_type, tuple(a.t.dims), nh.to_array(a.t).tobytes())
    return ("C2",) + _attrs(n)


def _dce(g):
    prod = {o: i for i, n in enumerate(g.node) for o in n.output if o}
    keep = set()
    stack = [o.name for o in g.output]
    while stack:
        t = stack.pop()
        i = prod.get(t)
        if i is None or i in keep:
            continue
        keep.add(i)
        for inp in g.node[i].input:
            if inp:
                stack.append(inp)
    newn = [g.node[i] for i in range(len(g.node)) if i in keep]
    del g.node[:]
    g.node.extend(newn)
    used = set(inp for n in g.node for inp in n.input)
    keepi = [ini for ini in g.initializer if ini.name in used]
    del g.initializer[:]
    g.initializer.extend(keepi)


def _dedup_cse(model):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    remap = {}

    def R(n):
        while n in remap:
            n = remap[n]
        return n

    # 1. initializer dedup
    seen = {}
    keep = []
    for init in g.initializer:
        k = (init.data_type, tuple(init.dims), nh.to_array(init).tobytes())
        if k in seen:
            remap[init.name] = seen[k]
        else:
            seen[k] = init.name
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)

    # 2. CSE
    canon = {}
    newn = []
    for node in g.node:
        for i in range(len(node.input)):
            node.input[i] = R(node.input[i])
        if node.op_type == "Constant":
            key = ("Constant",) + _const_key(node)
        else:
            key = (node.op_type, tuple(node.input), _attrs(node))
        if key in canon and len(canon[key]) == len(node.output):
            for old, new in zip(node.output, canon[key]):
                if old:
                    remap[old] = new
            continue
        canon[key] = list(node.output)
        newn.append(node)
    del g.node[:]
    g.node.extend(newn)

    _dce(g)
    del g.value_info[:]
    return m


def _noop(model):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    try:
        inf = shape_inference.infer_shapes(m, strict_mode=True)
    except Exception:
        return m
    types, shapes = {}, {}
    for v in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output):
        types[v.name] = v.type.tensor_type.elem_type
        if v.type.tensor_type.HasField("shape"):
            shapes[v.name] = tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
    g = m.graph
    remap = {}
    outn = set(o.name for o in g.output)

    def R(n):
        while n in remap:
            n = remap[n]
        return n

    for node in g.node:
        for i in range(len(node.input)):
            node.input[i] = R(node.input[i])
        o = node.output[0] if node.output else None
        if not o or o in outn:
            continue
        if node.op_type == "Identity":
            remap[o] = node.input[0]
        elif node.op_type == "Cast":
            to = [a.i for a in node.attribute if a.name == "to"][0]
            if types.get(node.input[0]) == to:
                remap[o] = node.input[0]
        elif node.op_type in ("Reshape", "Squeeze", "Unsqueeze", "Flatten"):
            if (node.input[0] in shapes and o in shapes
                    and shapes[node.input[0]] == shapes[o]):
                remap[o] = node.input[0]
    newn = [n for n in g.node if not (n.output and n.output[0] in remap)]
    del g.node[:]
    g.node.extend(newn)
    _dce(g)
    del g.value_info[:]
    return m


# --------------------------------------------------------------------------- #
# structural (algebraic) rewrites -- each is an exact identity                 #
# --------------------------------------------------------------------------- #
def _const_map(g):
    d = {}
    for i in g.initializer:
        d[i.name] = nh.to_array(i)
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    d[n.output[0]] = nh.to_array(a.t)
    return d


def _toposort(g):
    avail = set(i.name for i in g.initializer) | {inp.name for inp in g.input}
    ordered, remaining, prog = [], list(g.node), True
    while remaining and prog:
        prog, nxt = False, []
        for n in remaining:
            if all((i in avail) or i == "" for i in n.input):
                ordered.append(n)
                avail.update(o for o in n.output if o)
                prog = True
            else:
                nxt.append(n)
        remaining = nxt
    ordered.extend(remaining)  # cycles shouldn't happen; keep leftovers
    del g.node[:]
    g.node.extend(ordered)


def _conv_clip_to_maxpool(model):
    """Conv(all-ones kernel, no bias) -> Clip(min<=0, max==1) == MaxPool, for a
    0/1 mask input (both compute "any in-window neighbor set").  Removes the
    intermediate Clip tensor and drops the conv kernel from params."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    outset = set(o.name for o in g.output)
    consts = _const_map(g)
    uc = {}
    for n in g.node:
        for i in n.input:
            uc[i] = uc.get(i, 0) + 1
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    drop, repl, rep = set(), {}, 0
    for idx, C in enumerate(g.node):
        if C.op_type != "Conv" or len(C.input) > 2:
            continue
        w = C.input[1]
        if w not in consts:
            continue
        wv = consts[w]
        if wv.ndim != 4 or wv.shape[0] != 1 or wv.shape[1] != 1 or not np.all(wv == 1.0):
            continue
        co = C.output[0]
        if co in outset or uc.get(co, 0) != 1:
            continue
        clip = cons.get(co, [None])[0]
        if clip is None or clip.op_type != "Clip":
            continue
        cmin = cmax = None
        for a in clip.attribute:
            if a.name == "min":
                cmin = a.f
            if a.name == "max":
                cmax = a.f
        if len(clip.input) >= 2 and clip.input[1] and clip.input[1] in consts:
            cmin = float(np.asarray(consts[clip.input[1]]).ravel()[0])
        if len(clip.input) >= 3 and clip.input[2] and clip.input[2] in consts:
            cmax = float(np.asarray(consts[clip.input[2]]).ravel()[0])
        if cmin is None or cmin > 0 or cmax is None or abs(cmax - 1.0) > 1e-6:
            continue
        attrs = {}
        for a in C.attribute:
            if a.name == "kernel_shape":
                attrs["kernel_shape"] = list(a.ints)
            elif a.name == "pads":
                attrs["pads"] = list(a.ints)
            elif a.name == "strides":
                attrs["strides"] = list(a.ints)
        if "kernel_shape" not in attrs:
            continue
        mp = oh.make_node("MaxPool", [C.input[0]], [clip.output[0]], **attrs)
        drop.add(id(C))
        drop.add(id(clip))
        repl[idx] = mp
        rep += 1
    if rep == 0:
        return model
    out = []
    for i, n in enumerate(g.node):
        if id(n) in drop:
            if i in repl:
                out.append(repl[i])
            continue
        out.append(n)
    del g.node[:]
    g.node.extend(out)
    _dce(g)
    del g.value_info[:]
    return m


_REDUCE = {"ReduceSum", "ReduceMax", "ReduceMin", "ReduceMean", "ReduceProd"}


def _slice_through_reduce(model):
    """Slice(X, axis a) whose output feeds ONLY spatial Reduce ops (axes disjoint
    from a) == Reduce(X) then Slice.  Reducing first kills the full-grid Slice
    output (a channel-slice of a 30x30 grid) and leaves only tiny reduced tensors."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    outset = set(o.name for o in g.output)
    changed, guard = True, 0
    while changed and guard < 100:
        changed = False
        guard += 1
        cons = {}
        for n in g.node:
            for i in n.input:
                cons.setdefault(i, []).append(n)
        consts = _const_map(g)
        for S in list(g.node):
            if S.op_type != "Slice" or len(S.input) < 4 or not S.input[3]:
                continue
            T = S.output[0]
            if T in outset:
                continue
            cs = cons.get(T, [])
            if not cs or not all(c.op_type in _REDUCE for c in cs):
                continue
            if S.input[3] not in consts:
                continue
            saxes = set(int(x) for x in np.asarray(consts[S.input[3]]).ravel().tolist())
            ok = True
            for c in cs:
                ra = None
                for a in c.attribute:
                    if a.name == "axes":
                        ra = set(a.ints)
                if ra is None or (saxes & ra):
                    ok = False
                    break
            if not ok:
                continue
            X = S.input[0]
            news = []
            for c in cs:
                pre = c.output[0] + "_pre"
                nr = oh.make_node(c.op_type, [X], [pre])
                nr.attribute.extend(c.attribute)
                sl = oh.make_node("Slice", [pre] + list(S.input[1:]), [c.output[0]])
                news.append((c, nr, sl))
            for c, nr, sl in news:
                g.node.remove(c)
            g.node.remove(S)
            for c, nr, sl in news:
                g.node.append(nr)
                g.node.append(sl)
            changed = True
            break
    _toposort(g)
    _dce(g)
    del g.value_info[:]
    return m


def _optimize(model):
    m = _dedup_cse(model)
    for _ in range(4):
        n0 = len(m.graph.node)
        m = _dedup_cse(_noop(m))
        if len(m.graph.node) == n0:
            break
    # algebraic (exact) structural rewrites, each cost-reducing
    try:
        m = _conv_clip_to_maxpool(m)
    except Exception:
        pass
    try:
        m = _slice_through_reduce(m)
    except Exception:
        pass
    m = _dedup_cse(m)
    return m


# --------------------------------------------------------------------------- #
# task identification (md5 of train examples -> task number)                  #
# --------------------------------------------------------------------------- #
def _sig(ex):
    return hashlib.md5(json.dumps(ex.get("train", []), sort_keys=True).encode()).hexdigest()


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
# exactness self-check: rewrite must match incumbent bit-for-bit on (out > 0) #
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


def _inputs(ex):
    xs = []
    for e in ex.get("train", []) + ex.get("test", []) + ex.get("arc-gen", []):
        gi = np.asarray(e["input"])
        if gi.ndim != 2 or max(gi.shape) > HEIGHT:
            continue
        xs.append(_onehot(gi))
    return xs


def _sess(model):
    so = _ort.SessionOptions()
    so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    return _ort.InferenceSession(model.SerializeToString(), so)


def _matches(inc, opt, xs):
    if _ort is None:
        return False
    try:
        s1, s2 = _sess(inc), _sess(opt)
    except Exception:
        return False
    for x in xs:
        try:
            y1 = s1.run(["output"], {"input": x})[0] > 0.0
            y2 = s2.run(["output"], {"input": x})[0] > 0.0
        except Exception:
            return False
        if y1.shape != y2.shape or not np.array_equal(y1, y2):
            return False
    return True


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def candidates(ex):
    t = _sig_map().get(_sig(ex))
    if t is None:
        return []
    path = os.path.join(_ONNX_DIR, f"task{t:03d}.onnx")
    if not os.path.exists(path):
        return []
    inc = onnx.load(path)
    try:
        opt = _optimize(inc)
        onnx.checker.check_model(opt, full_check=True)
    except Exception:
        return []
    xs = _inputs(ex)
    if xs and _matches(inc, opt, xs):
        return [(f"golfe2_{t}", opt)]
    return []
