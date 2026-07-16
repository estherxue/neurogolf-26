"""family_lsp_3: LABEL-SPACE + FP16 golf on memory-dominated incumbents.

Slice mine = [g[0] for g in golf_targets][3::7]. For each target we load the exact
out_p6 incumbent ONNX and emit CHEAPER, byte-identical rewrites (harness keeps the
cheapest; the unchanged incumbent is always emitted as a fallback so we never regress).

Three mechanical rewrites, all preserving exact grader numerics (held-out validated on
the untouched 30% arc-gen split, and re-checked by the grader on train+test+arc-gen):

  * labelize_110  -- the flagship. The incumbent carries a 10-channel one-hot
    [1,10,30,30] through 64 period-correlation blocks. Each block computes
        reducesum14 = sum_c  x[c] * shift(x)[c] * c1[c]         (c1 = [0,1,1,..,1])
    i.e. "1 iff current & shifted cells share the same NON-ZERO colour". We build an
    injective label map labI = sum_c x[c]*(c+1) once, shift it with the block's own
    Pad/Slice, and replace the three [1,10,30,30] one-hot intermediates per block with
        reducesum14 = mask * (1 - min(|labI - shift(labI)|, 1))     (all [1,1,30,30])
    Opset-10 Equal rejects float16, so equality is done arithmetically (exact for
    integer labels). 2.89M -> 0.79M bytes.

  * to_fp16 (full)  -- lower every FLOAT32 tensor to FLOAT16; wrap I/O with Cast so the
    contract FLOAT[1,10,30,30] is preserved. Wins on the large float32 incumbents.

  * to_fp16_bnd  -- boundary-aware: keep the geometry prefix from 'input' and the
    geometry suffix into 'output' in float32 (Slice/Pad/Transpose/... are dtype-neutral),
    lower only the arithmetic core to float16. This pushes the two full-grid boundary
    casts inward past size-changing Slice/Pad, so the cast overhead shrinks below the
    halving savings on smaller float32 crop incumbents.
"""
from __future__ import annotations

import os
import numpy as np
import onnx
from onnx import helper as oh, numpy_helper as nh, TensorProto as TP, shape_inference

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX_DIR = os.path.join(_HERE, "out_p6", "onnx")

# golf_targets[3::7]
_MINE = [110, 205, 284, 394, 379, 17, 208, 64, 202, 206, 115, 148, 156, 48, 243,
         224, 184, 244, 221, 310, 105, 122, 368, 297, 215, 265, 344, 389, 27, 251,
         336, 225, 180, 28]

GEOM = {'Slice', 'Pad', 'Transpose', 'Gather', 'Reshape', 'Concat', 'Squeeze',
        'Unsqueeze', 'Flatten', 'Identity', 'Expand', 'Tile', 'Split',
        'DepthToSpace', 'SpaceToDepth'}


# --------------------------------------------------------------------------- #
# whole-graph float16                                                          #
# --------------------------------------------------------------------------- #
def to_fp16(model):
    m = onnx.ModelProto(); m.CopyFrom(model); g = m.graph
    new = []
    for init in g.initializer:
        if init.data_type == TP.FLOAT:
            new.append(nh.from_array(nh.to_array(init).astype(np.float16), init.name))
        else:
            new.append(init)
    del g.initializer[:]; g.initializer.extend(new)
    for n in g.node:
        if n.op_type == 'Constant':
            for a in n.attribute:
                if a.name == 'value' and a.t.data_type == TP.FLOAT:
                    a.t.CopyFrom(nh.from_array(nh.to_array(a.t).astype(np.float16)))
        if n.op_type == 'Cast':
            for a in n.attribute:
                if a.name == 'to' and a.i == TP.FLOAT:
                    a.i = TP.FLOAT16
    inh = 'input__h'
    for n in g.node:
        for i, x in enumerate(n.input):
            if x == 'input':
                n.input[i] = inh
    outh = 'output__h'
    for n in g.node:
        for i, o in enumerate(n.output):
            if o == 'output':
                n.output[i] = outh
    nodes = ([oh.make_node('Cast', ['input'], [inh], to=TP.FLOAT16, name='cast_in')]
             + list(g.node)
             + [oh.make_node('Cast', [outh], ['output'], to=TP.FLOAT, name='cast_out')])
    del g.node[:]; g.node.extend(nodes)
    del g.value_info[:]
    return m


# --------------------------------------------------------------------------- #
# boundary-aware float16                                                       #
# --------------------------------------------------------------------------- #
def _float_names(m):
    g = m.graph; fl = set()
    try:
        mi = shape_inference.infer_shapes(m)
    except Exception:
        mi = m
    for vi in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output):
        if vi.type.tensor_type.elem_type == TP.FLOAT:
            fl.add(vi.name)
    for init in g.initializer:
        if init.data_type == TP.FLOAT:
            fl.add(init.name)
    for n in g.node:
        if n.op_type == 'Constant':
            for a in n.attribute:
                if a.name == 'value' and a.t.data_type == TP.FLOAT:
                    fl.add(n.output[0])
    return fl


def to_fp16_bnd(model):
    m = onnx.ModelProto(); m.CopyFrom(model); g = m.graph
    fl = _float_names(model)
    f32in = {'input'}; changed = True
    while changed:
        changed = False
        for n in g.node:
            if n.op_type not in GEOM:
                continue
            fin = [i for i in n.input if i in fl]
            if fin and all(i in f32in for i in fin):
                for o in n.output:
                    if o in fl and o not in f32in:
                        f32in.add(o); changed = True
    f32out = {'output'}; changed = True
    while changed:
        changed = False
        for n in g.node:
            if n.op_type not in GEOM:
                continue
            fout = [o for o in n.output if o in fl]
            if fout and all(o in f32out for o in fout):
                for i in n.input:
                    if i in fl and i not in f32out:
                        f32out.add(i); changed = True
    keep = f32in | f32out
    new = []
    for init in g.initializer:
        if init.data_type == TP.FLOAT and init.name not in keep:
            new.append(nh.from_array(nh.to_array(init).astype(np.float16), init.name))
        else:
            new.append(init)
    del g.initializer[:]; g.initializer.extend(new)
    for n in g.node:
        if n.op_type == 'Constant' and n.output[0] not in keep:
            for a in n.attribute:
                if a.name == 'value' and a.t.data_type == TP.FLOAT:
                    a.t.CopyFrom(nh.from_array(nh.to_array(a.t).astype(np.float16)))
        if n.op_type == 'Cast' and n.output[0] not in keep:
            for a in n.attribute:
                if a.name == 'to' and a.i == TP.FLOAT:
                    a.i = TP.FLOAT16
    stored = {t: (TP.FLOAT if t in keep else TP.FLOAT16) for t in fl}
    stored['input'] = TP.FLOAT; stored['output'] = TP.FLOAT
    newnodes = []; cache = {}; ctr = [0]

    def get_cast(t, tgt):
        k = (t, tgt)
        if k in cache:
            return cache[k]
        nm = f"__bc{ctr[0]}"; ctr[0] += 1
        newnodes.append(oh.make_node('Cast', [t], [nm], to=tgt, name=nm))
        cache[k] = nm; return nm

    for n in g.node:
        fouts = [o for o in n.output if o in fl]
        if fouts:
            opdt = stored[fouts[0]]
        elif n.op_type == 'Cast':
            opdt = None
        else:
            opdt = TP.FLOAT16
        n2 = onnx.NodeProto(); n2.CopyFrom(n)
        for i, inp in enumerate(n2.input):
            if inp in fl:
                want = stored[inp] if (n.op_type == 'Cast' or opdt is None) else opdt
                if stored[inp] != want:
                    n2.input[i] = get_cast(inp, want)
        newnodes.append(n2)
    del g.node[:]; g.node.extend(newnodes)
    del g.value_info[:]
    return m


# --------------------------------------------------------------------------- #
# flagship task-110 label-space rewrite                                        #
# --------------------------------------------------------------------------- #
def _topo(nodes, seeds):
    have = set(seeds); ordered = []; remaining = list(nodes); prog = True
    while remaining and prog:
        prog = False; nxt = []
        for nd in remaining:
            if all((i in have or i == '') for i in nd.input):
                ordered.append(nd)
                for o in nd.output:
                    have.add(o)
                prog = True
            else:
                nxt.append(nd)
        remaining = nxt
    if remaining:
        raise RuntimeError("topo failed")
    return ordered


def labelize_110(model):
    m = onnx.ModelProto(); m.CopyFrom(model); g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    xin = None
    for n in g.node:
        if n.op_type == 'Cast' and 'input' in n.input:
            xin = n.output[0]
    assert xin
    rs3 = None; rs3node = None
    for n in g.node:
        if n.op_type == 'ReduceSum':
            pn = prod.get(n.input[0])
            if pn and pn.op_type == 'Mul' and xin in pn.input:
                rs3 = n.output[0]; rs3node = n; break
    assert rs3
    wI = '__wI'; ONE = '__one_f16'
    g.initializer.append(nh.from_array(np.arange(1, 11, dtype=np.float16).reshape(1, 10, 1, 1), wI))
    g.initializer.append(nh.from_array(np.array(1, np.float16), ONE))
    labMul = '__labmul'; labI = '__labI'
    setup = [oh.make_node('Mul', [xin, wI], [labMul], name=labMul)]
    rsn = oh.make_node('ReduceSum', [labMul], [labI], name=labI)
    for a in rs3node.attribute:
        rsn.attribute.append(a)
    setup.append(rsn)
    ctr = [0]; newnodes = []; kill = set()

    def nid(p):
        ctr[0] += 1; return f"__L{p}{ctr[0]}"

    for n in list(g.node):
        if n.op_type == 'Mul' and xin in n.input:
            other = [x for x in n.input if x != xin]
            if len(other) != 1:
                continue
            sx = other[0]; pn = prod.get(sx)
            if not (pn and pn.op_type == 'Slice'):
                continue
            m1 = n; c = cons.get(m1.output[0], [])
            if len(c) != 1 or c[0].op_type != 'Mul':
                continue
            m2 = c[0]; c2 = cons.get(m2.output[0], [])
            if len(c2) != 1 or c2[0].op_type != 'ReduceSum':
                continue
            rs = c2[0]; rsout = rs.output[0]; c3 = cons.get(rsout, [])
            if len(c3) != 1 or c3[0].op_type != 'Sub':
                continue
            sub = c3[0]; mlo = [x for x in sub.input if x != rsout]
            if len(mlo) != 1:
                continue
            mlp = prod.get(mlo[0])
            if not (mlp and mlp.op_type == 'Mul' and rs3 in mlp.input):
                continue
            sliceLabel = [x for x in mlp.input if x != rs3][0]
            slNode = prod.get(sliceLabel)
            if not (slNode and slNode.op_type == 'Slice'):
                continue
            padNode = prod.get(slNode.input[0])
            if not (padNode and padNode.op_type == 'Pad'):
                continue
            sPad = nid('pd'); sSl = nid('sl')
            pnode = oh.make_node('Pad', [labI], [sPad], name=sPad)
            for a in padNode.attribute:
                pnode.attribute.append(a)
            slnew = oh.make_node('Slice', [sPad] + list(slNode.input[1:]), [sSl], name=sSl)
            for a in slNode.attribute:
                slnew.attribute.append(a)
            df = nid('df'); ab = nid('ab'); mn = nid('mn'); ind = nid('in'); rn = nid('rs')
            newnodes += [pnode, slnew,
                         oh.make_node('Sub', [labI, sSl], [df], name=df),
                         oh.make_node('Abs', [df], [ab], name=ab),
                         oh.make_node('Min', [ab, ONE], [mn], name=mn),
                         oh.make_node('Sub', [ONE, mn], [ind], name=ind),
                         oh.make_node('Mul', [rs3, ind], [rn], name=rn)]
            for i, x in enumerate(sub.input):
                if x == rsout:
                    sub.input[i] = rn
            kill |= {id(m1), id(m2), id(rs), id(pn)}
            padx = prod.get(pn.input[0])
            if padx and padx.op_type == 'Pad' and xin in padx.input:
                kill.add(id(padx))
    if not newnodes:
        raise RuntimeError("no blocks matched")
    kept = [nd for nd in g.node if id(nd) not in kill]
    seeds = set(init.name for init in g.initializer) | {'input'}
    ordered = _topo(kept + setup + newnodes, seeds)
    del g.node[:]; g.node.extend(ordered)
    del g.value_info[:]
    return m


# --------------------------------------------------------------------------- #
# detection: fingerprint train inputs -> task number                          #
# --------------------------------------------------------------------------- #
def _fp(train):
    return tuple((tuple(np.asarray(e["input"]).shape),
                  np.asarray(e["input"], dtype=np.int64).tobytes())
                 for e in train)


def _build_index():
    idx = {}
    try:
        from ng_utils_shim import tasks_dir
        import json
        td = tasks_dir()
    except Exception:
        return idx
    for t in _MINE:
        p = os.path.join(_ONNX_DIR, f"task{t:03d}.onnx")
        jp = td / f"task{t:03d}.json"
        if not (os.path.exists(p) and jp.exists()):
            continue
        try:
            ex = json.load(open(jp))
            idx[_fp(ex["train"])] = t
        except Exception:
            continue
    return idx


_INDEX = _build_index()


def _has_f32(model):
    for init in model.graph.initializer:
        if init.data_type == TP.FLOAT:
            return True
    for n in model.graph.node:
        if n.op_type == 'Constant':
            for a in n.attribute:
                if a.name == 'value' and a.t.data_type == TP.FLOAT:
                    return True
    return False


def candidates(examples):
    train = examples.get("train", [])
    t = _INDEX.get(_fp(train))
    if t is None:
        return
    path = os.path.join(_ONNX_DIR, f"task{t:03d}.onnx")
    if not os.path.exists(path):
        return
    inc = onnx.load(path)
    yield (f"inc{t}", inc)
    if t == 110:
        try:
            yield ("lab110", labelize_110(onnx.load(path)))
        except Exception:
            pass
    if _has_f32(inc):
        try:
            yield (f"f16f{t}", to_fp16(onnx.load(path)))
        except Exception:
            pass
        try:
            yield (f"f16b{t}", to_fp16_bnd(onnx.load(path)))
        except Exception:
            pass
