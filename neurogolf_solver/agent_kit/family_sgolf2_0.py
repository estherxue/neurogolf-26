"""family_sgolf2_0 -- ALGORITHMIC golf: replace an O(N^2) pairwise-rank block with a
cheap O(N) 2-D prefix-count.

Several accepted baselines compute, for every cell i, its RANK among the non-background
cells under a position key (used to gather scattered dots into a compacted output, e.g.
"read the coloured dots in column-major order and lay them into a small grid").  The
baseline materialises this rank with a pairwise comparison matrix:

    A = Reshape(key, [N,1]);  B = Reshape(key, [1,N])
    R = ReduceSum(Cast(Less(B, A)), axis=1)      # rank_i = #{ j : key_j < key_i }

with N = H*W = 900, so the [900,900] intermediates (Less + its Cast) cost 2*3.24 MB and
dominate the whole model (~6.5 MB of ~7 MB).

When the key is a monotone POSITION index over the HxW grid (key = c*H + r for a
column-major scan, or r*W + c for row-major), the rank of a non-background cell is exactly
a separable 2-D prefix count of the presence mask M:

    column-major:  rank[r,c] = (#nonzero in columns < c) + (#nonzero in column c, rows < r)
    row-major:     rank[r,c] = (#nonzero in rows    < r) + (#nonzero in row    r, cols < c)

Both are computed with two tiny [H,H]/[W,W] triangular matmuls over M, so the [N,N]
intermediates vanish (3.24 MB -> a few kB each).  Background cells get a wrong rank but are
masked out downstream exactly as before, so the transform is byte-for-byte equivalent.

We fetch the baseline from its own family module (so this stays self-contained and
re-derives the rule on every grid), pattern-match the rank block, verify the key is an
arange position index, splice in the prefix version, and let the shared harness validate
EXACTness on train+test+arc-gen.  Anything that does not match / does not stay exact is
dropped, so this is pure upside and private-safe (no grid-size assumption; H=W=30 always).
"""
from __future__ import annotations

import importlib

import numpy as np
import onnx
from onnx import helper as oh, numpy_helper as nh, TensorProto as TP

# Baseline family modules that may carry an O(N^2) pairwise-rank block for one of our
# target tasks.  We try each, patch any rank block we find, and yield the cheaper model.
_SOURCE_FAMILIES = ["family_golf7_3"]


def _as_int_shape(inits, name):
    t = inits.get(name)
    if t is None:
        return None
    return nh.to_array(t).astype(np.int64).ravel().tolist()


def _detect_orientation(idx, H, W):
    """idx: [H,W] float position-index initializer. Return 'col'|'row'|None."""
    col = (np.arange(W)[None, :] * H + np.arange(H)[:, None]).astype(np.float64)
    row = (np.arange(H)[:, None] * W + np.arange(W)[None, :]).astype(np.float64)
    if idx.shape == (H, W) and np.array_equal(idx, col):
        return "col"
    if idx.shape == (H, W) and np.array_equal(idx, row):
        return "row"
    return None


def _reachable(nodes, outputs):
    """Return set of node indices whose outputs are (transitively) needed for `outputs`."""
    producer = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            producer[o] = i
    need = set()
    stack = list(outputs)
    while stack:
        t = stack.pop()
        i = producer.get(t)
        if i is None or i in need:
            continue
        need.add(i)
        stack.extend(nodes[i].input)
    return need


def _patch_rank(model):
    """Find an O(N^2) pairwise-rank block and replace it with a 2-D prefix count.
    Returns a new ModelProto, or None if no patchable block is present."""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    nodes = list(g.node)
    by_out = {o: n for n in nodes for o in n.output}

    # locate: ReduceSum(axis=[1]) <- Cast <- Less(B,A) ; A=Reshape(K,[N,1]) B=Reshape(K,[1,N])
    for rs in nodes:
        if rs.op_type != "ReduceSum":
            continue
        axes = next((list(a.ints) for a in rs.attribute if a.name == "axes"), None)
        if axes != [1]:
            continue
        cast = by_out.get(rs.input[0])
        if cast is None or cast.op_type != "Cast":
            continue
        cmp = by_out.get(cast.input[0])
        if cmp is None or cmp.op_type != "Less":
            continue
        b_in, a_in = cmp.input[0], cmp.input[1]   # Less(B < A) -> rank = #{key_j < key_i}
        rA, rB = by_out.get(a_in), by_out.get(b_in)
        if rA is None or rB is None or rA.op_type != "Reshape" or rB.op_type != "Reshape":
            continue
        key = rA.input[0]
        if rB.input[0] != key:
            continue
        shpA = _as_int_shape(inits, rA.input[1])
        shpB = _as_int_shape(inits, rB.input[1])
        if not shpA or not shpB:
            continue
        N = int(np.prod([d for d in shpA if d != -1])) if -1 not in shpA else None
        # shpA == [N,1], shpB == [1,N]
        if not (len(shpA) == 2 and shpA[1] == 1 and len(shpB) == 2 and shpB[0] == 1):
            continue
        N = shpA[0]

        # key producer must be Add(Mul(mask,IDX), Mul(1-mask, SENT)); find IDX arange + mask.
        kn = by_out.get(key)
        if kn is None or kn.op_type != "Add":
            continue
        idx_name = mask_name = None
        for side in kn.input:
            mn = by_out.get(side)
            if mn is None or mn.op_type != "Mul":
                continue
            for x in mn.input:
                if x in inits:
                    arr = nh.to_array(inits[x])
                    if arr.size == N:  # candidate position index
                        idx_name = x
                        mask_name = mn.input[1] if mn.input[0] == x else mn.input[0]
        if idx_name is None:
            continue
        kdt = nh.to_array(inits[idx_name]).dtype   # match baseline dtype (e.g. float16)
        idxarr = nh.to_array(inits[idx_name]).astype(np.float64)
        # infer H,W from key/mask tensor shape via value_info; assume square 30x30 grid
        H = W = int(round(N ** 0.5))
        if H * W != N:
            continue
        orient = _detect_orientation(idxarr.reshape(H, W), H, W)
        if orient is None:
            continue

        # Build replacement: rank[1,1,H,W] via prefix, then Reshape to [N,1] (reuse shpA).
        pre = f"sg2_{rs.output[0]}_"
        new_inits = []
        new_nodes = []
        if orient == "col":
            B = np.triu(np.ones((W, W), kdt), 1)        # B[c',c]=1 if c'<c (strict upper)
            L = np.tril(np.ones((H, H), kdt), -1)        # L[r,r']=1 if r'<r
            new_inits += [nh.from_array(L, pre + "L"), nh.from_array(B, pre + "B")]
            new_nodes += [
                oh.make_node("ReduceSum", [mask_name], [pre + "cs"], axes=[2], keepdims=1),
                oh.make_node("MatMul", [pre + "cs", pre + "B"], [pre + "bef"]),
                oh.make_node("MatMul", [pre + "L", mask_name], [pre + "wit"]),
                oh.make_node("Add", [pre + "wit", pre + "bef"], [pre + "rank"]),
            ]
        else:  # row-major
            B = np.tril(np.ones((H, H), kdt), -1)         # B[r,r']=1 if r'<r (before rows)
            L = np.triu(np.ones((W, W), kdt), 1)          # L[c',c]=1 if c'<c (within row)
            new_inits += [nh.from_array(L, pre + "L"), nh.from_array(B, pre + "B")]
            new_nodes += [
                oh.make_node("ReduceSum", [mask_name], [pre + "rs2"], axes=[3], keepdims=1),
                oh.make_node("MatMul", [pre + "B", pre + "rs2"], [pre + "bef"]),
                oh.make_node("MatMul", [mask_name, pre + "L"], [pre + "wit"]),
                oh.make_node("Add", [pre + "wit", pre + "bef"], [pre + "rank"]),
            ]
        # Match the original rank dtype (baseline casts the pairwise bool up, e.g. to FLOAT32)
        # before feeding the downstream compare, then Reshape to the ReduceSum output name.
        out_dt = next((a.i for a in cast.attribute if a.name == "to"), None)
        rank_src = pre + "rank"
        if out_dt is not None:
            new_nodes.append(oh.make_node("Cast", [pre + "rank"], [pre + "rankc"], to=out_dt))
            rank_src = pre + "rankc"
        new_nodes.append(oh.make_node("Reshape", [rank_src, rA.input[1]], [rs.output[0]]))

        # Assemble: keep all nodes except the removed rank block, insert new nodes, DCE.
        removed = {id(rA), id(rB), id(cmp), id(cast), id(rs)}
        kept = [n for n in nodes if id(n) not in removed]
        # insert new nodes right after the mask producer so topo order holds
        mask_prod = by_out.get(mask_name)
        insert_at = 0
        if mask_prod is not None:
            for i, n in enumerate(kept):
                if n is mask_prod:
                    insert_at = i + 1
                    break
        kept[insert_at:insert_at] = new_nodes

        all_inits = list(g.initializer) + new_inits
        # dead-node + dead-init elimination
        need = _reachable(kept, [o.name for o in g.output])
        kept = [n for i, n in enumerate(kept) if i in need]
        used = {x for n in kept for x in n.input}
        all_inits = [t for t in all_inits if t.name in used]

        ng = oh.make_graph(kept, g.name, list(g.input), list(g.output), all_inits)
        m2 = oh.make_model(ng, ir_version=model.ir_version, opset_imports=model.opset_import)
        try:
            onnx.checker.check_model(m2, full_check=True)
        except Exception:
            return None
        return m2
    return None


def candidates(examples):
    out = []
    for fam in _SOURCE_FAMILIES:
        try:
            mod = importlib.import_module(fam)
        except Exception:
            continue
        try:
            base = list(mod.candidates(examples) or [])
        except Exception:
            base = []
        for name, model in base:
            try:
                m2 = _patch_rank(model)
            except Exception:
                m2 = None
            if m2 is not None:
                out.append((f"{name}_prefixrank", m2))
    return out
