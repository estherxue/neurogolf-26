"""family_pgolf_2 — cheaper generalizing solvers for a slice of golf targets.

Slice = golf_targets.json[2::4]. Only fires on those task-shapes.

Implemented technique
---------------------
Fixed-size spatial COPY-MAP: when a task has a single input size (Hi x Wi) and a
single output size (Ho x Wo) across all examples, and every output cell is a
content-independent copy of one fixed input cell (i.e. output[p] == input[q(p)]
for the SAME q(p) in every example), the whole task is a static spatial
permutation/copy. It is realized as Reshape->Gather->Reshape on the flattened
30x30 grid. This is exact and generalizes by construction (the map is
position-only and is verified to hold on the held-out arc-gen split before
emitting).

Padding-safe: output cells outside the Ho x Wo content region, and any output
cell that is background(0) in every example, are mapped to a guaranteed-zero
input cell (the input's padding region), so they never pick up spurious colour.
"""
import numpy as np
import onnx
from onnx import helper as oh
from builders import _model
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64


def _pad30(g):
    return np.pad(g, ((0, HEIGHT - g.shape[0]), (0, WIDTH - g.shape[1])))


def _gather_model(idx, R, C):
    """Slice input to the top-left RxC box, Reshape [1,10,R*C], Gather axis=2 by
    idx (box->box flat map), Reshape [1,10,R,C], Pad back to [1,10,30,30]."""
    cs = oh.make_tensor("cs", INT64, [2], [0, 0])
    ce = oh.make_tensor("ce", INT64, [2], [R, C])
    ca = oh.make_tensor("ca", INT64, [2], [2, 3])
    shp1 = oh.make_tensor("shp1", INT64, [3], [1, 10, R * C])
    shp2 = oh.make_tensor("shp2", INT64, [4], [1, 10, R, C])
    ii = oh.make_tensor("gidx", INT64, [len(idx)], [int(v) for v in idx])
    n0 = oh.make_node("Slice", ["input", "cs", "ce", "ca"], ["box"])
    n1 = oh.make_node("Reshape", ["box", "shp1"], ["flat"])
    n2 = oh.make_node("Gather", ["flat", "gidx"], ["gath"], axis=2)
    n3 = oh.make_node("Reshape", ["gath", "shp2"], ["boxo"])
    n4 = oh.make_node("Pad", ["boxo"], ["output"], mode="constant", value=0.0,
                      pads=[0, 0, 0, 0, 0, 0, HEIGHT - R, WIDTH - C])
    return _model([n0, n1, n2, n3, n4], [cs, ce, ca, shp1, shp2, ii])


def _copymap(prs):
    """Detect a content-independent spatial copy map. Returns (idx_box, R, C) where
    idx_box maps each cell of the top-left RxC box (output side) to a flat position
    inside the same RxC box (input side), or None if no such map exists."""
    Hi, Wi = prs[0][0].shape
    Ho, Wo = prs[0][1].shape
    n = len(prs)
    IN = np.stack([_pad30(a) for a, _ in prs]).reshape(n, HEIGHT * WIDTH)
    OUT = np.stack([_pad30(b) for _, b in prs]).reshape(n, HEIGHT * WIDTH)
    buckets = {}
    for q in range(HEIGHT * WIDTH):
        buckets.setdefault(tuple(IN[:, q]), []).append(q)
    # first resolve a full-grid source for every output-content cell
    src = {}
    max_r, max_c = Ho - 1, Wo - 1
    for r in range(Ho):
        for c in range(Wo):
            p = r * WIDTH + c
            vec = OUT[:, p]
            if not vec.any():
                continue  # background output cell handled via forced zero later
            cand = buckets.get(tuple(vec))
            if not cand:
                return None
            q = cand[0]
            qr, qc = divmod(q, WIDTH)
            src[(r, c)] = (qr, qc)
            max_r = max(max_r, qr)
            max_c = max(max_c, qc)
    R, C = max_r + 1, max_c + 1
    if R > HEIGHT or C > WIDTH:
        return None
    # guaranteed-zero source inside the RxC box (a box cell outside input content)
    zero_src = None
    for r in range(R):
        for c in range(C):
            if r >= Hi or c >= Wi:
                zero_src = r * C + c
                break
        if zero_src is not None:
            break
    idx = np.zeros(R * C, dtype=np.int64)
    for r in range(R):
        for c in range(C):
            bp = r * C + c
            if (r, c) in src:
                qr, qc = src[(r, c)]
                idx[bp] = qr * C + qc
            else:
                # output cell is background (0) in every example -> force zero
                if zero_src is None:
                    return None
                idx[bp] = zero_src
    return idx, R, C


def _apply(idx, R, C, g):
    box = _pad30(g)[:R, :C].reshape(R * C)
    return box[idx].reshape(R, C)


def candidates(ex):
    train = [(np.array(e["input"]), np.array(e["output"])) for e in ex.get("train", []) + ex.get("test", [])]
    arc = [(np.array(e["input"]), np.array(e["output"])) for e in ex.get("arc-gen", [])]
    if not train:
        return []
    ish = {a.shape for a, _ in train}
    osh = {b.shape for _, b in train}
    if len(ish) != 1 or len(osh) != 1:
        return []
    Hi, Wi = next(iter(ish))
    Ho, Wo = next(iter(osh))
    if max(Hi, Wi, Ho, Wo) > 30 or min(Hi, Wi, Ho, Wo) < 1:
        return []
    res = _copymap(train)
    if res is None:
        return []
    idx, R, C = res
    # generalization self-check on held-out arc-gen (fit used train+test only)
    for a, b in arc:
        if a.shape != (Hi, Wi) or b.shape != (Ho, Wo):
            return []
        pred = _apply(idx, R, C, a)[:Ho, :Wo]
        if not np.array_equal(pred, b):
            return []
    return [("copymap", _gather_model(idx, R, C))]
