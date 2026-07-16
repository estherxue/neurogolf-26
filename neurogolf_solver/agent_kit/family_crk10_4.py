"""family_crk10_4 — hardest remaining tasks (slice U[4::8] = 44,86,137,163,208,319).

Solvers (each gated by an exact-detector, validated EXACT on train+test+arc-gen):

  * task137 concentric_rings: input is 3 same-colour single-pixel dots, collinear and
    equally spaced along a diagonal (two opposite corners + centre). Output = concentric
    Chebyshev square outlines centred at the middle dot, at radii 0,s,2s,... (s = dot
    spacing), clipped to the HxW grid region.  Rule per cell:
        D = max(|r-Cr|, |c-Cc|);   fill colour F iff (D mod s == 0).
    where Cr=(minr+maxr)/2, Cc=(minc+maxc)/2, s=max(maxr-minr,maxc-minc)/2.
"""
from __future__ import annotations
import numpy as np
import onnx
from onnx import helper as oh

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _pairs(examples):
    return [(np.array(e["input"]), np.array(e["output"]))
            for e in examples.get("train", []) + examples.get("test", [])]


# ============================================================================
# task137 — concentric Chebyshev rings from 3 diagonal dots
# ============================================================================
def _t137_solve(grid):
    H, W = grid.shape
    ys, xs = np.where(grid != 0)
    if len(ys) == 0:
        return grid.copy()
    Fc = grid[ys[0], xs[0]]
    minr, maxr, minc, maxc = ys.min(), ys.max(), xs.min(), xs.max()
    Cr = (minr + maxr) / 2.0
    Cc = (minc + maxc) / 2.0
    s = max(maxr - minr, maxc - minc) / 2.0
    out = np.zeros_like(grid)
    R = np.arange(H)[:, None]
    C = np.arange(W)[None, :]
    D = np.maximum(np.abs(R - Cr), np.abs(C - Cc))
    ring = (np.mod(D + 1e-4, s) < 1e-2)
    out[ring] = Fc
    return out


def _t137_detect(prs):
    if not prs:
        return False
    for a, b in prs:
        if a.shape != b.shape:
            return False
        ys, xs = np.where(a != 0)
        cols = set(int(a[y, x]) for y, x in zip(ys, xs))
        if len(cols) != 1:            # dots all one colour
            return False
        if len(ys) < 2:
            return False
        if not (_t137_solve(a) == b).all():
            return False
    return True


def _t137_build():
    N = oh.make_node
    Rg = (np.arange(HEIGHT, dtype=np.float32)[None, None, :, None]
          * np.ones((1, 1, 1, WIDTH), np.float32))
    Cg = (np.ones((1, 1, HEIGHT, 1), np.float32)
          * np.arange(WIDTH, dtype=np.float32)[None, None, None, :])
    inits = [
        oh.make_tensor("Rgrid", F, [1, 1, HEIGHT, WIDTH], Rg.flatten().tolist()),
        oh.make_tensor("Cgrid", F, [1, 1, HEIGHT, WIDTH], Cg.flatten().tolist()),
        oh.make_tensor("BIG", F, [1, 1, 1, 1], [1e6]),
        oh.make_tensor("HALF", F, [1, 1, 1, 1], [0.5]),
        oh.make_tensor("ONE", F, [1, 1, 1, 1], [1.0]),
        oh.make_tensor("sl1", INT64, [1], [1]),
        oh.make_tensor("sl10", INT64, [1], [10]),
        oh.make_tensor("slax", INT64, [1], [1]),
    ]
    nodes = []
    nodes.append(N("ReduceSum", ["input"], ["gmask"], axes=[1], keepdims=1))
    nodes.append(N("Slice", ["input", "sl1", "sl10", "slax"], ["inp9"]))
    nodes.append(N("ReduceSum", ["inp9"], ["mask"], axes=[1], keepdims=1))         # dots
    nodes.append(N("ReduceMax", ["inp9"], ["present9"], axes=[2, 3], keepdims=1))  # [1,9,1,1]
    # penalty field for cells that are NOT dots
    nodes.append(N("Sub", ["ONE", "mask"], ["invmask"]))
    nodes.append(N("Mul", ["invmask", "BIG"], ["penBIG"]))
    nodes.append(N("Mul", ["Rgrid", "mask"], ["rmask"]))
    nodes.append(N("Mul", ["Cgrid", "mask"], ["cmask"]))
    nodes.append(N("Add", ["rmask", "penBIG"], ["rminf"]))
    nodes.append(N("Sub", ["rmask", "penBIG"], ["rmaxf"]))
    nodes.append(N("Add", ["cmask", "penBIG"], ["cminf"]))
    nodes.append(N("Sub", ["cmask", "penBIG"], ["cmaxf"]))
    nodes.append(N("ReduceMin", ["rminf"], ["minr"], axes=[2, 3], keepdims=1))
    nodes.append(N("ReduceMax", ["rmaxf"], ["maxr"], axes=[2, 3], keepdims=1))
    nodes.append(N("ReduceMin", ["cminf"], ["minc"], axes=[2, 3], keepdims=1))
    nodes.append(N("ReduceMax", ["cmaxf"], ["maxc"], axes=[2, 3], keepdims=1))
    nodes.append(N("Add", ["minr", "maxr"], ["rsum"]))
    nodes.append(N("Add", ["minc", "maxc"], ["csum"]))
    nodes.append(N("Mul", ["HALF", "rsum"], ["Cr"]))
    nodes.append(N("Mul", ["HALF", "csum"], ["Cc"]))
    nodes.append(N("Sub", ["maxr", "minr"], ["rspan"]))
    nodes.append(N("Sub", ["maxc", "minc"], ["cspan"]))
    nodes.append(N("Max", ["rspan", "cspan"], ["span"]))
    nodes.append(N("Mul", ["HALF", "span"], ["s"]))
    # Chebyshev distance field
    nodes.append(N("Sub", ["Rgrid", "Cr"], ["dRr"]))
    nodes.append(N("Sub", ["Cgrid", "Cc"], ["dCc"]))
    nodes.append(N("Abs", ["dRr"], ["aRr"]))
    nodes.append(N("Abs", ["dCc"], ["aCc"]))
    nodes.append(N("Max", ["aRr", "aCc"], ["D"]))
    nodes.append(N("Mod", ["D", "s"], ["modv"], fmod=1))
    nodes.append(N("Less", ["modv", "HALF"], ["ringb"]))
    nodes.append(N("Cast", ["ringb"], ["ring"], to=F))
    nodes.append(N("Mul", ["ring", "gmask"], ["ring_in"]))
    nodes.append(N("Mul", ["ring_in", "present9"], ["colored"]))                    # [1,9,30,30]
    nodes.append(N("Sub", ["ONE", "ring_in"], ["notring"]))
    nodes.append(N("Mul", ["gmask", "notring"], ["ch0"]))
    nodes.append(N("Concat", ["ch0", "colored"], ["output"], axis=1))
    return _model(nodes, inits)


# ============================================================================
# task163 — copy the cell holding the unique colour-4 marker to the meta-cell
# indexed by the marker's within-cell position (3x3 grid of 3x3 cells, 5-seps).
# ============================================================================
def _t163_solve(grid):
    ys, xs = np.where(grid == 4)
    if len(ys) != 1:
        return None
    R4, C4 = int(ys[0]), int(xs[0])
    pr, pc = R4 % 4, C4 % 4
    if pr == 3 or pc == 3:
        return None
    Sr, Sc = R4 - pr, C4 - pc
    Dr, Dc = 4 * pr, 4 * pc
    src = grid[Sr:Sr + 3, Sc:Sc + 3]
    if src.shape != (3, 3):
        return None
    out = np.zeros_like(grid)
    out[grid == 5] = 5
    out[Dr:Dr + 3, Dc:Dc + 3] = src
    return out


def _t163_detect(prs):
    if not prs:
        return False
    for a, b in prs:
        if a.shape != b.shape or a.shape != (11, 11):
            return False
        if int((a == 4).sum()) != 1:
            return False
        p = _t163_solve(a)
        if p is None or not (p == b).all():
            return False
    return True


def _t163_build():
    N = oh.make_node
    Rg = (np.arange(HEIGHT, dtype=np.float32)[None, None, :, None]
          * np.ones((1, 1, 1, WIDTH), np.float32))
    Cg = (np.ones((1, 1, HEIGHT, 1), np.float32)
          * np.arange(WIDTH, dtype=np.float32)[None, None, None, :])
    inits = [
        oh.make_tensor("Rgrid", F, [1, 1, HEIGHT, WIDTH], Rg.flatten().tolist()),
        oh.make_tensor("Cgrid", F, [1, 1, HEIGHT, WIDTH], Cg.flatten().tolist()),
        oh.make_tensor("ONE", F, [1, 1, 1, 1], [1.0]),
        oh.make_tensor("FOUR", F, [1, 1, 1, 1], [4.0]),
        oh.make_tensor("NHALF", F, [1, 1, 1, 1], [-0.5]),
        oh.make_tensor("P25", F, [1, 1, 1, 1], [2.5]),
        oh.make_tensor("HALF", F, [1, 1, 1, 1], [0.5]),
        oh.make_tensor("s4", INT64, [1], [4]),
        oh.make_tensor("s5", INT64, [1], [5]),
        oh.make_tensor("s6", INT64, [1], [6]),
        oh.make_tensor("s1", INT64, [1], [1]),
        oh.make_tensor("s10", INT64, [1], [10]),
        oh.make_tensor("sax", INT64, [1], [1]),
    ]
    nodes = []
    # locate the single colour-4 pixel
    nodes.append(N("Slice", ["input", "s4", "s5", "sax"], ["mask4"]))
    nodes.append(N("Mul", ["mask4", "Rgrid"], ["m4r"]))
    nodes.append(N("Mul", ["mask4", "Cgrid"], ["m4c"]))
    nodes.append(N("ReduceSum", ["m4r"], ["R4"], axes=[2, 3], keepdims=1))
    nodes.append(N("ReduceSum", ["m4c"], ["C4"], axes=[2, 3], keepdims=1))
    nodes.append(N("Mod", ["R4", "FOUR"], ["pr"], fmod=1))
    nodes.append(N("Mod", ["C4", "FOUR"], ["pc"], fmod=1))
    nodes.append(N("Sub", ["R4", "pr"], ["Sr"]))
    nodes.append(N("Sub", ["C4", "pc"], ["Sc"]))
    nodes.append(N("Mul", ["pr", "FOUR"], ["Dr"]))
    nodes.append(N("Mul", ["pc", "FOUR"], ["Dc"]))
    # row-selection matrix L[o,i]=1 iff i in [Sr,Sr+2] and o==Dr+(i-Sr)
    #   (dim2=o -> Rgrid, dim3=i -> Cgrid)
    nodes.append(N("Sub", ["Rgrid", "Cgrid"], ["oi"]))          # o - i
    nodes.append(N("Sub", ["oi", "Dr"], ["oi1"]))
    nodes.append(N("Add", ["oi1", "Sr"], ["Ldiff"]))            # o-i-Dr+Sr
    nodes.append(N("Abs", ["Ldiff"], ["Ldabs"]))
    nodes.append(N("Less", ["Ldabs", "HALF"], ["Lmb"]))
    nodes.append(N("Sub", ["Cgrid", "Sr"], ["Lso"]))            # i - Sr
    nodes.append(N("Greater", ["Lso", "NHALF"], ["Llo"]))
    nodes.append(N("Less", ["Lso", "P25"], ["Lhi"]))
    nodes.append(N("Cast", ["Lmb"], ["Lmf"], to=F))
    nodes.append(N("Cast", ["Llo"], ["Llof"], to=F))
    nodes.append(N("Cast", ["Lhi"], ["Lhif"], to=F))
    nodes.append(N("Mul", ["Lmf", "Llof"], ["Lt"]))
    nodes.append(N("Mul", ["Lt", "Lhif"], ["L"]))               # [1,1,30,30]
    # col-selection matrix R[j,jo]=1 iff j in [Sc,Sc+2] and jo==Dc+(j-Sc)
    #   (dim2=j -> Rgrid, dim3=jo -> Cgrid)
    nodes.append(N("Sub", ["Cgrid", "Rgrid"], ["jj"]))          # jo - j
    nodes.append(N("Sub", ["jj", "Dc"], ["jj1"]))
    nodes.append(N("Add", ["jj1", "Sc"], ["Rdiff"]))            # jo-j-Dc+Sc
    nodes.append(N("Abs", ["Rdiff"], ["Rdabs"]))
    nodes.append(N("Less", ["Rdabs", "HALF"], ["Rmb"]))
    nodes.append(N("Sub", ["Rgrid", "Sc"], ["Rso"]))            # j - Sc
    nodes.append(N("Greater", ["Rso", "NHALF"], ["Rlo"]))
    nodes.append(N("Less", ["Rso", "P25"], ["Rhi"]))
    nodes.append(N("Cast", ["Rmb"], ["Rmf"], to=F))
    nodes.append(N("Cast", ["Rlo"], ["Rlof"], to=F))
    nodes.append(N("Cast", ["Rhi"], ["Rhif"], to=F))
    nodes.append(N("Mul", ["Rmf", "Rlof"], ["Rt"]))
    nodes.append(N("Mul", ["Rt", "Rhif"], ["Rmat"]))            # [1,1,30,30]
    # moved = L @ input @ R
    nodes.append(N("MatMul", ["L", "input"], ["Y"]))            # [1,10,30,30]
    nodes.append(N("MatMul", ["Y", "Rmat"], ["moved"]))         # [1,10,30,30]
    # reassemble output channels
    nodes.append(N("Slice", ["input", "s5", "s6", "sax"], ["sep"]))       # ch5
    nodes.append(N("ReduceSum", ["input"], ["gmask"], axes=[1], keepdims=1))
    nodes.append(N("Slice", ["moved", "s1", "s10", "sax"], ["moved19"]))
    nodes.append(N("ReduceSum", ["moved19"], ["mnb"], axes=[1], keepdims=1))
    nodes.append(N("Sub", ["ONE", "sep"], ["t1"]))
    nodes.append(N("Sub", ["t1", "mnb"], ["t2"]))
    nodes.append(N("Mul", ["gmask", "t2"], ["out0"]))
    nodes.append(N("Slice", ["moved", "s1", "s5", "sax"], ["moved14"]))
    nodes.append(N("Slice", ["moved", "s6", "s10", "sax"], ["moved69"]))
    nodes.append(N("Concat", ["out0", "moved14", "sep", "moved69"], ["output"], axis=1))
    return _model(nodes, inits)


# ============================================================================
def candidates(examples):
    prs = _pairs(examples)
    out = []
    if _t137_detect(prs):
        try:
            out.append(("concentric_rings", _t137_build()))
        except Exception:
            pass
    if _t163_detect(prs):
        try:
            out.append(("marker_copy_cell", _t163_build()))
        except Exception:
            pass
    return out
