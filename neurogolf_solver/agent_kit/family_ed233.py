"""family_ed233 - ENRICHED-ARSENAL deep-rebuild attempt for task233 (verify_97a05b5b).

RESULT: GENUINELY FLOORED.  No model that satisfies ALL of the task's hard gates
can beat the incumbent's *reported* 14.55.  The reason is not lack of arsenal
tricks - it is that the incumbent's 14.55 is not a legitimate, reproducible score.

============================================================================
What the task is
============================================================================
A solid red(=2) square carries bgc(=0) "anti-shape" holes; each hole is one D4
orientation of a scattered 3x3 marker (unique minority colour `col`, distinct
hole-count 4..8 -> colour<->count is a bijection).  Output = the square, every
hole filled by the matching marker's 3x3 (obj cells -> red, frame cells -> col)
in the hole's orientation.  Placement is a FORCED EXACT COVER (family_sr233
proved independent per-marker matching is NOT equivalent: adjacent 3x3 blocks and
count-4/5 sprites with 2x3 / 3x2 obj bboxes need the joint D4 exact-cover).

============================================================================
Dissection of the incumbent  out_blend10/onnx/task233.onnx
============================================================================
238 nodes, opset 13.  It is NOT the K=6 D4-propagation net of family_r233
(that one measures memory=8_629_388 -> 9.03 pts).  It is a *different, indexed*
net: value-image Conv -> u8; a 3x3 red/colour Conv that (via Greater) marks the
<=5 sprite windows; TopK+Gather pull each 3x3 marker; an 18x18 correlation +
TopK picks placements; ScatterElements paints the frames; fold_A/fold_B set the
interior out-of-bounds strip to 10; Pad+Equal -> one-hot.  Its static memory is
exactly 33_637 and params 763 -> cost 34_400 -> 25-ln(34400)=14.5543 ~= 14.55.
Per-dtype bytes: f32 6736, u8 10252, bool 6149, f16 7988, i32 2376, i64 136.

============================================================================
DEFECT 1 - the incumbent does not load in the mandated ORT 1.23.2
============================================================================
score_network runs under ORT 1.23.2 (hard gate a).  That build has NO uint8
kernel for Min or Max (verified: Min/Max u8 -> NOT_IMPLEMENTED; only
f32/f16/i32/i64 Min/Max exist).  The incumbent uses:
  * safe_name_82 = Min(x_u8, 1_u8)            (the 8-flood clamp)
  * safe_name_301 = Max(scatter_u8, fold_A, fold_B)   (colour/oob overlay)
so InferenceSession construction fails ("Could not find an implementation for
Min(13)").  i.e. on the grader named by the task the incumbent scores 0, not
14.55; its report.json 14.55 was produced by a DIFFERENT ORT.

I can make it load with two dtype-faithful rewrites (see _load_and_patch):
  * Min(x,1) u8  ->  Clip(x, 0, 1) u8            (u8 Clip IS implemented; free)
  * Max(sc, fold_A[row-oob->10], fold_B[col-oob->10]) u8
        == Where(row_inb, Where(col_inb, sc, 10), 10)   (u8 Where IS implemented)
    reusing the existing row/col in-bounds masks (safe_name_99/100) and dropping
    the two fold Wheres.  This is the CHEAPEST correct rewrite; it still adds one
    unavoidable [20,20] combine tensor: +360 bytes net.
Result: LOADS, memory 33_997, params 763, cost 34_760 -> 14.5438 pts, and it is
EXACT on train+test (4/0) and the embedded arc-gen (262/0).  But 14.5438<14.5543.

Why the patched net cannot be pushed back below 33_637 (verified exhaustively):
  * no dead node outputs, no no-op Casts, no identity Reshape/Squeeze, no
    duplicate subexpressions (CSE), no unused initializers, no constant-foldable
    intermediates - the author already minimised all of these.
  * the two fat f32 Convs (value-image 3600 + red/colour 3136) are optimal: the
    grader feeds FLOAT[1,10,30,30] and ORT rejects any non-float feed, so the
    first op that touches `input` must emit f32; casting `input` to f16/u8 to get
    cheaper Conv outputs costs 18000/9000 bytes, dwarfing the 3.4K saved.
  * every remaining f16 tensor feeds a TopK, which has no uint8 kernel, so it
    cannot be down-dtyped.
So making the incumbent merely LOAD already costs it the tie (14.544 < 14.55),
and there is no offsetting shave.

============================================================================
DEFECT 2 - the incumbent is OVERFIT: it FAILS the fresh-generator gate (c)
============================================================================
The base grader (evaluate.py) only checks train+test + the 262 EMBEDDED arc-gen
pairs (which the net passes).  The task additionally requires exactness on >=3000
FRESH generator samples (anti-overfit / hidden-set proxy).  On 900 fresh samples
the (loadable-patched) incumbent is WRONG on 7 (~0.8%): it misplaces a frame and
leaves holes unfilled (e.g. a 29x25 input with 4-5 sprites yields spurious `333`
and unfilled `000` cells).  The exact reference _ref below is 900/900 on the same
samples, so these are genuine incumbent BUGS, not inherent ambiguity - the indexed
net skips part of the joint exact-cover and only happens to match the embedded 262
it was tuned on.  A net that passes gate (c) must implement the full D4 exact
cover, i.e. family_r233 (_ref), which is exact everywhere but costs 8.63M -> 9.03.

============================================================================
Why NO valid model beats 14.55 (the floor)
============================================================================
* A gate-(c)-valid model must do the joint D4 exact-cover placement.  Folded to
  channels that is a per-(colour,orientation) 3x3 correlation over the 30x30 grid.
  Even the leanest form - uint8, rotations-only (the data uses no reflections),
  all 8 colours - is [1, 8*4=32, 30, 30] = 28_800 B in ONE named tensor, and the
  grader's memory is the SUM of every named intermediate (calculate_memory).  A
  single pass needs the correlation AND the matched/footprint maps (2-3 such
  tensors) and the exact cover needs up to 5 passes; the sum is hundreds of KB.
  A full-D4 uint8 stack [1,64,30,30] is already 57_600 B > the whole 34_400
  budget.  So the correct placement core cannot be reproduced under budget -
  exactly family_sr233's conclusion, now reinforced by DEFECT 2.
* The enriched arsenal (runtime-bias QLinearConv binary detect, adjoint-conv
  paint, static Gather, uint8/bool masks) shrinks per-op CONSTANTS but cannot
  change that a correct D4 exact-cover materialises >34_400 B of named 30x30
  correlation activations.  Binary-detect turns one Conv+Equal+Cast chain into
  one u8 map; it does not remove the map, and 6 passes x [1,>=32,30,30] u8 stays
  far over budget.
* The only sub-34_400 net known (the incumbent's indexed one) achieves it ONLY
  by being incorrect on the generator (DEFECT 2).

Hence: strictly beating the reported 14.55 with a model that passes hard gates
(a) load-in-1.23.2, (b) exact train+test+embedded, AND (c) exact on fresh
generator samples is genuinely FLOORED.  candidates() emits nothing rather than
ship an overfit net (it would fail the hidden set) or a losing valid one.

The reproducible loadable-patched incumbent (_load_and_patch) and the exact numpy
reference (_ref) are kept below so the finding can be re-verified.
"""
from __future__ import annotations

import os
import numpy as np

COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
NPASS = 6
_INCUMBENT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "out_blend10", "onnx", "task233.onnx")


# --------------------------------------------------------------------------- #
# Reproducible: load the incumbent and apply the two dtype-faithful rewrites   #
# that make it LOAD in ORT 1.23.2 (u8 Min->Clip, u8 Max->nested Where).        #
# This model is EXACT on train+test+embedded (14.544 pts) but is NOT gate-(c)  #
# valid (it inherits the incumbent's ~0.8% fresh-generator errors), so it is   #
# deliberately NOT returned by candidates().                                   #
# --------------------------------------------------------------------------- #
def _load_and_patch():
    import onnx
    from onnx import helper as oh
    m = onnx.load(_INCUMBENT)
    g = m.graph
    out = []
    for n in g.node:
        if n.op_type == "Min" and n.output[0] == "safe_name_82":
            # Min(safe_name_81, 1) == Clip(safe_name_81, 0, 1); 0=safe_name_36, 1=safe_name_51
            out.append(oh.make_node("Clip",
                                    ["safe_name_81", "safe_name_36", "safe_name_51"],
                                    ["safe_name_82"]))
            continue
        if n.output[0] in ("fold_A", "fold_B"):
            continue  # dropped; use safe_name_99/100 (row/col in-bounds) directly
        if n.op_type == "Max" and n.output[0] == "safe_name_301":
            # Max(sc, fold_A(row-oob->10), fold_B(col-oob->10))
            #   == Where(row_inb, Where(col_inb, sc, 10), 10)   (safe_name_33 == 10)
            out.append(oh.make_node("Where",
                                    ["safe_name_100", "safe_name_300", "safe_name_33"],
                                    ["ed_colmask"]))
            out.append(oh.make_node("Where",
                                    ["safe_name_99", "ed_colmask", "safe_name_33"],
                                    ["safe_name_301"]))
            continue
        out.append(n)
    del g.node[:]
    g.node.extend(out)
    onnx.checker.check_model(m, full_check=True)
    return m


# =========================================================================== #
# numpy reference - the EXACT rule (mirror of family_r233._ref).               #
# 900/900 on fresh generator samples; used only to document DEFECT 2.         #
# =========================================================================== #
def _orient3(m, o):
    return np.rot90(m, o) if o < 4 else np.fliplr(np.rot90(m, o - 4))


def _dil8(m):
    p = np.pad(m.astype(bool), 1)
    out = np.zeros_like(m, bool)
    for di in range(3):
        for dj in range(3):
            out |= p[di:di + m.shape[0], dj:dj + m.shape[1]]
    return out


def _tight(x):
    ys, xs = np.where(x)
    return x[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _ref(a):
    V = np.asarray(a, int)
    if V.ndim != 2 or max(V.shape) > 30:
        return None
    H, W = V.shape
    kernels = {}
    objunion = np.zeros((H, W), bool)
    for c in COLORS:
        colc = (V == c)
        if not colc.any():
            continue
        thru = (V == 2) | (V == c)
        m = colc.copy()
        for _ in range(3):
            m = _dil8(m) & thru
        objunion |= m & (V == 2)
        ys, xs = np.where(m)
        if ys.max() - ys.min() > 2 or xs.max() - xs.min() > 2:
            return None
        K = np.zeros((3, 3), int)
        K[ys - ys.min(), xs - xs.min()] = V[ys, xs]
        kernels[c] = K
    if not kernels:
        return None
    sq2 = (V == 2) & ~objunion
    if not sq2.any():
        return None
    ys, xs = np.where(sq2)
    R0, R1, C0, C1 = ys.min(), ys.max(), xs.min(), xs.max()
    sgh, sgw = R1 - R0 + 1, C1 - C0 + 1
    inside = np.zeros((H, W), bool)
    inside[R0:R1 + 1, C0:C1 + 1] = True
    remaining = (V == 0) & inside
    info = {}
    for c, K in kernels.items():
        objb = (K == 2)
        tb = _tight(objb)
        s = sum(1 for o in range(8)
                if _orient3(objb, o).shape == objb.shape
                and np.array_equal(_tight(_orient3(objb, o)), tb))
        info[c] = s
    colimg = np.zeros((H, W), int)
    placed = set()
    for _p in range(NPASS):
        claimed = []
        for c, K in kernels.items():
            if c in placed:
                continue
            wins = []
            for o in range(8):
                objk = (_orient3(K, o) == 2)
                rel = list(zip(*np.where(objk)))
                ws = [(oy, ox) for oy in range(R0, R1 - 1) for ox in range(C0, C1 - 1)
                      if all(remaining[oy + ry, ox + rx] for ry, rx in rel)]
                wins.append(ws)
            tot = sum(len(w) for w in wins)
            if tot > 0 and tot == info[c]:
                claimed.append((c, K, wins))
        if not claimed:
            break
        for c, K, wins in claimed:
            placed.add(c)
            for o in range(8):
                ob = _orient3(K, o)
                for oy, ox in wins[o]:
                    for ry in range(3):
                        for rx in range(3):
                            if ob[ry, rx] == 2:
                                remaining[oy + ry, ox + rx] = False
            fo = next(o for o in range(8) if wins[o])
            oy, ox = wins[fo][0]
            ob = _orient3(K, fo)
            for ry in range(3):
                for rx in range(3):
                    if ob[ry, rx] != 2:
                        colimg[oy + ry, ox + rx] = c
    if len(placed) != len(kernels) or remaining.any():
        return None
    out = np.full((sgh, sgw), 2, int)
    ci = colimg[R0:R1 + 1, C0:C1 + 1]
    out[ci != 0] = ci[ci != 0]
    return out


# =========================================================================== #
def candidates(examples):
    # GENUINELY FLOORED (see module docstring):
    #   * the incumbent's 14.55 is unloadable in ORT 1.23.2 (u8 Min/Max) AND
    #     overfit (fails the fresh-generator gate ~0.8%);
    #   * the cheapest loadable+exact-on-embedded rebuild is 14.544 (< 14.55) and
    #     is itself gate-(c)-invalid;
    #   * a gate-(c)-valid net needs the full D4 exact-cover, whose named 30x30
    #     correlation activations exceed the 34_400 budget even in uint8.
    # No candidate strictly beats the incumbent under all hard gates, so emit none.
    return []
