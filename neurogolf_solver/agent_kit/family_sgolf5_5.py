"""family_sgolf5_5 -- cheaper EXACT solvers via a GRID-AGNOSTIC periodic overlay.

Round rule: NO CROPPING. Canvas stays [1,10,30,30] end to end.  We only cheapen
where the cheapening is byte-identical for ANY grid the generator makes.

Rule handled here (task 176, maskmap_per3x12, incumbent 14.48):
  The output equals the input plus a single fill colour F painted onto BACKGROUND
  (colour-0) cells that lie on a fixed, COLUMN-PERIODIC lattice.  Concretely there
  is a small period P and, per row r, a set of residues (c mod P) that get filled.
  This is grid-agnostic in width (the mask is a true periodicization, correct for
  any width up to 30) and the strip height is fixed by the generator.

Cheap graph (no crop, single-channel [1,1,30,30] intermediates, answer written
straight into the FREE `output` tensor via Concat):
    ch0   = Slice(input, channel 0)              # background plane
    fill  = MC * ch0                             # periodic mask gated to real bg
    ch0o  = ch0 - fill                           # bg loses the filled cells
    chF   = (input chF) + fill                   # fill colour gains them
    output= Concat[ch0o, <other input planes / zeros>, chF, ...]   -> "output"
Padding never leaks: ch0 is 0 outside the real grid, so fill is 0 there.

Detection re-derives (P, per-row residues, F) from train+test+arc-gen and fires
ONLY if a numpy mirror reproduces EVERY pair exactly; the harness then re-checks
EXACTNESS (the grader gate) on all splits.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, HEIGHT, WIDTH, CHANNELS

INT64 = onnx.TensorProto.INT64


# --------------------------------------------------------------------------- #
# detection                                                                    #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _detect_periodic_fill(prs):
    """Return (H0, F, P, pattern) where pattern[r] = set of residues (c mod P)
    filled, or None.  Rule: output = input + F on background cells whose
    (row, col mod P) is in pattern, exact on every pair."""
    if not prs:
        return None
    if any(a.shape != b.shape for a, b in prs):
        return None
    # constant height (a fixed-height strip generator)
    hs = {a.shape[0] for a, _ in prs}
    if len(hs) != 1:
        return None
    H0 = hs.pop()
    if H0 < 1 or H0 > HEIGHT:
        return None
    # additive: unchanged wherever input is non-background
    for a, b in prs:
        if not np.array_equal(b[a != 0], a[a != 0]):
            return None
    # single fill colour on (background -> non-background) cells
    Fs = set()
    for a, b in prs:
        m = (a == 0) & (b != 0)
        Fs |= set(b[m].tolist())
    if len(Fs) != 1:
        return None
    F = Fs.pop()
    if not (1 <= F <= 9):
        return None
    if not any((a != b).any() for a, b in prs):
        return None
    # smallest column period P whose per-row residue mask reproduces all pairs
    for P in range(1, 16):
        pattern = {r: set() for r in range(H0)}
        for a, b in prs:
            h, w = a.shape
            for r in range(h):
                for c in range(w):
                    if a[r, c] == 0 and b[r, c] == F:
                        pattern[r].add(c % P)
        ok = True
        for a, b in prs:
            h, w = a.shape
            pred = a.copy()
            for r in range(h):
                pr = pattern[r]
                for c in range(w):
                    if a[r, c] == 0 and (c % P) in pr:
                        pred[r, c] = F
            if not np.array_equal(pred, b):
                ok = False
                break
        if ok:
            return H0, F, P, pattern
    return None


# --------------------------------------------------------------------------- #
# ONNX construction                                                            #
# --------------------------------------------------------------------------- #
def _slice_channel(c):
    """Node + inits that Slice input channel `c` -> tensor name f'ch{c}'."""
    s = oh.make_tensor(f"s{c}", INT64, [1], [c])
    e = oh.make_tensor(f"e{c}", INT64, [1], [c + 1])
    ax = oh.make_tensor(f"ax{c}", INT64, [1], [1])
    n = oh.make_node("Slice", ["input", f"s{c}", f"e{c}", f"ax{c}"], [f"ch{c}"])
    return n, [s, e, ax]


def _build(H0, F, P, pattern, present):
    # periodic mask constant, correct for any width up to WIDTH (true tiling)
    M = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    for r in range(H0):
        pr = pattern[r]
        for c in range(WIDTH):
            if (c % P) in pr:
                M[0, 0, r, c] = 1.0
    nodes, inits = [], []
    inits.append(oh.make_tensor("MC", DATA_TYPE, [1, 1, HEIGHT, WIDTH], M.ravel().tolist()))
    inits.append(oh.make_tensor("Z", DATA_TYPE, [1, 1, HEIGHT, WIDTH],
                                np.zeros(HEIGHT * WIDTH, np.float32).tolist()))

    # background plane
    n, ci = _slice_channel(0); nodes.append(n); inits += ci
    nodes.append(oh.make_node("Mul", ["MC", "ch0"], ["fill"]))        # gated periodic fill
    nodes.append(oh.make_node("Sub", ["ch0", "fill"], ["ch0o"]))      # bg minus filled

    # which non-background input planes must be preserved
    keep = sorted(present - {0})
    for c in keep:
        if c == F:
            continue
        n, ci = _slice_channel(c); nodes.append(n); inits += ci

    if F in present:
        n, ci = _slice_channel(F); nodes.append(n); inits += ci
        nodes.append(oh.make_node("Add", [f"ch{F}", "fill"], ["chFo"]))
        fill_out = "chFo"
    else:
        fill_out = "fill"

    order = []
    for c in range(CHANNELS):
        if c == 0:
            order.append("ch0o")
        elif c == F:
            order.append(fill_out)
        elif c in keep:
            order.append(f"ch{c}")
        else:
            order.append("Z")
    nodes.append(oh.make_node("Concat", order, ["output"], axis=1))
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    res = _detect_periodic_fill(prs)
    if res is None:
        return []
    H0, F, P, pattern = res
    present = set()
    for a, _ in prs:
        present |= set(np.unique(a).tolist())
    try:
        m = _build(H0, F, P, pattern, present)
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [(f"perfill_P{P}_F{F}", m)]
