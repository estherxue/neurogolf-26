"""Per-task solving: infer candidate transforms from the train/test pairs (numpy),
map each to a minimal ONNX builder, validate against ALL splits, keep the cheapest
that is 0-wrong everywhere.
"""
from __future__ import annotations

import numpy as np

import builders
import learner
from evaluate import evaluate
from ng_utils_shim import CHANNELS


def _pairs(examples):
    out = []
    for split in ("train", "test"):
        for e in examples.get(split, []):
            out.append((np.array(e["input"], dtype=int), np.array(e["output"], dtype=int)))
    return out


def _infer_color_map(prs):
    """Consistent per-cell color map input_color -> output_color, or None."""
    m = {}
    for a, b in prs:
        if a.shape != b.shape:
            return None
        for iv, ov in zip(a.ravel().tolist(), b.ravel().tolist()):
            if iv in m and m[iv] != ov:
                return None
            m[iv] = ov
    return m


def _full_map(m):
    """Extend a partial color map to all 10 colors (unseen -> identity)."""
    return [m.get(i, i) for i in range(CHANNELS)]


def candidates(examples):
    """Yield (name, model) candidates whose numpy transform matches all train/test pairs."""
    prs = _pairs(examples)
    if not prs:
        return
    out = []

    def matches(fn):
        for a, b in prs:
            try:
                t = fn(a)
            except Exception:
                return False
            if t.shape != b.shape or not np.array_equal(t, b):
                return False
        return True

    # --- pointwise / geometric param-free hypotheses ---
    if matches(lambda a: a):
        out.append(("identity", builders.identity()))
    if matches(lambda a: a.T):
        out.append(("transpose", builders.transpose_hw()))
    if matches(lambda a: a[:, ::-1]):
        out.append(("flip_w", builders.flip_w()))
    if matches(lambda a: a[::-1, :]):
        out.append(("flip_h", builders.flip_h()))
    if matches(lambda a: a[::-1, ::-1]):
        out.append(("rot180", builders.rot180()))

    # --- recolor ---
    m = _infer_color_map(prs)
    if m is not None and any(k != v for k, v in m.items()):
        cmap = _full_map(m)
        # bijection over 10 channels -> cheap Gather; else 1x1 Conv
        if sorted(cmap) == list(range(CHANNELS)):
            inv = [0] * CHANNELS
            for i, o in enumerate(cmap):
                inv[o] = i
            out.append(("recolor_gather", builders.recolor_gather(inv)))
        out.append(("recolor_conv", builders.recolor_conv(cmap)))

    # --- translate (search small shifts, zero fill) ---
    def shift(a, dy, dx):
        r = np.zeros_like(a)
        h, w = a.shape
        ys0, ys1 = max(dy, 0), min(h, h + dy)
        xs0, xs1 = max(dx, 0), min(w, w + dx)
        r[ys0:ys1, xs0:xs1] = a[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
        return r
    found_shift = None
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            if dy == 0 and dx == 0:
                continue
            if matches(lambda a, dy=dy, dx=dx: shift(a, dy, dx)):
                found_shift = (dy, dx)
                break
        if found_shift:
            break
    if found_shift:
        out.append((f"translate{found_shift}", builders.translate(*found_shift)))

    return out


def solve_task(task_num, examples):
    """Return best dict {name, points, params, memory, model} or None.

    Symbolic candidates first (cheap + perfectly generalizing). Only if none solve
    the task do we try the local-rule learner (more params/memory, lower points)."""
    best = None
    for name, model in candidates(examples) or []:
        res = evaluate(model, examples, tag=f"t{task_num:03d}_{name}", full=True)
        if res.get("ok") and (best is None or res["points"] > best["points"]):
            best = dict(name=name, model=model, **{k: res[k] for k in
                        ("points", "params", "memory", "cost")})
    if best is not None:
        return best
    for name, model in learner.local_candidates(examples):
        res = evaluate(model, examples, tag=f"t{task_num:03d}_{name}", full=True)
        if res.get("ok") and (best is None or res["points"] > best["points"]):
            best = dict(name=name, model=model, **{k: res[k] for k in
                        ("points", "params", "memory", "cost")})
    if best is not None:
        return best
    # Final tier: per-task CNN trainer (needs torch; skipped if unavailable).
    try:
        import trainer_torch
    except Exception:
        return best
    for name, model in trainer_torch.cnn_candidates(examples):
        res = evaluate(model, examples, tag=f"t{task_num:03d}_{name}", full=True)
        if res.get("ok") and (best is None or res["points"] > best["points"]):
            best = dict(name=name, model=model, **{k: res[k] for k in
                        ("points", "params", "memory", "cost")})
    return best
