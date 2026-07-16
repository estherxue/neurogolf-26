"""Per-task CNN trainer (the general 'NeuroGolf' method): train a small CNN on a task's
arc-gen examples until it reproduces the transformation EXACTLY, then export to ONNX.

Why this generalizes (the private-set concern): the grader thresholds each output channel
at >0, so we train with per-channel BCEWithLogits against the one-hot target (padding cells
= all-zero = "no color") and threshold at 0 — an exact match to the grader. We FIT on part
of arc-gen and require EXACT correctness on the held-out remainder (same generator as the
private set) before accepting. Compute is free in the cost metric, so we can afford depth
(large receptive field) for global structure; we pay only params + intermediate memory.

Requires torch (use the py3.11 venv). Import is lazy so the rest of the package works
without torch installed.
"""
from __future__ import annotations

import io
import math

import numpy as np
import onnx

from ng_utils_shim import ng, IR_VERSION


def _device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _tensors(examples_list, dev=None):
    import torch
    X, Y = [], []
    for ex in examples_list:
        b = ng.convert_to_numpy(ex)
        if not b:
            continue
        X.append(b["input"][0])
        Y.append(b["output"][0])
    if not X:
        return None, None
    xt = torch.tensor(np.array(X), dtype=torch.float32)
    yt = torch.tensor(np.array(Y), dtype=torch.float32)
    if dev is not None:
        xt, yt = xt.to(dev), yt.to(dev)
    return xt, yt


def _build_net(ch, depth):
    import torch.nn as nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv2d(10, ch, 3, padding=1)
            self.blocks = nn.ModuleList([nn.Conv2d(ch, ch, 3, padding=1) for _ in range(depth)])
            self.out = nn.Conv2d(ch, 10, 1)

        def forward(self, x):
            import torch
            h = torch.relu(self.inp(x))
            for b in self.blocks:
                h = torch.relu(h + b(h))   # residual; Add+Conv+Relu all opset-10 legal
            return self.out(h)

    return CNN()


def _all_exact(net, X, Y):
    import torch
    if X is None:
        return True
    with torch.no_grad():
        pred = (net(X) > 0.0).float()
    return bool((pred == Y).all().item())


def train_task(examples, ch=48, depth=8, steps=1500, lr=5e-3, holdout=0.30, seed=0,
               max_fit=64, time_cap=None, threads=None):
    """Train a CNN; return (net, fit_exact, gen_exact). gen_exact is the anti-overfit
    signal: trained on the fit split, exactly correct on the held-out arc-gen split.

    Key fixes: BCE pos_weight to counter the heavy 0/1 imbalance (most one-hot target
    cells are 0 for small grids), an LR decay schedule, a fit-set cap for speed, and an
    optional wall-clock cap."""
    import time
    import torch
    import torch.nn as nn

    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    dev = _device()
    tt = list(examples.get("train", [])) + list(examples.get("test", []))
    arc = list(examples.get("arc-gen", []))
    if len(arc) < 4:
        return None, False, False
    nfit = max(1, math.ceil(len(arc) * (1 - holdout)))
    fit_examples = (tt + arc[:nfit])[:max_fit]
    Xf, Yf = _tensors(fit_examples, dev)
    Xh, Yh = _tensors(arc[nfit:], dev)
    if Xf is None:
        return None, False, False

    # pos_weight = (#negatives / #positives) so the rare active cells dominate the loss.
    pos = float(Yf.sum().item())
    neg = float(Yf.numel()) - pos
    pw = torch.tensor(min(max(neg / max(pos, 1.0), 1.0), 500.0), device=dev)

    net = _build_net(ch, depth).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    t0 = time.time()
    for i in range(steps):
        opt.zero_grad()
        loss = lossf(net(Xf), Yf)
        loss.backward()
        opt.step()
        sched.step()
        if i % 25 == 0 and _all_exact(net, Xf, Yf) and _all_exact(net, Xh, Yh):
            break
        if time_cap and (time.time() - t0) > time_cap:
            break
    net.eval()
    return net, _all_exact(net, Xf, Yf), _all_exact(net, Xh, Yh)


def export_onnx(net):
    import torch
    net = net.to("cpu").eval()
    x = torch.zeros(1, 10, 30, 30)
    buf = io.BytesIO()
    torch.onnx.export(net, x, buf, input_names=["input"], output_names=["output"],
                      opset_version=10, do_constant_folding=True)
    buf.seek(0)
    model = onnx.load_model_from_string(buf.read())
    model.ir_version = IR_VERSION
    return model


# Architecture ladder: small (cheap, generalizes) first, grow only if needed.
# Tuned for GPU: spans tiny->deep so easy tasks stay cheap (high pts) and hard tasks get capacity.
LADDER = [(8, 2), (16, 4), (24, 6), (32, 10), (48, 14), (64, 20), (96, 24)]


def cnn_candidates(examples, seeds=(0, 1), per_arch_cap=12.0, total_budget=150.0):
    """Yield (name, model) for the smallest CNN that fits AND generalizes (held-out exact).
    Architecture ladder small->large, multi-seed; the held-out arc-gen gate (in train_task)
    is the anti-overfit guard. Bounded by per-arch and total wall-clock budgets."""
    import time
    t0 = time.time()
    for ch, depth in LADDER:
        for seed in seeds:
            if time.time() - t0 > total_budget:
                return
            net, fit_ok, gen_ok = train_task(examples, ch=ch, depth=depth, seed=seed,
                                             steps=6000, time_cap=per_arch_cap)
            if net is not None and fit_ok and gen_ok:
                yield (f"cnn_c{ch}_d{depth}", export_onnx(net))
                return
