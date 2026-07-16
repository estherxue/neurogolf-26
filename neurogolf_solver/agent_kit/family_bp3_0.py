"""family_bp3_0 — cost-recompile bench for tasks 191, 233, 366, 158.

Goal was to turn each solver into a label-space + downsampled graph and beat the
current out_blend6 cost.  Dissection of the CURRENT models (per-tensor memory via
the ORT profiler, exactly mirroring neurogolf_utils.calculate_memory) shows these
are NOT naive fp16 [1,10,30,30] canvas solvers — they are already heavily golfed,
low-param, small-intermediate graphs:

  task191 (7df24a62, size=23 template match, opset17): mem 35335, params 90.
      dominated by the 8-orientation D4 match maps  safe_name_77/79 = [1,8,23,23]
      fp16 (8464 B each) + bool 78 (4232 B) ~= 21 KB.  Output already bool one-hot.
  task233 (97a05b5b, exact-cover hole fill, opset13): mem 33637, params 763.
      full constraint-propagation solver (see family_r233); memory is spread over
      the 8-slot x 8-orientation correlation stack — all algorithmically load-bearing.
  task366 (e6721834, two-grid box fill, opset16): mem 29962, params 203.
      401 nodes; memory is the SUM of ~200 small (<=510 B) int/uint8 intermediates,
      biggest a single [1,1,30,30] fp32 (3600 B).  No dominant canvas to kill.
  task158 (6aa20dc0, magnified-sprite reconstruction, opset18): mem 26178,
      params 2305.  QLinearConv stamp banks (w_pair* = 968+512+200 elems) + [1,4,*]
      stamp maps; variable sprite count/mag/flip detection.

None of the four carries a free [1,10,30,30] one-hot canvas to collapse — that lever
is already applied (all four emit a bool one-hot output via Equal, which is FREE).
The residual memory is genuine algorithmic working set (D4 correlation stacks, slot
propagation, per-box fills).  A verified strict improvement would require a full
re-derivation of each solver AND re-verification exact on train+test+ >=1500 fresh
ARC-GEN samples with NO regression and NO overfit.  That was not completed to the
required confidence in this pass, so — per the hard gate "NEVER ship a regression or
an unverified graph" — candidates() intentionally emits nothing here.  The harness
keeps the current (better) out_blend6 models; no task regresses.

Identified concrete win vector for a future pass (task191): the two stacked fp16
[1,8,23,23] D4 match maps (~17 KB) can be replaced by 8 independently-named uint8
[1,1,23,23] per-orientation tensors (529 B each = 4232 B total) since deep chains on
tiny tensors are FREE — recompute each orientation's correlation and OR its
box-stamp into the accumulator instead of materializing the full stacked map. That
alone targets ~35335 -> ~18000 (points 14.52 -> ~15.1) if the box-draw stays exact.
"""
from __future__ import annotations


def _fingerprint(example):
    """Route a task to one of the four handlers from a single train example.

    Returns one of 'sq23' (191), 'holefill' (233), 'twogrid' (366),
    'megasprite' (158), or None if it matches none.
    """
    train = example.get("train", [])
    if not train:
        return None
    inp = train[0]["input"]
    out = train[0]["output"]
    ih, iw = len(inp), len(inp[0])
    oh_, ow = len(out), len(out[0])
    colors = set(v for row in inp for v in row)

    same_shape = (ih, iw) == (oh_, ow)
    if same_shape:
        # 191: {black, blue(1), yellow(4)} only; 158: 4-colour magnified sprites.
        if colors <= {0, 1, 4}:
            return "sq23"
        return "megasprite"
    # output strictly smaller than input -> a cropped sub-region.
    # 366: input is exactly one grid stacked on/next to another (2x in one axis).
    if ih == 2 * oh_ or iw == 2 * ow:
        return "twogrid"
    # 233: red(2) reference box carved out of a solid square.
    if 2 in colors:
        return "holefill"
    return None


def candidates(example):
    """Dispatch table for the four target tasks.

    Every handler returns [] for now: the current out_blend6 models are already
    label-space / small-intermediate golfed graphs, and no re-derivation reached the
    verified strict-improvement bar (exact on train+test+>=1500 fresh ARC-GEN, no
    regression) in this pass.  Emitting nothing means the harness keeps the better
    existing model — never a regression.  See module docstring for the per-task
    memory breakdown and the identified task191 win vector.
    """
    kind = _fingerprint(example)
    handler = {
        "sq23": _sq23,
        "holefill": _holefill,
        "twogrid": _twogrid,
        "megasprite": _megasprite,
    }.get(kind)
    if handler is None:
        return []
    return handler(example)


def _sq23(example):        # task191 — see docstring (win vector identified, unverified)
    return []


def _holefill(example):    # task233 — exact-cover solver, no dominant canvas
    return []


def _twogrid(example):     # task366 — 400-node sum of small intermediates
    return []


def _megasprite(example):  # task158 — variable sprite detect + magnify
    return []
