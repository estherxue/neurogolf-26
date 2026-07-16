"""Faithful evaluation: mirror neurogolf_utils.verify_network's scoring path exactly
(minus the IPython/onnx_tool display branch) so local points == leaderboard points.
"""
from __future__ import annotations

import math
import pathlib
import tempfile

import onnx
import onnxruntime

from ng_utils_shim import ng, FILESIZE_LIMIT

_WORK = pathlib.Path(tempfile.mkdtemp(prefix="ng_eval_"))


def _verify_subset(session, subset):
    right, wrong = 0, 0
    for example in subset:
        benchmark = ng.convert_to_numpy(example)
        if not benchmark:           # grid > 30x30 -> ignored by the grader
            continue
        out = ng.run_network(session, benchmark["input"])
        if out.shape == benchmark["output"].shape and (out == benchmark["output"]).all():
            right += 1
        else:
            wrong += 1
    return right, wrong


def _cleanup(fn, trace=None):
    """Remove the temp onnx and any profiling trace JSON so _WORK never grows unbounded
    (ORT profiling traces over a full arc-gen run are large; leaking them fills the disk)."""
    try:
        fn.unlink(missing_ok=True)
    except Exception:
        pass
    if trace:
        try:
            pathlib.Path(trace).unlink(missing_ok=True)
        except Exception:
            pass


def evaluate(model, examples, tag="cand", full=True):
    """Returns dict(ok, params, memory, points, agi, gen, reason).
    Short-circuits: checks train+test first (cheap); only runs full arc-gen + scoring
    if those pass. `full=False` validates on a small arc-gen sample (fast triage)."""
    fn = _WORK / f"{tag}.onnx"
    onnx.save(model, str(fn))
    if not ng.check_network(str(fn)):
        _cleanup(fn)
        return dict(ok=False, reason="filesize", points=0.0)

    try:
        sanitized = ng.sanitize_model(onnx.load(str(fn)))
        if not sanitized:
            _cleanup(fn)
            return dict(ok=False, reason="sanitize", points=0.0)
        opts = onnxruntime.SessionOptions()
        opts.enable_profiling = True
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.profile_file_prefix = str(_WORK / f"{tag}_prof")
        session = onnxruntime.InferenceSession(sanitized.SerializeToString(), opts)
    except Exception as e:
        _cleanup(fn)
        return dict(ok=False, reason=f"load:{e}", points=0.0)

    agi_r, agi_w = _verify_subset(session, examples["train"] + examples["test"])
    if agi_w:
        _cleanup(fn, session.end_profiling())
        return dict(ok=False, reason="arc-agi", agi=(agi_r, agi_w), points=0.0)

    gen = examples.get("arc-gen", [])
    sample = gen if full else gen[:25]
    gen_r, gen_w = _verify_subset(session, sample)
    if gen_w:
        _cleanup(fn, session.end_profiling())
        return dict(ok=False, reason="arc-gen", agi=(agi_r, agi_w),
                    gen=(gen_r, gen_w), points=0.0)

    trace = session.end_profiling()
    memory, params = ng.score_network(sanitized, trace)
    _cleanup(fn, trace)
    if memory is None or params is None or memory < 0 or params < 0:
        return dict(ok=False, reason="unmeasurable", points=0.0)
    cost = memory + params
    points = max(1.0, 25.0 - math.log(max(1.0, cost)))
    return dict(ok=True, params=params, memory=memory, cost=cost, points=points,
                agi=(agi_r, agi_w), gen=(gen_r, gen_w), reason="ok")


def save_solution(model, task_num, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"task{task_num:03d}.onnx"
    onnx.save(model, str(fn))
    return fn
