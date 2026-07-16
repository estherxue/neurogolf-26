"""Scaling family: integer upscale (pixel/nearest), downscale (block-subsample),
and constant output. All origin-anchored. Reference example for the agent kit."""
import numpy as np
from builders import upscale, downscale, constant
from ng_utils_shim import ng


def _pairs(ex):
    return [(np.array(e["input"]), np.array(e["output"]))
            for e in ex.get("train", []) + ex.get("test", [])]


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []
    for k in range(2, 6):
        if all(b.shape == (a.shape[0] * k, a.shape[1] * k)
               and np.array_equal(b, np.kron(a, np.ones((k, k), int))) for a, b in prs):
            out.append((f"upscale{k}", upscale(k)))
            break
    for k in range(2, 6):
        if all(a.shape[0] >= k and a.shape[1] >= k and np.array_equal(b, a[::k, ::k])
               for a, b in prs):
            out.append((f"downscale{k}", downscale(k)))
            break
    outs = [b for _, b in prs]
    if len(outs) > 1 and all(o.shape == outs[0].shape and np.array_equal(o, outs[0])
                             for o in outs):
        b = ng.convert_to_numpy({"input": prs[0][1].tolist(), "output": prs[0][1].tolist()})
        if b:
            out.append(("constant", constant(b["output"])))
    return out
