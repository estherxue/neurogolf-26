"""From rv14_results.json pick, per task, the CHEAPEST ADOPTABLE pool variant that
beats the incumbent's points; emit swapspec entries (to merge into _cands/swapspec.json).
usage: python pick_adopt.py
"""
import json, os, sys
import numpy as np
import onnx
from onnx import shape_inference, TensorProto as TP

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = {TP.FLOAT: 4, TP.FLOAT16: 2, TP.UINT8: 1, TP.INT8: 1, TP.BOOL: 1,
      TP.INT32: 4, TP.INT64: 8, TP.UINT32: 4, TP.UINT16: 2, TP.INT16: 2}


def points(path):
    m = onnx.load(path)
    m = shape_inference.infer_shapes(m, strict_mode=True)
    g = m.graph
    vi = {v.name: v for v in list(g.value_info) + list(g.output)}
    outn = {o.name for o in g.output}
    mem = 0
    for n in g.node:
        for o in n.output:
            if o in outn:
                continue
            t = vi[o].type.tensor_type
            dims = [d.dim_value for d in t.shape.dim]
            if not dims or any(d == 0 for d in dims):
                raise RuntimeError("unresolved " + o)
            mem += int(np.prod(dims)) * DT.get(t.elem_type, 4)
    params = sum(max(1, int(np.prod(i.dims) if list(i.dims) else 1)) for i in g.initializer)
    return 25 - float(np.log(mem + params))


res = json.load(open(KIT + "/_tools/rv14_results.json"))
spec = []
for tn_s, tres in res.items():
    tn = int(tn_s)
    inc_pts = tres["inc"]["pts"]
    best = None
    for c in tres["cands"]:
        if c["verdict"] != "ADOPTABLE":
            continue
        p = f"{KIT}/_pools/{c['pool']}/task{tn:03d}.onnx"
        try:
            pts = points(p)
        except Exception as e:
            print(f"task{tn:03d} {c['pool']}: cost-err {str(e)[:60]}")
            continue
        if best is None or pts > best[0]:
            best = (pts, p, c["pool"], c["fails"])
    if best is None:
        continue
    pts, p, pool, fails = best
    if pts <= inc_pts + 0.01:
        print(f"task{tn:03d}: adoptable {pool} but NOT cheaper ({pts:.2f} <= {inc_pts:.2f}) — skip")
        continue
    print(f"task{tn:03d}: ADOPT {pool}  {inc_pts:.2f} -> {pts:.2f}  (dirt {fails}/{tres['inc']['n']} vs inc {tres['inc']['fails']})")
    spec.append({"task": tn, "path": p, "name": f"rv14_{pool}", "pts": round(pts, 2)})

json.dump(spec, open(KIT + "/_cands/adopt_spec.json", "w"), indent=1)
print(f"\n{len(spec)} adoptions -> _cands/adopt_spec.json")
