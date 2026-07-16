"""Runtime-probe cost (== grader method): promote all node outputs to graph outputs,
run ALL official examples, mem = sum over named intermediates of per-tensor MAX bytes;
params = initializer (+Constant) element count. pts = max(1, 25 - ln(mem+params)).
usage: python probe3.py <task_int> <onnx_path> [more paths...]
importable: probe_cost(path, tn) -> (mem, params, pts)
"""
import sys, os, json
import numpy as np
import onnx, onnxruntime as ort
from onnx import numpy_helper as nh

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = "/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26/f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad"


def to_np(gr):
    b = np.zeros((1, 10, 30, 30), np.float32)
    for r, row in enumerate(gr):
        for c, v in enumerate(row):
            b[0][v][r][c] = 1.0
    return b


def probe_cost(path, tn, verbose=False):
    m = onnx.load(path)
    outn = {o.name for o in m.graph.output}
    names = [o for n in m.graph.node for o in n.output if o and o not in outn]
    m2 = onnx.ModelProto(); m2.CopyFrom(m)
    for nm in names:
        m2.graph.output.add().name = nm
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 4
    s = ort.InferenceSession(m2.SerializeToString(), so)
    meta = [o.name for o in s.get_outputs()]
    d = json.load(open(f"{SC}/data/task{tn:03d}.json"))
    exs = [e for sp in ("train", "test", "arc-gen") for e in d.get(sp, [])]
    mx = {}
    for e in exs:
        for nm, arr in zip(meta, s.run(None, {"input": to_np(e["input"])})):
            if nm in outn or arr.size == 0:
                continue
            b = int(np.prod(arr.shape)) * arr.dtype.itemsize
            if b > mx.get(nm, (0,))[0]:
                mx[nm] = (b, str(arr.dtype), tuple(arr.shape))
    mem = sum(v[0] for v in mx.values())
    params = sum(max(1, int(np.prod(i.dims) if list(i.dims) else 1)) for i in m.graph.initializer)
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    params += int(nh.to_array(a.t).size)
    pts = max(1.0, 25 - np.log(max(1, mem + params)))
    if verbose:
        for nm, (b, dt, shp) in sorted(mx.items(), key=lambda kv: -kv[1][0]):
            print(f"   {b:7}B {dt:9} {shp} {nm[:40]}")
    return mem, params, pts


if __name__ == "__main__":
    tn = int(sys.argv[1])
    for p in sys.argv[2:]:
        mem, params, pts = probe_cost(p, tn, verbose=("-v" in sys.argv))
        print(f"task{tn:03d} {os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}: mem={mem} params={params} cost={mem+params} pts={pts:.2f}")
