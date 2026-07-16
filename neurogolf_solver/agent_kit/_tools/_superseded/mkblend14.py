"""Assemble out_blend14 = out_blend13 + validated candidate swaps.
Each swap is applied ONLY if the candidate file exists and its static cost beats
the incumbent (static profiler verified == official on e017/ed209).
usage: python mkblend14.py swapspec.json
  swapspec.json = [{"task":17,"path":"/abs/path.onnx","name":"e017_dilated"}, ...]
"""
import sys, os, json, shutil, zipfile
import numpy as np
import onnx
import onnxruntime as ort

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = "/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26/f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad"


def _to_np(gr):
    b = np.zeros((1, 10, 30, 30), np.float32)
    for r, row in enumerate(gr):
        for c, v in enumerate(row):
            b[0][v][r][c] = 1.0
    return b


def points(path, tn):
    """Runtime-probe cost — the grader's method: max named-intermediate bytes over all official examples."""
    m = onnx.load(path)
    outn = {o.name for o in m.graph.output}
    names = [o for n in m.graph.node for o in n.output if o and o not in outn]
    m2 = onnx.ModelProto(); m2.CopyFrom(m)
    for nm in names:
        m2.graph.output.add().name = nm
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 4
    s = ort.InferenceSession(m2.SerializeToString(), so)
    meta = [o.name for o in s.get_outputs()]
    d = json.load(open(f"{SC}/data/task{tn:03d}.json"))
    exs = [e for sp in ("train", "test", "arc-gen") for e in d.get(sp, [])]
    mx = {}
    for e in exs:
        for nm, arr in zip(meta, s.run(None, {"input": _to_np(e["input"])})):
            if nm == "output":
                continue
            mx[nm] = max(mx.get(nm, 0), int(np.prod(arr.shape)) * arr.dtype.itemsize if arr.size else 0)
    mem = sum(mx.values())
    params = sum(max(1, int(np.prod(i.dims) if list(i.dims) else 1)) for i in m.graph.initializer)
    return 25 - float(np.log(mem + params)), mem, params


def main():
    spec = json.load(open(sys.argv[1]))
    src, dst = os.path.join(KIT, "out_blend13"), os.path.join(KIT, "out_blend14")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    rep = json.load(open(os.path.join(dst, "report.json")))
    by = {t["task"]: t for t in rep["tasks"]}
    total_delta = 0.0
    for s in spec:
        tn, path, name = s["task"], s["path"], s["name"]
        if not os.path.exists(path):
            print(f"task{tn:03d}: SKIP (candidate missing: {path})")
            continue
        new_pts, mem, params = points(path, tn)
        old_pts = by[tn]["points"]
        if new_pts <= old_pts + 0.01:
            print(f"task{tn:03d}: SKIP ({name} {new_pts:.2f} <= incumbent {old_pts:.2f})")
            continue
        shutil.copy(path, os.path.join(dst, "onnx", f"task{tn:03d}.onnx"))
        by[tn]["points"] = round(new_pts, 2)
        by[tn]["name"] = name
        total_delta += new_pts - old_pts
        print(f"task{tn:03d}: SWAP {name}  {old_pts:.2f} -> {new_pts:.2f} (mem {mem} par {params})")
    rep["total_points"] = round(sum(t["points"] for t in rep["tasks"]), 2)
    json.dump(rep, open(os.path.join(dst, "report.json"), "w"), indent=1)
    zp = os.path.join(dst, "submission.zip")
    if os.path.exists(zp):
        os.remove(zp)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        od = os.path.join(dst, "onnx")
        for f in sorted(os.listdir(od)):
            if f.endswith(".onnx"):
                z.write(os.path.join(od, f), f)
    print(f"\nblend14 total={rep['total_points']} (delta +{total_delta:.2f})  zip={zp}")


if __name__ == "__main__":
    main()
