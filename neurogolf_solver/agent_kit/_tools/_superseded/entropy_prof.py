"""M2: 3-axis dataflow entropy profiler.
Per named intermediate, over ALL official examples:
  axis3 (dtype): value range + integrality -> u8-demotable bytes (as surg_scout)
  axis2 (support): content bbox (max extent of nonzero/non-constant region) vs declared shape -> crop slack
  axis1 (structure): number of distinct values ever seen -> mask-like (<=2), small-alphabet (<=10)
Summary per task: mem, params, pts, u8_slack, crop_slack (bytes recoverable if all spatial dims cropped
to observed content bbox +0), mask_bytes (intermediates that are 0/1 masks — fusion candidates).
usage: python entropy_prof.py <task_int> [onnx_path] [--brief]
"""
import sys, os, json
import numpy as np
import onnx, onnxruntime as ort

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = "/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26/f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad"

tn = int(sys.argv[1])
path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else f"{KIT}/out_blend14/onnx/task{tn:03d}.onnx"
BRIEF = "--brief" in sys.argv


def to_np(gr):
    b = np.zeros((1, 10, 30, 30), np.float32)
    for r, row in enumerate(gr):
        for c, v in enumerate(row):
            b[0][v][r][c] = 1.0
    return b


m = onnx.load(path)
outn = {o.name for o in m.graph.output}
producer = {o: n.op_type for n in m.graph.node for o in n.output if o}
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

info = {}
for e in exs:
    for nm, arr in zip(meta, s.run(None, {"input": to_np(e["input"])})):
        if nm == "output" or arr.size == 0:
            continue
        b = int(np.prod(arr.shape)) * arr.dtype.itemsize
        a = arr.astype(np.float64)
        # content bbox on trailing 2 dims if 4D spatial
        if arr.ndim == 4 and arr.shape[2] > 1 and arr.shape[3] > 1:
            bg = a.reshape(-1, a.shape[2], a.shape[3])
            # background = the modal border value; use "non-most-common" support
            vals, cnts = np.unique(a, return_counts=True)
            bgv = vals[cnts.argmax()]
            nz = np.argwhere((bg != bgv).any(axis=0))
            if nz.size:
                h = int(nz[:, 0].max()) + 1
                w = int(nz[:, 1].max()) + 1
            else:
                h = w = 1
        else:
            h = w = -1  # n/a
        u = np.unique(a)
        if nm not in info:
            info[nm] = {"b": b, "isz": arr.dtype.itemsize, "dt": str(arr.dtype),
                        "vmin": float(a.min()), "vmax": float(a.max()),
                        "int": bool(np.all(a == np.round(a))),
                        "h": h, "w": w, "H": arr.shape[2] if arr.ndim == 4 else -1,
                        "W": arr.shape[3] if arr.ndim == 4 else -1,
                        "vals": set(u[:64].tolist()) if len(u) <= 64 else None}
        else:
            r = info[nm]
            r["b"] = max(r["b"], b); r["vmin"] = min(r["vmin"], float(a.min()))
            r["vmax"] = max(r["vmax"], float(a.max()))
            r["int"] = r["int"] and bool(np.all(a == np.round(a)))
            r["h"] = max(r["h"], h); r["w"] = max(r["w"], w)
            if r["vals"] is not None:
                if len(u) <= 64:
                    r["vals"] |= set(u[:64].tolist())
                    if len(r["vals"]) > 64:
                        r["vals"] = None
                else:
                    r["vals"] = None

mem = sum(v["b"] for v in info.values())
params = sum(max(1, int(np.prod(i.dims) if list(i.dims) else 1)) for i in m.graph.initializer)
pts = 25 - np.log(mem + params)

u8_slack = crop_slack = mask_bytes = 0
for nm, v in info.items():
    if v["isz"] > 1 and v["int"] and v["vmin"] >= 0 and v["vmax"] <= 255:
        u8_slack += v["b"] - v["b"] // v["isz"]
    if v["H"] > 0 and v["h"] > 0 and (v["h"] < v["H"] or v["w"] < v["W"]):
        eff = v["b"] * (1 - (v["h"] * v["w"]) / (v["H"] * v["W"]))
        crop_slack += int(eff)
    if v["vals"] is not None and len(v["vals"]) <= 2:
        mask_bytes += v["b"]

both = mem - u8_slack - int(crop_slack * (1 - u8_slack / max(mem, 1)))  # rough combined
opt_pts = 25 - np.log(max(both, 1) + params)
print(f"task{tn:03d} mem={mem} par={params} pts={pts:.2f} | u8_slack={u8_slack} crop_slack={crop_slack} "
      f"mask_bytes={mask_bytes} | combined_opt≈{opt_pts:.2f} (+{opt_pts-pts:.2f})")
if not BRIEF:
    rows = sorted(info.items(), key=lambda kv: -kv[1]["b"])[:10]
    for nm, v in rows:
        nv = len(v["vals"]) if v["vals"] is not None else ">64"
        print(f"   {producer.get(nm,'?'):13} {v['dt']:8} {v['b']:6}B bbox {v['h']}x{v['w']}/{v['H']}x{v['W']} "
              f"nvals={nv} rng[{v['vmin']:.0f},{v['vmax']:.0f}] {nm[:24]}")
