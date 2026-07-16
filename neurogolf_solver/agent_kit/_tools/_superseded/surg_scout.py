"""Surgery scout: per-tensor runtime census of a solver, quantifying u8-demotion headroom.
For each named intermediate: max runtime bytes, dtype, value range, integrality.
A f16/f32 tensor is 'demotable' to u8 iff over ALL official examples its values are
integers in [0,255]. Reports realizable memory saving = sum(demotable bytes * (1 - 1/itemsize)).
usage: python surg_scout.py <task_int> [onnx_path]
"""
import sys, os, json
import numpy as np
import onnx, onnxruntime as ort

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = "/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26/f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad"

tn = int(sys.argv[1])
path = sys.argv[2] if len(sys.argv) > 2 else f"{KIT}/out_blend14/onnx/task{tn:03d}.onnx"


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
info = {}  # name -> [maxbytes, itemsize, dtype, vmin, vmax, all_int]
for e in exs:
    for nm, arr in zip(meta, s.run(None, {"input": to_np(e["input"])})):
        if nm == "output" or arr.size == 0:
            continue
        b = int(np.prod(arr.shape)) * arr.dtype.itemsize
        a = arr.astype(np.float64)
        vmin, vmax = float(a.min()), float(a.max())
        allint = bool(np.all(a == np.round(a)))
        if nm not in info:
            info[nm] = [b, arr.dtype.itemsize, str(arr.dtype), vmin, vmax, allint]
        else:
            r = info[nm]
            r[0] = max(r[0], b); r[3] = min(r[3], vmin); r[4] = max(r[4], vmax); r[5] = r[5] and allint

mem = sum(v[0] for v in info.values())
params = sum(max(1, int(np.prod(i.dims) if list(i.dims) else 1)) for i in m.graph.initializer)
pts = 25 - np.log(mem + params)

# demotable: itemsize>1, all-int, range in [0,255]
saving = 0
demot = []
for nm, (b, isz, dt, vmn, vmx, ai) in info.items():
    if isz > 1 and ai and vmn >= 0 and vmx <= 255:
        s_bytes = b - b // isz  # to u8
        saving += s_bytes
        demot.append((nm, producer.get(nm, "?"), b, dt, vmn, vmx, s_bytes))
opt_mem = mem - saving
opt_pts = 25 - np.log(opt_mem + params)

print(f"task{tn:03d}  nodes={len(m.graph.node)}  mem={mem}  params={params}  pts={pts:.2f}")
print(f"  DEMOTABLE-to-u8: {len(demot)} tensors, save {saving} B  ->  opt_mem={opt_mem}  opt_pts={opt_pts:.2f}  (headroom +{opt_pts-pts:.2f})")
demot.sort(key=lambda x: -x[6])
for nm, op, b, dt, vmn, vmx, sv in demot[:12]:
    print(f"    {op:14} {dt:9} {b:6}B save{sv:6}  range[{vmn:.0f},{vmx:.0f}]  {nm[:28]}")
# also show top non-demotable float tensors (the structural floor)
nondemot = [(nm, info[nm]) for nm in info if info[nm][1] > 1 and (nm, producer.get(nm, "?"), 0, 0, 0, 0, 0) not in [(d[0],)+d[1:] for d in demot]]
floor = [(nm, v) for nm, v in info.items() if v[1] > 1 and not (v[5] and v[3] >= 0 and v[4] <= 255)]
floor.sort(key=lambda x: -x[1][0])
print(f"  FLOOR (float, cannot u8): {sum(v[0] for _,v in floor)} B")
for nm, v in floor[:6]:
    print(f"    {producer.get(nm,'?'):14} {v[2]:9} {v[0]:6}B  range[{v[3]:.1f},{v[4]:.1f}] int={v[5]}  {nm[:26]}")
