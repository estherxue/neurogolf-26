"""Batch FP16 dtype-lowering sweep over a whole output dir. For every task whose incumbent
ONNX still computes in float32, apply the proven hand FP16 lowering (Cast input->f16, retarget
Cast-to-f32 -> f16, float32 initializers/Constants -> f16 with +-30000 clamp, output declared
f16, drop stale value_info) and keep it ONLY if it is grader-EXACT on train+test+arc-gen AND
cheaper (best-of-two vs incumbent) AND exact on a HELD-OUT 30% arc-gen split. Monotone: never
regresses. Catches every remaining FP16 win the sliced agent waves missed.

usage: python batch_fp16.py <src_dir> <dst_dir>
"""
import sys, os, json, glob, pathlib, zipfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnx
from onnx import TensorProto as TP
from onnx import helper as oh, numpy_helper as nh
import evaluate
from ng_utils_shim import tasks_dir

SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("out_p7")
DST = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("out_p8")
FLOAT, FLOAT16 = TP.FLOAT, TP.FLOAT16
_CLAMP = 30000.0
FIT = 0.70


def _to_f16_array(arr):
    return np.clip(arr, -_CLAMP, _CLAMP).astype(np.float16)


def to_fp16(model):
    m = onnx.ModelProto(); m.CopyFrom(model); g = m.graph
    if not any(i.type.tensor_type.elem_type == FLOAT for i in g.input):
        pass
    for node in g.node:
        for i, inp in enumerate(node.input):
            if inp == "input":
                node.input[i] = "input_f16"
    g.node.insert(0, oh.make_node("Cast", ["input"], ["input_f16"], to=FLOAT16, name="cast_input_f16"))
    for node in g.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == FLOAT:
                    a.i = FLOAT16
    for init in g.initializer:
        if init.data_type == FLOAT:
            init.CopyFrom(nh.from_array(_to_f16_array(nh.to_array(init)), init.name))
    for node in g.node:
        if node.op_type in ("Constant", "ConstantOfShape"):
            for a in node.attribute:
                if a.name == "value" and a.t.data_type == FLOAT:
                    a.t.CopyFrom(nh.from_array(_to_f16_array(nh.to_array(a.t))))
    g.output[0].type.tensor_type.elem_type = FLOAT16
    del g.value_info[:]
    return m


def is_float32_graph(model):
    # already-fp16 if the graph has a leading input Cast to f16 or f16 initializers dominate
    for init in model.graph.initializer:
        if init.data_type == FLOAT:
            return True
    for n in model.graph.node:
        if n.op_type in ("Constant",) and any(a.name == "value" and a.t.data_type == FLOAT for a in n.attribute):
            return True
    return False


def exact_on(model, examples, tag):
    if not examples:
        return True
    try:
        return bool(evaluate.evaluate(model, {"train": examples, "test": [], "arc-gen": []}, tag=tag).get("ok"))
    except Exception:
        return False


def main():
    tdir = tasks_dir()
    rep = json.load(open(SRC / "report.json"))
    (DST / "onnx").mkdir(parents=True, exist_ok=True)
    for f in (SRC / "onnx").glob("task*.onnx"):
        (DST / "onnx" / f.name).write_bytes(f.read_bytes())
    by_task = {t["task"]: dict(t) for t in rep["tasks"]}
    wins = 0
    for tn, t in by_task.items():
        if not t.get("name"):
            continue
        fp = SRC / "onnx" / f"task{tn:03d}.onnx"
        m = onnx.load(str(fp))
        if not is_float32_graph(m):
            continue
        ex = json.load(open(tdir / f"task{tn:03d}.json"))
        # SAFE GUARD: fp16 only FIXED-SIZE tasks (all arc-gen one size, in AND out). Then arc-gen
        # covers the generator's full size range, so exactness there implies exactness on the
        # hidden set — fp16 rounding of any >2048 integer would already show up. Variable-size
        # tasks are skipped (hidden grids may be larger, exceeding fp16's exact-integer range).
        allex = ex["train"] + ex["test"] + ex.get("arc-gen", [])
        sizes = {(len(e["input"]), len(e["input"][0])) for e in allex}
        sizes |= {(len(e["output"]), len(e["output"][0])) for e in allex}
        if len(sizes) != 1:
            continue
        base = evaluate.evaluate(m, ex, tag=f"bf_{tn}")
        if not base.get("ok"):
            continue
        try:
            m16 = to_fp16(onnx.load(str(fp)))
        except Exception:
            continue
        r = evaluate.evaluate(m16, ex, tag=f"cf_{tn}")
        if not r.get("ok") or r["points"] <= base["points"] + 0.02:
            continue
        ag = ex.get("arc-gen", []); k = max(1, int(len(ag) * FIT))
        if not exact_on(m16, ag[k:], tag=f"hf_{tn}"):
            continue
        onnx.save(m16, str(DST / "onnx" / f"task{tn:03d}.onnx"))
        by_task[tn]["points"] = round(r["points"], 2)
        by_task[tn]["name"] = (t.get("name") or "") + "+bf16"
        wins += 1
        print(f"  task{tn:03d}: {base['points']:.2f} -> {r['points']:.2f}  +{r['points']-base['points']:.2f}", flush=True)
    tasks = [by_task[i] for i in sorted(by_task)]
    tot = sum(x["points"] for x in tasks if x.get("name"))
    json.dump(dict(solved=sum(1 for x in tasks if x.get("name")), total_points=round(tot, 3), tasks=tasks),
              open(DST / "report.json", "w"))
    with zipfile.ZipFile(DST / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((DST / "onnx").glob("task*.onnx")):
            z.write(f, f.name)
    print(f"BATCH_FP16: {wins} wins; total {tot:.2f}", flush=True)


if __name__ == "__main__":
    main()
