"""GPU CNN campaign: for tasks the symbolic/family pipeline did NOT solve, train a per-task
CNN (architecture search + multi-seed, held-out gated) in parallel across workers, score with
the official grader, and merge the new solves into an existing combined output dir.

Run AFTER `combined_solve.py <out_dir>` has produced <out_dir>/report.json + onnx/.
usage: python gpu_cnn.py <out_dir> [num_workers]
Env: NG_DATA_DIR must point at the data (tasks + neurogolf_utils).
"""
import os, sys, json, glob, pathlib, zipfile
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)

OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (pathlib.Path(HERE) / "out_pod")
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 6


def _train_one(tnum):
    """Worker: train a CNN for one task; return (tnum, name, points, params, memory) or None."""
    import json as j
    from ng_utils_shim import tasks_dir
    from evaluate import evaluate
    import trainer_torch
    import onnx
    try:
        ex = j.load(open(tasks_dir() / f"task{tnum:03d}.json"))
    except Exception:
        return None
    best = None
    for name, model in trainer_torch.cnn_candidates(ex):
        res = evaluate(model, ex, tag=f"gpu_{tnum}_{name}")
        if res.get("ok") and (best is None or res["points"] > best[2]):
            onnx.save(model, str(OUT / "onnx" / f"task{tnum:03d}.onnx"))
            best = (tnum, name, res["points"], res.get("params"), res.get("memory"))
    return best


def main():
    report = json.load(open(OUT / "report.json"))
    solved = {t["task"] for t in report["tasks"] if t.get("name")}
    unsolved = [t["task"] for t in report["tasks"] if not t.get("name")]
    print(f"[gpu_cnn] families solved {len(solved)}; training CNN on {len(unsolved)} unsolved "
          f"with {WORKERS} workers", flush=True)
    (OUT / "onnx").mkdir(parents=True, exist_ok=True)

    new = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(WORKERS) as pool:
        for r in pool.imap_unordered(_train_one, unsolved):
            if r:
                tnum, name, pts, par, mem = r
                new[tnum] = dict(task=tnum, name=name, points=round(pts, 2), params=par, memory=mem)
                print(f"  [cnn] task{tnum:03d} {name} {pts:.2f}  (new total solves {len(new)})", flush=True)

    # Merge: update report tasks with new CNN solves, rebuild submission.zip.
    by_task = {t["task"]: t for t in report["tasks"]}
    for tnum, rec in new.items():
        by_task[tnum] = rec
    tasks = [by_task[i] for i in sorted(by_task)]
    total = sum(t["points"] for t in tasks if t.get("name"))
    nsolved = sum(1 for t in tasks if t.get("name"))
    json.dump(dict(solved=nsolved, total_points=round(total, 3), tasks=tasks),
              open(OUT / "report.json", "w"), indent=2)
    with zipfile.ZipFile(OUT / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((OUT / "onnx").glob("task*.onnx")):
            z.write(f, f.name)
    print(f"[gpu_cnn] DONE: +{len(new)} CNN solves -> total solved {nsolved}, "
          f"total_points {total:.2f}", flush=True)


if __name__ == "__main__":
    main()
