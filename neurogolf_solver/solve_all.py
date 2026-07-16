"""Driver: solve a range of tasks, write taskNNN.onnx for each solved task,
emit a report, and package submission.zip.

Usage:
    python solve_all.py [--start 1] [--end 400] [--out ../submission]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import zipfile

from ng_utils_shim import tasks_dir
from evaluate import save_solution
from solve import solve_task


def load_task(tnum, tdir):
    with open(tdir / f"task{tnum:03d}.json") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=400)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(__file__).parent / "out")
    args = ap.parse_args()

    tdir = tasks_dir()
    onnx_dir = args.out / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    report, total_points, solved = [], 0.0, 0
    t0 = time.time()
    for tnum in range(args.start, args.end + 1):
        examples = load_task(tnum, tdir)
        best = solve_task(tnum, examples)
        if best:
            save_solution(best["model"], tnum, onnx_dir)
            total_points += best["points"]
            solved += 1
            report.append(dict(task=tnum, **{k: best[k] for k in
                          ("name", "points", "params", "memory", "cost")}))
            tag = f"{best['name']:<16} {best['points']:6.2f}pts  (cost={best['cost']})"
        else:
            report.append(dict(task=tnum, name=None, points=0.0))
            tag = "UNSOLVED"
        print(f"task{tnum:03d}: {tag}")

    dt = time.time() - t0
    summary = dict(range=[args.start, args.end], solved=solved,
                   attempted=args.end - args.start + 1,
                   total_points=round(total_points, 3), seconds=round(dt, 1))
    (args.out / "report.json").write_text(json.dumps(
        dict(summary=summary, tasks=report), indent=2))

    # Package submission.zip of all produced onnx files.
    zpath = args.out / "submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(onnx_dir.glob("task*.onnx")):
            z.write(f, f.name)

    print("\n" + "=" * 60)
    print(f"solved {solved}/{summary['attempted']}  "
          f"total_points={summary['total_points']}  ({dt:.1f}s)")
    print(f"submission: {zpath}")
    print("=" * 60)


if __name__ == "__main__":
    main()
