"""ngpatterns — catalog of proven generator-idiom -> cheap-ONNX-primitive mappings,
plus a scanner that finds which OTHER tasks match each idiom by grepping the 400
ARC-GEN generator sources. Turns "can task X use method Y?" into a systematic sweep.

Each PATTERN records:
  name         short id
  primitive    the ngbuild primitive(s) that implement it cheaply
  won_on       tasks already solved/proven with it
  signature    regex(es) over the generator source that flag a candidate
  antisig      regex(es) that DISQUALIFY (a known trap that breaks the method)
  note         why it works / the catch

Usage:
  python ngpatterns.py                 list patterns + candidate tasks (cross-ref cost)
  python ngpatterns.py <pattern>       detail one pattern's candidates
"""
import os, re, sys, json, glob
import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENDIR = os.path.join(KIT, "_arcgen", "tasks")
HM = json.load(open(os.path.join(KIT, "task_hash_map.json")))
HASH2TASK = {v: int(k) for k, v in HM.items()}

PATTERNS = [
    dict(name="periodic-texture",
         primitive="residue-class dilated MaxPool (e017)",
         won_on=[17],
         signature=[r"%\s*\w*length", r"\*\s*\w*\)\s*%\s*\w*mod", r"r\s*\*\s*r\s*\+\s*c\s*\*\s*c"],
         antisig=[r"len\(colors\)\s*//\s*mod"],   # anisotropic colors-branch (t110): row-period != col-period
         note="whole grid doubly-periodic v(r,c); class-max recovers cutouts. TRAP: a "
              "colors[] override with row-period != col-period (t110) breaks single-L."),
    dict(name="directional-ray-fill",
         primitive="directional MaxPool / dir_reach (e138)",
         won_on=[138],
         signature=[r"dr,\s*dc", r"r\s*\+\s*dr.*c\s*\+\s*dc", r"while True:.*break"],
         antisig=[],
         note="seeds cast rays in ONE direction until a boundary = one-sided prefix-max. "
              "TRAP: rays that stop at OBSTACLES (not just the frame) need bounded fill."),
    dict(name="guillotine-rectangles",
         primitive="wall-distance MaxPool + Equal-vs-extremum (t145)",
         won_on=[145],
         signature=[r"bisect", r"cut", r"randint.*wide", r"randint.*tall"],
         antisig=[],
         note="rectangles bounded by lines -> per-cell width*height via 4 directional "
              "MaxPools; select min/max area by Equal-vs-Reduce."),
    dict(name="magnified-stamp",
         primitive="runtime-weight QLinearConv detect+paint (e191/e319/e101)",
         won_on=[191, 319, 209, 5],
         signature=[r"mag", r"for dr in range\(.*mag", r"bmag|megarot|magnif|upsample"],
         antisig=[],
         note="a sprite stamped at anchors with mag/rotation. Correlate the (dilated) "
              "template, adjoint-paint. COST-WIN NEEDS A SMALL CANVAS: when copies span the "
              "full grid (t101/t255) the full-canvas paint planes cost MORE than the "
              "incumbent's bbox-coordinate tensors -> cleaner but costlier (hedge-only). "
              "Wins on cost only when the work fits a cropped canvas (t319 19x19, t005/t209). "
              "TRAP: red/marker periodicity -> ghost anchors; needs a private-red filter."),
    dict(name="radius-enumeration",
         primitive="per-R QLinearConv channels + grouped paint (t349)",
         won_on=[349],
         signature=[r"radius", r"for.*range\(.*radius", r"factor"],
         antisig=[],
         note="a variable size drawn from a small set {1..k} -> one exact-match channel "
              "per size collapses the size search to k static branches."),
    dict(name="bit-plane-flood",
         primitive="bit-packed u32 shift/AND/OR flood (bpk286/t002)",
         won_on=[286, 2],
         signature=[r"queue", r"while queue", r"flood|bfs|connected"],
         antisig=[],
         note="reachability/flood on a maze -> bit-packed rows, log-depth doubling. "
              "TRAP: nearest-SEED (distance) needs the front, not just reachability."),
    dict(name="marker-shape-match",
         primitive="exact-cover matching (HARD — sometimes irreducible, t233/t285)",
         won_on=[],
         signature=[r"exact.?cover|hole|marker", r"D4|orientation|reflect|rotate"],
         antisig=[],
         note="fill holes by matching marker shapes. Often needs propagation/joint "
              "constraint -> the 'spaghetti' IS the floor (t285 proven no-ship)."),
]


def gensrc(task):
    p = os.path.join(GENDIR, "task_%s.py" % HM[str(task)])
    return open(p).read() if os.path.exists(p) else ""


def scan_pattern(pat, points=None):
    """Return tasks whose generator matches all signatures and no antisig."""
    hits = []
    for f in glob.glob(os.path.join(GENDIR, "task_*.py")):
        h = os.path.basename(f)[5:-3]
        tn = HASH2TASK.get(h)
        if tn is None:
            continue
        src = open(f).read()
        gen_body = src[src.find("def generate"):src.find("def validate")] if "def validate" in src else src
        if not all(re.search(s, gen_body) for s in pat["signature"]):
            continue
        if any(re.search(a, gen_body) for a in pat["antisig"]):
            continue
        hits.append(tn)
    return sorted(hits)


def main(argv):
    # cost cross-ref (which candidates are still expensive = worth a rebuild)
    rep = {}
    for d in ("out_blend19", "out_blend16"):
        rp = os.path.join(KIT, d, "report.json")
        if os.path.exists(rp):
            rep = {t["task"]: t["points"] for t in json.load(open(rp))["tasks"]}
            break
    only = argv[0] if argv else None
    for pat in PATTERNS:
        if only and pat["name"] != only:
            continue
        hits = scan_pattern(pat)
        won = set(pat["won_on"])
        fresh = [t for t in hits if t not in won]
        # rank fresh candidates by how expensive they still are (low points = headroom)
        fresh_ranked = sorted(fresh, key=lambda t: rep.get(t, 25))
        print("\n=== %s  ->  %s" % (pat["name"], pat["primitive"]))
        print("    won: %s" % (sorted(won) or "-"))
        print("    NEW candidates (by rising score; low=more headroom):")
        for t in fresh_ranked[:14]:
            pts = rep.get(t, None)
            flag = "  <-- cheap already" if pts and pts >= 18 else ""
            print("      t%03d  %s%s" % (t, ("%.2f" % pts) if pts else "?", flag))
        if len(fresh_ranked) > 14:
            print("      ... +%d more" % (len(fresh_ranked) - 14))
        print("    note: %s" % pat["note"])


if __name__ == "__main__":
    main(sys.argv[1:])
