"""family_gw2_a — from-scratch cheap-ONNX attempts for golf ranks 0..14.

Assigned targets (golf_targets.json[0:15]), with the MINIMAL true rule read from
_rearc/verifiers.py and empirically confirmed against train+test+arc-gen:

  158 6aa20dc0  grow/extend objects along a learned trajectory        (obj, variable)
  285 b775ac94  fractal reflection expansion of a seed shape           (obj)
  233 97a05b5b  match/place sub-objects into a container, crop         (obj, VARIABLE crop)
   18 0e206a2e  reflect a 4-colour key onto single-cell markers        (obj, CC + template)
   46 234bbc79  gravity/merge of coloured bars                          (obj, VARIABLE crop)
  319 ce602527  select object matching a criterion, recolour, crop     (obj, VARIABLE crop)
  396 fcb5c309  pick the box whose border is complete, recolour, crop  (obj, VARIABLE crop)
  216 8efcae92  select object with most minority-colour cells, crop    (obj, VARIABLE crop)
  365 e50d258f  select object by minority-colour count, crop           (obj, VARIABLE crop)
  173 72322fa7  2-colour template stamp on every colour-component
                occurrence (marker cells AND full arm-shape matches)   (CC + runtime multi-template)
  219 90f3ed37  shoot the widest horizontal ray, continue in colour 1  (obj, ray mechanic)
  133 57aa92db  upscaled 2-colour template stamp on markers            (CC + template + scale)
   54 264363fd  draw a 2/3 cross through the marker inside each frame   (frame/rectangle detect)
   66 2dd70a9a  draw an L-pipe of 3 connecting two markers             (path tracing)
  392 f8c80d96  complete a boustrophedon/spiral fill from a seed        (spiral trace)

CONCLUSION (why nothing is yielded):
Every one of these is object-level.  The variable-output tasks (233/46/319/396/216/365)
require cropping a data-dependently-sized selected object — explicitly banned for
arc-gen sets with >1 distinct grid size.  The same-size tasks require either
connected-component isolation (unavailable in opset-10 without NonZero/Loop), runtime
extraction of a *variable number* of per-example templates (173/133/18), or
path/ray/spiral tracing (219/66/392/54).  None reduce to a global pointwise / flip /
tile / fixed-kernel-Conv graph, and none can be expressed CHEAPLY (a correct build
would need many full-grid correlations + multi-template ConvTranspose stamps, whose
intermediate memory would not reliably beat the incumbents and whose exactness over
~266 arc-gen cases is not achievable by a static graph).

The confirmed pure-numpy rule for task 173 is retained below (`_ref_173`, exact
266/266) as documentation for a future wave that has a connected-components primitive;
it is NOT emitted because there is no cheap opset-10 realisation.

`candidates()` therefore yields nothing — a strictly no-regression module.
"""
from __future__ import annotations

from collections import deque

import numpy as np


# --------------------------------------------------------------------------- #
# Confirmed numpy reference for task 173 (72322fa7) — exact on all 266 pairs.  #
# Kept for documentation only; not expressible as cheap opset-10 ONNX.         #
# --------------------------------------------------------------------------- #
def _components(a, bg):
    H, W = a.shape
    seen = np.zeros_like(a, bool)
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
          (0, 1), (1, -1), (1, 0), (1, 1)]
    out = []
    for i in range(H):
        for j in range(W):
            if a[i, j] == bg or seen[i, j]:
                continue
            q = deque([(i, j)])
            seen[i, j] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in nb:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and a[ny, nx] != bg:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            out.append(cells)
    return out


def _ref_173(a):
    a = np.asarray(a, int)
    H, W = a.shape
    vals, cnts = np.unique(a, return_counts=True)
    bg = vals[cnts.argmax()]
    out = a.copy()
    for cells in _components(a, bg):
        cols = sorted({int(a[y, x]) for y, x in cells})
        if len(cols) != 2:
            continue
        full = [(y, x, int(a[y, x])) for y, x in cells]
        for col in cols:
            sub = [(y, x) for y, x in cells if a[y, x] == col]
            sy = min(y for y, x in sub)
            sx = min(x for y, x in sub)
            subn = [(y - sy, x - sx) for y, x in sub]
            sh = max(y for y, x in subn) + 1
            sw = max(x for y, x in subn) + 1
            for i in range(H - sh + 1):
                for j in range(W - sw + 1):
                    if all(a[i + y, j + x] == col for y, x in subn):
                        for (fy, fx, fc) in full:
                            ny, nx = fy - sy + i, fx - sx + j
                            if 0 <= ny < H and 0 <= nx < W:
                                out[ny, nx] = fc
    return out


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def candidates(examples):
    """No cheap, exact opset-10 realisation exists for any assigned target.

    Yielding nothing keeps this module strictly no-regression: the grader takes
    the max over families, so an empty family can only ever leave incumbents
    untouched.
    """
    return
    yield  # pragma: no cover  (marks this a generator)
