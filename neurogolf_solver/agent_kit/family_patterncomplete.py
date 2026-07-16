"""PATTERN COMPLETION: repair a periodic or symmetric mosaic that has an occluded
region (a "marker" colour, or a background hole), restoring the full pattern.

Idea (origin-anchored, opset-10)
--------------------------------
The input grid is (mostly) a regular pattern -- either spatially PERIODIC with a
period (p, q), or SYMMETRIC under a mirror / rotation -- except for an occluded
region whose cells all carry one fixed MARKER colour ``C`` (a special non-bg
colour, or the background 0).  The output keeps every VISIBLE cell verbatim and
restores every occluded cell from the pattern:

        output[i, j] = source[i, j]   if input[i, j] == C   (occluded)
        output[i, j] = input[i, j]    otherwise             (visible / kept)

``source`` is reconstructed from the VISIBLE cells only, after REMOVING the marker
(``clean`` = input with colour channel ``C`` zeroed, so occluded cells contribute
nothing):

  * PERIODIC repair (SIZE-INDEPENDENT).  Fold ``clean`` onto its (p, q) residues
    (Reshape + ReduceMax over the two block axes -- the opset-10 trick), recover the
    period block, Tile it back over the whole 30x30, and ``Where(input==C, tile,
    input)``.  This is origin-safe for grids of ANY size: the occluded mask is 0 on
    the zero padding (padding is all-zero, incl. channel 0), so ``Where`` leaves the
    pad untouched and only real occluded cells are filled.  The period (p, q) is the
    structural constant; the grid size may vary across examples.

  * DIAGONAL symmetry repair (SIZE-INDEPENDENT) via ``Transpose`` (origin-safe).

  * MIRROR / ROTATION symmetry repair (fliplr / flipud / rot180 / anti-transpose /
    4-fold quad / full D4).  These reflect about the grid CENTRE, which depends on
    the (variable) size, so -- per the padding gotcha -- they are only emitted for
    tasks whose grid size is CONSTANT across every split, realised as WINDOWED
    transforms on the [0:H,0:W] sub-grid zero-padded back to 30x30.  ``source`` is
    the one-hot OR of the windowed copies of ``clean``, so a cell occluded in one
    copy is filled from whichever mirror image of it is still visible.

The marker colour and the period / symmetry are detected STRUCTURALLY from the
visible cells, and every candidate is validated for EXACT equality on all available
train+test+arc-gen pairs before it is emitted (the grader's gate), so wrong
hypotheses are dropped before scoring.  This extends ``family_symfixed`` (single
mirror, background hole only) and ``family_periodic`` (full periodic rewrite) with
arbitrary-marker, multi-copy, Where-anchored repair.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL
_NEG = -(1 << 31)


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals):
        nm = self.name("c")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(dims),
                                         [float(v) for v in np.asarray(vals).ravel()]))
        return nm

    def i(self, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)], [int(v) for v in vals]))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


# --------------------------------------------------------------------------- #
# shared building blocks                                                       #
# --------------------------------------------------------------------------- #
def _remove_marker(g, src, C):
    """clean = src with colour channel C zeroed -> occluded cells contribute 0."""
    mvals = [1.0] * CHANNELS
    mvals[C] = 0.0
    return g.node("Mul", [src, g.f([1, CHANNELS, 1, 1], mvals)])


def _colored_merge(g, copies):
    """OR (one-hot) the colour channels 1..9 of the given copies and recompute a
    valid background channel -> a [1,10,*] one-hot tensor (no leftover marker)."""
    cs, ce, cax = g.i([1]), g.i([CHANNELS]), g.i([1])
    cols = [g.node("Slice", [c, cs, ce, cax]) for c in copies]      # [1,9,*]
    colored = cols[0] if len(cols) == 1 else g.node("Max", cols)
    present = g.node("ReduceMax", [colored], axes=[1], keepdims=1)   # [1,1,*]
    bg = g.node("Sub", [g.f([1, 1, 1, 1], [1.0]), present])
    return g.node("Concat", [bg, colored], axis=1)                   # [1,10,*]


def _where_fill(g, src, C, out="output"):
    """output = Where(input==C, src, input).  Padding stays zero (channel C is 0
    on all-zero padding)."""
    chanC = g.node("Slice", ["input", g.i([C]), g.i([C + 1]), g.i([1])])  # [1,1,30,30]
    cond = g.node("Cast", [chanC], to=BOOL)
    return g.node("Where", [cond, src, "input"], out)


# --------------------------------------------------------------------------- #
# window helpers (constant-size windowed flips)                                #
# --------------------------------------------------------------------------- #
def _window(g, src, H, W):
    if H == HEIGHT and W == WIDTH:
        return src
    return g.node("Slice", [src, g.i([0, 0]), g.i([H, W]), g.i([2, 3])])


def _unwindow(g, src, H, W, out):
    if H == HEIGHT and W == WIDTH:
        return g.node("Identity", [src], out)
    return g.node("Pad", [src], out, mode="constant", value=0.0,
                  pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W])


def _copy_win(g, win, key):
    """Transform a [1,10,H,W] window (square required for T / antiT)."""
    if key == "id":
        return win
    if key == "T":
        return g.node("Transpose", [win], perm=[0, 1, 3, 2])
    if key == "antiT":
        r = _copy_win(g, win, "rot180")
        return g.node("Transpose", [r], perm=[0, 1, 3, 2])
    axes = {"fliplr": [3], "flipud": [2], "rot180": [2, 3]}[key]
    n = len(axes)
    return g.node("Slice", [win, g.i([-1] * n), g.i([_NEG] * n),
                            g.i(axes), g.i([-1] * n)])


# --------------------------------------------------------------------------- #
# ONNX builders                                                               #
# --------------------------------------------------------------------------- #
def build_per(C, p, q):
    """SIZE-INDEPENDENT periodic repair (period (p,q) constant; any grid size)."""
    g = _G()
    clean = _remove_marker(g, "input", C)                       # [1,10,30,30]
    Hp = ((HEIGHT + p - 1) // p) * p
    Wp = ((WIDTH + q - 1) // q) * q
    if Hp != HEIGHT or Wp != WIDTH:
        clean = g.node("Pad", [clean], mode="constant", value=0.0,
                       pads=[0, 0, 0, 0, 0, 0, Hp - HEIGHT, Wp - WIDTH])
    resh = g.node("Reshape", [clean, g.i([1, CHANNELS, Hp // p, p, Wp // q, q])])
    base = g.node("ReduceMax", [resh], axes=[2, 4], keepdims=0)  # [1,10,p,q]
    block = _colored_merge(g, [base])                           # one-hot [1,10,p,q]
    rh = (HEIGHT + p - 1) // p
    rw = (WIDTH + q - 1) // q
    tiled = g.node("Tile", [block, g.i([1, 1, rh, rw])])        # >= 30x30
    src = g.node("Slice", [tiled, g.i([0, 0]), g.i([HEIGHT, WIDTH]), g.i([2, 3])])
    _where_fill(g, src, C)
    return _model(g.nodes, g.inits)


def _roll_block(g, src, p, q, main):
    """Wrap-shift a [1,9,p,q] block one step along the (anti)diagonal (rows down by
    1; cols left for anti / right for main), via Slice + Concat (opset-10 safe)."""
    r_last = g.node("Slice", [src, g.i([p - 1]), g.i([p]), g.i([2])])
    r_head = g.node("Slice", [src, g.i([0]), g.i([p - 1]), g.i([2])])
    rowsh = g.node("Concat", [r_last, r_head], axis=2)
    if not main:                                   # anti: cols shift left (wrap)
        c_tail = g.node("Slice", [rowsh, g.i([1]), g.i([q]), g.i([3])])
        c_head = g.node("Slice", [rowsh, g.i([0]), g.i([1]), g.i([3])])
        return g.node("Concat", [c_tail, c_head], axis=3)
    c_last = g.node("Slice", [rowsh, g.i([q - 1]), g.i([q]), g.i([3])])  # main: right
    c_head = g.node("Slice", [rowsh, g.i([0]), g.i([q - 1]), g.i([3])])
    return g.node("Concat", [c_last, c_head], axis=3)


def build_per_roll(C, p, mode):
    """SIZE-INDEPENDENT diagonal-stripe periodic repair (period (p,p)): fold to a
    block, OR all (anti)diagonal rolls of its colour channels, then tile + fill."""
    q = p
    g = _G()
    clean = _remove_marker(g, "input", C)
    Hp = ((HEIGHT + p - 1) // p) * p
    Wp = ((WIDTH + q - 1) // q) * q
    if Hp != HEIGHT or Wp != WIDTH:
        clean = g.node("Pad", [clean], mode="constant", value=0.0,
                       pads=[0, 0, 0, 0, 0, 0, Hp - HEIGHT, Wp - WIDTH])
    resh = g.node("Reshape", [clean, g.i([1, CHANNELS, Hp // p, p, Wp // q, q])])
    base = g.node("ReduceMax", [resh], axes=[2, 4], keepdims=0)   # [1,10,p,q]
    colored = g.node("Slice", [base, g.i([1]), g.i([CHANNELS]), g.i([1])])  # [1,9,p,q]
    full = colored
    cur = colored
    for _ in range(p - 1):
        cur = _roll_block(g, cur, p, q, mode == "main")
        full = g.node("Max", [full, cur])
    present = g.node("ReduceMax", [full], axes=[1], keepdims=1)
    bg = g.node("Sub", [g.f([1, 1, 1, 1], [1.0]), present])
    block = g.node("Concat", [bg, full], axis=1)                  # [1,10,p,q]
    rh = (HEIGHT + p - 1) // p
    rw = (WIDTH + q - 1) // q
    tiled = g.node("Tile", [block, g.i([1, 1, rh, rw])])
    src = g.node("Slice", [tiled, g.i([0, 0]), g.i([HEIGHT, WIDTH]), g.i([2, 3])])
    _where_fill(g, src, C)
    return _model(g.nodes, g.inits)


def build_diag(C):
    """SIZE-INDEPENDENT diagonal (transpose) symmetry repair."""
    g = _G()
    clean = _remove_marker(g, "input", C)
    src = g.node("Transpose", [clean], perm=[0, 1, 3, 2])        # [1,10,30,30]
    _where_fill(g, src, C)
    return _model(g.nodes, g.inits)


def build_sym(C, keys, H, W):
    """Constant-size windowed mirror / rotation repair: source = OR(copies(clean))."""
    g = _G()
    win = _window(g, "input", H, W)
    clean = _remove_marker(g, win, C)
    copies = [_copy_win(g, clean, k) for k in keys]
    filled = _colored_merge(g, copies)                          # [1,10,H,W]
    chanC = g.node("Slice", [win, g.i([C]), g.i([C + 1]), g.i([1])])
    cond = g.node("Cast", [chanC], to=BOOL)
    out = g.node("Where", [cond, filled, win])
    _unwindow(g, out, H, W, "output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX semantics exactly)                         #
# --------------------------------------------------------------------------- #
def _transform(a, key):
    if key == "id":
        return a
    if key == "fliplr":
        return a[:, ::-1]
    if key == "flipud":
        return a[::-1, :]
    if key == "rot180":
        return a[::-1, ::-1]
    if key == "T":
        return a.T
    if key == "antiT":
        return a[::-1, ::-1].T
    raise ValueError(key)


def _recon_per(a, C, p, q):
    H, W = a.shape
    clean = a.copy()
    clean[a == C] = 0
    base = np.full((p, q), -1, np.int64)
    for i in range(H):
        ip = i % p
        for j in range(W):
            v = clean[i, j]
            if v == 0:
                continue
            r, c = ip, j % q
            if base[r, c] == -1:
                base[r, c] = v
            elif base[r, c] != v:
                return None                      # two colours per residue
    base[base == -1] = 0
    src = base[np.arange(H)[:, None] % p, np.arange(W)[None, :] % q]
    occ = (a == C)
    return np.where(occ, src, a)


def _recon_per_roll(a, C, p, mode):
    """Mirror build_per_roll: residue fold (colours only) + diagonal roll-OR."""
    H, W = a.shape
    q = p
    clean = a.copy()
    clean[a == C] = 0
    oh_ = np.zeros((CHANNELS - 1, p, q), np.int64)
    for i in range(H):
        for j in range(W):
            v = clean[i, j]
            if v != 0:
                oh_[v - 1, i % p, j % q] = 1
    main = (mode == "main")

    def roll(b):
        b2 = np.concatenate([b[:, p - 1:p, :], b[:, 0:p - 1, :]], axis=1)
        if main:
            return np.concatenate([b2[:, :, q - 1:q], b2[:, :, 0:q - 1]], axis=2)
        return np.concatenate([b2[:, :, 1:q], b2[:, :, 0:1]], axis=2)

    full = oh_.copy()
    cur = oh_.copy()
    for _ in range(p - 1):
        cur = roll(cur)
        full = np.maximum(full, cur)
    if (full.sum(axis=0) > 1).any():               # colour conflict on a residue
        return None
    blk = np.zeros((p, q), np.int64)
    for v in range(1, CHANNELS):
        blk[full[v - 1] > 0] = v
    src = blk[np.arange(H)[:, None] % p, np.arange(W)[None, :] % q]
    occ = (a == C)
    return np.where(occ, src, a)


def _recon_diag(a, C):
    H, W = a.shape
    if H != W:
        return None
    clean = a.copy()
    clean[a == C] = 0
    occ = (a == C)
    return np.where(occ, clean.T, a)


def _recon_sym(a, C, keys):
    H, W = a.shape
    if any(k in ("T", "antiT") for k in keys) and H != W:
        return None
    clean = a.copy()
    clean[a == C] = 0
    copies = [np.asarray(_transform(clean, k)) for k in keys]
    arr = np.stack(copies, axis=0)                 # [K,H,W]
    nz = arr != 0
    anynz = nz.any(axis=0)
    mx = arr.max(axis=0)
    masked = np.where(nz, arr, 99)
    mn = masked.min(axis=0)
    occ = (a == C)
    if (anynz & (mx != mn) & occ).any():           # ambiguous fill -> invalid
        return None
    filled = np.where(anynz, mx, 0)
    return np.where(occ, filled, a)


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _matches(prs, fn):
    for a, b in prs:
        o = fn(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if any(a.shape != b.shape for a, b in prs):          # repair preserves shape
        return []
    if all(np.array_equal(a, b) for a, b in prs):         # identity -> not us
        return []

    # marker colour: the single input colour present at every CHANGED cell
    ci = set()
    for a, b in prs:
        d = a != b
        if d.any():
            ci |= set(a[d].tolist())
    if len(ci) != 1:
        return []
    C = ci.pop()

    out, seen = [], set()

    def add(name, builder):
        if name in seen:
            return
        try:
            m = builder()
            onnx.checker.check_model(m, full_check=True)
        except Exception:
            return
        seen.add(name)
        out.append((name, m))

    # ---- PERIODIC repair (size-independent; smallest matching period) ------- #
    maxH = max(a.shape[0] for a, _ in prs)
    maxW = max(a.shape[1] for a, _ in prs)
    pmax = min(maxH, 15)
    qmax = min(maxW, 15)
    done_per = False
    for s in range(2, pmax + qmax + 1):                  # prefer small p*q-ish
        for p in range(1, pmax + 1):
            q = s - p
            if q < 1 or q > qmax:
                continue
            if p == 1 and q == 1:
                continue
            if _matches(prs, lambda a, pp=p, qq=q: _recon_per(a, C, pp, qq)):
                add(f"per_{p}x{q}_C{C}", lambda pp=p, qq=q: build_per(C, pp, qq))
                done_per = True
                break
        if done_per:
            break

    # ---- diagonal-stripe periodic repair (size-independent; roll fill) ------ #
    if not done_per:
        done_roll = False
        for p in range(2, min(maxH, maxW, 15) + 1):
            for mode in ("anti", "main"):
                if _matches(prs, lambda a, pp=p, mm=mode: _recon_per_roll(a, C, pp, mm)):
                    add(f"perroll_{p}_{mode}_C{C}", lambda pp=p, mm=mode: build_per_roll(C, pp, mm))
                    done_roll = True
                    break
            if done_roll:
                break

    # ---- DIAGONAL (transpose) symmetry repair (size-independent) ------------ #
    if all(a.shape[0] == a.shape[1] for a, _ in prs):
        if _matches(prs, lambda a: _recon_diag(a, C)):
            add(f"diag_C{C}", lambda: build_diag(C))

    # ---- MIRROR / ROTATION repair (constant size only; windowed) ------------ #
    shapes = {a.shape for a, _ in prs}
    if len(shapes) == 1:
        H, W = next(iter(shapes))
        square = (H == W)
        keysets = [("lr", ["fliplr"]), ("ud", ["flipud"]), ("rot180", ["rot180"])]
        if square:
            keysets += [("antidiag", ["antiT"])]
        keysets += [("quad", ["fliplr", "flipud", "rot180"])]
        if square:
            keysets += [("d4", ["fliplr", "flipud", "rot180", "T", "antiT"])]
        for tag, keys in keysets:
            if _matches(prs, lambda a, k=keys: _recon_sym(a, C, k)):
                add(f"sym_{tag}_C{C}", lambda k=keys: build_sym(C, k, H, W))

    return out
