"""task367 (ARC e73095fd) — fill 'cleanly drawn' rectangular rooms with 4.
Rule decoded from Hodel's verifier (validated 266/266 on NeuroGolf data):
  fill a background component iff (a) it is a SOLID RECTANGLE and (b) no wall cell sits just
  beyond any corner of its outbox (offsets (+-2,+-1),(+-1,+-2) from the component corner cells).
  (b) distinguishes deliberately-drawn boxes (walls stop at corners) from incidental pockets
  between crossing wall lines (walls extend past corners).
ONNX: taint = concave 2x2 (3-bg window, via Conv+ConvTranspose) OR corner-with-wall-beyond
(shift masks); propagate taint through bg by 24-step 4-dilation (max needed on arc-gen: 17);
fill = bg AND NOT taint. Single-channel [1,1,30,30] float chain throughout.
"""
import numpy as np
import onnx
from onnx import helper as oh
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = onnx.TensorProto.FLOAT
I64 = onnx.TensorProto.INT64
STEPS = 24


# ---------- numpy reference (gate) ---------- #
def _solve(inp):
    x = np.asarray(inp); H, W = x.shape
    B = (x == 0); Wl = (x == 5)
    if not set(np.unique(x)).issubset({0, 5}):
        return None
    T = np.zeros((H, W), bool)
    for i in range(H - 1):
        for j in range(W - 1):
            win = B[i:i + 2, j:j + 2]
            if win.sum() == 3:
                T[i:i + 2, j:j + 2] |= win
    def nb(i, j): return B[i, j] if 0 <= i < H and 0 <= j < W else False
    def wl(i, j): return Wl[i, j] if 0 <= i < H and 0 <= j < W else False
    for i in range(H):
        for j in range(W):
            if not B[i, j]:
                continue
            up, dn, lf, rt = nb(i - 1, j), nb(i + 1, j), nb(i, j - 1), nb(i, j + 1)
            if not up and not lf and (wl(i - 2, j - 1) or wl(i - 1, j - 2)): T[i, j] = True
            if not up and not rt and (wl(i - 2, j + 1) or wl(i - 1, j + 2)): T[i, j] = True
            if not dn and not lf and (wl(i + 2, j - 1) or wl(i + 1, j - 2)): T[i, j] = True
            if not dn and not rt and (wl(i + 2, j + 1) or wl(i + 1, j + 2)): T[i, j] = True
    while True:
        Td = T.copy()
        Td[1:, :] |= T[:-1, :]; Td[:-1, :] |= T[1:, :]
        Td[:, 1:] |= T[:, :-1]; Td[:, :-1] |= T[:, 1:]
        Td &= B; Td |= T
        if (Td == T).all():
            break
        T = Td
    y = x.copy(); y[B & ~T] = 4
    return y


def _const(name, arr, dt=F):
    a = np.asarray(arr)
    a = a.astype(np.int64) if dt == I64 else a.astype(np.float32)
    return oh.make_tensor(name, dt, list(a.shape) if a.shape else [1], a.flatten().tolist())


def _build():
    nodes, inits = [], []

    def C(name, arr, dt=F):
        inits.append(_const(name, arr, dt)); return name

    def shift(src, dst, dr, dc):
        """dst[i,j] = src[i-dr, j-dc] (shift content down/right by (dr,dc)), zero-filled."""
        pt, pb = max(dr, 0), max(-dr, 0)
        pl, pr = max(dc, 0), max(-dc, 0)
        nodes.append(oh.make_node("Pad", [src], [dst + "_p"], mode="constant", value=0.0,
                                  pads=[0, 0, pt, pl, 0, 0, pb, pr]))
        sr, sc = pb, pr
        C(dst + "_s", [0, 0, sr, sc], I64); C(dst + "_e", [1, 1, sr + 30, sc + 30], I64)
        C(dst + "_a", [0, 1, 2, 3], I64)
        nodes.append(oh.make_node("Slice", [dst + "_p", dst + "_s", dst + "_e", dst + "_a"], [dst]))
        return dst

    # B = channel0, W = channel5  (out-of-grid one-hot rows are all-zero -> not bg, not wall: OK)
    C("s0", [0], I64); C("e1", [1], I64); C("ax1", [1], I64)
    nodes.append(oh.make_node("Slice", ["input", "s0", "e1", "ax1"], ["B"]))       # [1,1,30,30]
    C("s5", [5], I64); C("e6", [6], I64)
    nodes.append(oh.make_node("Slice", ["input", "s5", "e6", "ax1"], ["W"]))

    C("half", [0.5]); C("one", [1.0])
    nodes.append(oh.make_node("Sub", ["one", "B"], ["nB"]))                        # not-bg

    # ---- taint 1: 2x2 windows with exactly 3 bg ---- #
    C("k22", np.ones((1, 1, 2, 2)))
    nodes.append(oh.make_node("Conv", ["B", "k22"], ["w22"], kernel_shape=[2, 2]))  # [1,1,29,29]
    C("c25", [2.5]); C("c35", [3.5])
    nodes.append(oh.make_node("Greater", ["w22", "c25"], ["g25"]))
    nodes.append(oh.make_node("Less", ["w22", "c35"], ["l35"]))
    nodes.append(oh.make_node("And", ["g25", "l35"], ["eq3b"]))
    nodes.append(oh.make_node("Cast", ["eq3b"], ["eq3"], to=F))
    nodes.append(oh.make_node("ConvTranspose", ["eq3", "k22"], ["spread"],
                              kernel_shape=[2, 2]))                                 # [1,1,30,30]
    nodes.append(oh.make_node("Mul", ["spread", "B"], ["t1f"]))                     # >0 where tainted

    # ---- taint 2: component corners with wall beyond outbox corner ---- #
    shift("B", "Bup", 1, 0)    # Bup[i,j]=B[i-1,j]  (bg above)
    shift("B", "Bdn", -1, 0)
    shift("B", "Blf", 0, 1)
    shift("B", "Brt", 0, -1)
    for nm in ["Bup", "Bdn", "Blf", "Brt"]:
        nodes.append(oh.make_node("Sub", ["one", nm], ["n" + nm]))
    # corner masks (float products)
    nodes.append(oh.make_node("Mul", ["B", "nBup"], ["c_u"]))
    nodes.append(oh.make_node("Mul", ["c_u", "nBlf"], ["tl"]))
    nodes.append(oh.make_node("Mul", ["c_u", "nBrt"], ["tr"]))
    nodes.append(oh.make_node("Mul", ["B", "nBdn"], ["c_d"]))
    nodes.append(oh.make_node("Mul", ["c_d", "nBlf"], ["bl"]))
    nodes.append(oh.make_node("Mul", ["c_d", "nBrt"], ["br"]))
    # walls beyond corners: for tl need W[i-2,j-1] or W[i-1,j-2]  -> shift W by (2,1)/(1,2)
    pairs = [("tl", (2, 1), (1, 2)), ("tr", (2, -1), (1, -2)),
             ("bl", (-2, 1), (-1, 2)), ("br", (-2, -1), (-1, -2))]
    tsum = ["t1f"]
    for nm, o1, o2 in pairs:
        shift("W", f"W{nm}a", o1[0], o1[1])
        shift("W", f"W{nm}b", o2[0], o2[1])
        nodes.append(oh.make_node("Add", [f"W{nm}a", f"W{nm}b"], [f"W{nm}s"]))
        nodes.append(oh.make_node("Mul", [nm, f"W{nm}s"], [f"t_{nm}"]))
        tsum.append(f"t_{nm}")
    # total taint seed (float, >0 where tainted)
    acc = tsum[0]
    for i, t in enumerate(tsum[1:]):
        nodes.append(oh.make_node("Add", [acc, t], [f"acc{i}"])); acc = f"acc{i}"
    nodes.append(oh.make_node("Greater", [acc, "half"], ["T0b"]))
    nodes.append(oh.make_node("Cast", ["T0b"], ["T0"], to=F))

    # ---- propagate taint through bg: T = B * clip(T + dilate4(T)) , STEPS times ---- #
    plus = np.zeros((1, 1, 3, 3)); plus[0, 0, 1, :] = 1; plus[0, 0, :, 1] = 1
    C("kplus", plus)
    cur = "T0"
    for s in range(STEPS):
        nodes.append(oh.make_node("Conv", [cur, "kplus"], [f"d{s}"], kernel_shape=[3, 3],
                                  pads=[1, 1, 1, 1]))
        nodes.append(oh.make_node("Mul", [f"d{s}", "B"], [f"db{s}"]))
        nodes.append(oh.make_node("Greater", [f"db{s}", "half"], [f"gb{s}"]))
        nodes.append(oh.make_node("Cast", [f"gb{s}"], [f"T{s+1}"], to=F))
        cur = f"T{s+1}"

    # ---- output: ch0 = B*T (tainted bg stays 0-color), ch4 = B*(1-T), ch5 = W ---- #
    nodes.append(oh.make_node("Mul", ["B", cur], ["out0"]))
    nodes.append(oh.make_node("Sub", ["B", "out0"], ["out4"]))
    nodes.append(oh.make_node("Sub", ["B", "B"], ["Z"]))                            # zero plane
    nodes.append(oh.make_node("Concat",
                              ["out0", "Z", "Z", "Z", "out4", "W", "Z", "Z", "Z", "Z"],
                              ["output"], axis=1))

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "hc367", [x], [y], inits)
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return
    for a, b in prs:
        r = _solve(a)
        if r is None or r.shape != b.shape or not np.array_equal(r, b):
            return
    yield ("hc367_cleanroom", _build())
