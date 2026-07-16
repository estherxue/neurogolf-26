"""family_lsp_5: FLOAT16-intermediate rebuilds of memory-dominated FP32 solvers.

Slice [5::7] of golf_targets. Every incumbent in out_p6/onnx was inspected: the
memory-heavy one-hot-carrying graphs (110 period-completion, 280 ray-smear, 268
diagonal rays, 117 symmetry-fill, 88 overlay ...) were already float16 and carry
MULTI-colour one-hot arithmetic (channel unions / max-projections) that does NOT
collapse to a single label channel, so a label-space rewrite could not be proven
byte-exact and the incumbent is kept (monotone, never regress).  The incumbents
that were still FLOAT32 were rebuilt so every intermediate tensor is float16
(2 bytes vs 4): a Cast lowers ``input`` to f16 at the top, all float initializers
are re-emitted f16, interior Cast-to-float are retargeted to f16, and a final Cast
lifts the result back to float32 before ``output``.  Only tasks whose f16 rebuild
is strictly CHEAPER and reproduces every train+test+arc-gen pair EXACTLY after the
grader (>0) threshold are embedded (183, 358, 39); float16 is exact for the small
one-hot sums (<=900 < 2048) these graphs compute, so behaviour is byte-identical.
Detection is behavioural + self-contained (each embedded graph is run and proposed
only when it matches), and the unchanged incumbent is emitted as a 2nd candidate so
the harness keeps whichever is cheaper.
"""
from __future__ import annotations

import base64
import gzip

import numpy as np
import onnx

try:
    import onnxruntime as _ort
except Exception:  # pragma: no cover
    _ort = None

# task -> gzip+base64 of the rebuilt float16 model (strictly cheaper, exact).
_FP16 = {
    39: "H4sIAAqgSGoC/6VVTW/aQBBdG5KYASnIQVWUqrRCPXHCu+uviINDvxKCiVRuvUTEOBVqEiJhpBz9U/JTuPbWn9SdtTe2KJWoKrQL83bmzZvxeDHg9GcDerA3f3hcJWZNfl3fWs5JPZouk+v5Axqd6gdhdGugJ4tjeNZ08KBwNfXE7dS+xrNVFE9W9906VKdP8TLQnrWD7iEYP+L4cTa/Xx5rGPm+FAn63BPLh8rc6pmVxLI6e5O7eRTDkUjlAiII005lsroBhgBFgKl84fTpJV9la74iiG8L0rcGHWEQAz2S6e1OJVzdZUw2As7uTKYgcRSRm9XRytgRQNQr0XsI+P8iFOkxRgTSXsYv1XNxgmVTq6CnqILS3Zun1EsiVlLPkY0hykv0HAF7d3qlnmJXqVPilyiOBHWLmqgr/GUO0bKz2Uy6Uoqbg6hfcvVzV9bLXCUohi1C0SzvCYKsp0CqQFGv4mSsCGdqIBjPEh0iKJOICTm7WUo9zFZ84rUZxcslvEZUkrl/vkitPBvDFjOvHIKzwPxtIUiFGz503itkW6pr3FKyM3aOsjktZHNMx1khm7NcNuclDRyr4/Z22ZiNSeJypRwr5e522dzGDaee51P/pnwdYDV45nf2w2mCx23EPNx8ExarJPd8OWdQQs397PdJQ95dC0TFUy0rwbEzte/dkaGJT9vQmjAQlQz7hJA+CciAfCSfyGfyhZyn5+QivSDDdEgu00syCkbpaD0iYRCm4Tok42CcjtdjchVc5WyCT7LR/2Qzc7ZMGxvqxNvAuMD6G5gtsGADc4b6r6hbF9bBqUYG4rZVhiYMv9tQBt6/396qf4FX0DI0swm6oYkFYrVx3byDvMF/8xhUgTThN74H5PlUBgAA",
    183: "H4sIAAqgSGoC/6VX608bRxC/J5hJ09AtIbxEWn9pZVWVY/qgER/cS5s4xCaRiUTVL9axXvA1Zx/4zoGP/kcq8afwp3V2dvd8fkWtiuWdm9l5/fZmZk0Jnv+9A1Xwo8HVKGNrRDoXz37aecDDNOtEA8mUvRfIVNbAyZItuLMd+A4mquDzHu+kighFQuYiKfunccQF1EByzLm4LK+1RXfExemoX3kAXngr0rp9Z69WHkHpgxBX3aifbtkywo+A6mxlmNw0wtSYtcLb3Mz9lBlP4iVmzkKzXdCRiEbdW+YNG52o7LZGMfwMxLCVfnjbTm7+k1eVCFHyyhudv3KvkiGvL5L430PcBJ0IuMlAMKfRK7u/drsox0fwemF8wXy5NlQgpY8htP7ZTa5/dlPUP1P6u+YMQHlhXpZcBWWvKdIU9oA45uI6XxZ7BqgyPWN+LC4yY7sPimWeJPPWG5QgSNfMPU+ysns6OocnSko2zB9Glz29sZOfg8bQH3biZPleL1K49yf4yIJ5/eFlUF59NRRhJoYSohQwF9f5JHenrHuRtI4LpyM5aRrPm26DdCmXmJWitKWqic58J39HJl8+g2V6r4DFHDhZYDZ8FgsnLHwxloI1YeFTWDhh4UuwcImFGyyyhgnLD8XZsPq++VHI6aAfhHkImU8PZkZUi1byiKudVFOhaYhNidRYYIKShTw8VgfyfdNeisMwbRlmweBZ3F7TieD5UCJEhaaYiKSFRCQL+TtlvuQniRDH/KC5JJHF0+PricMCxpXzYStMPyjfT6dyVTvMPx+eCv0ufgHFYfDZU3hogi85h0mb0NhYiWSvFErrK9Ai5hOdr5Enk0GV3SSyKWqqojdlD9R0OWPH1PQE28tDkpBi9iJTkCog8hQQO2BBPatUQCmg/SDLW2zSKhoPn8fDNR7+CTw0SBUeXsDDJ3h4AY8OSUKKOY2Hazx8KR6u8HCDJ2+zp6DhgRZjHw4QSJQMlcK3xeLw+odYxnIVtIbM6R+aAt4CZCA3R2SH+up7TNNYD1/3ffPaXCkkVtMY5e3rXB3ntlEPCupSbNSDgjpGAumXebgcmgtIjQYgGdMDI5mxaUubdtGmrWzaZNNeYBPIOEExTqDiBBQnWBQnkHGCYpxAxQkoTlCMs2eGWwImAeZeV5+ZajARwJjhbu1A7WI81AQpYN51Ouor8fbktVBCeDFeVs3F6KdxklVBipiHS2IuE3IAJGKrySh7OYpj5e97MDzW5jC5otkmqdBUzjakpjT+AGIZoJWupbL7LuxWvsQySrqiXOLJIM3CQXZnu5Vt8K7Cblq3Ch/HjDf/YxiPxGML/+5sGw6g4JOtqOedz+hnZyKlB7WpjpBDidmXlYclex0C2cPHjnVU+ZxYaj3kD802Niiy9cq7ko2ffRLq4XJ8hBkcYWqB9Zv1u/XSemU1xg3r9fi1dTw+tt6M31jNenPcvG9arXpr3LpvWSf1k/HJ/Yn1tv5We0Sf0qNq7//p8Rv0BtInelSv9HjDki5n/gjr6nPbDtQvbsOD4sXMflj5ouQg71hWYK5hI0IdcyEbkeMaUVh5pBxZgb6IjcDWAmEEjhYUTdSVWTBRd6cRuFoQ0qtCQSmg4WTYNWKFYW1iw8q6zhIjqro1kv19LRG5jqsl4Z9PzX82m7BRstk6OCUbv4Dfffk9xyGsKm+ZRuCBtQ7/AHjTZvYoDQAA",
    358: "H4sIAAqgSGoC/6VWXU/bMBR10kKD2USVoQ11WjflYQ+dNiW2kyaIh1LGgNIWCd54qUKbsWi0BZJKaE/9KfwUftk0X+cTmm5Ia+XUPr733Ot7bKcK3v5dxTpe8SfXs1BdEz+D74ZVWx+6QTjwJzDQynt80FjDcjjdwveSjDWcmWLZN3mD36Yqh7a2cnblDz1scHObA462duqNZkPvbDZurOOye+cFLeleqjQ2sPLT865H/jjYkoDWydGqpdDQE9eee/cPV8KjOeBjFIUrFfpQDPbgRIoCyX8PRIsCycsDUXBiRYGKs3sFTjzaUGRoaqWz2QXeEinDwwTU0ioHt54berf4LYCias1FvQQXS7jsHJdIzQbUecIFiyR6ERfXFQyaYGBopd7sKgWJDiCJQJUHhKCwckKjoIKaAMAeUVeAGiaJsDYXJz/DJMMvB+PpaPDLu50OfIuplcH4cuDd6LWko63s38zcK/wFJ4iqQMefDPVa2luk/xjRpxYR8yiImXlHK+2ORqJqxMQJCMlafLnTUZQ9KEAKFNiASVExXv3diwBvAmDHkhBHK3e9IIg4oPJ0SeWdpPI0rjwISIGDkudvSFgEJbjkU6g2ZVrl1At+uNeeSIvq8GAwY0ZB3kBasO0obDtqaas9N0wkpiAzwM2cxAKwiyWmIn+nWGJqL5HYSCQ2FiQ2UomNVGKjWGJOn1okEhuJxEYmMXUSiaG2TM8kZqA5M4olZsKaZBKz5AQzmpOYQdkZWy6xOEfMzCRmUHhmPf9yg0Uwi0vMoNrMfiwxY/CAc8+cTGImQsPpNPVUYrEjOBHHYAK2nXsnUFNcRAIlWaam8KfPv7r5mocEHKEoJos20aagBhR2khlX4hMAZhQET2dh/K7QVvemk6EbRoH89NbNmairUb/2QrzYpoDyE5MXAJxU6bKhKlL0reI2164jI/sJRji20+gKpB5jtLODENpBLdRGX9E++oYO0OH8EB3Nj1Bn3kHH82PUbXXn3Ycu6rV6895DD/Vb/Xn/oY9OWicxG+cTbOw/2dZ5VpVtSWrzl3My4Ly+lZ9pNjaUMh+UJakuteEyyACp3oat03gH5u3HJ7KjoPhz/j756/AabyqSWsWyIvGGeatDu/iA48Ivs2iXMariP2CtJT+JCAAA",
}

# task -> gzip+base64 of the unchanged incumbent (monotone fallback candidate).
_INC = {
    39: "H4sIAAqgSGoC/8VVPW/TUBS1k7RNXys1ciPEh1Qij57s957tuEuTwEgkRDYm8mGhiJZGsi1FnTJ2ZGTsyMjI2JGRgYGxIz+Dc5/9sFWMFCaSnCf5+N5zz72+ctrs9Pshk2xn+X6VpVYjDe39V/Eim8eT7MI5YK3pOk4G5o255xyx9rs4Xi2WF8lDEA3WK7JYY9kHItZceq7VTD3P3pmcL+cxO2YQZMQQze3mJJsxQQQnQuha4+n6d61mba0ySdYlNWqTjilJsMZclfft5jg7z5V8IoLtlSyIBFoozPvo5upEENuvyPeJiP7FKMlTDhK5m+sr9xJ3qG3ulfKcXHC+/fC0eyUkKu4lqQliZUVeEuFvL6/dc5oqDyr6iqWV4GHZEw8Rr2pgZMPFQoVyTkdAbFQJjYpQ4eahisSyzcm0KGZCpHA1yTWJfrWmEGW60AshZF7oiEhVBBsynCXKj/C1XmC3XsRJwp4Qq8RCu/VsmqTOPrb7Mh9Bt6gmaMSiX02hXRBRXQpJ0UEPXbqlbU9PTXradq4uybbkpW1J5aQobUtR2Jay4kFSd9Kvt03VhBKudiqpUxnW25Y+HbT1stj6R/pVQJ0QH9m742lKtx4T16cjsnYvsxRR+p5lvnWuzTZ9T9pmx7TXhvpsznAM8AM2wA1wC9wBxtAwOkAPcIEB8BJ4A6yADXANfAA+AjfAJ+Az8AW4Bb4C34AfwB3wczjCHLQVmPnPVrjTLZzQUFqodAZW3Gc3xMo/Ygdg/fvs1XOwgXOA671T0xjhla0vTFxEzqG+oJf466f6D+EBg5DVYY22CTDghDDrseJZ/i1i1GJGh/0CrSIYE18GAAA=",
    183: "H4sIAAqgSGoC/8VWvW8cVRC//bLPg0LMI3HiD13CiSJaIXQYIUwobL9I4MKWonMkSzRm/e7Zt2Tv1r7di126pKCgpHRJhSgpU1JSUqbkz2DevDd7a/suQMWd583N7Myb+b2dmecmPP1lBZ5AlA5PxyVEqq8OC8u0ZYkIkLWj/SxVGtbBSMI/PmkvdHVvrPT+eBC/A2FyoYst78qbj+9C86XWp710UDxEhQ+fAZqLuVF+vpMU7LaXXFRuwdvcVJ7NcPOnuq2Ci0Q87V2IcLRzmLaDvXEGnwMJYm6QXHTz8/+0q02EOO2qdg6/q3Y1Au36LM/+PcQlcIlAkA+18Hf67WC710M9/oSwn2THIjLrjg1k7TGEsz84r+wPzuv2B9Z+lc8A7C4iLPNT2Q53dVHAGpAkAlzb4bOkKOMF8MvcprbGQK3rgYgyfVyybwusKELDbnvfowTBbC2Co7xsB/vjI3hgteQjolF60ncPVqpzcBgGo8Msn/2sn1rcrQk+8hDhYHQi2/Nfj3RS6pGBaBQiwPV2kqvXvPup8c5qp2Mk45rddl0Gs6VZMtFMiz1bTXTmK9U74nzVDSzXn9Ww8IGTB2ajbmJRhEVNx1LzJizqGhZFWNQMLMpgUYzF1DBh6fBcmH+x+0qbyeB+aP6RiIh+8Hz4iD3M0XYOC8e14wk2I3K2xsSMCFVYrAqUB9xWVsIQXRNiysCZ3laTJPBMKAni2nFMwvBaEkaE6j2KyMiTJEgSkdydkcT0ifHBZMMavrmj0V5SvOT25DytVkRHo33tzv4LsBIGvon+DgeegX/SFjQm5lLTG7VSegxOJSLit2viwWQwlee5aYJ1W8FLpubXXflih6y7ibVWhSQlxeynXIA2IMoUECt+Sv3aVMAaoP+wrFpq0hoOj7qNRzk86i14aHBaPKqGR03wqBoeF5KUFPM6HuXwqJl4lMWjGE/VVo/AwQOnxr4bIpA0H1mDD7kwwsEGlq9ZNa2J8AcbXLgPAQWoXBHVhrvm7tPkdYM2eLF7xtcHqe3kRX33rDLHGc3msmZu1Gwua+YYCcy+IsRlo6pmGgVAOuEGRH7Dp2t8unWfrvXpkk93io80cWQ9jrRxJMWR0+JIE0fW40gbR1IcWY+zxsMsB05ABGedT7gSOAKwGz5d/9Q+xXhoCUYhwrNiPLDq5clroYTwEjzp8CUYFVledsCoRIhLzhcHbQCkEvP5uPxqnGV2v4+BZazLUX5KM81w7biZaci5NF4AiWIOvbCO2sHzpBe/jyWU93S7qfJhUSbD8soL4mUIT5NesdWofX0eadGrJBvr+w38XHme8E7iu01v0WuHjcblpjTNGC+yorEpqYsmJo0tabot/sFrmm+L9BcN+lxumuf4h3SJdIX0GukNUmO70VhEeozUQdpCeo70LdIp0iXS90g/Iv2EdIX0M9KvSL8hvUb6HekPpD+R3iD9tS3dgOJ0MKH/Nx07X+IvMRMw+WA2T1wm//iRtobid9Ft/qnnSfuPO8tgZX3jeRK/1/RR9nEDvtFZhTZ8t7PKD1iV0EvFjRrS3eus8JxCs8J3irqLvYVrLvY6ZkXgFEl8xyqakuYeiwskahY9EhOsPJslRrQtwZpWy2l0ZRM4TfLNIzdaxRLca3piEfymhwRILUNHONtt08yykCE0FuFvOXIDGzkNAAA=",
    358: "H4sIAAqgSGoC/8VVMW/TQBi1k7RJr0INoYKqiFB5AMkSyL47O3GXNqkQEqISajeW4CZWsWiatHakiClj2RgZOzIyMnZkZGTsyM/ge+c4SVsHupH2Wfa7+9733ffu7BLb/HSXVdlCeNwfxCwXOgSXUKvk4rqxsH8UtgNmM3ogwjOW9oLOoB3sD7rmMiv4wyDa1s/1ornCSh+CoN8Ju9EaETnmjCUr+di20rBdf/iPME6ZPMTYWanymTGCYT6CeFai3N8TiaxEufmJBIJkVqLs6u4hiLK1VYWOkd8fHLA1VTIuDljXKL48Dfw4OGUPQboga0Zhx49ic4kq7c1oyVSrPqOlSquD9a5pYZHcytIiTzGhhgm2kd8dHE1IboHkCVmhhEiKlXORJFXSHIS8Il2ENAa5mu3cHHyGQcnutLq9TutjcNprha6sFFvdw1ZwYq2nN8bCi5OBf8Ses5SplHATHret9cndTfknifxkRqLcicbKdGPkG52O6hp3WEqiWJeW2+sk1cMBnuHACgZVx6j7jYOIrYKojy3hnlF4HURRooHOizmd99LOi3HnYaCAhuC335BYhOAsHwp0W0ijuBdE7/1+oMoSFi4SI06S5AHKwrYT2HbCNRZ3/Ti1WMBm0LUZixVRz7ZYqPq9bItFfY7FdmqxfcNie2KxPbHYzraY5CczUovt1GJ7arHwUovRW2lNLZbwXNrZFks1m08tlukJlmLGYom2SznfYnWOpDO1WKLx0r39yw2LkC5ZLNFtWb9qsZS44NxLb2qxVKlxOh1rYrHaESREHAaw7fyhYh31IlIsn1bqqHhx+1c3rbnNEYimODLZRKtKGix2kjPuxFMQTpJksTeI6TthLO70jtt+nCQJE82KfmiulvTkr6wbBU3TtppkxHV2BJabZwlZVfRQU7/RFl226Z8wIpwTLgiXBK2haWXCBsEibBPeEN4R+oQR4YzwmfCFcE74SvhG+E64IPwg/CT8IlwSfjeoFJGWQsX851KkuUztKG7qepO+7ukDowd3dqRmrpQK9FDQ9arexBtlSujVJvaf+QjTm1eP9auSNv69fZx+8u8zsqdSZrmSTmCEKnCwwcZmz5vRLDCtzP4AkI6mrpQIAAA=",
}

_SESS = {}


def _raw(blob):
    return gzip.decompress(base64.b64decode(blob))


def _session(key, blob):
    s = _SESS.get(key)
    if s is None:
        so = _ort.SessionOptions()
        so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.log_severity_level = 3
        s = _ort.InferenceSession(_raw(blob), so)
        _SESS[key] = s
    return s


def _onehot(grid):
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    x = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r in range(h):
        row = g[r]
        for c in range(w):
            x[0, int(row[c]), r, c] = 1.0
    return x


def candidates(examples):
    if _ort is None:
        return []
    pairs = list(examples.get("train", [])) + list(examples.get("test", []))
    data = []
    for e in pairs:
        gi = np.asarray(e["input"]); go = np.asarray(e["output"])
        if gi.ndim != 2 or go.ndim != 2:
            continue
        if max(gi.shape) > 30 or max(go.shape) > 30:
            continue
        data.append((_onehot(gi), _onehot(go)))
    if not data:
        return []
    out = []
    for store, prefix in ((_FP16, "lsp5_f16"), (_INC, "lsp5_inc")):
        for t, blob in store.items():
            try:
                sess = _session((prefix, t), blob)
            except Exception:
                continue
            ok = True
            for x, tg in data:
                try:
                    y = sess.run(["output"], {"input": x})[0]
                except Exception:
                    ok = False
                    break
                yb = (np.asarray(y) > 0.0)
                if yb.shape != tg.shape or not np.array_equal(yb, tg > 0.0):
                    ok = False
                    break
            if ok:
                try:
                    out.append((f"{prefix}_{t}", onnx.load_from_string(_raw(blob))))
                except Exception:
                    pass
    return out
