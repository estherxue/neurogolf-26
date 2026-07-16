"""family_fp16_2: float16-intermediate rebuilds of memory-dominated incumbent solvers.

For each target task the exact incumbent ONNX graph (float32) is re-emitted with
every FLOAT intermediate lowered to FLOAT16 (2 bytes instead of 4), halving the
dominant intermediate-memory term of the cost (cost = params + memory; the grader
only thresholds ``output > 0`` so the reduced float precision is numerically
harmless once validated).  The graph is rebuilt FROM SCRATCH (no auto-converter):

  * ``input`` stays FLOAT[1,10,30,30]; a single Cast lowers it to float16 at the top.
  * every float32 initializer/Constant -> float16 (values clamped to +/-65504 so
    large sentinels stay finite), every intermediate Cast-to-FLOAT retargeted to
    FLOAT16, so every op keeps a single (float16) float dtype -> valid SSA graph.
  * the output is declared FLOAT16 (grader thresholds it), skipping a final Cast
    and its full-grid intermediate -- matching the working hand-built f16 solvers.
  * pure-integer/bool tensors (Slice indices, masks, ArgMax) keep their dtype.

Each rebuilt graph was self-checked EXACT (byte-identical ``>0`` output vs the
incumbent) on the task's full train+test+arc-gen set before embedding, so wrong
tasks never fire and the fp16 precision is validated.  Detection is behavioural:
each embedded graph is run on the task's train+test pairs and proposed only when
it reproduces every pair exactly.

Tasks where fp16 lost exactness (Resize lacks an fp16 kernel; a couple of graphs
whose reductions crossed the fp16 integer-exact boundary) or where the graph is
so small that the input-cast overhead outweighs the halving, retain the exact
incumbent unchanged (no regression).
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

_SESS = {}

# task -> gzip+base64 of the embedded ONNX model (fp16 rebuild, or exact incumbent)
_MODELS = {
    17: (  # keep-incumbent
        "H4sIAKp2SGoC/+2dSXMbRRTHZ5GtcSeOFEWxDQEFpkig5iRNd49GrhwUsSUKviScuKQUSwQV2FJZUpWPPlF8jHwAvgtVcObEFvYd"
        "whKg38ykJUU2jglJbOv/XDP15k3/XvdbRl3lSzts+ebbFjvNZtrr3UE/N3P16mZ73U093+j1vTlm9TtL7IZpsTyL3zBr1c9Zfe7a"
        "K4O3mEisyiLcucut5mC1dWWw5h1hqcZmq1c1b5hpL8OcN1utbrO91lsyyVdJueUKkfeOnE2Q0J17daOx3ut2ei3vOEt1WxtrVaNq"
        "Vc2qrUC2oMaFzGpXcna/VHTTl1u9NxrdFltUdsnsdqlEL/zhi7OMBpKRj7rOjLomx4vROLpxGizc2ZVGn1IQOfDJJnd34NNN0uBA"
        "O8iTLaCboBdl174yuMZOkKGssh2QMYzTLckYkqEymrr5O6mrWncnz4hrR4TyRcH7RTf1SqvXY6cYPZClNFlumt5XdVot0wA/np7C"
        "9ylUn7uz5zeurzQ247K144kmy0Yz+1zl3afYfOna55tNtkRWqoZPwfnlYTWejcqsbDvU2U5ijF2E5IIqzYvjLijDvLS9C9UoQxe8"
        "pFxwConz8Z7glBkudi0pp6Rwio7LsZ7gFAcPdnfA6UaJ4OWxnuBlukWhhMOe4GHSE7wy7AlOSRDFvfaEKCY9IUojPSEii799Twg/"
        "6QnBhz0hKFQh9tATQqi8C4pNBMOeEAFZI+/hsBpPk7FC9aY3sqTm7TRpjtfXOs04GFq2pGXLbZZ9kl6qD29VUkklHwlV0sKlmGT0"
        "nIK+Nil3mpMikMEOcwZ6zvLonFEU4SSzQC9F8gHIiu4FahJJa5G0lqCoXxAQqBL2RW62M+irX++oJDnzunfUMbOspopbt4zQe8Yx"
        "Haau2ObX84ZhnLv7z3vnA9PJOAWnEI0K6rfeN6Nx9yOgQYMGDRr0QaaROdCgQYMGfRj2MuyEoEGDBg16uvcy7ISgQYMGDXq69zLs"
        "hKBBgwYNerr3MuyEoEGDBg0ago4BDRo0aOxlEPQbaNCgQWMvg6BbQYMGDRp7GQS9Dho0aOxlEAi+FNCgQWMvg0DwnYEGDRp7GQSC"
        "rxQ0aNAQCATfOGjQ2MsgEAh+IUCDxl4GgUypeM85megUpnK9sHXJuLRVN+pbF42LWxeMC8bLxkvGi8YLRs2oGue8Y46dTS+nzMK7"
        "Vs1qV5Jn2ywUanTaXXQSVHrZNGt0Bps371jqyYofAy8zAtNRauM090dpIcdoUfY+PBOdFpVefu+M8fcdMf/S2m2t/am1P7T2u9Z+"
        "09otrf2qtV+09rPWftLaj1r7QWvfa+07rX2rtW+09rXWvtLal1r7Qmufa+2m1j7T2qda+0RrH2vtI62ZyNEOObKQmvHU2MhIpKWm"
        "OxEzUxn/7DSFnZ6CaJ3DG+TcoYuNHZaQjhzwSI4ezADmD9S6jx2E5Wb28Sqz+29xx/fNmnKPeiknHtEK8g934pMPZb6FBznN4gPw"
        "vvT/OX3svn09/l9dnNoj+cS9AU/+6zj6/1bFyyRHptMp7nSO+mun2Ux7vTvo5xZY3jFzWWY5prqYugp0XXuKJQexRyPY5IhaihlZ"
        "9g/UcdjOdIMAAA=="
    ),
    21: (  # fp16
        "H4sIAKp2SGoC/6WUy27aQBSGzwykmNNKRS5qolRKK3fHosIzvhGxcOkliWOI1Oy6QcS4FWouKBipSz9KHiNLHqGP1Dk2FBdcCalC"
        "Y838c/7fZz5sa3j8iPgO9ya303mi14fDb1PTGU5uD9U0Gs0SNTWqH9SkVUee3B3gA+P4FteFyCeuGp4aHb2SmG1j7/J6EsXYKRTR"
        "hmnUv8TjeRRfzm9aT7E6+hnPfPbAaq3nqP2I4+l4cjM7YJR/WMynTPILo9KfX6MkQZAgdw/MTJJMVpmp8m+TSSZ7d1OTTBbyKLub"
        "Y1TDeDbDV6Q6pLjbNPdp015ZPKN2ch+Pkvg+d3kkdrZd2Y1cuhB20V7TEYRLlOLmpT2/IJOpGrDIqDi/H4/XyITcPekPMmHtbqKT"
        "CLk8v7ALyIRNilOOTKwoC/dvZMIl0StHJhy6EFTRKSAjhrK9+99MyGR7iUyaObKmWpuURo+nFIWDyEyRZS3xSFABQZNW0ZIl2+Wn"
        "kNS0JDrSyU9BRKSTvyxP7uaJepuzDZ19b4UaU78jjTWwpzoMugDQBR968BE+wWc4gdP0FM7SMwjSAM7Tcwj9MA0XIfT9ftpf9GHg"
        "D9LBYgAX/sUyTeVlaeI/0/RlWt6bDDh4G5qltO6GZivN39CcgP+KWs803qgdc4Ce+i6tVoyplbda8Ypadb6+Xn30XmJTY3oDucbU"
        "QDWOaFy9wSXIrAK3K3pVhAb+BjsIJwFDBQAA"
    ),
    55: (  # fp16
        "H4sIAKp2SGoC/+1Yy07bQBQdOzySoQKUooJaibZWV1lU8TxtxCINbYGQgFR23aAQ3CoqLxFHYpllf6GLSnwKn9BPqo/HSYxx2khd"
        "NoMcjY/P3HvumTt5UKRbP1/Tt3S+e3ndD8ulk5Mv16466V4+j6addi+Mps7cTjSplKgdXm3QO8umHh0Ty3aondKn4KzfCY77F5Ul"
        "Ote+DXo1685arKzQ4rcguD7rXvQ2LKx8k1pJ7a4XXT4tdN1quRC6rjN/fN7tBPQptTuMAgHMnMJx/xRgqAEygNwptPrnlI9YIk+E"
        "nSuCIRLWyOnXrCORiHTFyZSzuHsTtMPghr7AAwVQPzZqzaxCKjA8U8kKAA+A7xTenfYMzU+Cs6oz1wx6vTgygzHMnRAZdrB4DcvY"
        "wXheaYU/2sHE9GtgB+NDxfKhHQzFMpUvmkEjE2DosR0sFuCN7WDeMLiftsOPEF6dEFlRPATDNXagj+A+Q8twNgbZCOSj5upAGOcA"
        "hWEiKAeVQy6Xkbqzs9hjjgq5GvrV6l7+xa+HSnSeEi+jBA3C/ZQS7DaHAaJqqKAJ1CvckZL27RRKJCLBXJHxxIAZTwQ8ESlPBKgC"
        "noiUJwKeiFxP7CmU6DwlGU8EPBEpTwQ8EfBEpjyR8ETmejLpcJvd4TjHkj3sZondkTyv58wqAalSpNpUwhsp89tUxhJhllSmkvXE"
        "CBGn15n0OBnSy0+PVTJOnz4lEoaoCadEehQPwXCHRqbSK5YKpFC6yil9PZ1biYeKFapXcrJhsc1KpRPFiM5PNPJYeZlEaAflT6gT"
        "7wYKy3R13DAuzNeoXrtj1HDR5pplubBA8ywX26dFlovCtRyjysULNlCrLBcVa53lYue0l+WiTu1nNaA2D7W1b0fHxkNpnmvOwiiA"
        "F1OT0poAwNXYQR0n8PGioxgIrN1k4saT8sJVP4y+HDgLO1eXnXZozlLXHJ2y9bVSLlrmb5XWo/fshk28DMYibDuD8Yb9q1NpJthm"
        "jInGNiFkm9RInbwnH8hHskv2Bntkf7BPGoMGORgckGatOWjeN0mr1hq07lvksHY4OLw/JEe1oyTaZpJB/mO0H8tJOCNONb4vk9mY"
        "jdmYjdmYjf94VJaiz9jFLatYj37DD29K0Y1feWJurDp+1X9+OfzPwjO6VrTKq9QuWtFFo2sT1+krmny9iBn0MaM+R8kq/Q0gfS8l"
        "qBAAAA=="
    ),
    59: (  # keep-incumbent
        "H4sIAKp2SGoC/+1VzU7bQBC2vYaYDaghgYoKiSKLQ7W9xN51friQpoeeqCpy66Uy8aqNCEkUOwj11Efh1mfpe/QZeu58dlwSkkI4"
        "cMPWrDLfznzzt+s4/PhnhSu+1huMJknZSjx340xHk67uTC5FkdvhtY5b5o1ZEC+4c6H1KOpdxnsEWPxg6sWtnk8iSRQxBO5ap9/r"
        "ar7NiY4kILDmss7knH8ktcatbp2ghmu/Hw6uxC7fvNDjge5/ib+FI91iLYZw29wehVHcMrIXUIkX4mTciygju2UTQikQD8Vtctbz"
        "qli8Mks8P08h4NAAqbyu0/BabE3rsrJQC5XtwE1x1vUkfIMs+b2UDEsAtOYWPox1mOgx3wdYA1inosI4ERuU2DAje4vNOsgasGi6"
        "62c67n3Xoszty2Gk3cJAh2MdJzcm44cwRjU+qvE9LMjfl3lJyM2XROcr4NTY00kfKAHYQhp+Pcu4ns+VoMbsYFdpgN8AZ5N8ZdVl"
        "76KIU1T6DcCb7eb9pwRUkuroStQh5W0vZRULOizVfC8lSpPBYi9TsgBkKFPW/xVPdWOrTqiqzqDou0TflXeLKgRWOCnKzypL25fy"
        "ol4lZ2wxcIUklcpsOwBUeX04Sai1LvsURqIyHaXTHQ7iJBxgluLV/AlO30qrknVp7SrsT/SuQc+NaZbNr6LomKXCsWm06TrlikmK"
        "nFWUUI5JL3NYyXSPDOPHyUPSpusmNh2LKCwD7E2xlWmMtXFtctVKVU/sphHopQg25XfSxkUQ+45NQGmGugXBZiOnIH4c3VwtFqF6"
        "cwF8X/xhKX/RKRLhbzafMJ779KeyfczznO9jbdv4PC0O/m6S9+lPZfvwDVre9Od8V7HF4APRpKnz6SfljXGU/FpgWYLBtbn0ayT9"
        "O4wrNgautbuuq10muDaWRX2oHZlr8/Pr/L/4Jd9xzHKJW45JwkkOIOeHfPqX8j+Lts2NEv8LT6OACrIJAAA="
    ),
    71: (  # fp16
        "H4sIAKp2SGoC/+2dQW/bRhCFtSQl0VPEFdaO47hNY7A9EWiRpInRBD7Yqq0AQQW1Vk7JQVhTS4uIRCokZRlBDr7kXvTWm39qd5ei"
        "bCOp0aKnhm8MUpyZNyvt7EfCF2JdevbnB4t+oHoUT2c5XxkMwunDnUEUb6nLQGS5uvScn9WFv0JWnmzSBdP6SyHZwbyvT4f6tM/1"
        "SINsywnmg8yr98dRIOk7KqLkdIPFWZqz4FY3KFU7pBzejJP4nUwTb+VIDmeB7M8m/hfkiDOZ7bEL1vS/JPeNlNNhNMk2mf41T0xd"
        "I03mI5GVZV1xtiyzbyoLkvHflFmfLPuaFt9Ei1JeT2a5TD27OxvThh6Uigh3hlEYenZ/dkycjMProojtH2fqBxQeZwdXJ3ur/Pob"
        "f7czTWX2b8rWiR2QMxLjkNsH49xzfpFZRvfIDLRINPT1Se41n6dSqBmoyWotd9Qp/BiDbVoU8Lr+/ITiDplSKvLcnqvBTZ9uk742"
        "M6knQRCMivCPVHic9f75+q9TyQyxHmedouPbxDrc7mRvvWb/7UzKd3LZopoZSCt63O7doPiW9ADc6uTeystUxNk0yYxoKtPJHtur"
        "aZFa8k5eCO3f1No2uiJfoKDjvSKeLOP3SMuoGY5F/vTBA+4oL/SaRzIbiaks0sm1dHIl/ZXq6ETkZKp4PQxPZbAc+koy0cnkanKD"
        "CjXZWXTG7TDcKXp+hwohOWF0KlUieVIkOGkR6QBnfcXscEjfE+tzK4+8xn56srxZomxTtcK6tjQ6oG/8o2MRv1FERNw6OvEaz0U+"
        "kum1MtoilaJm+thMUekeX873bvnoUFHupDIcLye0RnYSS72ITpzkvfJGYz0yOm6Jh+VNaQSLkVT8URFfJyVRxyNzC6vHlZniayo8"
        "3lAf6qFoHmVTz/5VDP01cibJUHpukMRZLuL8gtn+XcWDGGpoLv9W91YLVuunYjyTt2vKLhjj7MT//b173z23WtQ2i/Xi/H1ttwar"
        "tu3+hyys6gSADxAAPkAA+AAB4AMEgA8QAD5AAPgAAeADBIAPEIAsCED2/8YHVqLqfKDXVecD3QQf6BcIAB8gAHyAAPABAsAHCAAf"
        "IAB8gADwAQLABwiAgQDY58IHrOp8wKrOB6zqfMAqwIf/h34lvOW2WtQu3pfHS+EwPDhg+NcCBIAPEAA+QAD4AAHgAwSADxCALAhA"
        "FgQgWzU+sIpV5wPrVHU+sBJV5wO9rjof6Cb4QL9AAPgAAeADBIAPEAA+QAD4AAHgAwSADxAAPkAADATAPhc+YFXnA1Z1PmBV5wNW"
        "AT78VZfpPcJHYhy+sGo/+beMr7c8V+5u6WbRmXI7pVpv4a78Q5NuPmOs7XSDQVa6ZFx5PSt87lrKtc8t1i73nffXXEfFHMZarXa5"
        "Nbuq00KrVmvbwbxfukqh3MPStWzt7r+6bzZcn+V8g9ZdxltkuUwdpI5v9HG8TYst1o2CPla0Haq16C9OVu9MpYIAAA=="
    ),
    91: (  # fp16
        "H4sIAKp2SGoC/6VVTU/bQBDddRIwixAo0IpSFZCPPlSxdzd2EIeQfkFIQlVuvSCTuFVUSCLZqTjmp+Sn8BN67M/pPDsmBlIpVRPt"
        "amd25r03s5uNKY5+b4i3otQfjMZxee3q6tvIqV71B3u07AZRTEur+I4W9pow4uGumHJDWGIeKIx+lYZHwy8bcc0qXd70u6G4oPBa"
        "uRA7FavwOejZ26J4O+yFltkdDqI4GMRTXrBfieIo6EV1lvvyOpvyVXtTlH4GN+PwBaPPlHNxKAAmCn3HweRikiBQzyj1kpR8Tvt3"
        "Sg2iKiYPkw+Chyp3QYkohQnkLtXbDu6EFFjD4VhrX8LeuBuS214XxeAujOqFlND8EYajXv822uVo7DzJXZRkLEzaRpIjjK6DREn0"
        "45sUCe1x1fJIZQLRGRA18XJ8LXZSdDjgrebgq3B4/yIU8MhBop/iJ+pd2nHhrOXg0U1ZWb55mXoASSennq4KOeB15/AyCZPLw2fq"
        "JboqVQ4fZ+/iXkg9r0lCioKTWnbS6yWhsoIp8Xq5UC8L9dNQULkPkbXMifXsdFQlTd8UWMNBBZ9cRwmLwmWASuVaxVYYReI1vChY"
        "yee/5p2MDSUolU8Bv9KLUgCFCZdCzS4FZDtZL5T3uBaFQ1d+TnYSVcvJrs1k60pOg0Z12lksG2wS11DnK9WoVMvFsjVydEKjUtlv"
        "8q8ZqsGetlbaQYztPcTjiLUurwzHMb2T2V6Zf7dbJqfvvsm3RIMENY/p7TimB6XB3rMP7CP7xE4np+xscsaakyY7n5yzVr01ad23"
        "WLvenrTv26xT70w69x12Ub+YoRFegub+J1p5hpZqk02D+U98inzHT3y6afzq2utkrR7xUoPe98xYIcPLDE6Gb2+YBhkGZw08zJl5"
        "sA/TzUyjAFNmJkuCq5m5nwR7j4P9rwfZn9JLsWPy8pYwTE5D0NjHuD4Us+NIIsTziEZRsC3xB4h6egHjBgAA"
    ),
    114: (  # fp16
        "H4sIAKp2SGoC/6VVy27TQBQdO6Gkg2gjU0EFoiCLVRbInocfqAs3vNo0TiW6YxPlBYqgDymO1KU/pZ/Sf+IHmDOxMyZNKxCJbNkn"
        "9557zzlO0qDvfj2mb+mD6fnlPHM2+/1vl37Qn54/V5ejwSxTl279vbpobVI7u9il15ZNI2oKHTuL3c0vk/F8NDmdn7Ue0frgajJL"
        "rGvrYWubNn5MJpfj6dls10InUySxU8t8r+xJB1fLntr9Pf66HnttzxOKGdQe+Whkbi2d/6QcIAPA/5HJV0y6UVSYBAD59zocrKMO"
        "3Re4tdP5kO6AKMCJAw0V/fRczbRHIcAQYKTAwZUGGcAIYLzo3wYAc5jn1g6GM03IIBx8zHfr3clsRl8AhRWM3U5TK+SqJUABVzzj"
        "seLBsrgHKKo82J/J2zx6NDZkEhXBwqqXAILqA6PuQ3cjHWT42NGqFqawyJjCImylieI/TWGQyz1jCtbkHkDfmMKhljNjCmeFKZxX"
        "xHCNiDtMkYUpXBpTIJBjLx5UeXRZuN4ULnBCljxamPIMOiCGI0weL+3YKh5ZUYlTlHGKapxCl90XJ5YURZylT0LzCOOTgPFCVqbJ"
        "clpVn4A+cYc+AU8E9IloMW6r+L6IuEIcF8TSqxBLBCf9ewKADMmMDAyTIJfcyJCaWJhpUpTTZHWaJgvWy5AwSEKoDMu8IQkoxMki"
        "PCyBRCWyk7F5ZnkMFI9n4JnSqCwNfBN+4OHkOxsX80z98OpxjvW91W1Y6r3XsJq0rRLr7BNC9klC2uQD+Ug+kc/kMD8kR/kR6eQd"
        "cpwfk27Szbs3XZImaZ7epKSX9PLeTY+cJCcFm+LTbOw/2ZyCbbEb79gkWsGEwvZXMKmwZAULFHawgoUKI6036p4usaizQ7Dzyuvr"
        "q/L/6indaVhOk9oNSx1UHXs4hq9pYayuoLcr2nVKmvQ32FSYgf4GAAA="
    ),
    121: (  # keep-incumbent
        "H4sIAKp2SGoC/6VUy07bUBC9tvNwhlZQE1EqJIosdeNV7Hv9CGJh0tJCiIPU7LpBeVjI4tnGllj6U/iAfgT/1U1nru0QHl2gJrpX"
        "nuM5Z+6Zsa3D7u8V+Aj15OomS4366eltcmXWPo/nqdUCNb3ehDtFhW0o7oCauLg8XL6hpoFZH10k0xjeYWoA6tRGsGtqUXYBAqGu"
        "oaV2x2x9j2fZNB5ll9ZbqI1v43mohtqd0rRWQT+P45tZcjnfVKiQUQo5xLQLJRfomgDnNVLrROugliAqN7VRNilApwJFBWJBuiFB"
        "39T2Z7Mi05ee8CIoMlcJDAhAl/uTObQJ6GIWR9DpmLVBPJ/DFlBAiP28l+2i2tT2KMFZppBrh79EISnaZBlR9OVDNRSCCHfNRjRO"
        "6RY5onxb5nuVo8KmIwv7D44cn4DgwZETVI66y8ejcfLOy45scuRSgr1E4dQ87rzsiHdoI8+cF47eE8CplFQSCz/0CHDyyN3XPAKy"
        "CnKnnCzzcrKyLFnmgdnY/3UWjW+tFVJKCtpznQ1iBKAlnFogOmb94Gc2vpAWBc1Z2P+wKLC4IIuitLhToomQm0sbHU741btEPOHT"
        "Rs0TwUNrBLVGBEbjOkvxbZVuDOXMGugK/rd1ZQ16OIn+HmNsj4Wsx76wA/aVfWOH+SE7yo9YP++z4/yYDcJBPrgfsCiM8ug+YsNw"
        "mA/vh+wkPCnVUE+qOf+pZpRqxdl4X2XBE0wgtmetYNTcVfQefmGqoIWBVwUKBr61vkSlBw65vaegJ8FPCMAC5F6/PfrDHv0otrYW"
        "ac1dYIqq1eqNpt7q0ayfaAin38ZOPP5jb6w31dlpqlVEEsKtIoUi78fiQ7sBbV0x1kDVFVyAa5vWZAfK4coMeJ7RqwFbg7/7eqNl"
        "twUAAA=="
    ),
    143: (  # fp16
        "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"
    ),
    161: (  # keep-incumbent
        "H4sIAKp2SGoC/8VVO28jVRT2eOzYuQsiOA92iZSNpkDRVDP3OU6zjhHaKhBtqKiYtUe71ia2sccoZcotKSldUlLSsSUlJQ3SlvwM"
        "zjcPz0xsg6iw/V35fj5zHt8597rNzv/cZ5o1R+PpIu7UY+XsvoiGi0F0vbh1P2SN8C6a96yevbRa7kes/SaKpsPR7fyxtbTq7JDR"
        "A6w+EPSgdlrPZ1EYRzP2hGhNlHEan4fz2N2l/SR9ohQp2BypvjVSkEXqFpE+JbrbsWPfWw/1FX01+M137Ktw6O6zxu1kGDntwWQ8"
        "j8NxvLRs9wlrTMPhvFcrva1eLc2h+X14s4gOa/RaWhY7ZXDG7JHPsQgsEgGU07y+GQ0itk8ZSlgp0NqxrxcvQcYGpAZpHPtyccOu"
        "QHgggv+YHeW3PbsAOXVp4R4WnwJw/jA7zkGLNLuDJBGwAqxM0zsiUz+RmzhV6H0MQxTH9brgeaEchfKs0JUkaAUPiqA8MQ3Adoug"
        "PAsqvGpQAa2Evx50lb9AsYJXo3L0R5RKFRxLEiEr9QxEYkaj//UsHM+nk3nkfkzSR7PbZBzsZChprNPZxQPQQGhn5zKMcyccgyjM"
        "vzj5BE+jdAGVRLByYfKTQWx309GobzmEicduppt8oJuEbnKLbpKGeSAxDVKkYiQSBWChiFS5mrCApCB1oabUMEVnZVA4kAoLOiup"
        "sxfDIUN/JORRnrNzMXt1Gd65j1DWKK1hvajHDMY0wwptVdxpvYjmr8NpxD4ruqCQuhLOzvMwfh3NKi6T86BwRpXEorBAcmXy83CQ"
        "WaSzqbIKQCgkrypdeJR3YWMPVsE0cta4JDSS02ItWDKSWhbBdEKoTcE2NzxxhRFSkFTrQnmd+MNsalO0TuvsJtSlJmmT30C6W2ID"
        "LHBrvLR1mC7jZdNl/Op0GTTH8O1XQRLAZMN1nPQUhNw8jwZH1kA3o6rn2CTRS5NnMGMG/TQmZZN8UKUJKt5b8H6EH3E5GtQWeE7z"
        "i+8W4Q07Ydh1diaLmEZqLauO9cp9a7XxPmlbe5ZzV0te989o6dGHcE9YEt4R3hNqF7XaHuGU4BF6hCvCt4Qp4Z7wlvAD4UfCkvAT"
        "4WfCL4R3hN8IvxP+ILwn/HXRp/s4T4WS+Z9T4e5BlglEaVCkZ8SKh+w9WOl+QPvWuWX18c+Z755iJ/JdHbuqZbdsyb18Z2Pnu12K"
        "wrJIZ4US/4w+bjv38EGSv4KWVfrszfhL0KackvLzXQ27SrpKlUtRumypvbKl9sulaO4er0ppnbOaVbcbzZ1We7ePYf3maf6vcMRI"
        "3M4eq7ctAiOcAC9PWTa82yz6DVbbY38DJkQmQOgJAAA="
    ),
    165: (  # fp16
        "H4sIAKp2SGoC/81WTW7TQBT2T9K6j1YENyltkQBZrLxA8czYTqou0kApBCohsoNF5DqmRE2TKHbaLnOUXIAFEgdgw0U4CfPGcRKS"
        "CA078skTzzdv3v+MbMDR1xI8h3ynNxgl5lar9XngeK1O75C/hkGc8Fcr94K/2FugJf19mKgal58Lgh7eNnE4xeHERE2t+DAX3rZi"
        "K9/sdsIIziFlQQtdU0s8rrHfu7G3IX857I8GQqldgu2raNiLuq34SzCIanpNn6ib9gPIDYJ2XFNTcApM7onHdTlcl2/l3kVxDAec"
        "8/m8suptcWY9qXCJqqWfj7rA+LRq6olTtrY+RO1RGDVH1/YO5IK7KK5pqfH7YFxF0aDduY73VdTlT3XhRudfNj4E3CGc5i/E2jwb"
        "RkESDeERLhAk6arru7hI+S6KAiz1vMgT7bi4wpD1LL05ukhFMStClz8T5QHi4CNbsfSTdhsoEhUkqlkI552efW8agro2AKG/Og2A"
        "lFNNpUw/KSPrTIuBMREhR9aVAz1HCfSU0AVPBYuhEpYGtZeVLuPdVBojIC4S3mIR/h7BojLMHPEXlGF+SEVemTtvBFKV34ZtQLIs"
        "0vKfbUAxidRZ1wZaKHLjoQCZu00xhZTOqhjczexra+3TmRYmvykzT7EE1E1Lg5FQN4vEW4pEmPDXF5/6OGAFaGVefFJBFruSVhcC"
        "xCPKyvK+CgPoLMNkMmdugKIBht4yMmcZwSXBzjsxZEIDhsvclU5k2HjMW9BBs5Zi05baQ9bDwTfz/VHCr0JxXj5BOjM3+B+/cMU1"
        "ObD090Hb3oXcdb8dWUbY78VJ0Esmqm4fTG8/ZQFQgzT0/E3QHUUlhf8mqmqql7ZpqCkKUOeVaWhKZYkjDe3XwH7G5zDjaKPIVRwv"
        "w/62ITYWjaIQY43JhiL7O5ZETRJ1SbyUxKkkXkniTBKv5TCWhPJGDmNJKA05jCWhvJXDWBLKOznUJDGWxA9J2D/Tg6UbujgzbuO7"
        "Ouv11VPx3zL27sKtgZ8b/CpZJhkT5I6hFTaPNEWp45dgNi0WcXqaTTUdpycfn2RfmntQNFSzAJqh8gf48xifi6cwvRqFBKxK1HOg"
        "FOA3NRA5T7gKAAA="
    ),
    166: (  # fp16
        "H4sIAKp2SGoC/6WUzW7aQBDH14YUMxxKDWqjVk0rS734UOFdY0yUg0s/kjg2kcqtl4gYp0LNBwpGytGPkgdpJR6hj9SZxRgnEAmp"
        "oEU7f8/85oNda7D/pwYfYWd8PZklevXs7GJiOWfj69e4jYbTBLdG+TNuzCqoyc0u3CsquLBy1NVEGNXv8WgWxYPZlVmD8vAunnrK"
        "vVIxn4P2K44no/HVdFehSKMQCerYxtXG5SClY+wMLsdRDC8wkcDVQdE1SoPZOVhoumh2l5nC4V2eqbQxE5chpcRqbYpRN8boGNMF"
        "NbIoThilcHYJAmhPgr09qAGlyOJLUnvRREPSySbRKeAdEjrb45sLPAVRpLvkU7eYlJPYLfC7KPDW9rPLyycStxb45gJPAql8xefS"
        "TWzPz+vnNFhuFxK49EgmyKb2iqZIrjQ17hiVw9t4mMS38IZEmhzvrJ9QybJzFk7o02iEasaig8G7RjmIp9MFiEYkWptBvAP0kDys"
        "RddSzfGCr0qlrgTNQ4iHpQpqVdhPZGjlrPaqVMmitoVTKFVQz+KJnoVNP3QqhLsqVVDTgpoW2bGgsyK68pKVErvQlY2ukU0N2NkF"
        "eFu8siTrz25mCb4tZKG68tMMNAW/e5pShx4O2D9gjB0wj/XYF/aVfWOH7Cg9YsfpMfNTn52kJyzwgjSYByz0wjSch6zv9dP+vM9O"
        "vdOMhjxJ4/9Jq2FVlX2F9fBlszQUNNpFwzEbWVJqgf4JX2XuY5H76t/I/IAC5KLd8puMyQoffB67cXT7ve72493yzfsSmpqi10HV"
        "FFyAa4/W+XvIpi09YN2jVwZWh3++8KbNyAUAAA=="
    ),
    174: (  # fp16
        "H4sIAKp2SGoC/6Va224cxxHdXVISNY5hiZYox0lsgwHywIdgurq7utvQA8M4sS2TMhC95UWgKCYg4otgkoAf+Sn+k+gT8knpOj09"
        "U7PcRTQbCNvYrr5UV9U5s6eH2mk+//cXzR+bOxc/vLm+2r3/8uU/3hh+efHDx/nr2enlVf66v/3n/OXgfrO4+vGj5pf5onnYLM5M"
        "/tDu4or3t/70+nXz+2ZY2ywuQv7E/Em7W1em3b/z4ruLs/PGNtITk9m//7fz19dn5yenPx+812yf/nx+ebj4ZX7v4INm51/n529e"
        "X3x/+dFcnH0oi4q3/IX2t06uvys7wWBX7bS1cqfdvImvG7n9rRfXr5pHZXcxiNWr7b0Y+N23/xDbyxpZGNT+2C6INZZs5alXLMYo"
        "xlSmfiAGSRi1edarS6ylNu9qxWj2t4/PLy+b34jViIVuV+YJMiwzJEiy+3dPTq8kokf9ADZzg0uSwMkXl3J+ksCpD/zF9fcH79cK"
        "rQnd18JSmLIMAXINMOoAJTGUVgdIoVti2/17X/50fnp1/hNWWTmCNbdXwVGSGZI424FIrJbq0a0t1j9oJG9dWCeNl0bqaoMGs5Wi"
        "2jgNzDZ2GLRpQJuVwrt2MzA7M4At7y4GsSquOEyz08HsJM9OkcUJiJxgxvkxmJ3gxvGALCcJc2EAs6uFc7rWTmrt1tTaBpkhyfHt"
        "CMwY8FI4bwaXXgL3NIDZS+DeTgUzCuvdVDB72wXovQrQS2I8rw7Qu7okjMHscYS4GsxeEu4lcT4NYPapHp1bBXEBAEum2JSKLUGc"
        "SRorjdSVvYY4y+GZp0GcuUMmhwGDjIPFzSDOaYAgS/AsmAjtsH2QCIOZDvEgmAmk9hcKBfEa7BjiQQoV1MMzSMKCHyAefFfOwAoB"
        "QZgQwmoEMA4hyQlxBPEyIEUO6iciSOCxHSAeJfBopkIchY00FeLRdAFGqwKMsLjVAUaqS/wY4hFH4NUQj04aSVwMA5hj6I8eByvL"
        "L3iUTMW0EuKplcZII3VNVkM8ydmSmwbx5DpkJqUZkhws8WYQT0oyJAk+CSZSVNtLhCm9+/aPOointLudf+Xa4mBPLNzAAnv3WHgM"
        "lMMAc0eJhzARTLbAbg+m8rCTb66Dwu9gd7D522X9NZKNOR5zuMf73jDEGAraeYApFucBpghTmgJfLrXO60w7ZV2JNtVojdHRGuTK"
        "0Kpo4alfZgfwl5XFvJI02BItkll1KU5i/BAHV7sgH4PInqhPqeiBZsH2hTERbZKWUPssKzsm+AZdGOndufAYywgYlm+diiq74ZDk"
        "pj6T616+YOBJ5wImDLB2gogH4fm/nTyGEyzC0qjdOLRAF6UlXhBoZFsFTYssWqN4YU0teRaaCikWQWWZeaveH5fUY1JZ6HpiPFFj"
        "wIL12j3yYVkxwyIfdpIQD0PtBzX7ztSwoQ84jQJGurKoXU0N6GAsy/p1TA2HozhaQ42sQTGMSR3gyoAdAhHR2g8YROiQwKpbl8nh"
        "GG1ACwBkUarJ4RCPbyeSw7cV0KJUe9x6nNLThuTImnZArUdCPJLpnXaCiLManU4OD2h51m7AQQ+A+bBEDg8y+ajQ6ZFFUag9OXz/"
        "GM0SVWGFwSM2a8jh8HBjpIxpTI5uDFlhq9xz8eMUORj5YD+ZHKX2zJPJwb4POIwCRro4riEHc78sLZGDcZTQriFHVq8YxiSjOBDM"
        "EEggzRrUOyCBVfEukyM4tB4tABDCiBwB8YQ4kRwhVkCHpHAbcMrYbkiOaBRqAxISgZ1Iykksc+0G5IgoTXTKTQQHIwAW/RI5IsgU"
        "WaEzIosxKHLE/kEao8ZKBI9iWkOOAIRGpCy1Y3KUsQQ0JKPcJ+QjkSJHQj6SnUyOUvvkJpMj9RIyeR1wQroSryFHcv2ysESOVI4S"
        "15CjyN6EfKakOJBSHwi1rWaN/KQQhDJVobxEDmoJrUXrMNNrchDkLrU8jRx5QQdoasOAW2rLKeNm5KA2DagVFzDJgGmVE4OIjZlO"
        "DoIkJkPaTcQA/Bs7JgdBCFN9I/sQJmTR+IEcudMVnQwrrBAUL5mwmhw59ZiElJk4IkcdixhL2j3yUd/DCsYJcpnITCVHV3uiqeTI"
        "vmrAZHXAVGxuNTmIqF/mx+QgKkfh1eQgaF+CniYKAwdyRwUS1UBrMIwEVrW8TA7bojVoAQBrR+SA5CXrJpLDugpo6xVuoYjJ8obk"
        "sEGh1iIhFtixUTtBxDZtQA6IYnKtdgMWOADMLd3FCVKYnLqLE4QvOXUXz51adKfv4gTFS86vIYfFw80hZY7H5OjGgAYXtHvkw6nb"
        "OEEuk0uTyVFq7ydfx7OvGrDX13GCqia/5jpOvu2XLV3HyRfzmus4QfsS9DR5rzjgFYg9a9YkDCOBVS1jLpQy+QnPb70sTeSJF4WF"
        "rLB+vkNwE0/6Q1zhSdlLP909nu5Q4MRWO0FWeBK1O55AqRN3l90SSaosZdZOkGEOGzKeo44ENWPAmZN2AtqGdgPGQ4pTUJI0px0t"
        "eFffPCPEIA9vRF6FeJkPsgRkvr5+LvNdP3+QnGekZnM1o1PLFzShi+oJUT1PQr2bU9BXeoIup7ji7rHX+0VUccRKiG+Ka+7zBOlM"
        "kOMUO/w8xt+Y+yyJ1B6FB6FNUb0MoVhmsooj1msUxTA6EEKOcXUc8BvKslH40NiU2nVxgAcQ25S6q9eno59DiQ+jwx32t1gDl1mA"
        "3/3x+urN9VUd3Z3/8+B4Z57/fbIzf9Ac5bM9ezqbzZ7ODmdHsy9mf5n9dfbl7Kubr2Zf33w9e3bzbPbNzTez48Pjm+O3x7OTw5Ob"
        "k7cns+eHz2+ev30++/bw2263vB92o/9zt91ut3I2+2wxi0s2l21Pl2z+2eI/Zwfv5d69z+fzo8VFqJ1F7kQ9kg5+VUfkb7G1tyU9"
        "X3tz6bEeY6q9belZPZOdHktt7d2RntEzEx28X8fw9rR276Kband+hPepo1HHtXsP3TCa7OJoNLja3UHXjyYH1qP5tlG799G1enK+"
        "f4xGbX+qBl0zmmzp75/W//ux1zzame8+aBY78/xp8ucT+bz6rOkwiRnN7RlH283sQfNfrnUCxkoiAAA="
    ),
    185: (  # fp16
        "H4sIAKp2SGoC/+1YW2/bNhQ+lGRbOUOTlE3bdB68Qt3DoAJDLtilRTHY2oABQQIMdvrSl4CmmDqIb5XsOvsH+wF93EN+6g5J2ZZh"
        "dxjAYU+mQZrn8PsOP/Fiygzx9aeX+B1Wbobj6YTvXF1dj49/uLoZfklVKfIJVaPgF6rEO+hNRod4zzz8GZdAXpWj6XCSRzttlU6l"
        "6kwH8QMMxJ3Km17Tv2e1eA/DW6XG6c0gP2Sa/xILEgby+MqWypSCh7bl+FVU6fRvpMJXuHDxYCDu5LynC3EXf1H0xDb201j0U+mJ"
        "/vX3PBhnKo9qv2VKTFSG9aUOHZhXuqo/mkXBucpzIhs0r+jyen0QnqOF86r52oA4RMvFAsG92VHkX0z7eIBUpc57A5Hfcjaz3mel"
        "YUU240FL9Pu26QSNwVmrPM7//PSnyFq8ko1mvdNNQ7Y2NbAkyVF/M8nbSKqj7Qb97PSIB7oe1doq74mx0o0mXNGo68vGQzRoZG9J"
        "aSaGt1H1Qkz0E9PgaZumzvi5n6XXkd+ZdnEPdZ0cohv5rW5OUF1fzHGmPnSLOfwKjUVY9WF9fg5Q+60C7mUdO9QkSWs0kuRnJEkr"
        "SZYkSS1JliTJkiS5IkkaSfIzkqSWpBVwTxaSXiBVud+Rl9HOJXWdj0e50rtsrLJBkzVpEmr4BOkRimXidVoL1U+RLNRkHlzoBTVv"
        "eIHGxuDyXO9BKpUpBfcuz+e7bwlqG1DbgNoG1F4DJSZSYiIlJlKyHikxkRITKTGRkkWkh0h9U25zb3xsH55cCbkS7TqxrkdIrZRP"
        "9IYepdZ5icbg1bFIU0XO30UaP6KdPUpVRD8hw3wihpN75sfPCCnSvAmlT71Zt0u78lH0p+oxULpnDH/EIh4PxPAP+e93Hy0WvblP"
        "0fBoCfSOPtrF8hiNgZ46Mm5p9dfnPZlmyauj6YR+k2k9pSln7+NPu2EjbOxjwt6e/bkLbxw+4MQFJy44ccGJC05ccOKCExecuODE"
        "BScuOHHBiQtOXHDighMXnLjgxAUnLjhxwYkLTlxw4oITF5y44MQFJy44ccGJS+z4aRiEjE5F+2p2FlK4JiTwa7xn3eYN7MyDn+Jv"
        "QxYiZe0uXr3PDjatmfibEpJOaUJt0Bj/tashxaFsD/vlwfz/D+Ubx6Hcpm3apm3apm36D1L8gI7P2mvGEnPHNzfRmGq1VZDpkemx"
        "RqJvieJdawIk5nJibvu+sdXc9qwtFngKp+8pFvjA2Kv49hLPKH5Sih/4xl7BJ6X4Wm5Sih8Exl7FU/x3X88vVJ/gQcj4Pnoho4yU"
        "Gzp3n2Px994gcB2RBAj7+DfvvYu9nxUAAA=="
    ),
    193: (  # keep-incumbent
        "H4sIAKp2SGoC/31QTU/CQBTsdossDyOlqCExQbLxYPZkoSAQE0g9eNGLHEy8mAKLNHw13dZ49Kd49V/403wLNHqR3Uw2M/vevJ1l"
        "0PuywINcuIrSxDETlxce5SQdy2G6FEWwgnepBuST5EUJ2FzKaBIuVRUFE2q7LjDDBqKJ8NChxXPDRTiWUAa0Q7RQbHM6TEdwg7QN"
        "5vQapQ63bterN3ECh3MZr+TiRc2CSA7ogOpxZbCiYIKztxslqGB3B0zVdWjiXnHrXioFZ6CJVlw0DFQiCli23j6xspmn73RBg9OH"
        "dAFVLTSAKrepVY/n72IZJDKGur7xgE7dVvYjB+s0wZPnnmYylg55FUVG7HyPGD7GzghB0vxLPOExgpsyahN+YWzWR3/f6eO3iCO0"
        "INwyjO++j0FFKeNG39cPFl00BW2N8uWvxf7l60TP51mmUzhmxLHBZAQBiJrGqA67tP9V+BYYNvwAu0GjgDQCAAA="
    ),
    198: (  # keep-incumbent
        "H4sIAKp2SGoC/52WzW7TQBCA/ZO27rQSqVtQEVJBPiGfYs/YTqoKSjkgVeqF3rhEbuJA1DaNEgflcfoQHHk4dmxPWJEUwdpaJzu7"
        "8+04n+WNB6c/DuA1bI0n00Xpb/X7y/EkaH3M52W4C075cAyPtgMnUI+AM0bVSLXEd8o02Lq+Gw8KoGZcxbJg93MxXAyK68V9uAet"
        "fFnMz+1Heyd8Bt5tUUyH4/v5sc3UA7VAplqq0rqBe724gYhDqtvbRHE3UmKV0vPdMupIzlW+XOU4T+ZknBNtWmdzDgLP56R400JP"
        "F9flHPy/G6py6N+LO+TiOuCMqgKTwL1a3MExB6uyE46mwc6nWZGXxQxecTDlYLYuu2LFwupqLOJLVVrvDxYLiDvrrCMezIAHeUak"
        "YPmyWiGOKvPqS1yv8I6DvGzMQVSsh8n38Dns3xazSXHXn3/Lp8W5W/9oB9Ca5kP1YNWnCtVQFCgpqHpeKygJNDGDJgJNNWgq0MwM"
        "mgm0q0G7Au2ZQXsNFDu/odhpoBgZQVFEYaxBRRSaiUIRhZooFFFoJgpFFGqiUEShmSgUUaiJQhGFZqJQRJEmikQUmYkiEUWaKBJR"
        "ZCaKRBRpokhEkZkoElGkiSIRRWaiSESRJopEFJmJIhGVNKIOuQ/c52BU71pvV0G+8GsziYNttdwgL+sX97h5T7/nCTG4owR5Fj1R"
        "VF3BqiirPrmol7ITc7a//bAo1b4duB+GQ9/+Gu57dhsu1Jv70hmlYeDZ6nQ9t4rFl75lWWdVO2s+rXBPZeyc2taF2tmlY6sO6Z0k"
        "vPJAgewKz8Vfnlk/rU1Hw/378WX1h+MFHHm23wbHs1UD1U643byB5taqGbA+46IFVht+ATMBUVq/CAAA"
    ),
    204: (  # fp16
        "H4sIAKp2SGoC/+2a3W4bRRTHvbbbutNSgvsJVUNl4MZIKPM9U1UQ0otKRZUqeoHETeQmDliEJGqciste8gxwk0fhQXgQ7sqc/3on"
        "Z52aRt3c4bV26zm75/j85ievJ6v2xIO/n4mvxIXJ3sHRtH95c3PnQLrNyd4n6e3W6HCa3g66j9Kb4WXRnu7fEcdFWwzEyYWiPZFp"
        "V2nX/fbUDC48351sjeevsWl3affpmlBd0xftnZjKhn5nKtcGnedHL8QNQe9FZ0dKiqpB5+nRrviGooqimqIm9bS/92p4U1z9Zfxy"
        "b7y7efjz6GC83lnvHBeXhh+J7sFo+3C9KF8plMumXCpgU9nUF8raXNY1KOuogGdlfS4bGpTF1ERWNlZl1dr7l1VrVECelFUyl1UN"
        "yioqoFlZncs2UKZImWLKVFamGihTpEwxZSorUw2UKVKmmDKVlekGyjQp00yZzsp0A2WalGmmTGdluoEyTco0U6azMt1AmSZlminT"
        "WZluoEyTMs2U6azMNFBmSJlhykxWZhooM6TMMGUmKzMNlBlSZpgyk5WZBsoMKTNMmcnKTANlhpQZpsxkZbaBMkvKLFNmszLbQJkl"
        "ZZYps1mZbaDMkjLLlNmszDZQZkmZZcpsVmYbKLOkzDJlNitzDZQ5UuaYMpeVuQbKHClzTJnLylwDZY6UOabMZWWugTJHyhxT5rIy"
        "10CZI2WOKXNZmW+gzJMyz5T5rMw3UOZJmWfKfFbmGyjzpMwzZT4r8w2UeVLmZ8pSdMcHinqKxnItfD2tjQ0FYwqGtXIp/EzQewqk"
        "CXw22h5eF91f97fHg97W/t7hdLQ3PS46w49nn9zCq5j926KmPhQXXo12j8Y3W2k7LgpxnyomA5Og6KDpQMzBVgv1O/jM1GOgpoMf"
        "XHr8cjyajl+Ku3SGeg7h9N8JxBqIKli6IpYAN2bFUiBFY8L6dnsbWJGw4tmx2u/EioQVCSsSViSsWMOKhBUJK85hRcKKC7AiYUXC"
        "ijOsm7NiKdDvJsUzru8FBgidnaz7n2RfoKYU3YlcUzhqHA0+JePdLT9adHfkmsMpRngPJz3Cb2G8jdMBR4uLZpi3q6IUohOSg0qA"
        "yrOD9t4NKgEqASoBKgEq66ASoBKgch5UAlQuApUAlQCVHFQCVAJUcVAFUHV20JV3gyqAKoAqgCqAqjqoAqgCqJoHVQBVi0AVQBVA"
        "FQdVAFUA1TPQYflFTX1odKbRmUZnGp3p3BkmRaNuWuCfz30Jk6IdPs7jGHBEk2lhzicl/X2RJsVInFJzk5JWzHTUCyYlrZ9REBcZ"
        "NilUlEI4YZl9A1DjzulOBVADUANQA1ADUFsHNQC1ALXzoBagdhGoBagFqOWgBqAWoJaDWoBad543LgtQC1ALUAtQVwe1AHUAdfOg"
        "DqBuEagDqAOo46AWoA6gjoM6gDp3njcuB1AHUAdQB1BfB3UA9QD186AeoH4RqAeoB6jnoA6gHqCeg3qAeneeNy4PUA9QD1AP0FAH"
        "9QANAA3zoAGgYRFoAGgAaOCgHqABoGEG+iVClhoJaC2gtYDWAlqLubXym47bX0R5WoBQFZxIC4jUc0RzUZfLsls4oct1GL2d9fM5"
        "4gYJmOWYZvnp/vbwSgqlab5TlKuHclE3y/VlzdvIwt054s4eY/Vh5U2YIimu1mbLQBZ3iMsy/gNCnp6qClyeCitaJZw+UhKP9C/u"
        "H00PjqaDi2lNuzWaUt+j3yaH6Ltf/DS80itWLj0oWhvtiawGRRooPtB8YKtBOw0cP+OHV9NAbLR34pN26+Hwj2u9Ir1We6spSE9/"
        "n/x+rfWw0et9t2XuMneZu8xd5i5zzzP3vV/Dz/Db2Ol1yt9G/aSPkq38Q5f24Qf4PaXnSPhBvVr9WNNznWq0SiNdjTo0MjkxuJQY"
        "eGKsJcZaYjxJjJT4NYZIxHOJariKoa6GHQzN8Bpy8WgiJT+qJct6sqwnS5YsU/Lrx7VkVU9W9WTFklVK/uu76vS9DfyhO7yFoXid"
        "tn/epK1AXNWqaF1d9qbaystMrRXta63oUC8ST1oxElPPk0092dSTDUu28tT023qyrSdbluzkqel39WRXT3Ys2ctT0+/ryb6e7Fly"
        "kKemP7i3T3/wtSohvH36A6seFb4FJ2ObxutsXH5L/uTLTqyCad253Jbbcltuy225/Y+3Hz+t/mPgLXGjV/RXRLtXpF2kfZX2F/fF"
        "7JkRrhCnr9joitaK+BccHcnaZygAAA=="
    ),
    213: (  # keep-incumbent
        "H4sIAKp2SGoC/8VVv2/TQBT22U7tHtCG9AdFSKGyGJCn+O7iOF3qBiEWKlXtxgJufUDU/KK2q44ZOzIyZmRkZOwGIyNjR/4LeO+c"
        "NGmaIjJh53N8391337v3zrZNt74t0W1aaHZ6WUr1t7ykp75jPut2Tt01evdYnnRk63XyPurJkIRkQCz3PjV7UZyEWn4CRT0KKlDW"
        "nMV9GWdHcjc6c+9QMzqTSWigaJnax1L24mY72YBZ9CtJMEuiz5Ssg6RG9Wa9ZKRexbH2pYqLcoptJL1/938AkwXUaHoMhXxqNo6k"
        "mGc2dIf8eSisOtaLExml8oQ+wo4qkpjVKEndRTDu5qoqdvrYeZW4g6zt3htl4W9mYmQWTJkFSNZvMcPUsco8ZsOkKy/mXfdiimSz"
        "vRgmlvF5vYKRl5jyEkhWb/HCDDN/Hq81lGGtGRaA1RzzpUyS3KqGTHDTak9VGTvrjrEXxe4KNdvdWDr2UbeTpFEnHRDDfXj9+VDn"
        "qnpOIIrCadTK5JoGx4AQupGX0mhy3L/cG29D9OLoxdmcXuA202sVMivQD5fH+cSCOW53Lm4uGCUqvyq66qQEM879WRJMD47AgvGa"
        "Y+xmLcVyHy+4K3gwZlmArAqpnrMrYMpUD5Ci4hgH2aEaKio4VLHexLRoJjBTAjK1E8cwAd4P95HgOYlBC1ynuL5OK994yEMZBC5L"
        "+E7h+YcsatEy8n5poZul8Hq8sdgSeecu2aRIHBNyvN0Ax3G7j23mBjaxKQDZp5o6+ttwCeEH6Ic5dwH/l3i/o2nFHVBy95zY+VkG"
        "6dls6SCclmraJqACCAF7gDeAHqAPOAd8BHwCDACfAV8AXwEXgO+AH4CfgEvALwxFjEIpq1X8z1CqkF+zaG2ZsJ+NBnwK3OWJNr7N"
        "xwShDXyu3CdXFbC2Vn6PDkJ0wywsWPZiA8v+6vHwG1hap6s2KRWpbhMABZQRh5t0uA1uG9EwqVakfwAr4mDNUgcAAA=="
    ),
    240: (  # fp16
        "H4sIAKp2SGoC/+1Wy27bRhTlQ3LkcYG4kuMkLdAURFEUXBTiDGeGCrJQlD4SxQaKOqtuDFpiC6G2JIRU4aU+xd/UH8hH9Ad6z/A1"
        "iVzE6yIShtCce869cx8k1WNP/zlm37PuYrneFP398/Pf15E6Xyy/oJ+zNC/oZ9B5QT/CfeYVq0fsxvVYwFoi8xYRLU5L9L0iDrpn"
        "l4tZxp4QPWb+IhriEuHC+34RiZrAQQASB/u/ZvPNLDtNr8MD1kmvs3zs3rj3wvus92eWreeLq/yRi8CCgQ+RvE3k/7dIQqRq0dnm"
        "qhF5t4oGECnmzRSEOvDPNhclqAnUAJMapD0MCcBR4J9SURomQD5smQnDHmDUMjlVcIYTch74z+dz9gDyESyoGRdB5yTLc/YlEAEk"
        "3m3KoNRUjmTriLwDAKpsR8iM611HbfAIjMTWmHRGt2tMnGhEDDG0NALZimhX0zeFK0skeFmi+xAgZyEogYucHQEQVVYitv1iDoT8"
        "iF9V1sG4UaYpQLUVzQCJFS2po43saMgrHu5GMxLJYAQDXU2vTasRLkYJ4yo5UCNMY4wMY8rwdHNpoYYbt2gMlJvA8gNUGK4q0WOg"
        "kiIaqg72TtOiYWM0OUoVJ5aPpDnFqPZBcoCYCjl8z4ccwgfqJKPWh4xqH5JbnmNccBApylIYLs4sDTcuUbtJUrZNkrJuklRtk6Q5"
        "lW6bJHXVJGmPpzTebhnPhzDiZBInU216xjAyVhiiXYNCWxVvDN81ZBHsv3mbLvP1Ks/Cz1lnnb29Gjtjd+ybZ4phlur4Lkx0VMm7"
        "MFFHpT7CPKpTLo+g22Yo3YRLLDRpXI8s1JQAjxw9bFGNgVAogo4sFAOh0BTNLZQDRf+0NRC6eZLrauABaAPIu78P6ttMm6jKus2Q"
        "uTZRden/GwC6CZIEey9Wy1lalBEWlcMzEJL+3mpT0Psw8H9J5+GAda5W8yzozVbLvEiXxY3rh4+p4uk8p4q334PxQXnK7l/p5SZ7"
        "4NDnxnX77h/hZz3v8N5Tz3Em9Lqsd4MB7Xi983zaibDfc8vvIZvQiE89J/kAU4Q9C5VBBhWmp99SsGd0ionzg/Oj85Pzs/Ny+9J5"
        "tX3lTLdT5/X2tXMyPtmeVDpSGl1yZ927LgUrRaPp313SOJ/Wp/X/WXQfunQfuu4Ef1rrHcMues/Gf3tS/18+Zkc9t3/IvJ5Li9H6"
        "Cuvia1Y9QQyD7TImHeYcsn8ByCiObX4LAAA="
    ),
    244: (  # keep-incumbent
        "H4sIAKp2SGoC/8VVPW8aQRDlDmzwWpHR2cqn5FinVFTc7izcuTGQIooUpCh0qcLHKULxBzKHREnp0mVKypQpU7pMmSZSSpf5GZm3"
        "d+c72ZCPKsA7ad/Om503s0BFHH6/J0hsjE8ns8ixo4a79SYczYZhb3ZS2xal/jyctqylVa7tiMqHMJyMxifTh0zY4mmiEva4yfAZ"
        "gVOMvLq70TseD0OxKzifAAPac4u92UBIkFjLVScVV56kkMSDSP29aA8iiYeCkuLjd0AQCO0W24NpHKaFPTRRDbf0KpxOxROwpsym"
        "W3ren0a1La77LM6s03bxtv+PLnyIglTU7c//0GRTXmCssFLWMxcSXZVe5kJ6iQspcy6k0am7Lh7E/Ukk5JZfnIf9KDyPVWiR1HdV"
        "pp4mIoys4Ra7s+P4eIxaarDNmN0FwVdjiNFJPyXtIZkNkEHsx0QGvAO5queS+gIEWDgdjUwXFRKqmwvUHZ/edNFe2UXkV/AKW0pl"
        "mUwVilYN8TeZKM2k40zw5GEDPVGNzCiun8I1Us1scMqc6WeDU34yBRXkBqfwZaL6qhEkp6FZ5OUkhL6QXD01quOB20Aqm4/SiRei"
        "bBQEgyZSx6TR69Qh5RyiEIJDyjkkOKScQ0odUt4hwaFe41CmDnXeoYZDvcahhkONunXi8FH6EwUKPLmb3X6ErccoAFdAk7N5Nos4"
        "Kt1zrPe1C6uC937FqlruvGBeiyN+tPjDWDCWjCvGNaPQLhSqjANGndFivGa8Y0wYC8YF45LxkbFkfGJ8ZnxhXDG+Mr4xfjCuGT/b"
        "HW5wWgoX859LkbW9pBI0pcQnHTGrbrMLsHSLfRa9ZFbXtnldPrQKHf7TSBcWL/z8Inib/rc49wXncarCrlgMwdgHBgciGdq6iE5J"
        "FKriF/WllPbgBgAA"
    ),
    246: (  # keep-incumbent
        "H4sIAKp2SGoC/+1cTW/aQBBlbULcTaUiK0p7QLT10Se8/gBySUXPXJpbb4SYympKIsVIOeYnRFXPVdRfmh0bg5NSBAnwDp2HdsXO"
        "ePf5Y7HnzSa25PHPX0I25V4yvpqk0kg8XZQuvm2kgbN3epEM47I/1CXSpa39ncL/ce7vSjPxWlR5tpl6qthEj5kGZPGdV1/i88kw"
        "7g9u3ANZHdzE15+Me7HvvpHW9zi+Ok9+XL8T98LI+nSoT7Ben4wnXNTHXM4Trd7nUNKxUEVsynNM3Ue+lfRdmiOvTVbl1PqDtD+5"
        "KDmISPkzB42jFFU+OQI9TjLORw+pisgazkYfeV3aNiRrVB59pFozR/vx6BFV2f50noyusn3vaut0Wy8gK+2i35pbFXH6LbJOj5Os"
        "vkcdFFlVaVw6Fj+z+o55OjmTfTLoMzXyA7o2dK7zxtNKd8u+2bXLSapnk1P7fDkeDtL8SiT5ibfFN/fAEvX9Y2H09HQtGqZuqKIh"
        "dMMve8KiUdWNqLxZu2hUdKPrvi48NIsftTz3d8NqWs26cO4alcrtCaYQULwI7jJQvLvkXgQU7y64lwHFu03uVYDi3Qb3OkDxbpL7"
        "OUDxboL7JUDxvoR7E0DxPod7k0DxrsO9DaB4V+HeJlC8y7h3ARTvIu5dAsVb5kbg9qRHiv9vgYjZGewFQE465A8NeXNB3lCRDxHk"
        "gxMZLCADJGRQiAyEkcE/UvAgRR5S2CLFPDKBgUzaIBNVyOQcMiGJTMIiE8/IZDtygQG5qIJcSEIunuUCscMCkQUiC0QWiCwQWSCy"
        "QGSByAKRBSILRBaILBBZIJJA7PKfmCInHPJHhryxIG+myAcI8qGJDBSQwREyIEQGwcjAHyl2kAIPKWqRQh6ZvEAmbJBJKmRiDpmM"
        "RCZgkUlnZKIdubiAXFBBLiIhF85y7h79i6f7p2EJ/ZmLRAaDwWAwGAwGg8Fg/I/o0ftzvr6fvo7JPpKHlrDr0rCELlKXJpWzD3L6"
        "ip1/bdGrykpdPgCpopptHkoAAA=="
    ),
    260: (  # keep-incumbent
        "H4sIAKp2SGoC/5VWXW7bRhDmnyx62iDqRnYcN25S1kALPjlpDCRpi8hsgzRAjRbxQ4P0gViKa0kwuRRE0tKjj9AjCD1CL1Dlrb1F"
        "DtEDdHaXS0ppUrAUZsFvdma+b0e7K7nw+HcCPnQmfFoWYA/nZ2J4KoYTIrxhvu8M52Hudc6SyZDBV6C8ZHs0m8RhSvMLb/sFi8sh"
        "OytT/wNw6ILlA3Npdv3r4F4wNo0nab6HDgsOockCZ0yTc+IKR5Rlidd9NmO0YDP4WlO4wzHlYV6m6wzXKgZrYL+T4zOo0zSFxNMs"
        "byg+h9pJQL+F557zLc0LfxusIlPV7sDaNHR5VgzHR8fEZuFLzz4tE/i0EosdO5YDw+EeJdbpsW7ZHiCotNinx1Ej42adjPWIs8hZ"
        "oqp+CRJglUX77gqeRc2zWOP5GAQG+7soB4cHz58RK154nZ/HbMbgASAgZt3kU7rA2hXRe9u8q6qZKbFHRdpQ9UEsEYST2DS69OwT"
        "HgsB+L4hgF5qAY8AAdlK6ULE/y8Ve1ClgV3MM+JwNp8iYxzD3Xqmw8eCcGtM85CuNeVGs4IEV+D8wPK8lp8I+VEyV/L3QbyreFuq"
        "j9bVR0L9hItwrX7CW6qXaY361LPPyghu1zNOIz5KKpE7tfKtjIdpGKukHahg5aYoHsP2KzettoYjUFXplv5KsG+kixPTI13tJmis"
        "J6p6t/WELtiRsKq4DwpC1W9JN1VtXGNLZVH+FhvXbPwtNr7JxjfZeMUWJZKNK7ZdkCuV41ROUM/6cYZdl+9y5LJVNEnkzAFUCOpb"
        "SeaNVMEDmTKSrndcFQcbl5uIQa3n5xgqV9hXPnXUreKeOuh9UDFgsSP03tdeDEC7jxXKAi9fuaV/AYVQc1ngdS0vZmztTzT2b4CT"
        "ZjHz3GHG84LyYmna/i1wpjTOB8bapz/oq73YuaRJyXYMfJamSbaHs4tHYTyhI7/nmj3Tc3DiSSB77l+vPYNAbNXacVg8D8SBqHMO"
        "i78CecAbz9/fB3IT+x/Vnj8DdSj9b1zTBTQx8YUhn6snm/ZvX6CvYf/hZroO/+8nwFb7H2JO97HZCcS1rdGWQEwjMxAXuf+bLQiQ"
        "BpDkV7sRhN00BmhXaEu0FdobNOPEMHpoV3+0izVW7WIHq3axV6t2sctVu9jVql3sm1W7WON1u9je6zaxgbhS/GuuhV+ZJSD+hdEQ"
        "QMCnGlq2gCev7lR/d8gu9F2T9MByTTRA+0RYhL8d6oS9LyJwwOjBP/Nr09I9CQAA"
    ),
    265: (  # fp16
        "H4sIAKp2SGoC/+2a3WoTQRTHz+xHsj1WjGNaYgVb9sLCCFK9ECy52MRWJSgU6oWosE52t2ZJulmym8bLvoFX3udBfDhndhNsjQpV"
        "xH6cGWYyZ+c/Z2bOjywJHAe3vzzDB2jHSTrO+ZLvH6QPH/txsqaGgcxyNXStp2ogltDIhw2cMkPpvwvRDCb7utvVXYtrT362ZgUT"
        "P3Pt/UEcRNjG8inaEz/obXGj+1E5HSZHYgWX+9EoiQZ+1pNp5DGPTVlV3EQrlWHmQVnVI+VDrVIOgmEYcVv3B7/wYXrWSR/MM2Zu"
        "cR3LheX6+NTFqvpiAu1cdgdRqYu5fShH/S238lzmvWgkrqElP8VZA7R2FctZfS5u6aFrvhoPcAcLA62JDENeUd2hTM983TvzkM0c"
        "cHs4zlVEzVYY4jssLV5RH4pbEe3UNfdkKG6p7dXZXScYJlkuk3zKTHH79AZFXfaW9d430D6Sg3G0AqpMGeOVYNR/4j8Smw5zUDVW"
        "w3bJrVOHJiwUcV+LHNOxCqEOXKehhB7swAt4CXvwGt7Ae/gAIfTEZ8853tDCIs6dYw8uU2n+EKCz2uftLr8738/mT9p/Mk/8iT/x"
        "J/7En/gTf+JP/In/RWD+N/Wi87+s9/+3/IH40/uf3v/En/gTf+J/ZfgD8afv/7ni3/zPlfgTf/r/T/yJ/5XkL+6VWQZFnkGRM9Gp"
        "w9fF30riumPUqtsGQFtnnMxNzrW5OzcNU5utt+vzjJZVrDuM19BwmGqo2l3duhs4y50oFLioaFsINfwGWhWpjiAjAAA="
    ),
    268: (  # fp16
        "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"
    ),
    275: (  # fp16
        "H4sIAKp2SGoC/6VX224bVRQdX9o6B6EmJjgGRKksHpAf0MycO8rDNJReXDuVKE+8RE7sIosmjnxBffSn9FPywAfwSZy153bsTC8I"
        "RR6N1+zbWnufPU6L/fR3l/3I7syurter9t7Z2evrSJ3Nrr52txfj5crd9po/u5v+Hquv5l32rlZnhpWG7frK9PZ+nU7WF9NX68v+"
        "Z6w5fjtdJrV3tXv9+6z153R6PZldLrs1eHY9T1a/kM7b9hqj9RsWu/C23VhF4adHg4+BT5T7jMZvC5/Gh33iKp96pc8XrlTFkAeO"
        "PC2YA+AAxH+JBHsXLoaj7DUeTSYUXuMBgcoLrwDoTydH4XUe3qThDwEaXCRQp/er9Tk7AmCdKTjFYe/e08V0vJou2DcM3wFGtzuP"
        "WHGUx4rjtFYQiOkJQJ4mIFNiAbljkZoSSrYCaKEA7l0U6BmrPGoGUlSdR01bEVN+02sOp8tlWjOlsVU1p/KSCw89Fw6a/H00LSyQ"
        "m8dl7VEIFPLybA6O8vAaoNgWkoMllx8oCi3myi+KEF1dFJe4UCbjCUqlgj+3u6XiTImw7FOhhIjK6YOiAlRFnMp8HwBYCsfy0fmS"
        "AgqeNUMIr14BiqKCIiZMoAZB6VTv7mi8yhVTuWJCbysmCDTViqlcMWH9CsBRhtWKCUy+RKNlVGojo7y5cre5ErQlLxUrxk2K7fMq"
        "wVzKUjFJVqpUTKpMMam9eiUoSlOtmKR6cQ6kLRQrjp0AeRVWHDsVlcdOwVRCFeWxU2CnwE7xlAjyqbypamd0Fdipir5K8sJD5W+m"
        "z/PF997dlFarIJLKjjNAARCSKG+kFTaHwkgrmxZLacFJF2+I0ezqo2kpGFqoMQM6KrulwVrHZbd0nEmhudctDapa3NYhJ6QhlJal"
        "/FrkNLXKdwT2RR59Z+I1uGtTPb4atDV00N7R1kQIYpiwXPKUl6Nc4426LvplvGEwGBGDYTB8ew/QqBs36o9nf7EDWIGfkb07T97M"
        "5wtaywZrGWUbVeppMJtGl3oanTE2/p42IGMq9vRDvwCLGZ9P8L57fTmflK83G2aprddKiyzWa6XNW2n9VlqoYCta+QPqgpxW9vZ+"
        "W4yvltfz5bR/wJrX08VlEiS1pEFvdLK0EMSqj1geUUxcKKv21h++4kJEzfYD9NBgVG15/Lt4gFBWtZtuS4XFkw5AwwikR1njv0zf"
        "nSQm4KzzHZKYAIJzcb4ljBNWIU+HFl7pJrfc0gyqap1RPLqmheu8OiqVfgDizqSdPCDYEGTTXnYISn+muLso9DNHRDmqeH9/RY8j"
        "ctbtu/P1yv26pdzt2h/9dquW/u2zExd4UA/MDhY77HgH4w5LdjAxqP9z0f/efWcFJgeHQRAcB0lwEjwOfgmeBE+DZ5tn/SH5Pcis"
        "1OC4yip4vnkeDDaD4MXmRTBMhpvhzTAYJaPN6GYUnCanm9Ob0+Bl8jKL5uJRNP3/ov3+Xf4PQIcdtmrtfVZv1dyHuc8DfM4fskxE"
        "smC3LU6aLNhn/wI7ydXfTwwAAA=="
    ),
    298: (  # fp16
        "H4sIAKp2SGoC/41US2vbQBBerZxGnrbEqCaYFpoietKhaB96hR4S9xVCe6lvvZjYVopoHgZJ0GN+Sn5Wfk5nVl5LwaUUMct838x8"
        "nhmt7MHxgwfvYK+8WTe1P5zPL9cimZc3L9FdXlQ1usHgAzrhEHh9O4F7h0MGXaLPaxUMvxerZlnMmuvwKQwufhfViXPv7IcH4P0q"
        "ivWqvK4mDlVOepXAlxKrdeB+a65AoLxGGP+/2CssiYGXCVqKlmF5HuzNrsplAUcm6JYiokPQIX23FmonQdNhvIQSUpvgt/J1TmwW"
        "uLNmAQdAPhF54J4uKhgTkeMoAkkZBYOvRVVhZwSIEbvbe9H+MLZCCbITltSgVJ2wVFZY94U1MfE/hFNKSHrCNJhMe8KpFc76wjSY"
        "zP8mTKKbVaioFSYdFVEZCSnRvkbbgzIkDne6WhkytxMr1atXVB8Tu7kGhpV00JQqbgXMlpUdTSU9hYQUTF9pT8G0YHIz2wL55sqh"
        "k3e7UVSro243OtrsRovebrRh5O5uDikoESv/yW1T41dk2vCdn6HvOe0zgilKnnOWhW8Rw5aT52PG2Ht2wqbsI/vEPrMv7OzuLHzm"
        "8dH+MWdsirfbIsdBlFrEXURZ+Hwbo6tuIecExRa6BOXjqLbQNdH4cXLy48j+KRzC2HP8EXDPQQO012SLN7AZ2GTAbsZ0AGwEfwAx"
        "b/lsYwQAAA=="
    ),
    301: (  # fp16
        "H4sIAKp2SGoC/6VWXW8aRxSdXbDBU1W2CHWsREor1L7wULHzvYkfMG2ahIAjJW99sTDepqj5cAxIeeSn5Kf4ra/9P31o5szusgNs"
        "FFCNQOszc8/cc+6dC3X68O9j+jPdm7y7ns8aBxcXf1xH6mLy7p59HI+mM/vYqv5iH9oHNJy9P6GfgpAaWmxshLO4dfAyuZqPk+Ho"
        "Y/sbWh19TKbdyqeg1j6k9b+S5Ppq8nZ6EiAy9iMrs6hTFhqWhjbs+TENxxHiolZlOH9DOcUzALYT0VjlRLxVeTW/pHccOf4HKDx2"
        "AUBuz37HsSMGgSqnh1a7wgBqj14DMNsbmCfviOKUvZmyA7Ao6xT0rAMg2p4+z57BHMZS/kerRWM853s1f9v+NnfjC4x3wcYtLZxl"
        "olV7cpOMZskNvY8FuMvkZoc5w5iNMthgXTy7uqKQyBQA3do/u3m9lDNJz9o8/BgRmlYmzPHErb3HH+ajN+nZMIt3ys9m0p6N2vCs"
        "02CMoAgAyArjeYQPlIPzwngOtVz4RuXGB9tcDC63D3Xnodu42qXQ2J/dAu51JHeqzS4Nj/2wB4FxWqplTqKzG5PIb4nwrrhAkoLt"
        "pk6wLCfB05xO0k6sTIRDZav2Mpn+ObpOvBV0l9CrK9ytoIFEXKx8hyM0PiBT2j4aJNOp6yyJHpHRZmehXyT6RaD3JCsUSkiWvKzo"
        "5QqRmERiEmKkJ8bVI860S1V0qsTFli5bnTriUGjgbq9ZzsIxkpSQLLMZcwgAWSs7Xs4upy5WdbJ7rSJPvkK1FCuXL9xYcjG8mI0q"
        "HxBKpOBddAEWkJiSq1NDQYNSZQekUcJRaT8ndLUy5TkpGKOgVsVpSRzqzodk3SlQ3clvt15OBs9vzVY91Kir5oWH2oWKwkMtMula"
        "evlqSNRfk6h9iRoS9RckakjUkKg9iTrOJ5rxJSJx49CouM0GVTVs+7HUzCzgsNBwL1OD3I34ijjj+2Hgh1Hl4gwGs8HVNXp1WhvY"
        "b0xxBwy63cAH4/lgrA/jGIrjqMyHmKU+4LssRoJx1Nh/P5/Zn0xuoRG8bg/qgX09qAdHtGeF908JIaekS3rkV/KY/EaekKeLp+TZ"
        "4hnpL/rk+eI5GXQHi8HtgAy7w8XwdkjOu+eL89tz8qL7ImOzfI6N/U+2RsaW5sb7ITFrmLDY6RomLdZdw1Q//Gfc/tH+T5eY7jdd"
        "dmuvtV2m3/zvX7Lx176/3FV7SEkQVqp7+7X6QQ/f2e3DetXC1SCgQQ9zuwACCkCtA0UIDRAiRfsnLw1U2WZ7upnH79/nv4GPabMe"
        "NI5oWA/sm9r3A7wvf6BZyd0OurmjV6XkiH4Gr054XVILAAA="
    ),
    310: (  # keep-incumbent
        "H4sIAKp2SGoC/6VV207bQBBdOwHCthUQAkWtRKuoT1Yf7L3YToVUQy9AiINU3vqCTGK1ERAi7FSR+uJP4R/6A3xCP6kz60sMWGpR"
        "Ze3Ke3bmzJzZ8bpB3/1aoa/owmg8mcbNhdPT2Wjcrn8IothYpnp8tUVvNJ3aNN1p6jFvL38Jh9NBeDK9NJ7RejALI0/3ajfakrFC"
        "G+dhOBmOLqMtDf3eAgUHJwmUF6MJ2Ncug9kGIcn7G01Ty9EYloTAkq6BtaT6wAIPu13zpxcIDQTANkBOu3YyPVNWDsAuQJ3Uqolh"
        "lFUttswUW6f4DmAHQatd2x0OKUfQQoDlMvzR2HiSydAqRSgmBhEZOvKUaUMx4cQRFe16L4wi+hIRgYh8WEVFlAqEl0zhblZa3FP5"
        "O+B4Nf5hbNCn5+H1OLw4jb4Hk9DT0vTWaH0SDCOPpA9AqSoHfJlVqApmharqo5k7sSonvdKpRTEIrQ0srCrjqQSkYlgGJh5FNbDc"
        "gkqmh5sHAABRuxQAi8OcxwcAJ/R0SwEYbDET0U4pAObBzX+vYKFAUXHrbgAAEGXzABz7h/PHBwAn9BRpAOwi5mbtyLO6balCoinW"
        "jdvtpf3rMIjDa9WRHGvHneqOZCLncovWzrhcRDul1uZYI2E+JELV3KG4iRbW/BvkOb1g81SZiaYK5XdTFShViOpUuZlzyXmqKRfK"
        "FnYpVYGahVOdqhA4YV8IN01Vodh4AkWLrC825x+nQOnSLKzTEjG0ltnt0sqzkXj0MtO7ggAmLfHqOItUKMkzJbJ8c0i8OaSszDll"
        "5srCnsdTWUjUKp1SPJQm3VK8vGFk+TglarIrjvM5bkqc8Dhtq73oBzEKf0FxiZPZXLyaxvDTyPea2jfjTUNrUBjaKt2Da67bgpt9"
        "5/5jNNGisGJdnbj3PAV47vzV0+3qJz+N9RKG30tX/z0wfAVt52Cnu6My8cge+Ug+kc9knxwkB+QwOSTdpEuOkiPS83pJ77ZHfM9P"
        "/Fuf9L1+0r/tk2PvOKMDQkXHzP+j+1r8bzdpq6E1V6ne0GBQGNs4zl7TrLjKgj602KtTskr/AKWm/5G+BwAA"
    ),
    330: (  # keep-incumbent
        "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"
    ),
    342: (  # fp16
        "H4sIAKp2SGoC/51Xy27bRhTlQ7blKQIbspw4KdAWRLvRoiA578AL262bwGiAIt61C0OW2FSoH4IpxVnmM7r0J/V/umjnzJDmRKJj"
        "uyJIk4f3nnPvnTvDcZe8/Os5+Z6sTC6m81lv/eTk92kmTiYXL8ztaFjOzG3S+cHcDNZJNLvcITdhZOwbQxKPro9xOcRlvwemk/JF"
        "Z3R9UiYrx2eTUUEYcWgvmslk/W0xno+K4/n54AvSGX4oyr3wJlwbbJDun0UxHU/Oy50QKqLxUr7Xk8or2otb/XomUEWike7FsyxN"
        "4v3xmKQE9wCyZHX/6t2b4QenPnFOyyxfkniS5fCgywXYAR2FBYMFT9beFuUfw2kBN6MBUCy79fFS4MJhIZP4eH5KNgBIAMoEe1o6"
        "M2RAAeqk83NRlo4ZOeXpMvPTqlYEr2GTJfGb+RmhABBOnj+88NYJqee0drqtl3GK73FibU5Rq9OW6Zmcm0xtiMLVA+nnlk4AlV4m"
        "qFKuHs7fd/xwgqd2AreqCJemniojAIB69aMIjuYPL8WtKrUC9FbVRGBU0TKUub4ESGkNcgc2gzmiKAGtSoAQqaxHlKo6cFM+yw3Q"
        "7xYKhLV0y7PKhUKVZcnaq6tiOCuurBcDOcvbutdUzMYLA+oJMYuwdiHrgn5nfEHIgndME4YZy2x8fvYKKLqAKYeizAyDy/Rj1ohG"
        "AgXm6YIERw/wrJHgKArPHy2R11lwuiiBmnHmSVgz/v8kbBZiUcKi0pNA5fijFtMtv7u4blYsDkCkzYol0mrFEpnXGgKFEy3dtOV3"
        "k6ANsbAA84hZTcx9YnSPuKN7BJgF0hdV+p/kIZQnh+YR2pPTlZxMPTmJjpDZZ/JAPDJviCUmv6QNsaQ1MfOJMeySt+chEbK0zOLT"
        "POzUld4XRGJspfcFkfUXRPprgkT+Kr1nPFTWECuMoMobYpVXxMpfA5RFWHseClNNIWTFW/JQwpPDqCnpyclaTvlyGDal7xkPnTbE"
        "GiOos4ZYZxWxzj1ijWHT9I48NCysD2smGwMqUH/NG5QjColstDcxNeahtqh0q721BapsxMpjwCTWKJDWDt0GYFHd65gNTVp/MEBH"
        "LGLxzOHbNi4XBuC8gRV33ICpg59ZgtxeqX3BanvzfbKws+euqAaeOVFuYVHHGI2EhZ1oNQF3LJTZq+ytXM5nZn9o6X8j7qm3av6Y"
        "XajdO06T+JfheLBFOueX4yLpji4vytnwYnYTxoPnpDMdjsu9wDv6e323fq28H57Ni+3A/G7CsBe+G3zbDc1BuuEmOTBNd9Q3r3aN"
        "z0HwY3AY/BS8Cl5/fF1ZGTtrld9h1ausHBc9igK1gDGD7Vo2covxz2g2VsJY7QZLvwUrfdT/958Wq20T+dpLEoRR3FlZXeuuH2D/"
        "OtjodgzcCUkYAmCDLS9abFOOor9Hg+88CWw6qngXjsGTbmTIoiA4wKa/fiTwuT6sH6MYj/u/fl3/U/GU9Lthb5NE3dCcxJxf4Tz9"
        "hlQDbi3IssVBhwSb5D/kjCE8owwAAA=="
    ),
    353: (  # fp16
        "H4sIAKp2SGoC/6VWTU8aURR9M2DF50KCtDFtaptJumHRMO9rBuMC6YeKgEnddWMQpg2pVSND4pKf4s/oogt/QtNf1HsGBp6IiVDI"
        "S947c++595575yPHd/7m+Xu+0ru4GsSFtdPTb1e+Oe1dvKRtp92PaetlP9CmtMbd+HKL3zou9/jUkLs9n5agJQturLyVk/NeJ5q1"
        "0bQMrYBswtSGnGJVyMS+9Na+RN1BJ2q2b0rrPNu+ifrVzK2zWtrguR9RdNXt/exvOYg98VHzfNy5PpscMbjbqcBRe5nm4JxLgBqA"
        "WYip44uUKfAyJ4MzXhzRAwAaWvwhgMrT+YsjfjiRpyhbARQulYH60wDCByCeLl8aYEwl7wcgAKiyAkBroRcPQE7wNKMA6Bq0EMFi"
        "nU58wsU6LYJxf0TFqgOALC/ZaelPdSJ6AEDFlF8KAHKJTktILpUVIEzbI61RlZBTmmU7LYP7AQgAas2qhNayskSnJSRS1qzKZJQQ"
        "Vo2F2+LYwwHCKeGt7l9H7Ti65q9wBeIp+fBB8zx1UxhDpb1sI+r3Rz7QQ5mHPkhAoSBlYGHVrcpIK0FDK61wklZlJi3UpcuPpBWm"
        "aWnfSkuDSIv5aWkkoFGstm48AVShHZqmYK/bHaEGtpgNrUfoJoYRYBJ0fGNtAEBJmgrdO+snuekgLUmHdm7osK7MzY2mBBYQ1ZSn"
        "1AZNNP6U2kyaaIRFbVCUkY+UjawNSjHjJwtuU6PGT3Kjp1oYTfQG9ZhgZPrafo8ALjy7HMT0skpEKTjfS+s5J7+642Rq9CZKD1k6"
        "iPTg0EHaV3R6WKGDsc2CUiPn0H+bIF4jveu7jLFdVmU19pF9Yp/ZPjsYHrDD4SGrD+vsaHjEGtXGsHHXYM1qc9i8a7JWtTVs3bXY"
        "cfW41EzYiI/YoPF/0m2O6ZwRnV93WTgLirr7pzMDKkWWv0rvCOAT0Jh6kf1mLMnI+n19k34OvODFnFPIczfn0OK0trHO3vJxDxIL"
        "/tCiluUsz/8BPixOe10IAAA="
    ),
    357: (  # keep-incumbent
        "H4sIAKp2SGoC/8WUvW/TUBTFnz8ozmVo5BaEhFSKR0/xsx0nLM3HSgWiW6cmjqmikjqSbSljxo6MHTN2ZGTsyMjI2BH+C871w6Yq"
        "DiuOf0/WyTm+974XxaLXPy0K6NH8clnktp53ndb7ZFbEyUmxcJ+QOVkl2UDbaI/dXbIukmQ5my+y5xB08gh2RKIqcjxZ1RF9ayRC"
        "pNdUxfhnlX5TleaIRKRvG7nXaSrT3NkesZ/02OOg5xgnxVSJaCAOWJRKtPH2XmX07xl9iJLFwDGOi4/0ikXJSxkPoaYzbuPDIp2p"
        "qvv8bViG2dJVb9tloctC5BjDaaZsPi8Rq72qqB6X2R6L/XtZHl52/mQlD8YVpOeYb5Isoxes8gRSOuZ4kuVuC2Ol1V5gy9nA00hf"
        "TcPVvFoM6haUkyeUYe1kMZY8goyU+JZdES9+/bR1CcsneyctcvwqnZ1xehlPcnWCc3Vgtnbu7lua+rQ1xxRifTRCiw9VMYAq/1LZ"
        "67tXSjwo5ZUor/URh3CDNdiAW3AHxFCINjgEHTAA78AZWII1uAKfwDXYgBvwGXwBt+Ar+Aa+gzvwY4hWgqoVNPOfWwndpw/2Soz4"
        "ME9fVv8Szwi7abdJtzRA4ICZHtLvE9vmGJkk2vQLWVx+1HQEAAA="
    ),
    365: (  # fp16
        "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"
        "TP3vvQdnpgb/PTwzNTf7xCDoA/sGJX/fgAc80Xqy9a+tz7T+rbV/fX/r39f/vXVg/UDrP9b/o3Vw4eD6wc2DracWnlp/avOp1tML"
        "T68/vfl067MLnx1aG9iDNfX/aW3H0FoVmz4w3bLRGA3G9kVjfGD6+rG9fzW4uvfxqdYT06fM6GJqcGHrF9neb98/xO8j9r8nduDC"
        "/f/vEa9/dvOzrcX5xYVFt7i+uLG4ubi12Hpm/pmFZ9wz689sPLP5zNYzrUPzhxYOuUPrhzYObR7aOtQ6PH944bA7vH544/Dm4a3D"
        "raW5pfmlZGlhaXHJLa0trS9dXNpYurK0uXR9aWvp1lLryNyR+SPJkYUji0fckbUj60cuHtk4cuXI5pHrR7aO3DrSWp5bnl9OlheW"
        "F5fd8try+vLF5Y3lK8uby9eXt5ZvLbeOzh2dP5ocXTi6eNQdXTu6fvTi0Y2jV45uHr1+dOvoraOt9kx7rr2rPd/e007atr3Q3t9e"
        "bLfbrn2yvdY+315vX2hfbF9qb7Qvt6+0r7Y329fa19s32lvtm+1b7dvt1srMytzKrpX5lT0ryYpdWVjZv7K40l5xKydX1lbOr6yv"
        "XFi5uHJpZWPl8sqVlasrmyvXVq6v3FjZWrm5cmvl9kqrM9OZ6+zqzHf2dJKO7Sx09ncWO+2O65zsrHXOd9Y7FzoXO5c6G53LnSud"
        "q53NzrXO9c6NzlbnZudW53antTqzOre6a3V+dc9qsmpXF1b3ry6utlfd6snVtdXzq+urF1Yvrl5a3Vi9vHpl9erq5uq11eurN1a3"
        "Vm+u3lq9vdrKt+cz+X35XL4z35U/lM/nj+Z78sfyJKfc5vvyhfzJfH9+MF/Ml/J2nucuP56fzE/na/nZ/Hz+fL6ev5BfyF/ML+Yv"
        "5Zfyl/ON/JX8cv5qfiV/Lb+av55v5m/k1/I38+v5W/mN/O18K38nv5m/m9/K38tv5+/nre727kz3vu5cd2d3V/eh7nz30e6e7mPd"
        "pEtd293XXeg+2d3fPdhd7C51292867rHuye7p7tr3bPd893nu+vdF7oXui92L3Zf6l7qvtzd6L7Svdx9tXul+1r3avf17mb3je61"
        "7pvd6923uje6b3e3uu90b3bf7d7qvte93X2/2+pt78307uvN9Xb2dvUe6s33Hu3t6T3WS3rUs719vYXek739vYO9xd5Sr93Le653"
        "vHeyd7q31jvbO997vrfee6F3ofdi72Lvpd6l3su9jd4rvcu9V3tXeq/1rvZe72323uhd673Zu957q3ej93Zvq/dO72bv3d6t3nu9"
        "2733e63+9v5M/77+XH9nf1f/of58/9H+nv5j/aRPfdvf11/oP9nf3z/YX+wv9dv9vO/6x/sn+6f7a/2z/fP95/vr/Rf6F/ov9i/2"
        "X+pf6r/c3+i/0r/cf7V/pf9a/2r/9f5m/43+tf6b/ev9t/o3+m/3t/rv9G/23+3f6r/Xv91/v99y0267u8fNuFl3n7vfzbkdbqd7"
        "wO1yD7qH3MNu3u12j7pPuj1ur3vMfcolTjly4qx73O1zn3YL7gn3pPuM2+8OuIPuabfoDrklt+zaruNy13POle64+5w76T7vTrtn"
        "3Zp7zp11X3Ln3Zfd8+4rbt191b3gvuYuuK+7F9033EX3TfeS+5a75L7tXnbfcRvuu+4V9z132X3fvep+4K64H7rX3I/cVfdj97r7"
        "idt0P3VvuJ+5a+7n7k33C3fd/dK95X7lbrhfu7fdb9yW+617x/3O3XS/d++6P7hb7o/uPfcnd9v92b3v/uJaxXSxvbinmClmi/uK"
        "+4u5Ykexs3ig2FU8WDxUPFzMF7uLR4tPFnuKvcVjxaeKpFAFFVLY4vFiX/HpYqF4oniy+EyxvzhQHCyeLhaLQ8VSsVy0i06RF73C"
        "FWVxvPhccbL4fHG6eLZYK54rzhZfKs4XXy6eL75SrBdfLV4ovlZcKL5evFh8o7hYfLN4qfhWcan4dvFy8Z1io/hu8UrxveJy8f3i"
        "1eIHxZXih8VrxY+Kq8WPi9eLnxSbxU+LN4qfFdeKnxdvFr8orhe/LN4qflXcKH5dvF38ptgqflu8U/yuuFn8vni3+ENxq/hj8V7x"
        "p+J28efi/eIvRaucLreX95Qz5Wx5X3l/OVfuKHeWD5S7ygfLh8qHy/lyd/lo+clyT7m3fKz8VJmUqqRSSls+Xu4rP10ulE+UT5af"
        "KfeXB8qD5dPlYnmoXCqXy3bZKfOyV7qyLI+XnytPlp8vT5fPlmvlc+XZ8kvl+fLL5fPlV8r18qvlC+XXygvl18sXy2+UF8tvli+V"
        "3yovld8uXy6/U26U3y1fKb9XXi6/X75a/qC8Uv6wfK38UXm1/HH5evmTcrP8aflG+bPyWvnz8s3yF+X18pflW+Wvyhvlr8u3y9+U"
        "W+Vvy3fK35U3y9+X75Z/KG+VfyzfK/9U3i7/XL5f/qVsHZs+tv3YPcdmju3965npQaWcHhRR/wuYo8tHHvaXMrqc3uYvzeiy5Ser"
        "ZHT5sJ+s0mCyUncmT/nLO5YffsRfhpZVaFmHlnVoWasgZm2DmHUWTKYksEw6sEwUTuYgZrJBzBRa5tAyh5Y5tMwcxCxpELOoYLLo"
        "wLJIYFlMONkGMZs0iNmElk1o2YSWTWjZ2CBmS0HMloPJVgLLNgssZ0kwOUuDmDMKYs5Cy1loOcv23n/HMn7vbXQ9mI1ffhtdD6LG"
        "b6iNrh+p5tto/h17LcxPVWg/1eH8lO7Mn8L1HfuD4PG7ZdH8yL6K7KvIvqIwfp2E8es0nK9VaF9zaF9LNN+E8VMSxk+RfYrsU2Sf"
        "IvtkwvhZh/EzhfOZQ/tsQ/uchfMlCeMXHcYvkX2J7EtkXyL7JgnjNxLGb0w034b2bRratyqcb3UYv5UwfhvZt5H9LLKfRfYzHcY/"
        "3i+IX0X7RY33C+yrhAL7KuFovgTxq/H+Qfwqjeynkf00sp9G9lMJ4lfj/VLFH+0XNd4vlX1lQvvKRvOzMP7x/qni15F9HdnXkX0d"
        "2ddZGP94v1TxR/tFjfdLZZ+T0D6H+1GxCuMf758qfo7sc2RfIvsS2RcVxj/eL1X80X5R4/1S2Tc6tG8oms9h/OP9U8VvIvs2sm8j"
        "+zaybzmMf7xfqvij/aLG+6Wyn0loPzPRfBvEr8f7B/HrJLSvk9C+TkL7OjHRfBvEr8f7BfHraL/o8X6p7KdhfdQq3I9apWH84/1T"
        "xa8i+yqyryL7OrKvw/qrdVh/dbRftA7ro6awPmoK96OmsP5qCuuvpsg+RfY5ss+RfQ7rr5aw/upov2gJ66OWsD5qkWh+WH+1Ceuv"
        "NpF9E9k3kX0T2Tdh/dU2rL862i/ahvVR27A+ahvuR52F9VdnYf3VWWQ/i+xnkf0stE9JWH8pCesvRfuFkrA+UhrWR0rD/UhpWH8p"
        "DesvpZH9NLKvIvsqsq/C+ksqrL8U7RfSYX0kHdZH0hzND+sv6bD+EkX2KbJPkX2K7FNYf4nD+kvRfiEO6yNxWB+JbTQ/rL8kYf0l"
        "iexLZF8i+xLZl7D+kgnrL0X7hUxYH8mG9ZFsuB/JhvWXbFh/yUb2bWQ/i+xnkf0srL+UhfWXov3CSVgfOQnrIycUzQ/rLydh/eUk"
        "sp9G9tPIfhrZT8P6yyqsvxztF1ZhfWQV1kdWJpof1l/WYf1lHdnXkX0d2deRfR3WX6aw/nK0X5jC+sgU1kfmcD8yh/WXOay/zJF9"
        "juxzZF8i+xLWX5aw/nK0X1jC+sgmrI9swv3IJqy/bML6yyaybyL7NrJvI/s2rL+chfWXo/3CWVgfOQvrI2cSzQ/rryRh/ZUktC9J"
        "aF+S0L4kEs0P66+kYf2VaL9IGtZHScP6KGm4H0WF9VdUWH9FRfZVZF9F9lVkX4f1V3RYfyXaL6LD+igU1kehcD8KhfVXKKy/QpF9"
        "iuxzZJ8j+xzWX+Gw/kq0X0TC+igS1kcRjuaH9VckrL9iIvsmsm8i+yayb8L6KzasvxLtF7FhfRQb1kexNpof1l/JwvorWWQ/i+xn"
        "kf0ssj+oT3+D620v3OsdmEEBGg5MvXAvBrygh+7s9BP4xtTochsu9egSd6cUmRvwtbmZ7YOB7VNTiMAMdsjOWm8YXf4Dg2g7j4z+"
        "T64emB1M2DE3Oz0zNfg3O/j3sP9Xzs8Oe+WYMXv3jCe2z7bmZv8PKhc9+zRrAAA="
    ),
    379: (  # fp16
        "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"
    ),
}


def _raw(t):
    return gzip.decompress(base64.b64decode(_MODELS[t]))


def _session(t):
    s = _SESS.get(t)
    if s is None:
        so = _ort.SessionOptions()
        so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.log_severity_level = 3
        s = _ort.InferenceSession(_raw(t), so)
        _SESS[t] = s
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
            continue  # grader ignores >30x30 grids
        data.append((_onehot(gi), _onehot(go)))
    if not data:
        return []
    out = []
    for t in _MODELS:
        try:
            sess = _session(t)
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
                out.append((f"fp16_{t}", onnx.load_from_string(_raw(t))))
            except Exception:
                pass
    return out
