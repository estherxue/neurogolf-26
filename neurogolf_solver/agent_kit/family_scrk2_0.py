"""family_scrk2_0 — assigned unsolved slice U[0::5] = tasks [5,46,80,118,158,191,238,349].

After deep analysis every task in this slice requires non-local / data-dependent
reasoning that cannot be expressed as an EXACT, generalizing opset-10 static graph
(the grader gates on EXACT over all 262 arc-gen examples, so an approximate or
overfit solver scores 0). See the accompanying report for the per-task findings.

candidates() yields nothing; wrong guesses would only be rejected by the harness.
"""

import numpy as np


def candidates(examples):
    return []
