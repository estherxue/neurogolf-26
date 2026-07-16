"""Locate and import the OFFICIAL `neurogolf_utils.py` so our local scoring is
byte-identical to the competition's. Set NG_UTILS_DIR / NG_DATA_DIR to override.

Layout we expect (downloaded via the Kaggle API):
    $NG_DATA_DIR/
        neurogolf_utils.py            (or neurogolf_utils/neurogolf_utils.py)
        tasks/task001.json ... task199.json
"""
import os
import sys
import pathlib

_DEFAULT_DATA = (
    "/private/tmp/claude-501/"
    "-Users-xingyuanxue1122-Documents-coding-neurogolf-26--claude-worktrees-kaggle-agent-harness/"
    "f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad/ng_data"
)

NG_DATA_DIR = pathlib.Path(os.environ.get("NG_DATA_DIR", _DEFAULT_DATA))


def _find_utils_dir():
    env = os.environ.get("NG_UTILS_DIR")
    cands = []
    if env:
        cands.append(pathlib.Path(env))
    cands += [
        NG_DATA_DIR,
        NG_DATA_DIR / "neurogolf_utils",
        NG_DATA_DIR / "tasks",
        NG_DATA_DIR / "tasks" / "neurogolf_utils",
    ]
    for d in cands:
        if (d / "neurogolf_utils.py").is_file():
            return d
    raise FileNotFoundError(
        "neurogolf_utils.py not found. Set NG_UTILS_DIR or NG_DATA_DIR. Tried: "
        + ", ".join(str(c) for c in cands)
    )


def tasks_dir():
    """Directory containing taskNNN.json files."""
    for d in [NG_DATA_DIR / "tasks", NG_DATA_DIR, NG_DATA_DIR / "tasks" / "tasks"]:
        if list(d.glob("task*.json")):
            return d
    raise FileNotFoundError(f"No task*.json under {NG_DATA_DIR}")


_utils_dir = _find_utils_dir()
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))

import neurogolf_utils as ng  # noqa: E402  (official module — scoring ground truth)

# Re-export the pieces we depend on.
GRID_SHAPE = ng._GRID_SHAPE
DATA_TYPE = ng._DATA_TYPE
IR_VERSION = ng._IR_VERSION
OPSET_IMPORTS = ng._OPSET_IMPORTS
CHANNELS = ng._CHANNELS
HEIGHT = ng._HEIGHT
WIDTH = ng._WIDTH
EXCLUDED_OP_TYPES = ng._EXCLUDED_OP_TYPES
FILESIZE_LIMIT = ng._FILESIZE_LIMIT_IN_BYTES
