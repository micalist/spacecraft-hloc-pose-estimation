"""Portable project paths shared by the experiment entry points.

Every path can be overridden with an environment variable.  The defaults keep
the original workspace layout while avoiding machine-specific absolute paths.
"""

from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = _path_from_env("POSE_ESTIMATION_ROOT", REPO_ROOT)
SHIRT_ROOT = _path_from_env("SHIRT_ROOT", WORKSPACE / "shirtv1")
PILOT_ROOT = _path_from_env(
    "POSE_PILOT_ROOT", WORKSPACE / "pilot_roe1_synthetic_v1"
)
HLOC_ROOT = _path_from_env(
    "HLOC_ROOT", WORKSPACE / "Hierarchical-Localization"
)
LIGHTGLUE_ROOT = _path_from_env("LIGHTGLUE_ROOT", WORKSPACE / "LightGlue")
TORCH_CACHE = _path_from_env("TORCH_HOME", WORKSPACE / ".torch_cache")
