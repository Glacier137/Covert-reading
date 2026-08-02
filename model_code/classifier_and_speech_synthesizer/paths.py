"""Portable paths for the classifier and speech-synthesis release.

All default inputs and outputs live below the repository-level ``workspace``
directory. Set ``COVERT_READING_WORKSPACE`` only when that directory must be
stored elsewhere. More specific environment variables remain available for
cluster or institutional deployments.
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]


def _configured_path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default.resolve()


WORKSPACE_ROOT = _configured_path(
    "COVERT_READING_WORKSPACE",
    REPOSITORY_ROOT / "workspace",
)
DATA_ROOT = _configured_path(
    "COVERT_READING_DATA_ROOT",
    WORKSPACE_ROOT / "model_data",
)
RESULTS_ROOT = _configured_path(
    "COVERT_READING_RESULTS_ROOT",
    WORKSPACE_ROOT / "model_results",
)
SPEECH_CHECKPOINT_ROOT = _configured_path(
    "COVERT_READING_SPEECH_CHECKPOINT_DIR",
    WORKSPACE_ROOT / "speech_checkpoints",
)
ELECTRODE_LIST_PATH = _configured_path(
    "COVERT_READING_ELECTRODE_LIST",
    WORKSPACE_ROOT / "private" / "electrode_lists.json",
)
