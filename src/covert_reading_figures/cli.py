from __future__ import annotations

import argparse
from pathlib import Path

from .notebook_runner import NOTEBOOK_ORDER, reproduce


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    project_root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Reproduce Covert Reading figures from workspace/Source_Data.xlsx."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "reproduction_run",
        help="New or empty output directory. Defaults to results/reproduction_run.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=NOTEBOOK_ORDER,
        help="Optional normalized notebook names to run.",
    )
    args = parser.parse_args()
    reproduce(project_root=project_root, run_dir=args.output_dir, only=args.only)


if __name__ == "__main__":
    main()
