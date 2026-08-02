from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import random
import re
import shutil
import time
import traceback
from pathlib import Path


NOTEBOOK_FILES = {
    "00_fig1_extended_data1": "00_fig1_extended_data1.ipynb",
    "01_extended_data2_beta1_erp": "01_extended_data2_beta1_erp.ipynb",
    "02_extended_data3_high_gamma_erp": "02_extended_data3_high_gamma_erp.ipynb",
    "03_fig2_fig3_beta1_spatial": "03_fig2_fig3_beta1_spatial.ipynb",
    "04_extended_data3_to6_high_gamma_spatial": "04_extended_data3_to6_high_gamma_spatial.ipynb",
    "05_fig2_fig4_beta1_single_effector": "05_fig2_fig4_beta1_single_effector.ipynb",
    "06_extended_data3_high_gamma_single_effector": "06_extended_data3_high_gamma_single_effector.ipynb",
    "07_fig3_cross_correlation_beta1": "07_fig3_cross_correlation_beta1.ipynb",
    "08_fig5_akt_decoding_source_data": "08_fig5_akt_decoding_source_data.ipynb",
    "09_fig5_accuracy_mcd_source_data": "09_fig5_accuracy_mcd_source_data.ipynb",
    "10_extended_data8_mcd_mos": "10_extended_data8_mcd_mos.ipynb",
    "11_fig5_spectrogram_source_data": "11_fig5_spectrogram_source_data.ipynb",
}

NOTEBOOK_ORDER = list(NOTEBOOK_FILES)

REQUIRED_FILES = [
    "Source_Data.xlsx",
    "elecs/MNI.png",
    "elecs/Brain2D/HS44_brain2D.png",
    "elecs/Brain2D/HS45_brain2D.png",
    "elecs/Brain2D/HS47_brain2D.png",
    "elecs/Brain2D/HS48_brain2D.png",
    "elecs/Brain2D/HS50_brain2D.png",
    "elecs/Brain2D/HS54_brain2D.png",
    "elecs/Brain2D/HS71_brain2D.png",
    "elecs/Brain2D/HS73_brain2D.png",
    "elecs/Brain2D/HS76_brain2D.png",
    "elecs/Brain2D/HS78_brain2D.png",
]

REQUIRED_DIRECTORIES = ["elecs/Brain2D"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_package(project_root: Path, run_dir: Path) -> tuple[Path, dict[str, Path]]:
    project_root = project_root.resolve()
    run_dir = run_dir.resolve()
    workspace = project_root / "workspace"
    notebooks_dir = project_root / "notebooks"

    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be new or empty: {run_dir}. "
            "Choose another directory or remove the previous generated run."
        )

    missing = [str(workspace / relative) for relative in REQUIRED_FILES if not (workspace / relative).is_file()]
    missing.extend(
        str(workspace / relative)
        for relative in REQUIRED_DIRECTORIES
        if not (workspace / relative).is_dir()
    )
    notebooks = {slug: notebooks_dir / filename for slug, filename in NOTEBOOK_FILES.items()}
    missing.extend(str(path) for path in notebooks.values() if not path.is_file())
    if missing:
        formatted = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Required package inputs are missing:\n{formatted}")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reproduced").mkdir()
    (run_dir / "logs").mkdir()
    (workspace / "figures").mkdir(exist_ok=True)
    (workspace / "figures_R3").mkdir(exist_ok=True)

    inventory = {
        "project_root": ".",
        "workspace": "workspace",
        "source_data": {
            "relative_path": "workspace/Source_Data.xlsx",
            "sha256": sha256(workspace / "Source_Data.xlsx"),
        },
        "notebooks": {
            slug: {
                "relative_path": path.relative_to(project_root).as_posix(),
                "sha256": sha256(path),
            }
            for slug, path in notebooks.items()
        },
        "required_files": [str(Path("workspace") / item) for item in REQUIRED_FILES],
        "scope": "Source Data figure redraw only; model training and inference are intentionally excluded.",
    }
    (run_dir / "input_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return workspace, notebooks


def clean_cell_source(source: str) -> str:
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
    )


def skip_reason(source: str) -> str | None:
    lowered = source.lower()
    if ".to_excel(" in lowered or "excelwriter(" in lowered:
        return "would modify the read-only Source_Data workbook"
    return None


def snapshot_outputs(workspace: Path) -> dict[str, tuple[int, int]]:
    result = {}
    for folder in (workspace / "figures_R3", workspace / "figures"):
        for path in folder.rglob("*"):
            if path.is_file():
                stat = path.stat()
                result[str(path)] = (stat.st_mtime_ns, stat.st_size)
    return result


def execute_notebook(
    slug: str,
    notebook_path: Path,
    project_root: Path,
    workspace: Path,
    output_dir: Path,
    log_dir: Path,
) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    plt.rcdefaults()
    os.chdir(project_root)
    random.seed(0)
    np.random.seed(0)

    excel_file = pd.ExcelFile(workspace / "Source_Data.xlsx", engine="openpyxl")
    original_read_excel = pd.read_excel

    def cached_read_excel(io_value, *args, **kwargs):
        try:
            same_source = Path(str(io_value)).name.lower() == "source_data.xlsx"
        except Exception:
            same_source = False
        if same_source:
            return original_read_excel(excel_file, *args, **kwargs)
        return original_read_excel(io_value, *args, **kwargs)

    pd.read_excel = cached_read_excel
    namespace = {
        "__name__": "__main__",
        "__file__": str(notebook_path),
        "pd": pd,
        "np": np,
        "plt": plt,
    }
    cell_logs = []

    try:
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = clean_cell_source("".join(cell.get("source", [])))
            reason = skip_reason(source)
            if reason:
                cell_logs.append({"cell_index": cell_index, "status": "SKIPPED", "reason": reason})
                continue
            if not source.strip():
                cell_logs.append({"cell_index": cell_index, "status": "EMPTY"})
                continue

            show_counter = {"value": 0}

            def capture_show(*_args, **_kwargs):
                for figure_number in list(plt.get_fignums()):
                    show_counter["value"] += 1
                    target = output_dir / f"cell_{cell_index:03d}_show_{show_counter['value']:02d}.png"
                    plt.figure(figure_number).savefig(target, dpi=300, bbox_inches="tight")
                plt.close("all")

            plt.show = capture_show
            before = snapshot_outputs(workspace)
            stdout, stderr = io.StringIO(), io.StringIO()
            started = time.perf_counter()
            status, error = "PASS", None
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    exec(compile(source, f"{slug}:cell{cell_index}", "exec"), namespace, namespace)
                except Exception as exc:
                    status = "ERROR"
                    error = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=12),
                    }
                finally:
                    if plt.get_fignums():
                        capture_show()

            copied = []
            after = snapshot_outputs(workspace)
            for raw, metadata in sorted(after.items()):
                if raw not in before or before[raw] != metadata:
                    source_path = Path(raw)
                    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_path.name)
                    target = output_dir / f"cell_{cell_index:03d}_{safe_name}"
                    shutil.copy2(source_path, target)
                    copied.append(
                        {
                            "target": target.relative_to(log_dir.parent).as_posix(),
                            "sha256": sha256(target),
                        }
                    )

            cell_logs.append(
                {
                    "cell_index": cell_index,
                    "status": status,
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "stdout": stdout.getvalue()[-12000:],
                    "stderr": stderr.getvalue()[-12000:],
                    "error": error,
                    "captured_show_count": show_counter["value"],
                    "copied_outputs": copied,
                }
            )
    finally:
        excel_file.close()
        pd.read_excel = original_read_excel

    record = {
        "slug": slug,
        "notebook": notebook_path.relative_to(project_root).as_posix(),
        "cells": cell_logs,
        "summary": {
            "pass": sum(item["status"] == "PASS" for item in cell_logs),
            "error": sum(item["status"] == "ERROR" for item in cell_logs),
            "skipped": sum(item["status"] == "SKIPPED" for item in cell_logs),
            "outputs": len(list(output_dir.glob("*"))),
        },
    }
    (log_dir / f"{slug}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return record


def reproduce(
    project_root: Path,
    run_dir: Path,
    only: list[str] | None = None,
) -> dict:
    project_root = Path(project_root).resolve()
    run_dir = Path(run_dir).resolve()
    workspace, notebooks = validate_package(project_root, run_dir)
    selected = [slug for slug in NOTEBOOK_ORDER if not only or slug in only]
    unknown = sorted(set(only or []) - set(NOTEBOOK_ORDER))
    if unknown:
        raise ValueError(f"Unknown notebook names: {unknown}")

    results = []
    started = time.perf_counter()
    for slug in selected:
        print(f"[{slug}] running", flush=True)
        record = execute_notebook(
            slug,
            notebooks[slug],
            project_root,
            workspace,
            run_dir / "reproduced" / slug,
            run_dir / "logs",
        )
        results.append(record)
        print(f"[{slug}] {record['summary']}", flush=True)

    summary = {
        "scope": "Source Data figure redraw; no model training or inference",
        "executed": {item["slug"]: item["summary"] for item in results},
        "unexpected_errors": sum(item["summary"]["error"] for item in results),
        "source_data_sha256": sha256(workspace / "Source_Data.xlsx"),
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary
