from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
from paths import ELECTRODE_LIST_PATH


if not ELECTRODE_LIST_PATH.is_file():
    raise FileNotFoundError(f"Electrode list file not found: {ELECTRODE_LIST_PATH}")

SUBJECTS = [45, 47, 48, 50, 54, 71, 73, 76, 78]

for band in ("hg", "b1"):
    for state in ("SI", "SA"):
        conditions = ["ECoG_overt"] if state == "SA" else ["ECoG_covert"]
        for subject in SUBJECTS:
            for condition in conditions:
                command = [
                    sys.executable,
                    str(SCRIPT_DIR / "train_model_sig_half.py"),
                    "--band",
                    band,
                    "--HS",
                    str(subject),
                    "--condition",
                    condition,
                    "--percent",
                    "100%",
                ]
                started = time.time()
                print("Launching:", " ".join(command), flush=True)
                subprocess.run(command, cwd=SCRIPT_DIR, check=True)
                runtime = int(time.time() - started)
                print(
                    f"Finished {band}-{subject}-{condition}-100% "
                    f"in {runtime // 60} min {runtime % 60} seconds",
                    flush=True,
                )
