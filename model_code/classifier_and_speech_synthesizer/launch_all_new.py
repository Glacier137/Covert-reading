from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
from paths import ELECTRODE_LIST_PATH


if not ELECTRODE_LIST_PATH.is_file():
    raise FileNotFoundError(f"Electrode list file not found: {ELECTRODE_LIST_PATH}")

subjects = [45, 47, 48, 50, 54, 71, 73, 76, 78]
percents = ["20%", "40%", "60%", "80%"]

for band in ("hg", "b1"):
    for state in ("SI", "SA"):
        conditions = ["ECoG_overt"] if state == "SA" else ["ECoG_covert"]
        for subject in subjects:
            for condition in conditions:
                for percent in percents:
                    command = [
                        sys.executable,
                        str(SCRIPT_DIR / "train_model_new.py"),
                        "--band",
                        band,
                        "--HS",
                        str(subject),
                        "--condition",
                        condition,
                        "--percent",
                        percent,
                    ]
                    started = time.time()
                    print("Launching:", " ".join(command), flush=True)
                    subprocess.run(command, cwd=SCRIPT_DIR, check=True)
                    runtime = int(time.time() - started)
                    print(
                        f"Finished {band}-{subject}-{condition}-{percent} "
                        f"in {runtime // 60} min {runtime % 60} seconds",
                        flush=True,
                    )
