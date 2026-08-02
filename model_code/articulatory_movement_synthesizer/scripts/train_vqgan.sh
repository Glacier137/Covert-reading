#!/usr/bin/env bash
set -uo pipefail

# Batch launcher for the standard fold-indexed VQGAN experiment.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPOSITORY_ROOT="$(cd -- "$PROJECT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${COVERT_READING_LOG_DIR:-$REPOSITORY_ROOT/workspace/model_results/movement_logs}"
read -r -a GPU_IDS <<< "${COVERT_READING_GPU_IDS:-0}"

READING_NAMES=("overt" "covert")
ELEC_TYPES=("downsample_overt_sig" "downsample_covert_sig")

mkdir -p "$LOG_DIR"
TASKS=()
for reading_name in "${READING_NAMES[@]}"; do
    for elec_type in "${ELEC_TYPES[@]}"; do
        if [[ "$reading_name" == "covert" && "$elec_type" == "downsample_overt_sig" ]]; then
            continue
        fi
        if [[ "$reading_name" == "overt" && "$elec_type" == "downsample_covert_sig" ]]; then
            continue
        fi
        TASKS+=("$reading_name|$elec_type")
    done
done

declare -A PIDS
for index in "${!TASKS[@]}"; do
    IFS='|' read -r reading_name elec_type <<< "${TASKS[$index]}"
    gpu_id="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
    log_file="$LOG_DIR/${reading_name}_${elec_type}_gpu${gpu_id}_high_gamma.log"

    echo "Starting $reading_name / $elec_type on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$PROJECT_DIR/models/training_vqgan.py" \
        --reading_name "$reading_name" \
        --elec_type "$elec_type" \
        --epochs 80 \
        --band "high gamma" \
        --device "cuda" > "$log_file" 2>&1 &
    PIDS["$!"]="$gpu_id"
done

status=0
for pid in "${!PIDS[@]}"; do
    if wait "$pid"; then
        echo "Completed PID $pid on GPU ${PIDS[$pid]}"
    else
        echo "Failed PID $pid on GPU ${PIDS[$pid]}" >&2
        status=1
    fi
done
exit "$status"
