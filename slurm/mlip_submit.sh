#!/bin/bash
set -euo pipefail

CONFIG="${1:-config.toml}"
shift || true

RUN_TAG="${1:-$(date +%F)}"
shift || true

mkdir -p slurm_output
TASKFILE="slurm_output/mlip_tasks_${RUN_TAG}.jsonl"

# Generate tasks via the dedicated MLIP entrypoint.
python -m moira.mlip make-tasks --config "$CONFIG" --run-tag "$RUN_TAG" --out "$TASKFILE" "$@"

NTASKS=$(wc -l < "$TASKFILE")
echo "Submitting $NTASKS tasks from $TASKFILE"

sbatch --array=0-$((NTASKS-1)) slurm/mlip_one.sbatch "$TASKFILE" "$CONFIG"
