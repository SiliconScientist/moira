#!/bin/bash
#SBATCH --job-name="moira"
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --cpus-per-task=32
#SBATCH --mem=10G
#SBATCH -t 24:00:00
#SBATCH -o slurm_output/%x_%A_%a.out
#SBATCH -e slurm_output/%x_%A_%a.err
#SBATCH --mail-user=<your-email@example.com>
#SBATCH --mail-type=END

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKFILE="$1"
CONFIG="${2:-config.toml}"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKFILE")"
echo "TASK: $LINE"

MODEL="$("${PYTHON_BIN:-python3}" -c '
import json
import sys

line = sys.argv[1].strip()
if not line:
    sys.exit(1)
try:
    payload = json.loads(line)
except json.JSONDecodeError:
    parts = line.split()
    if len(parts) in {2, 3, 4}:
        print(parts[0])
        sys.exit(0)
    sys.exit(1)
if isinstance(payload, dict):
    print(payload.get("model", ""))
' "$LINE")"
if [[ -z "$MODEL" ]]; then
    echo "error: unable to determine model from task line" >&2
    exit 1
fi

MODEL_PYTHON="$ROOT_DIR/envs/$MODEL/.venv/bin/python"
if [[ -x "$MODEL_PYTHON" ]]; then
    PYTHON_EXEC="$MODEL_PYTHON"
else
    PYTHON_EXEC="${PYTHON_BIN:-python}"
fi

cd "$ROOT_DIR"
"$PYTHON_EXEC" -m moira.mlip run-one --line "$LINE" --config "$CONFIG"
