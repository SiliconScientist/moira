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

MODEL="$(python - <<'PY'
import json, sys
line = sys.stdin.read().strip()
if not line:
    sys.exit(1)
payload = json.loads(line)
print(payload.get('model', ''))
PY
<<<"$LINE")"
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
