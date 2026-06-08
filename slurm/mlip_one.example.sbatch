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
CONFIG="${2:-mlip.toml}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$ROOT_DIR}"

if [[ "$TASKFILE" != /* ]]; then
    TASKFILE="$SUBMIT_DIR/$TASKFILE"
fi

if [[ "$CONFIG" != /* ]]; then
    CONFIG="$SUBMIT_DIR/$CONFIG"
fi

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
echo "MODEL=$MODEL"
echo "MODEL_PYTHON=$MODEL_PYTHON"
ls -l "$MODEL_PYTHON" || true
readlink -f "$MODEL_PYTHON" || true
if [[ -x "$MODEL_PYTHON" ]]; then
    echo "MODEL_PYTHON_EXECUTABLE=yes"
else
    echo "MODEL_PYTHON_EXECUTABLE=no"
fi
if [[ -x "$MODEL_PYTHON" ]]; then
    PYTHON_EXEC="$MODEL_PYTHON"
else
    PYTHON_EXEC="${PYTHON_BIN:-python}"
fi

cd "$ROOT_DIR"

echo "=== runtime ==="
date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PYTHON_EXEC=$PYTHON_EXEC"
echo

echo "=== nvidia-smi ==="
nvidia-smi || true
echo

echo "=== modules ==="
module list 2>&1 || true
echo

echo "=== env ==="
env | grep -E '^(SLURM|CUDA|NVIDIA|TORCH)_' | sort || true
echo

echo "=== torch diag ==="
"$PYTHON_EXEC" - <<'PY'
import sys

print("python:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda.is_available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"device[{i}]:", torch.cuda.get_device_name(i))
    else:
        print("cuda unavailable")
except Exception as exc:
    print("torch diag failed:", repr(exc))
PY
echo

"$PYTHON_EXEC" -m moira.mlip run-one --line "$LINE" --config "$CONFIG"
