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

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
TASKFILE="$1"
CONFIG="${2:-config.toml}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$ROOT_DIR}"

if [[ "$TASKFILE" != /* ]]; then
    TASKFILE="$SUBMIT_DIR/$TASKFILE"
fi

if [[ "$CONFIG" != /* ]]; then
    CONFIG="$SUBMIT_DIR/$CONFIG"
fi

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKFILE")"
echo "TASK: $LINE"

cd "$ROOT_DIR"
PROJECT_SRC="$ROOT_DIR/src"
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$PROJECT_SRC:$PYTHONPATH"
else
    export PYTHONPATH="$PROJECT_SRC"
fi

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

CONFIGURED_MODEL_PYTHON="$("${PYTHON_BIN:-python3}" -c '
import sys

from moira.mlip.registry import get_model_python

model = sys.argv[1]
config_path = sys.argv[2]
python = get_model_python(model, config_path)
print("" if python is None else python)
' "$MODEL" "$CONFIG")"
MODEL_PYTHON="$ROOT_DIR/envs/$MODEL/.venv/bin/python"
echo "MODEL=$MODEL"
echo "CONFIGURED_MODEL_PYTHON=${CONFIGURED_MODEL_PYTHON:-}"
echo "MODEL_PYTHON=$MODEL_PYTHON"
ls -l "$MODEL_PYTHON" || true
readlink -f "$MODEL_PYTHON" || true
if [[ -x "$MODEL_PYTHON" ]]; then
    echo "MODEL_PYTHON_EXECUTABLE=yes"
else
    echo "MODEL_PYTHON_EXECUTABLE=no"
fi
if [[ -n "$CONFIGURED_MODEL_PYTHON" ]]; then
    PYTHON_EXEC="$CONFIGURED_MODEL_PYTHON"
elif [[ -x "$MODEL_PYTHON" ]]; then
    PYTHON_EXEC="$MODEL_PYTHON"
else
    PYTHON_EXEC="${PYTHON_BIN:-python}"
fi

TF_GPU_LIB_PATHS="$("$PYTHON_EXEC" - <<'PY'
import importlib

modules = [
    "nvidia.cublas.lib",
    "nvidia.cuda_runtime.lib",
    "nvidia.cudnn.lib",
    "nvidia.cufft.lib",
    "nvidia.curand.lib",
    "nvidia.cusolver.lib",
    "nvidia.cusparse.lib",
]
paths = []
for module_name in modules:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        continue
    for path in getattr(module, "__path__", []):
        if path and path not in paths:
            paths.append(path)
print(":".join(paths))
PY
)"
if [[ -n "$TF_GPU_LIB_PATHS" ]]; then
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        export LD_LIBRARY_PATH="$TF_GPU_LIB_PATHS:$LD_LIBRARY_PATH"
    else
        export LD_LIBRARY_PATH="$TF_GPU_LIB_PATHS"
    fi
fi

echo "=== runtime ==="
date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PYTHON_EXEC=$PYTHON_EXEC"
echo "PYTHONPATH=${PYTHONPATH:-}"
echo "TF_GPU_LIB_PATHS=${TF_GPU_LIB_PATHS:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
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

echo "=== framework diag ==="
"$PYTHON_EXEC" - <<'PY'
import sys

print("python:", sys.executable)

try:
    import torch
except ModuleNotFoundError:
    print("torch: not installed in this model environment")
except Exception as exc:
    print("torch diag failed:", repr(exc))
else:
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda.is_available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"device[{i}]:", torch.cuda.get_device_name(i))
    else:
        print("cuda unavailable")

try:
    import tensorflow as tf
except ModuleNotFoundError:
    print("tensorflow: not installed in this model environment")
except Exception as exc:
    print("tensorflow diag failed:", repr(exc))
else:
    print("tensorflow:", tf.__version__)
    try:
        gpu_devices = tf.config.list_physical_devices("GPU")
    except Exception as exc:
        print("tensorflow gpu query failed:", repr(exc))
    else:
        print("tensorflow.gpu_count:", len(gpu_devices))
        if gpu_devices:
            for i, device in enumerate(gpu_devices):
                print(f"tensorflow.device[{i}]:", device)
        else:
            print("tensorflow gpu unavailable")
PY
echo

"$PYTHON_EXEC" -m moira.mlip run-one --line "$LINE" --config "$CONFIG"
