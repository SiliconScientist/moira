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

TASKFILE="$1"
CONFIG="${2:-config.toml}"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKFILE")"
echo "TASK: $LINE"

# Use the dedicated MLIP entrypoint to parse config + dispatch
python -m moira.mlip run-one --line "$LINE" --config "$CONFIG"
