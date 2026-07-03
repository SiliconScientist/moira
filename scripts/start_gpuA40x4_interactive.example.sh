#!/usr/bin/env bash

set -euo pipefail

PARTITION="${PARTITION:-gpuA40x4-interactive}"
ACCOUNT="${ACCOUNT:-your-account}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-8G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
SHELL_BIN="${SHELL_BIN:-bash}"
QOS="${QOS:-your-qos}"

usage() {
    cat <<'EOF'
Usage: ./scripts/start_gpuA40x4_interactive.sh [--] [extra srun args]

Environment overrides:
  PARTITION       Slurm partition to use. Default: gpuA40x4-interactive
  ACCOUNT         Slurm account to charge. Set this in the file. Default: your-account
  QOS             Slurm QoS to use. Set this in the file. Default: your-qos
  GPUS            GPU count. Default: 1
  CPUS_PER_TASK   CPU count. Default: 4
  MEM             Memory request. Default: 8G
  TIME_LIMIT      Wall time. Default: 00:30:00
  SHELL_BIN       Shell started by srun. Default: bash
EOF
}

if ! command -v srun >/dev/null 2>&1; then
    echo "error: srun was not found in PATH." >&2
    exit 1
fi

extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            extra_args=("$@")
            break
            ;;
        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done

echo "Starting interactive Slurm session:"
echo "  partition: $PARTITION"
echo "  account:   $ACCOUNT"
echo "  gpus:      $GPUS"
echo "  cpus:      $CPUS_PER_TASK"
echo "  mem:       $MEM"
echo "  time:      $TIME_LIMIT"
echo "  qos:       $QOS"

exec srun \
    --partition "$PARTITION" \
    --account "$ACCOUNT" \
    --qos "$QOS" \
    --gres "gpu:${GPUS}" \
    --cpus-per-task "$CPUS_PER_TASK" \
    --mem "$MEM" \
    --time "$TIME_LIMIT" \
    "${extra_args[@]}" \
    --pty "$SHELL_BIN" -l
