#!/usr/bin/env bash

set -euo pipefail

PARTITION="${PARTITION:-gpuA40x4-interactive}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-8G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
SHELL_BIN="${SHELL_BIN:-bash}"
DEFAULT_QOS="${DEFAULT_QOS:-your-qos}"

usage() {
    cat <<'EOF'
Usage: ./scripts/start_gpuA40x4_interactive.sh [--account ACCOUNT] [--qos QOS] [--] [extra srun args]

Environment overrides:
  PARTITION       Slurm partition to use. Default: gpuA40x4-interactive
  GPUS            GPU count. Default: 1
  CPUS_PER_TASK   CPU count. Default: 4
  MEM             Memory request. Default: 8G
  TIME_LIMIT      Wall time. Default: 00:30:00
  SHELL_BIN       Shell started by srun. Default: bash
  DEFAULT_QOS     Default QoS if --qos is not passed. Default: your-qos
EOF
}

if ! command -v srun >/dev/null 2>&1; then
    echo "error: srun was not found in PATH." >&2
    exit 1
fi

account_args=()
qos_args=(--qos "$DEFAULT_QOS")
extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)
            if [[ $# -lt 2 ]]; then
                echo "error: --account requires a value" >&2
                exit 1
            fi
            account_args=(--account "$2")
            shift 2
            ;;
        --qos)
            if [[ $# -lt 2 ]]; then
                echo "error: --qos requires a value" >&2
                exit 1
            fi
            qos_args=(--qos "$2")
            shift 2
            ;;
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
echo "  gpus:      $GPUS"
echo "  cpus:      $CPUS_PER_TASK"
echo "  mem:       $MEM"
echo "  time:      $TIME_LIMIT"
echo "  qos:       ${qos_args[1]}"

exec srun \
    --partition "$PARTITION" \
    --gres "gpu:${GPUS}" \
    --cpus-per-task "$CPUS_PER_TASK" \
    --mem "$MEM" \
    --time "$TIME_LIMIT" \
    "${account_args[@]}" \
    "${qos_args[@]}" \
    "${extra_args[@]}" \
    --pty "$SHELL_BIN" -l
