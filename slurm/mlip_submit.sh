#!/bin/bash
set -euo pipefail

CONFIG="${1:-config.toml}"
shift || true

RUN_TAG="${1:-$(date +%F)}"
shift || true

mkdir -p slurm_output
python -m moira.mlip --config "$CONFIG" --run-tag "$RUN_TAG" "$@"
