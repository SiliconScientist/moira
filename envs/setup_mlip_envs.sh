#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
ORB_V3_PYTHON_VERSION="${ORB_V3_PYTHON_VERSION:-3.12}"
UV_BIN="${UV_BIN:-uv}"
LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/moira-mlip-envs.XXXXXX")"
FORCE_REBUILD=0
GRACE_GIT_REF="${GRACE_GIT_REF:-ce505520e28b15daf3984c40bcf0992b148ce58f}"

usage() {
    cat <<'EOF'
Usage: ./setup_mlip_envs.sh [--force]

Options:
  --force    Remove and recreate existing virtual environments.

Environment:
  PYTHON_VERSION          Default Python version for MLIP environments.
  ORB_V3_PYTHON_VERSION   Python version for envs/orb_v3.
  GRACE_GIT_REF           Commit/tag to install for envs/grace.
EOF
}

cleanup() {
    rm -rf "$LOG_DIR"
}

trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE_REBUILD=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
    echo "error: uv executable '$UV_BIN' was not found in PATH." >&2
    echo "Set UV_BIN=/path/to/uv and rerun." >&2
    exit 1
fi

shopt -s nullglob

requirement_files=("$ROOT_DIR"/*/requirements.txt)

if [[ "${#requirement_files[@]}" -eq 0 ]]; then
    echo "error: no MLIP requirements files found under $ROOT_DIR." >&2
    exit 1
fi

create_env() {
    local model_dir="$1"
    local python_version="$2"
    local venv_dir="$model_dir/.venv"
    local ready_file="$venv_dir/.moira-ready"

    if [[ -d "$venv_dir" && -f "$ready_file" && "$FORCE_REBUILD" -eq 0 ]]; then
        return 1
    fi

    rm -rf "$venv_dir"
    "$UV_BIN" venv --python "$python_version" "$venv_dir"
}

install_requirements() {
    local model_dir="$1"
    local req_file="$model_dir/requirements.txt"
    local venv_python="$model_dir/.venv/bin/python"
    local ready_file="$model_dir/.venv/.moira-ready"

    "$UV_BIN" pip install --python "$venv_python" -r "$req_file"

    if [[ "$(basename "$model_dir")" == "grace" ]]; then
        "$UV_BIN" pip install --python "$venv_python" --no-deps \
            "git+https://github.com/ICAMS/grace-tensorpotential.git@${GRACE_GIT_REF}"
    fi

    touch "$ready_file"
}

models=()
logs=()
pids=()

for req_file in "${requirement_files[@]}"; do
    model_dir="$(dirname "$req_file")"
    model_name="$(basename "$model_dir")"
    python_version="$PYTHON_VERSION"

    if [[ "$model_name" == "orb_v3" ]]; then
        python_version="$ORB_V3_PYTHON_VERSION"
    fi

    if create_env "$model_dir" "$python_version"; then
        echo "==> Creating $model_name environment (Python $python_version)"
    else
        echo "==> Skipping $model_name environment (already exists)"
        continue
    fi

    log_file="$LOG_DIR/$model_name.log"
    models+=("$model_name")
    logs+=("$log_file")

    (
        echo "==> Installing $model_name dependencies"
        install_requirements "$model_dir"
    ) >"$log_file" 2>&1 &

    pids+=("$!")
done

failed=0

for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "==> ${models[$i]} dependencies installed"
    else
        echo "error: ${models[$i]} dependency install failed" >&2
        sed -n '1,200p' "${logs[$i]}" >&2
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    exit 1
fi

echo
echo "All MLIP environments created."
