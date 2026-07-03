#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AQCAT_DIR="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/envs/aqcat25/.venv/bin/python}"

usage() {
    cat <<'EOF'
Usage: ./envs/aqcat25/patch_fairchem.sh /path/to/aqcat25-ev2

This copies the gated AQCat25 fairchem patches into the installed aqcat25
virtual environment after you have:
  1. requested access to SandboxAQ/aqcat25-ev2 on Hugging Face
  2. downloaded the repository files locally
  3. created envs/aqcat25/.venv via ./envs/setup_mlip_envs.sh

Required AQCat25 subpaths:
  ev2_film/equiformer_v2_film.py
  patched_code/ase_utils.py
EOF
}

if [[ -z "$AQCAT_DIR" || "$AQCAT_DIR" == "-h" || "$AQCAT_DIR" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "error: aqcat25 python not found: $PYTHON_BIN" >&2
    exit 1
fi

AQCAT_DIR="$(cd "$AQCAT_DIR" && pwd)"
EQV2_SRC="$AQCAT_DIR/ev2_film/equiformer_v2_film.py"
ASE_UTILS_SRC="$AQCAT_DIR/patched_code/ase_utils.py"

if [[ ! -f "$EQV2_SRC" ]]; then
    echo "error: missing AQCat25 file: $EQV2_SRC" >&2
    exit 1
fi

if [[ ! -f "$ASE_UTILS_SRC" ]]; then
    echo "error: missing AQCat25 file: $ASE_UTILS_SRC" >&2
    exit 1
fi

FAIRCHEM_CORE_DIR="$("$PYTHON_BIN" -c '
from pathlib import Path
import fairchem.core

print(Path(fairchem.core.__file__).resolve().parent)
')"

EQV2_DEST="$FAIRCHEM_CORE_DIR/models/equiformer_v2/equiformer_v2_film.py"
ASE_UTILS_DEST="$FAIRCHEM_CORE_DIR/common/relaxation/ase_utils.py"

cp "$EQV2_SRC" "$EQV2_DEST"
cp "$ASE_UTILS_SRC" "$ASE_UTILS_DEST"

echo "Patched fairchem-core in:"
echo "  $FAIRCHEM_CORE_DIR"
