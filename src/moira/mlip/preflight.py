from __future__ import annotations

import subprocess
from pathlib import Path

from moira.mlip.registry import get_model_specs


LEGACY_MODEL_IMPORTS = {
    "allegro": "allegro",
    "chgnet": "chgnet",
    "aqcat25": "fairchem",
    "alphanet": "alphanet",
    "grace": "tensorpotential",
    "mace": "mace",
    "mattersim": "mattersim",
    "orb_v3": "orb_models",
    "sevennet": "sevenn",
    "uma": "fairchem",
}


def _validate_python_exists(model: str, python: str) -> None:
    python_path = Path(python).resolve()
    if not python_path.exists():
        raise FileNotFoundError(
            f"Python for model '{model}' does not exist: {python_path}"
        )


def _validate_import(model: str, python: str, module_name: str) -> None:
    result = subprocess.run(
        [python, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return

    stderr = result.stderr.strip() or result.stdout.strip() or "unknown import failure"
    raise RuntimeError(
        f"Model '{model}' failed preflight import in {python}: import {module_name!r} "
        f"failed with: {stderr}"
    )


def validate_model_envs(
    config_path: str | Path,
    *,
    show_progress: bool = False,
) -> None:
    config_path = Path(config_path).resolve()
    specs = get_model_specs(config_path)

    checks = [(model, spec) for model, spec in specs.items() if spec.python is not None]

    for index, (model, spec) in enumerate(checks, start=1):
        assert spec.python is not None
        if show_progress:
            print(f"Preflight [{index}/{len(checks)}]: checking {model}")

        if spec.python is None:
            continue

        _validate_python_exists(model, spec.python)

        module_name = LEGACY_MODEL_IMPORTS.get(model)
        if module_name is not None:
            _validate_import(model, spec.python, module_name)
