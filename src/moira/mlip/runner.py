# src/moira/mlip/runner.py

import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from moira.mlip.registry import (
    get_catbench_source_path,
    get_model_adapter_function,
    get_model_adapter_module,
    load_config,
    get_model_python,
)


@contextmanager
def _catbench_source_on_syspath(config_path: str):
    catbench_source = get_catbench_source_path(config_path)
    if catbench_source is None:
        yield
        return

    if not catbench_source.exists():
        raise FileNotFoundError(
            f"Configured CatBench source path does not exist: {catbench_source}"
        )
    if not (catbench_source / "catbench").exists():
        raise FileNotFoundError(
            "Configured CatBench source path must contain a 'catbench' package "
            f"directory: {catbench_source}"
        )

    catbench_source_str = str(catbench_source)
    already_present = catbench_source_str in sys.path
    if not already_present:
        sys.path.insert(0, catbench_source_str)
    try:
        yield
    finally:
        if not already_present:
            sys.path.remove(catbench_source_str)


def _load_adapter_callable(model: str, config_path: str):
    module_name = get_model_adapter_module(model, config_path)
    function_name = get_model_adapter_function(model, config_path)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _project_src_path() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_device(config_path: str) -> str:
    config = load_config(config_path)
    device = str(config.get("mlip", {}).get("device", "cuda")).strip().lower()
    if device != "cuda":
        return device

    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def _maybe_reexec_with_model_python(model: str, line: str, config_path: str) -> str:
    resolved_config_path = str(Path(config_path).resolve())
    target_python = get_model_python(model, resolved_config_path)
    if target_python is None:
        return resolved_config_path

    target_python_path = Path(target_python).absolute()
    if not target_python_path.exists():
        raise FileNotFoundError(
            f"Configured Python for model '{model}' does not exist: {target_python_path}"
        )

    current_python = Path(sys.executable).absolute()
    current_marker = os.environ.get("MOIRA_ACTIVE_MODEL_PYTHON")
    if current_python == target_python_path or current_marker == str(target_python_path):
        return resolved_config_path

    env = os.environ.copy()
    env["MOIRA_ACTIVE_MODEL_PYTHON"] = str(target_python_path)
    project_src = str(_project_src_path())
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([project_src, existing_pythonpath])
    else:
        env["PYTHONPATH"] = project_src
    os.execve(
        str(target_python_path),
        [
            str(target_python_path),
            "-m",
            "moira.mlip",
            "run-one",
            "--line",
            line,
            "--config",
            resolved_config_path,
        ],
        env,
    )
    raise AssertionError("os.execve returned unexpectedly")


def _parse_task_record(line: str) -> dict[str, str]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        parts = line.split()
        if len(parts) == 4:
            model, dataset_name, input_path, _task_work_path = parts
        elif len(parts) == 3:
            model, input_path, _task_work_path = parts
            dataset_name = Path(input_path).stem
        elif len(parts) == 2:
            model, dataset_name = parts
        else:
            raise ValueError(
                "Task line must be JSON or have 2, 3, or 4 fields: "
                "<model> <dataset_name>, "
                "<model> <input_path> <task_work_path> (legacy), or "
                "<model> <dataset_name> <input_path> <task_work_path>"
            ) from None
        return {
            "model": model,
            "dataset_name": dataset_name,
        }

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected task payload to be a JSON object, got {type(payload).__name__}"
        )

    required = {"model", "dataset_name"}
    missing = required.difference(payload)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Task payload missing required fields: {missing_str}")
    return {key: str(payload[key]) for key in required}


def run_one_task(line: str, config_path: str):
    task = _parse_task_record(line)
    model = task["model"]
    dataset_name = task["dataset_name"]
    resolved_config_path = _maybe_reexec_with_model_python(model, line, config_path)
    device = _resolve_device(resolved_config_path)

    print(f"Running adapter: {model} ({dataset_name})")
    with _catbench_source_on_syspath(resolved_config_path):
        run_adapter = _load_adapter_callable(model, resolved_config_path)
        run_adapter(
            model=model,
            dataset_name=dataset_name,
            device=device,
            config_path=resolved_config_path,
        )
