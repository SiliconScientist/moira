# src/moira/mlip/runner.py

import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

from moira.mlip.registry import (
    get_catbench_source_path,
    get_model_adapter_function,
    get_model_adapter_module,
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
        else:
            raise ValueError(
                "Task line must be JSON or have 3 or 4 fields: "
                "<model> <input_path> <task_work_path> (legacy) or "
                "<model> <dataset_name> <input_path> <task_work_path>"
            ) from None
        return {
            "model": model,
            "dataset_name": dataset_name,
            "input_path": input_path,
        }

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected task payload to be a JSON object, got {type(payload).__name__}"
        )

    required = {"model", "dataset_name", "input_path"}
    missing = required.difference(payload)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Task payload missing required fields: {missing_str}")
    return {key: str(payload[key]) for key in required}


def run_one_task(line: str, config_path: str):
    task = _parse_task_record(line)
    model = task["model"]
    dataset_name = task["dataset_name"]
    input_path = task["input_path"]

    print(f"Running adapter: {model} ({dataset_name})")
    with _catbench_source_on_syspath(config_path):
        run_adapter = _load_adapter_callable(model, config_path)
        run_adapter(
            model=model,
            input_path=input_path,
            dataset_name=dataset_name,
            config_path=str(config_path),
        )
