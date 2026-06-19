# src/moira/mlip/runner.py

import importlib
import json
import os
import sys
from contextlib import ExitStack, contextmanager
from time import perf_counter
from pathlib import Path
from tempfile import TemporaryDirectory

from moira.adapters.catbench_paths import resolve_results_dir
from moira.mlip.registry import (
    get_catbench_source_path,
    get_model_adapter_function,
    get_model_adapter_module,
    load_config,
    get_model_python,
)
from moira.mlip.shards import slice_json_obj
from moira.mlip.tasks import dataset_name_from_path


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
            return {
                "model": model,
                "dataset_name": dataset_name,
                "input_path": input_path,
            }
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
            "input_path": input_path,
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
    allowed = required | {
        "input_path",
        "shard_index",
        "shard_count",
        "shard_start",
        "shard_stop",
    }
    return {key: str(payload[key]) for key in allowed if key in payload}


def _task_int(task: dict[str, str], key: str) -> int | None:
    value = task.get(key)
    if value is None:
        return None
    return int(value)


def _materialize_task_dataset(
    task: dict[str, str],
    *,
    stack: ExitStack,
) -> tuple[str | None, str]:
    dataset_name = task["dataset_name"]
    dataset_path = task.get("input_path")
    shard_start = _task_int(task, "shard_start")
    shard_stop = _task_int(task, "shard_stop")
    if dataset_path is None or shard_start is None or shard_stop is None:
        return dataset_path, dataset_name

    source_path = Path(dataset_path).resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    shard_payload = slice_json_obj(payload, start=shard_start, stop=shard_stop)
    temp_dir = Path(stack.enter_context(TemporaryDirectory(prefix="moira-shard-")))
    shard_path = temp_dir / source_path.name
    with shard_path.open("w", encoding="utf-8") as handle:
        json.dump(shard_payload, handle, indent=2)
        handle.write("\n")
    return str(shard_path), dataset_name


def _task_results_dir(task: dict[str, str], config_path: str) -> str | None:
    if "shard_index" not in task:
        return None

    config = load_config(config_path)
    dev_run = bool(config.get("mlip", {}).get("dev_run", False))
    configured_results_dir = resolve_results_dir(
        config.get("mlip", {}).get("results_dir"),
        config_path=config_path,
        dev_run=dev_run,
    )
    if configured_results_dir is None:
        configured_results_dir = Path.cwd() / "result_shards"
    return str((configured_results_dir / task["dataset_name"]).resolve())


def _task_slab_cache_dir(task: dict[str, str], config_path: str) -> str | None:
    if "shard_index" not in task:
        return None

    input_path = task.get("input_path")
    if input_path is None:
        return None

    config = load_config(config_path)
    dev_run = bool(config.get("mlip", {}).get("dev_run", False))
    configured_results_dir = resolve_results_dir(
        config.get("mlip", {}).get("results_dir"),
        config_path=config_path,
        dev_run=dev_run,
    )
    if configured_results_dir is None:
        configured_results_dir = Path.cwd() / "result_shards"

    base_dataset_name = dataset_name_from_path(Path(input_path))
    return str(
        (configured_results_dir / base_dataset_name / "_shared" / "slab_cache").resolve()
    )


def _task_mlip_name(model: str, config_path: str) -> str | None:
    config = load_config(config_path)
    spec = config.get("mlip", {}).get("rootstock", {}).get("models", {}).get(model, {})
    mlip_name = spec.get("mlip_name")
    return str(mlip_name) if mlip_name is not None else None


def run_one_task(line: str, config_path: str):
    task = _parse_task_record(line)
    model = task["model"]
    resolved_config_path = _maybe_reexec_with_model_python(model, line, config_path)
    device = _resolve_device(resolved_config_path)

    with ExitStack() as stack:
        dataset_path, dataset_name = _materialize_task_dataset(task, stack=stack)
        results_dir_override = _task_results_dir(task, resolved_config_path)
        slab_cache_dir_override = _task_slab_cache_dir(task, resolved_config_path)
        mlip_name = _task_mlip_name(model, resolved_config_path)

        print(f"Running adapter: {model} ({dataset_name})")
        with _catbench_source_on_syspath(resolved_config_path):
            from moira.mlip.result_metadata import write_efficiency_summary

            run_adapter = _load_adapter_callable(model, resolved_config_path)
            time_init = perf_counter()
            run_adapter(
                model=model,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                device=device,
                config_path=resolved_config_path,
                results_dir_override=results_dir_override,
                slab_cache_dir_override=slab_cache_dir_override,
            )
            task_wall_seconds = perf_counter() - time_init
            result_dir = (
                Path(results_dir_override)
                if results_dir_override is not None
                else Path.cwd() / "result" / (mlip_name or model)
            )
            write_efficiency_summary(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                result_path=result_dir / f"{mlip_name or model}_result.json",
                mlip_name=mlip_name,
                model_name=model,
                task_wall_seconds=task_wall_seconds,
                shard_index=_task_int(task, "shard_index"),
                shard_count=_task_int(task, "shard_count"),
            )
