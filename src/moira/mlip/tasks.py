# src/moira/mlip/tasks.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from moira.mlip.registry import get_model_specs, load_config
from moira.mlip.shards import infer_shard_count, shard_bounds, slice_json_obj


def default_run_tag(cfg: dict) -> str:
    # allow config override; else caller provides; else fallback handled in CLI
    mlip = cfg.get("mlip", {})
    tag = mlip.get("run_tag", None)
    return str(tag) if tag else "run"


def resolve_datasets(datasets: list[str] | None, cfg: dict) -> list[Path]:
    if datasets and len(datasets) > 0:
        return [Path(d) for d in datasets]

    mlip = cfg.get("mlip", {})
    cfg_datasets = mlip.get("datasets", None)
    if cfg_datasets is None:
        cfg_datasets = mlip.get("dataset", None)
    if cfg_datasets:
        if isinstance(cfg_datasets, (str, Path)):
            cfg_list = [cfg_datasets]
        elif isinstance(cfg_datasets, list):
            cfg_list = cfg_datasets
        else:
            raise TypeError("mlip.dataset(s) must be a string or list of strings")
        paths = [Path(p) for p in cfg_list]
        missing = [p for p in paths if not p.exists()]
        if missing:
            missing_str = ", ".join(p.as_posix() for p in missing)
            raise FileNotFoundError(
                f"Config-specified dataset(s) not found: {missing_str}"
            )
        return paths


def dataset_name_from_path(p: Path) -> str:
    # remove the "_dev_adsorption" or "_adsorption" suffix before the json extension
    if p.name.endswith("_dev_adsorption.json"):
        return p.stem[: -len("_dev_adsorption")]
    if p.name.endswith("_adsorption.json"):
        return p.stem[: -len("_adsorption")]
    return p.stem


def shard_dataset_name(dataset_name: str, *, shard_index: int, shard_count: int) -> str:
    width = max(2, len(str(max(shard_count - 1, 0))))
    return f"{dataset_name}_shard_{shard_index:0{width}d}_of_{shard_count:0{width}d}"


def maybe_make_dev_dataset(dpath: Path, cfg: dict) -> Path:
    mlip = cfg.get("mlip", {})
    dev_run = bool(mlip.get("dev_run", False))
    dev_n = int(mlip.get("dev_n", 2))
    if not dev_run:
        return dpath
    # If already a dev dataset, use it as-is.
    if dpath.name.endswith("_dev_adsorption.json"):
        return dpath
    # Prefer "<base>_dev_adsorption.json" when the input is "<base>_adsorption.json".
    if dpath.name.endswith("_adsorption.json"):
        base = dpath.stem[: -len("_adsorption")]
        out = dpath.with_name(f"{base}_dev_adsorption{dpath.suffix}")
    else:
        out = dpath.with_name(f"{dpath.stem}_dev{dpath.suffix}")
    # Reuse existing dev dataset if present (keeps sbatch deterministic)
    if out.exists():
        return out
    with dpath.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    sliced = slice_json_obj(obj, start=0, stop=dev_n)
    with out.open("w", encoding="utf-8") as f:
        json.dump(sliced, f, indent=2)
        f.write("\n")
    return out


def make_task_lines(
    *,
    config_path: str | Path,
    run_tag: str,
    datasets: list[str] | None = None,
) -> list[str]:
    cfg = load_config(config_path)
    mlip_cfg = cfg.get("mlip", {})
    dev_run = bool(mlip_cfg.get("dev_run", False))
    shard_size = mlip_cfg.get("shard_size")
    num_shards = mlip_cfg.get("num_shards")
    shard_index = mlip_cfg.get("shard_index")
    specs = get_model_specs(config_path)
    dataset_paths = resolve_datasets(datasets, cfg)
    lines: list[str] = []
    for dpath in dataset_paths:
        task_dataset_path = maybe_make_dev_dataset(dpath, cfg)
        dname_base = dataset_name_from_path(task_dataset_path)
        dname_task = dname_base
        if dev_run and not dname_task.endswith("_dev"):
            dname_task = f"{dname_task}_dev"
        shard_records = _dataset_shard_records(
            task_dataset_path,
            dataset_name=dname_task,
            shard_size=shard_size,
            num_shards=num_shards,
            shard_index=shard_index,
        )
        for model in specs:
            for shard_record in shard_records:
                lines.append(
                    json.dumps(
                        {
                            "model": model,
                            **shard_record,
                        }
                    )
                )
    return lines


def _dataset_shard_records(
    dataset_path: Path,
    *,
    dataset_name: str,
    shard_size: int | None,
    num_shards: int | None,
    shard_index: int | None,
) -> list[dict[str, object]]:
    resolved_input_path = str(dataset_path.resolve())
    if shard_size is None and num_shards is None:
        return [
            {
                "dataset_name": dataset_name,
                "input_path": resolved_input_path,
            }
        ]

    with dataset_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    shard_count = infer_shard_count(obj, shard_size=shard_size, num_shards=num_shards)
    records: list[dict[str, object]] = []
    shard_indexes = range(shard_count) if shard_index is None else [shard_index]
    for current_shard_index in shard_indexes:
        start, stop = shard_bounds(
            obj,
            shard_size=shard_size,
            num_shards=num_shards,
            shard_index=current_shard_index,
        )
        records.append(
            {
                "dataset_name": shard_dataset_name(
                    dataset_name,
                    shard_index=current_shard_index,
                    shard_count=shard_count,
                ),
                "input_path": resolved_input_path,
                "shard_index": current_shard_index,
                "shard_count": shard_count,
                "shard_start": start,
                "shard_stop": stop,
            }
        )
    return records


def write_tasks(out_path: str | Path, lines: Iterable[str]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def make_tasks(
    *,
    config_path: str | Path,
    run_tag: str,
    out_path: str | Path,
    datasets: list[str] | None = None,
) -> None:
    # Optional: if caller passes run_tag="auto", use config default
    cfg = load_config(config_path)
    if run_tag == "auto":
        run_tag = default_run_tag(cfg)

    lines = make_task_lines(config_path=config_path, run_tag=run_tag, datasets=datasets)
    write_tasks(out_path, lines)
    print(f"Wrote {len(lines)} tasks to {out_path}")
