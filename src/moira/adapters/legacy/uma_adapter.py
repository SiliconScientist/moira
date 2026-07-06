# Restored from the last pre-Rootstock per-model adapter revision.

from __future__ import annotations

from pathlib import Path

from catbench.adsorption import AdsorptionCalculation
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

from moira.adapters.catbench_paths import patch_adsorption_paths, resolve_results_dir
from moira.mlip.result_metadata import enrich_result_file
from moira.mlip.registry import load_config
from moira.pathing import resolve_project_path

DEFAULT_MLIP_NAME = "uma-s-1p1"
DEFAULT_TASK_NAME = "oc20"


def _resolve_checkpoint_path(checkpoint: str, config_path: str) -> str:
    checkpoint_path = Path(checkpoint)
    if (
        not checkpoint_path.is_absolute()
        and any(sep in checkpoint for sep in ("/", "\\"))
    ):
        checkpoint_path = resolve_project_path(
            checkpoint_path,
            config_path=config_path,
        )

    if checkpoint_path.exists():
        return str(checkpoint_path)

    raise FileNotFoundError(
        "Legacy UMA requires mlip.rootstock.models.uma.checkpoint to point to an "
        f"existing checkpoint file, but got {checkpoint!r}. "
        "If you intended to use the Rootstock model alias "
        "('uma-s-1p1'), set mlip.adapter_backend = 'rootstock'."
    )


def run(
    *,
    model: str,
    dataset_name: str,
    dataset_path: str | None = None,
    device: str = "cuda",
    config_path: str = "config.toml",
    model_path: str | None = None,
    n_calcs: int = 3,
    results_dir_override: str | None = None,
    slab_cache_dir_override: str | None = None,
) -> None:
    config = load_config(config_path)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))
    save_files = bool(config.get("mlip", {}).get("save_files", True))
    dev_run = bool(config.get("mlip", {}).get("dev_run", False))
    n_calcs = 1 if dev_run else n_calcs
    results_dir = (
        Path(results_dir_override).resolve()
        if results_dir_override is not None
        else resolve_results_dir(
            config.get("mlip", {}).get("results_dir"),
            config_path=config_path,
            dev_run=dev_run,
        )
    )
    spec = config.get("mlip", {}).get("rootstock", {}).get("models", {}).get(model, {})
    metadata = spec.get("metadata", {})
    resolved_model_path = model_path or spec.get("checkpoint")
    if resolved_model_path is None:
        raise KeyError(f"mlip.rootstock.models.{model}.checkpoint is required")
    resolved_model_path = _resolve_checkpoint_path(resolved_model_path, config_path)
    mlip_name = str(spec.get("mlip_name", DEFAULT_MLIP_NAME))
    task_name = str(metadata.get("task_name", DEFAULT_TASK_NAME))

    calculators = []
    for _ in range(n_calcs):
        predict_unit = load_predict_unit(
            path=resolved_model_path,
            device=device,
        )
        calc = FAIRChemCalculator(
            predict_unit=predict_unit,
            task_name=task_name,
        )
        calculators.append(calc)

    with patch_adsorption_paths(
        dataset_path=dataset_path,
        results_dir=results_dir,
    ):
        adsorption_calc = AdsorptionCalculation(
            calculators,
            mlip_name=mlip_name,
            benchmark=dataset_name,
            optimizer=optimizer,
            save_files=save_files,
            model_name=model,
            slab_cache_dir=slab_cache_dir_override,
        )
        save_directory = Path(adsorption_calc.run())
        enrich_result_file(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            result_path=save_directory / f"{mlip_name}_result.json",
            mlip_name=mlip_name,
            model_name=model,
        )
