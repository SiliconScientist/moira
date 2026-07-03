# Restored from the last pre-Rootstock per-model adapter revision.

from __future__ import annotations

from pathlib import Path

from catbench.adsorption import AdsorptionCalculation
from sevenn.calculator import SevenNetCalculator

from moira.adapters.catbench_paths import patch_adsorption_paths, resolve_results_dir
from moira.mlip.result_metadata import enrich_result_file
from moira.mlip.registry import load_config

DEFAULT_MLIP_NAME = "7net-omni"
DEFAULT_MODAL = "mpa"


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
    del model_path

    config = load_config(config_path)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))
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
    mlip_name = str(spec.get("mlip_name", DEFAULT_MLIP_NAME))
    modal = str(metadata.get("modal", DEFAULT_MODAL))

    calculators = [
        SevenNetCalculator(model=mlip_name, modal=modal) for _ in range(n_calcs)
    ]

    with patch_adsorption_paths(
        dataset_path=dataset_path,
        results_dir=results_dir,
    ):
        adsorption_calc = AdsorptionCalculation(
            calculators,
            mlip_name=mlip_name,
            benchmark=dataset_name,
            optimizer=optimizer,
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
