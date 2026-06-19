# Restored from the last pre-Rootstock per-model adapter revision.

from __future__ import annotations

from pathlib import Path

from catbench.adsorption import AdsorptionCalculation
from mattersim.forcefield.potential import MatterSimCalculator, Potential

from moira.adapters.catbench_paths import patch_adsorption_paths, resolve_results_dir
from moira.mlip.result_metadata import enrich_result_file
from moira.mlip.registry import load_config

DEFAULT_MLIP_NAME = "mattersim-v1-5m"


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
) -> None:
    config = load_config(config_path)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))
    dev_run = bool(config.get("mlip", {}).get("dev_run", False))
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
    resolved_model_path = model_path or spec.get("checkpoint")
    mlip_name = str(spec.get("mlip_name", DEFAULT_MLIP_NAME))
    if resolved_model_path is not None:
        checkpoint_path = Path(resolved_model_path)
        if (
            not checkpoint_path.is_absolute()
            and any(sep in resolved_model_path for sep in ("/", "\\"))
        ):
            resolved_model_path = str((Path(config_path).parent / checkpoint_path).resolve())
    calculators = []
    for _ in range(n_calcs):
        potential = Potential.from_checkpoint(
            load_path=resolved_model_path,
            device=device,
        )
        calculators.append(MatterSimCalculator(potential=potential))

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
        )
        save_directory = Path(adsorption_calc.run())
        enrich_result_file(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            result_path=save_directory / f"{mlip_name}_result.json",
            mlip_name=mlip_name,
            model_name=model,
        )
