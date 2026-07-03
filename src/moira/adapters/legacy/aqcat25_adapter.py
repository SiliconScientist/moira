# Legacy AQCat25 adapter using SandboxAQ's patched fairchem ASE calculator.

from __future__ import annotations

from pathlib import Path

from catbench.adsorption import AdsorptionCalculation

from moira.adapters.catbench_paths import patch_adsorption_paths, resolve_results_dir
from moira.mlip.result_metadata import enrich_result_file
from moira.mlip.registry import load_config

DEFAULT_MLIP_NAME = "aqcat25-ev2"


def _resolve_required_checkpoint_path(
    path_value: str,
    *,
    config_path: str,
    field_name: str,
) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = (Path(config_path).parent / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return str(path)


def _load_patched_calc():
    try:
        from fairchem.core.common.relaxation.ase_utils import patched_calc
    except ImportError as exc:
        raise ImportError(
            "AQCat25 requires a patched fairchem-core install that provides "
            "'fairchem.core.common.relaxation.ase_utils.patched_calc'. "
            "Run ./envs/aqcat25/patch_fairchem.sh after downloading the gated "
            "AQCat25 Hugging Face files."
        ) from exc

    return patched_calc


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
    del device  # AQCat25's patched calculator manages devices internally.

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
    checkpoint = model_path or spec.get("checkpoint")
    if checkpoint is None:
        raise KeyError(f"mlip.rootstock.models.{model}.checkpoint is required")

    resolved_checkpoint = _resolve_required_checkpoint_path(
        checkpoint,
        config_path=config_path,
        field_name=f"mlip.rootstock.models.{model}.checkpoint",
    )
    mlip_name = str(spec.get("mlip_name", DEFAULT_MLIP_NAME))
    patched_calc = _load_patched_calc()

    calculators = [patched_calc(checkpoint_path=resolved_checkpoint) for _ in range(n_calcs)]

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
