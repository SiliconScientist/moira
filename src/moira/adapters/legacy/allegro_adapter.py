# Legacy Allegro adapter using NequIP's ASE calculator for compiled models.

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import allegro  # noqa: F401  # Ensure the Allegro NequIP extension is registered.
from catbench.adsorption import AdsorptionCalculation
from nequip.integrations.ase import NequIPCalculator

from moira.adapters.catbench_paths import patch_adsorption_paths, resolve_results_dir
from moira.mlip.result_metadata import enrich_result_file
from moira.mlip.registry import load_config
from moira.pathing import resolve_project_path

DEFAULT_MLIP_NAME = "allegro"


def _prepare_aotinductor_loader() -> None:
    # PyTorch 2.10's PT2 loader accesses this lazily loaded module as an attribute.
    import_module("torch._inductor.codecache")


def _resolve_required_checkpoint_path(
    path_value: str,
    *,
    config_path: str,
    field_name: str,
) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = resolve_project_path(path, config_path=config_path)

    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return str(path)


def _resolve_chemical_species_to_atom_type_map(
    metadata: object,
) -> dict[str, str] | bool | None:
    if not isinstance(metadata, dict):
        return None

    value = metadata.get("chemical_species_to_atom_type_map")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {str(key): str(mapped_type) for key, mapped_type in value.items()}

    raise ValueError(
        "mlip.rootstock.models.allegro.metadata.chemical_species_to_atom_type_map "
        "must be a boolean or a table"
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
    checkpoint = model_path or spec.get("checkpoint")
    if checkpoint is None:
        raise KeyError(f"mlip.rootstock.models.{model}.checkpoint is required")

    resolved_checkpoint = _resolve_required_checkpoint_path(
        checkpoint,
        config_path=config_path,
        field_name=f"mlip.rootstock.models.{model}.checkpoint",
    )
    mlip_name = str(spec.get("mlip_name", DEFAULT_MLIP_NAME))
    chemical_species_to_atom_type_map = _resolve_chemical_species_to_atom_type_map(
        spec.get("metadata", {})
    )

    _prepare_aotinductor_loader()
    calculators = [
        NequIPCalculator.from_compiled_model(
            compile_path=resolved_checkpoint,
            device=device,
            chemical_species_to_atom_type_map=chemical_species_to_atom_type_map,
        )
        for _ in range(n_calcs)
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
