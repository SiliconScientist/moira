# src/moira/adapters/rootstock_adapter.py

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

from catbench.adsorption import AdsorptionCalculation

from moira.adapters.catbench_paths import patch_adsorption_paths, resolve_results_dir
from moira.mlip.result_metadata import enrich_result_file
from moira.mlip.registry import load_config
from moira.pathing import resolve_project_path


def _resolve_checkpoint(checkpoint: str | None, config_path: Path) -> str | None:
    if checkpoint is None:
        return None

    checkpoint_path = Path(checkpoint)
    if (
        not checkpoint_path.is_absolute()
        and any(sep in checkpoint for sep in ("/", "\\"))
    ):
        return str(resolve_project_path(checkpoint_path, config_path=config_path))
    return checkpoint


def _get_rootstock_spec(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    rootstock = config.get("mlip", {}).get("rootstock", {})
    models = rootstock.get("models", {})
    if model_name not in models:
        known = ", ".join(sorted(str(name) for name in models))
        raise KeyError(
            f"No Rootstock config found for model '{model_name}'. "
            f"Known Rootstock models: {known}"
        )

    spec = dict(models[model_name])
    if "model" not in spec:
        raise KeyError(f"mlip.rootstock.models.{model_name}.model is required")
    if "mlip_name" not in spec:
        raise KeyError(f"mlip.rootstock.models.{model_name}.mlip_name is required")
    return spec


def run(
    *,
    model: str,
    dataset_name: str,
    dataset_path: str | None = None,
    device: str = "cuda",
    config_path: str = "config.toml",
    n_calcs: int = 3,
    results_dir_override: str | None = None,
    slab_cache_dir_override: str | None = None,
) -> None:
    from rootstock import RootstockCalculator

    resolved_config_path = Path(config_path).resolve()
    config = load_config(resolved_config_path)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))
    dev_run = bool(config.get("mlip", {}).get("dev_run", False))
    n_calcs = 1 if dev_run else n_calcs
    results_dir = (
        Path(results_dir_override).resolve()
        if results_dir_override is not None
        else resolve_results_dir(
            config.get("mlip", {}).get("results_dir"),
            config_path=resolved_config_path,
            dev_run=dev_run,
        )
    )
    rootstock_cfg = config.get("mlip", {}).get("rootstock", {})
    root = rootstock_cfg.get("root", "/projects/bchg/rootstock")
    spec = _get_rootstock_spec(config, model)
    checkpoint = _resolve_checkpoint(spec.get("checkpoint"), resolved_config_path)
    mlip_name = str(spec["mlip_name"])

    with ExitStack() as stack:
        calculators = [
            stack.enter_context(
                RootstockCalculator(
                    root=root,
                    model=str(spec["model"]),
                    checkpoint=checkpoint,
                    device=device,
                )
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
