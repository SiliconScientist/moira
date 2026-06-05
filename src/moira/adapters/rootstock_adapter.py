# src/moira/adapters/rootstock_adapter.py

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from catbench.adsorption import AdsorptionCalculation

from moira.mlip.registry import load_config


def _resolve_checkpoint(checkpoint: str | None, config_path: Path) -> str | None:
    if checkpoint is None:
        return None

    checkpoint_path = Path(checkpoint)
    if (
        not checkpoint_path.is_absolute()
        and any(sep in checkpoint for sep in ("/", "\\"))
    ):
        return str((config_path.parent / checkpoint_path).resolve())
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Rootstock adsorption predictions")
    parser.add_argument("--model", required=True, help="Configured MLIP model key")
    parser.add_argument("--input", required=True, help="Input dataset JSON")
    parser.add_argument("--output", required=True, help="Output result JSON")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name (passed from task runner)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="mlip.toml")
    parser.add_argument("--n-calcs", type=int, default=3)
    args = parser.parse_args()

    from rootstock import RootstockCalculator

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))
    rootstock_cfg = config.get("mlip", {}).get("rootstock", {})
    root = rootstock_cfg.get("root", "/projects/bchg/rootstock")
    spec = _get_rootstock_spec(config, args.model)
    checkpoint = _resolve_checkpoint(spec.get("checkpoint"), config_path)

    # --- Build calculators ---
    with ExitStack() as stack:
        calculators = [
            stack.enter_context(
                RootstockCalculator(
                    root=root,
                    model=str(spec["model"]),
                    checkpoint=checkpoint,
                    device=args.device,
                )
            )
            for _ in range(args.n_calcs)
        ]

        # --- Run CatBench adsorption workflow ---
        adsorption_calc = AdsorptionCalculation(
            calculators,
            mlip_name=str(spec["mlip_name"]),
            benchmark=args.dataset_name,
            optimizer=optimizer,
        )
        adsorption_calc.run()


if __name__ == "__main__":
    main()
