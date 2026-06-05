# Restored from the last pre-Rootstock per-model adapter revision.

from __future__ import annotations

import argparse

from catbench.adsorption import AdsorptionCalculation
from sevenn.calculator import SevenNetCalculator

from moira.mlip.registry import load_config

MLIP_NAME = "7net-omni"
MODAL = "mpa"


def run(
    *,
    input_path: str,
    dataset_name: str,
    device: str = "cuda",
    config_path: str = "config.toml",
    model_path: str | None = None,
    n_calcs: int = 3,
) -> None:
    del input_path, model_path

    config = load_config(config_path)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))

    calculators = [
        SevenNetCalculator(model=MLIP_NAME, modal=MODAL) for _ in range(n_calcs)
    ]

    adsorption_calc = AdsorptionCalculation(
        calculators,
        mlip_name=MLIP_NAME,
        benchmark=dataset_name,
        optimizer=optimizer,
    )
    adsorption_calc.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SevenNet adsorption predictions")
    parser.add_argument("--input", required=True, help="Input dataset JSON")
    parser.add_argument(
        "--output",
        required=True,
        help="Compatibility task work path; CatBench manages result artifacts.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name (passed from task runner)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Unused for SevenNet preset model IDs (kept for adapter interface compatibility).",
    )
    parser.add_argument("--n-calcs", type=int, default=3)
    args = parser.parse_args()
    run(
        input_path=args.input,
        dataset_name=args.dataset_name,
        device=args.device,
        config_path=args.config,
        model_path=args.model_path,
        n_calcs=args.n_calcs,
    )


if __name__ == "__main__":
    main()
