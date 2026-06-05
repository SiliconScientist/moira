# Restored from the last pre-Rootstock per-model adapter revision.

from __future__ import annotations

import argparse

from catbench.adsorption import AdsorptionCalculation
from sevenn.calculator import SevenNetCalculator

from moira.mlip.registry import load_config

MLIP_NAME = "7net-omni"
MODAL = "mpa"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SevenNet adsorption predictions")
    parser.add_argument("--input", required=True, help="Input dataset JSON")
    parser.add_argument("--output", required=True, help="Output result JSON")
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
    config = load_config(args.config)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))

    calculators = [
        SevenNetCalculator(model=MLIP_NAME, modal=MODAL) for _ in range(args.n_calcs)
    ]

    adsorption_calc = AdsorptionCalculation(
        calculators,
        mlip_name=MLIP_NAME,
        benchmark=args.dataset_name,
        optimizer=optimizer,
    )
    adsorption_calc.run()


if __name__ == "__main__":
    main()
