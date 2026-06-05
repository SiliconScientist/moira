# Restored from the last pre-Rootstock per-model adapter revision.

from __future__ import annotations

import argparse

from catbench.adsorption import AdsorptionCalculation
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

from moira.mlip.registry import load_config

MLIP_NAME = "uma-s-1p1"
TASK_NAME = "oc20"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UMA-S-1p1 adsorption predictions")
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
        required=True,
        help="Path to UMA-S-1p1 checkpoint (.pt)",
    )
    parser.add_argument("--n-calcs", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))

    calculators = []
    for _ in range(args.n_calcs):
        predict_unit = load_predict_unit(
            path=args.model_path,
            device=args.device,
        )
        calc = FAIRChemCalculator(
            predict_unit=predict_unit,
            task_name=TASK_NAME,
        )
        calculators.append(calc)

    adsorption_calc = AdsorptionCalculation(
        calculators,
        mlip_name=MLIP_NAME,
        benchmark=args.dataset_name,
        optimizer=optimizer,
    )
    adsorption_calc.run()


if __name__ == "__main__":
    main()
