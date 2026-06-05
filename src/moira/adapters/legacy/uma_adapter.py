# Restored from the last pre-Rootstock per-model adapter revision.

from __future__ import annotations

import argparse

from catbench.adsorption import AdsorptionCalculation
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

from moira.mlip.registry import load_config

MLIP_NAME = "uma-s-1p1"
TASK_NAME = "oc20"


def run(
    *,
    input_path: str,
    output_path: str,
    dataset_name: str,
    device: str = "cuda",
    config_path: str = "config.toml",
    model_path: str,
    n_calcs: int = 3,
) -> None:
    del input_path, output_path

    config = load_config(config_path)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))

    calculators = []
    for _ in range(n_calcs):
        predict_unit = load_predict_unit(
            path=model_path,
            device=device,
        )
        calc = FAIRChemCalculator(
            predict_unit=predict_unit,
            task_name=TASK_NAME,
        )
        calculators.append(calc)

    adsorption_calc = AdsorptionCalculation(
        calculators,
        mlip_name=MLIP_NAME,
        benchmark=dataset_name,
        optimizer=optimizer,
    )
    adsorption_calc.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UMA-S-1p1 adsorption predictions")
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
        required=True,
        help="Path to UMA-S-1p1 checkpoint (.pt)",
    )
    parser.add_argument("--n-calcs", type=int, default=3)
    args = parser.parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        dataset_name=args.dataset_name,
        device=args.device,
        config_path=args.config,
        model_path=args.model_path,
        n_calcs=args.n_calcs,
    )


if __name__ == "__main__":
    main()
