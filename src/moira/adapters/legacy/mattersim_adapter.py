# Restored from the last pre-Rootstock per-model adapter revision.

from __future__ import annotations

import argparse

from catbench.adsorption import AdsorptionCalculation
from mattersim.forcefield.potential import MatterSimCalculator, Potential

from moira.mlip.registry import load_config

MLIP_NAME = "mattersim-v1-5m"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MACE adsorption predictions")
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
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--n-calcs", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))
    calculators = []
    for _ in range(args.n_calcs):
        potential = Potential.from_checkpoint(
            load_path=args.model_path,
            device=args.device,
        )
        calculators.append(MatterSimCalculator(potential=potential))

    adsorption_calc = AdsorptionCalculation(
        calculators,
        mlip_name=MLIP_NAME,
        benchmark=args.dataset_name,
        optimizer=optimizer,
    )
    adsorption_calc.run()


if __name__ == "__main__":
    main()
