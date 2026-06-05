# Restored from the last pre-Rootstock per-model adapter revision.

from __future__ import annotations

from catbench.adsorption import AdsorptionCalculation
from mattersim.forcefield.potential import MatterSimCalculator, Potential

from moira.mlip.registry import load_config

MLIP_NAME = "mattersim-v1-5m"


def run(
    *,
    input_path: str,
    dataset_name: str,
    device: str = "cuda",
    config_path: str = "config.toml",
    model_path: str | None = None,
    n_calcs: int = 3,
) -> None:
    del input_path

    config = load_config(config_path)
    optimizer = str(config.get("mlip", {}).get("optimizer", "LBFGS"))
    calculators = []
    for _ in range(n_calcs):
        potential = Potential.from_checkpoint(
            load_path=model_path,
            device=device,
        )
        calculators.append(MatterSimCalculator(potential=potential))

    adsorption_calc = AdsorptionCalculation(
        calculators,
        mlip_name=MLIP_NAME,
        benchmark=dataset_name,
        optimizer=optimizer,
    )
    adsorption_calc.run()
