# Restored from the last pre-Rootstock per-model adapter revision.

from __future__ import annotations

from catbench.adsorption import AdsorptionCalculation
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

from moira.mlip.registry import load_config

MLIP_NAME = "uma-s-1p1"
TASK_NAME = "oc20"


def run(
    *,
    input_path: str,
    dataset_name: str,
    device: str = "cuda",
    config_path: str = "config.toml",
    model_path: str,
    n_calcs: int = 3,
) -> None:
    del input_path

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
