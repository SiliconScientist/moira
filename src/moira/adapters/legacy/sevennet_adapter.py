# Restored from the last pre-Rootstock per-model adapter revision.

from __future__ import annotations

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
