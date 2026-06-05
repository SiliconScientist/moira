# Restored from the last pre-Rootstock per-model adapter revision.

from __future__ import annotations

from catbench.adsorption import AdsorptionCalculation
from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.calculator import ORBCalculator

from moira.mlip.registry import load_config

MLIP_NAME = "orb-v3-conservative-inf-omat"


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
        orbff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
            weights_path=model_path,
            device=device,
            precision="float32-high",
        )
        calculators.append(
            ORBCalculator(model=orbff, atoms_adapter=atoms_adapter, device=device)
        )

    adsorption_calc = AdsorptionCalculation(
        calculators,
        mlip_name=MLIP_NAME,
        benchmark=dataset_name,
        optimizer=optimizer,
    )
    adsorption_calc.run()
