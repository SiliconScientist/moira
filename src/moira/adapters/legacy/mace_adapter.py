# Restored from the last pre-Rootstock per-model adapter revision.

from __future__ import annotations

import argparse
from pathlib import Path

from catbench.adsorption import AdsorptionCalculation
from mace.calculators import mace_mp

MLIP_NAME = "mace-mh-1"


def infer_benchmark(input_path: str) -> str:
    name = Path(input_path).stem
    if not name.endswith("_adsorption"):
        raise ValueError(
            f"Expected dataset name to end with '_adsorption.json', got {name}"
        )
    return name.replace("_adsorption", "")


def run(
    *,
    input_path: str,
    device: str = "cuda",
    config_path: str = "config.toml",
    model_path: str | None = None,
    n_calcs: int = 3,
) -> None:
    del config_path

    calculators = [
        mace_mp(
            model=model_path,
            device=device,
            default_dtype="float32",
            head="omat_pbe",
        )
        for _ in range(n_calcs)
    ]

    adsorption_calc = AdsorptionCalculation(
        calculators,
        mlip_name=MLIP_NAME,
        benchmark=infer_benchmark(input_path),
    )
    adsorption_calc.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MACE adsorption predictions")
    parser.add_argument("--input", required=True, help="Input dataset JSON")
    parser.add_argument(
        "--output",
        required=True,
        help="Compatibility task work path; CatBench manages result artifacts.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--n-calcs", type=int, default=3)
    args = parser.parse_args()
    run(
        input_path=args.input,
        device=args.device,
        config_path=args.config,
        model_path=args.model_path,
        n_calcs=args.n_calcs,
    )


if __name__ == "__main__":
    main()
