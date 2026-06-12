from __future__ import annotations

from pathlib import Path

from moira.config import get_config
from moira.ingest.models import DatasetBundle
from moira.ingest.sources.ase_db import load_ase_db_bundle
from moira.ingest.sources.vasp_mapping import load_vasp_mapping_bundle
from moira.ingest.transforms.catbench import build_catbench_coefficients
from moira.ingest.transforms.structural_references import (
    annotate_elemental_adsorption_bundle,
    build_diatomic_gas_record,
    synthesize_adsorption_references,
)
from moira.ingest.writers.catbench import write_catbench_dataset


def load_dataset(cfg) -> DatasetBundle:
    profile = cfg.ingest.profile
    source = cfg.ingest.source
    dataset_name = cfg.ingest.dataset_name

    if profile == "vasp_mapping":
        return load_vasp_mapping_bundle(
            source,
            dataset_name=dataset_name,
        )
    if profile == "elemental_n_ase_db":
        return load_elemental_n_ase_db_dataset(
            source,
            dataset_name=dataset_name,
        )
    raise ValueError(f"Unsupported ingest.profile: {profile}")


def load_elemental_n_ase_db_dataset(
    source: Path,
    *,
    dataset_name: str | None = None,
) -> DatasetBundle:
    bundle = load_ase_db_bundle(source, dataset_name=dataset_name)
    annotated = annotate_elemental_adsorption_bundle(
        bundle,
        adsorbate_symbol="N",
        structure_kind="adslab",
    )
    return synthesize_adsorption_references(
        annotated,
        adsorbate_symbol="N",
        gas_record=build_diatomic_gas_record(
            symbol="N",
            formula="N2",
            bond_length=1.10,
        ),
    )


def build_coefficients(cfg, bundle: DatasetBundle) -> dict[str, dict[str, int | float]]:
    return build_catbench_coefficients(
        bundle,
        elements=list(cfg.ingest.stoich.elements),
        basis_species=list(cfg.ingest.stoich.basis_species),
    )


def write_dataset(cfg, *, bundle: DatasetBundle, coeff_setting: dict[str, dict[str, int | float]]) -> Path:
    dest = cfg.ingest.catbench_folder
    if dest is None:
        raise ValueError("cfg.ingest.catbench_folder is not initialized")

    project_root = Path.cwd()
    output_dir = project_root / "data" / "raw_data"
    return write_catbench_dataset(
        bundle=bundle,
        dest=dest,
        coeff_setting=coeff_setting,
        output_dir=output_dir,
        output_name=cfg.ingest.dataset_name,
    )


def main():
    cfg = get_config()
    bundle = load_dataset(cfg)
    coeff_setting = build_coefficients(cfg, bundle)
    write_dataset(cfg, bundle=bundle, coeff_setting=coeff_setting)


if __name__ == "__main__":
    main()
