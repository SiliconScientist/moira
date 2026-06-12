from __future__ import annotations

from pathlib import Path

from moira.config import get_config
from moira.ingest.models import DatasetBundle
from moira.ingest.sources.ase_db import load_ase_db_bundle
from moira.ingest.sources.vasp_mapping import load_vasp_mapping_bundle
from moira.ingest.transforms.catbench import build_catbench_coefficients
from moira.ingest.transforms.structural_references import (
    annotate_elemental_adsorption_bundle,
    build_gas_reference_record,
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
    if profile in {"elemental_adsorption_ase_db", "elemental_n_ase_db"}:
        return load_elemental_ase_db_dataset(
            source,
            dataset_name=dataset_name,
        )
    raise ValueError(f"Unsupported ingest.profile: {profile}")


def load_elemental_ase_db_dataset(
    source: Path,
    *,
    adsorbate_symbol: str | None = None,
    dataset_name: str | None = None,
    row_limit: int | None = None,
) -> DatasetBundle:
    bundle = load_ase_db_bundle(
        source,
        dataset_name=dataset_name,
        row_limit=row_limit,
    )
    annotated = annotate_elemental_adsorption_bundle(
        bundle,
        adsorbate_symbol=adsorbate_symbol,
        structure_kind="adslab",
    )
    synthesized = synthesize_adsorption_references(
        annotated,
        adsorbate_symbol=adsorbate_symbol,
        gas_record=None if adsorbate_symbol is None else build_gas_reference_record(
            formula=f"{adsorbate_symbol}2",
        ),
    )
    return _annotate_elemental_catbench_layout(synthesized)


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


def _annotate_elemental_catbench_layout(bundle: DatasetBundle) -> DatasetBundle:
    for reference in bundle.references:
        surface_name = reference.id.replace(":", "_")
        adsorbate_name = str(reference.metadata.get("adsorbate", "adsorbate"))

        if reference.slab is not None:
            reference.slab.metadata["catbench_relpath"] = f"{surface_name}/slab"
        if reference.adslab is not None:
            reference.adslab.metadata["catbench_relpath"] = (
                f"{surface_name}/{adsorbate_name}/1"
            )
        for gas in reference.gas:
            gas_name = gas.formula or gas.label or gas.id.removeprefix("gas:")
            gas.metadata["catbench_relpath"] = f"gas/{gas_name}gas"
    return bundle


def main():
    cfg = get_config()
    bundle = load_dataset(cfg)
    coeff_setting = build_coefficients(cfg, bundle)
    write_dataset(cfg, bundle=bundle, coeff_setting=coeff_setting)


if __name__ == "__main__":
    main()
