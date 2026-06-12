from pathlib import Path

from moira.config import get_config
from moira.ingest.models import DatasetBundle
from moira.ingest.sources.vasp_mapping import load_vasp_mapping_bundle
from moira.ingest.transforms.catbench import build_catbench_coefficients
from moira.ingest.writers.catbench import write_catbench_dataset


def load_dataset(cfg) -> DatasetBundle:
    return load_vasp_mapping_bundle(
        cfg.ingest.source,
        dataset_name=cfg.ingest.dataset_name,
    )


def build_coefficients(cfg, bundle: DatasetBundle) -> dict[str, dict[str, int | float]]:
    return build_catbench_coefficients(
        bundle,
        elements=list(cfg.ingest.stoich.elements),
        basis_species=list(cfg.ingest.stoich.basis_species),
    )


def main():
    cfg = get_config()
    dest = cfg.ingest.catbench_folder
    if dest is None:
        raise ValueError("cfg.ingest.catbench_folder is not initialized")

    bundle = load_dataset(cfg)
    coeff_setting = build_coefficients(cfg, bundle)

    # CatBench's VASP preprocessor uses one parameter for both:
    # 1) input dataset directory traversal, and 2) output JSON filename stem.
    # Patch its path helpers so we can read from the real dataset path while
    # writing to a stable project-root output file.
    project_root = Path.cwd()
    output_dir = project_root / "data" / "raw_data"
    write_catbench_dataset(
        bundle=bundle,
        dest=dest,
        coeff_setting=coeff_setting,
        output_dir=output_dir,
        output_name=cfg.ingest.dataset_name,
    )


if __name__ == "__main__":
    main()
