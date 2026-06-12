from pathlib import Path

from moira.config import get_config
from moira.ingest.catbench_coefficients import build_coeff_setting
from moira.ingest.sources.vasp_mapping import load_vasp_mapping_bundle
from moira.ingest.writers.catbench import write_catbench_dataset


def main():
    cfg = get_config()
    source = cfg.ingest.source
    dest = cfg.ingest.catbench_folder
    if dest is None:
        raise ValueError("cfg.ingest.catbench_folder is not initialized")

    bundle = load_vasp_mapping_bundle(source, dataset_name=cfg.ingest.dataset_name)
    tag_map = bundle.metadata["tag_map"]

    coeff_setting = build_coeff_setting(
        tag_map=tag_map,
        elements=list(cfg.ingest.stoich.elements),
        basis_species=list(cfg.ingest.stoich.basis_species),
    )

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
