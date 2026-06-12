import shutil
from pathlib import Path
from catbench.adsorption.data import vasp as catbench_vasp

from moira.config import get_config
from moira.ingest.catbench_coefficients import build_coeff_setting
from moira.ingest.models import DatasetBundle, StructureRecord
from moira.ingest.sources.vasp_mapping import load_vasp_mapping_bundle


def copy_selected_files(src_dir: Path, dst_dir: Path, filenames=("CONTCAR", "OSZICAR")):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src_file = src_dir / name
        if src_file.is_file():
            shutil.copy2(src_file, dst_dir / name)


def materialize_catbench_layout(bundle: DatasetBundle, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for structure in bundle.structures:
        materialize_structure(structure, dest)


def materialize_structure(structure: StructureRecord, dest: Path) -> None:
    if structure.source_path is None:
        return
    relpath = structure.metadata.get("catbench_relpath")
    if not isinstance(relpath, str):
        return
    copy_selected_files(Path(structure.source_path), dest / relpath)


def main():
    cfg = get_config()
    source = cfg.ingest.source
    dest = cfg.ingest.catbench_folder
    if dest is None:
        raise ValueError("cfg.ingest.catbench_folder is not initialized")

    bundle = load_vasp_mapping_bundle(source, dataset_name=cfg.ingest.dataset_name)
    materialize_catbench_layout(bundle, dest)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{cfg.ingest.dataset_name}_adsorption.json"

    catbench_vasp.get_raw_data_directory = lambda: str(output_dir)
    catbench_vasp.get_raw_data_path = lambda _benchmark_name: str(output_path)
    catbench_vasp.vasp_preprocessing(
        dataset_name=str(dest),
        coeff_setting=coeff_setting,
    )


if __name__ == "__main__":
    main()
