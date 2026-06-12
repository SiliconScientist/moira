from __future__ import annotations

import shutil
from pathlib import Path

from catbench.adsorption.data import vasp as catbench_vasp

from moira.ingest.models import DatasetBundle, StructureRecord


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


def write_catbench_dataset(
    *,
    bundle: DatasetBundle,
    dest: Path,
    coeff_setting: dict[str, dict[str, int | float]],
    output_dir: Path,
    output_name: str,
) -> Path:
    materialize_catbench_layout(bundle, dest)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}_adsorption.json"

    catbench_vasp.get_raw_data_directory = lambda: str(output_dir)
    catbench_vasp.get_raw_data_path = lambda _benchmark_name: str(output_path)
    catbench_vasp.vasp_preprocessing(
        dataset_name=str(dest),
        coeff_setting=coeff_setting,
    )
    return output_path
