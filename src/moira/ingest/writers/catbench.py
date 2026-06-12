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


def _validate_emittable_references(bundle: DatasetBundle) -> None:
    structure_ids = {structure.id for structure in bundle.structures}
    missing_ids: list[str] = []
    missing_geometry: list[str] = []

    for reference in bundle.references:
        referenced_structures = [
            structure
            for structure in [reference.slab, reference.adslab, *reference.gas]
            if structure is not None
        ]
        for structure in referenced_structures:
            if structure.id not in structure_ids:
                missing_ids.append(structure.id)
                continue
            relpath = structure.metadata.get("catbench_relpath")
            if structure.source_path is None or not isinstance(relpath, str):
                missing_geometry.append(structure.id)

    if missing_ids:
        missing = ", ".join(sorted(set(missing_ids)))
        raise ValueError(f"Referenced structures must be present in bundle.structures: {missing}")
    if missing_geometry:
        missing = ", ".join(sorted(set(missing_geometry)))
        raise ValueError(f"Referenced structures must include source geometry metadata: {missing}")


def write_catbench_dataset(
    *,
    bundle: DatasetBundle,
    dest: Path,
    coeff_setting: dict[str, dict[str, int | float]],
    output_dir: Path,
    output_name: str,
) -> Path:
    _validate_emittable_references(bundle)
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
