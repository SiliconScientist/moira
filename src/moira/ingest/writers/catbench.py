from __future__ import annotations

from copy import deepcopy
import json
import shutil
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write
from catbench.adsorption.data import vasp as catbench_vasp
from catbench.utils.data_utils import detect_adsorbate_indices, save_catbench_json

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
    relpath = structure.metadata.get("catbench_relpath")
    if not isinstance(relpath, str):
        return
    dest_dir = dest / relpath
    if structure.source_path is not None and Path(structure.source_path).is_dir():
        copy_selected_files(Path(structure.source_path), dest_dir)
        return
    if structure.has_inline_geometry():
        materialize_inline_structure(structure, dest_dir)


def materialize_inline_structure(structure: StructureRecord, dest_dir: Path) -> None:
    atoms = _atoms_from_structure(structure)
    dest_dir.mkdir(parents=True, exist_ok=True)
    write(dest_dir / "CONTCAR", atoms, format="vasp", direct=True, sort=False)
    if structure.energy_ev is not None:
        _write_oszicar(dest_dir / "OSZICAR", structure.energy_ev)


def _atoms_from_structure(structure: StructureRecord) -> Atoms:
    if structure.symbols is None or structure.positions is None:
        raise ValueError(f"Structure '{structure.id}' is missing atomic geometry")
    if structure.cell is None or structure.pbc is None:
        raise ValueError(f"Structure '{structure.id}' is missing cell geometry")
    atoms = Atoms(
        symbols=structure.symbols,
        positions=structure.positions,
        cell=structure.cell,
        pbc=structure.pbc,
    )
    if structure.constraints not in (None, [], ()):
        atoms.set_constraint(deepcopy(structure.constraints))
    return atoms


def _write_oszicar(path: Path, energy_ev: float) -> None:
    path.write_text(
        f"1 F= 0 E0= {energy_ev} d E =0\n",
        encoding="utf-8",
    )


def _require_catbench_reference_paths(bundle: DatasetBundle) -> None:
    missing: dict[str, list[str]] = {}
    for reference in bundle.references:
        for _, structure in reference.energy_complete_components():
            relpath = structure.metadata.get("catbench_relpath")
            if not isinstance(relpath, str):
                missing.setdefault(reference.id, []).append(structure.id)
    if missing:
        problems = "; ".join(
            f"{reference_id}: {', '.join(sorted(set(ids)))}"
            for reference_id, ids in sorted(missing.items())
        )
        raise ValueError(
            "Referenced structures must include CatBench relpaths: " + problems
        )


def write_catbench_dataset(
    *,
    bundle: DatasetBundle,
    dest: Path,
    coeff_setting: dict[str, dict[str, int | float]],
    output_dir: Path,
    output_name: str,
) -> Path:
    bundle.require_geometry_complete_references()
    _require_catbench_reference_paths(bundle)
    materialize_catbench_layout(bundle, dest)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}_adsorption.json"

    if not _has_materialized_reference_energies(bundle, dest):
        _write_partial_energy_catbench_json(
            bundle=bundle,
            coeff_setting=coeff_setting,
            output_path=output_path,
        )
        _attach_reference_metadata(bundle=bundle, output_path=output_path)
        return output_path

    catbench_vasp.get_raw_data_directory = lambda: str(output_dir)
    catbench_vasp.get_raw_data_path = lambda _benchmark_name: str(output_path)
    catbench_vasp.vasp_preprocessing(
        dataset_name=str(dest),
        coeff_setting=coeff_setting,
    )
    _attach_reference_metadata(bundle=bundle, output_path=output_path)
    return output_path


def _has_materialized_reference_energies(bundle: DatasetBundle, dest: Path) -> bool:
    for reference in bundle.references:
        for _, structure in reference.energy_complete_components():
            relpath = structure.metadata.get("catbench_relpath")
            if not isinstance(relpath, str):
                return False
            if not (dest / relpath / "OSZICAR").is_file():
                return False
    return True


def _write_partial_energy_catbench_json(
    *,
    bundle: DatasetBundle,
    coeff_setting: dict[str, dict[str, int | float]],
    output_path: Path,
) -> None:
    data: dict[str, Any] = {}
    for reference in bundle.references:
        if reference.adslab is None or reference.slab is None:
            continue
        coeff = _reference_coefficients(reference, coeff_setting)
        adsorbate_name = _reference_adsorbate_name(reference)
        surface_name = reference.id.replace(":", "_")
        reaction_key = f"{surface_name}_{adsorbate_name}"
        raw = {
            "star": {
                "stoi": coeff["slab"],
                "atoms": _atoms_from_structure(reference.slab),
                "energy_ref": reference.slab.energy_ev,
            },
            f"{adsorbate_name}star": {
                "stoi": coeff["adslab"],
                "atoms": _atoms_from_structure(reference.adslab),
                "energy_ref": reference.adslab.energy_ev,
            },
        }
        for gas in reference.gas:
            gas_key = f"{gas.formula or gas.label or gas.id.removeprefix('gas:')}gas"
            if gas_key not in coeff:
                continue
            raw[gas_key] = {
                "stoi": coeff[gas_key],
                "atoms": _atoms_from_structure(gas),
                "energy_ref": gas.energy_ev,
            }

        adsorbate_indices = _reference_adsorbate_indices(reference)
        entry: dict[str, Any] = {
            "raw": raw,
            "ref_ads_eng": _reference_energy(raw),
        }
        if isinstance(adsorbate_indices, list):
            entry["adsorbate_indices"] = adsorbate_indices
        data[reaction_key] = entry

    save_catbench_json(data, str(output_path))


def _attach_reference_metadata(*, bundle: DatasetBundle, output_path: Path) -> None:
    if not output_path.is_file():
        return

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected CatBench JSON at {output_path} to be an object, got {type(payload).__name__}"
        )

    for reference in bundle.references:
        reaction_key = _reaction_key(reference)
        entry = payload.get(reaction_key)
        if not isinstance(entry, dict):
            continue
        entry["metadata"] = _reference_metadata_payload(reference)

    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _reference_metadata_payload(reference) -> dict[str, Any]:
    return {
        "reference": _json_safe_metadata(reference.metadata),
        "structures": {
            "slab": _json_safe_metadata(reference.slab.metadata if reference.slab is not None else {}),
            "adslab": _json_safe_metadata(reference.adslab.metadata if reference.adslab is not None else {}),
            "gas": {
                gas.id: _json_safe_metadata(gas.metadata)
                for gas in reference.gas
            },
        },
    }


def _json_safe_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(item) for item in value]
    return str(value)


def _reaction_key(reference) -> str:
    adsorbate_name = _reference_adsorbate_name(reference)
    surface_name = reference.id.replace(":", "_")
    return f"{surface_name}_{adsorbate_name}"


def _reference_coefficients(
    reference,
    coeff_setting: dict[str, dict[str, int | float]],
) -> dict[str, int | float]:
    if reference.adslab is None or reference.adslab.formula is None:
        raise ValueError(f"ReferenceSet '{reference.id}' is missing adslab formula")
    if reference.adslab.formula not in coeff_setting:
        raise ValueError(
            f"Missing CatBench coefficients for reference formula '{reference.adslab.formula}'"
        )
    return coeff_setting[reference.adslab.formula]


def _reference_energy(raw: dict[str, dict[str, Any]]) -> float | None:
    terms: list[float] = []
    for entry in raw.values():
        energy_ref = entry["energy_ref"]
        if energy_ref is None:
            return None
        terms.append(float(energy_ref) * float(entry["stoi"]))
    return sum(terms)


def _reference_adsorbate_name(reference) -> str:
    explicit = reference.metadata.get("adsorbate")
    if isinstance(explicit, str) and explicit:
        return explicit
    adslab = reference.adslab
    if adslab is None:
        return "adsorbate"
    explicit = adslab.metadata.get("adsorbate")
    if isinstance(explicit, str) and explicit:
        return explicit
    if isinstance(adslab.formula, str) and adslab.formula.startswith("*"):
        return adslab.formula.removeprefix("*")
    if isinstance(adslab.label, str) and adslab.label:
        return adslab.label
    return "adsorbate"


def _reference_adsorbate_indices(reference) -> list[int] | None:
    adslab = reference.adslab
    slab = reference.slab
    if adslab is None or slab is None:
        return None
    explicit = adslab.metadata.get("adsorbate_indices")
    if isinstance(explicit, list) and all(isinstance(index, int) for index in explicit):
        return explicit
    return detect_adsorbate_indices(
        _atoms_from_structure(slab),
        _atoms_from_structure(adslab),
    )
