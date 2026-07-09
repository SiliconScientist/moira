from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write

from moira.ingest.site_constraints import (
    atoms_from_atoms_json,
    extract_adsorbate_indices,
    resolve_structure_atoms_json,
    strip_adsorbate_from_adslab,
)
from moira.mlip.artifacts import load_result_json


def analyze_adsorption_dataset(
    dataset_path: str | Path,
    *,
    csv_output_path: str | Path,
    summary_output_path: str | Path,
    structures_output_path: str | Path,
) -> dict[str, Any]:
    dataset = load_result_json(dataset_path)
    rows: list[dict[str, Any]] = []
    structures: list[Atoms] = []

    adslab_formula_counts: Counter[str] = Counter()
    slab_formula_counts: Counter[str] = Counter()
    adsorbate_formula_counts: Counter[str] = Counter()
    adslab_element_counts: Counter[str] = Counter()
    slab_element_counts: Counter[str] = Counter()
    adsorbate_element_counts: Counter[str] = Counter()

    for reaction, entry in _iter_dataset_entries(dataset):
        adslab = _extract_adslab_atoms(dataset, reaction, entry)
        adsorbate_indices = extract_adsorbate_indices(entry, reaction)
        slab = strip_adsorbate_from_adslab(adslab, adsorbate_indices)
        adsorbate = adslab[adsorbate_indices]

        adslab_formula = adslab.get_chemical_formula(mode="hill")
        slab_formula = slab.get_chemical_formula(mode="hill")
        adsorbate_formula = adsorbate.get_chemical_formula(mode="hill")
        adsorbate_key = _adsorbate_structure_key(entry, reaction)

        row = {
            "reaction": reaction,
            "adsorbate_key": adsorbate_key,
            "adslab_formula": adslab_formula,
            "slab_formula": slab_formula,
            "adsorbate_formula": adsorbate_formula,
            "num_atoms": len(adslab),
            "num_slab_atoms": len(slab),
            "num_adsorbate_atoms": len(adsorbate),
            "slab_elements": ",".join(sorted(set(slab.get_chemical_symbols()))),
            "adsorbate_elements": ",".join(sorted(set(adsorbate.get_chemical_symbols()))),
        }

        metadata = entry.get("metadata")
        if isinstance(metadata, dict):
            reference_metadata = metadata.get("reference")
            if isinstance(reference_metadata, dict):
                for key in (
                    "adslab_id",
                    "parent_slab_id",
                    "host_element",
                    "surface_type",
                    "adsorbate",
                    "adsorbate_symbol",
                    "source_formula",
                ):
                    if key in reference_metadata:
                        row[key] = reference_metadata[key]

        rows.append(row)
        adslab_formula_counts[adslab_formula] += 1
        slab_formula_counts[slab_formula] += 1
        adsorbate_formula_counts[adsorbate_formula] += 1
        adslab_element_counts.update(adslab.get_chemical_symbols())
        slab_element_counts.update(slab.get_chemical_symbols())
        adsorbate_element_counts.update(adsorbate.get_chemical_symbols())

        labeled = adslab.copy()
        labeled.info["reaction"] = reaction
        labeled.info["adsorbate_key"] = adsorbate_key
        labeled.info["adslab_formula"] = adslab_formula
        labeled.info["slab_formula"] = slab_formula
        labeled.info["adsorbate_formula"] = adsorbate_formula
        labeled.info["adsorbate_indices"] = ",".join(str(index) for index in adsorbate_indices)
        structures.append(labeled)

    _write_rows_csv(rows, csv_output_path)
    _write_summary_json(
        dataset_path=dataset_path,
        entry_count=len(rows),
        adslab_formula_counts=adslab_formula_counts,
        slab_formula_counts=slab_formula_counts,
        adsorbate_formula_counts=adsorbate_formula_counts,
        adslab_element_counts=adslab_element_counts,
        slab_element_counts=slab_element_counts,
        adsorbate_element_counts=adsorbate_element_counts,
        output_path=summary_output_path,
    )
    _write_structures(structures, structures_output_path)

    return {
        "dataset_path": str(dataset_path),
        "entry_count": len(rows),
        "csv_output_path": str(csv_output_path),
        "summary_output_path": str(summary_output_path),
        "structures_output_path": str(structures_output_path),
    }


def _iter_dataset_entries(dataset: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for reaction, entry in dataset.items():
        if reaction.startswith("_"):
            continue
        if not isinstance(entry, dict):
            continue
        if not isinstance(entry.get("raw"), dict):
            continue
        entries.append((reaction, entry))
    return entries


def _adsorbate_structure_key(entry: dict[str, Any], reaction: str) -> str:
    raw = entry.get("raw")
    if not isinstance(raw, dict):
        raise TypeError(f"Entry '{reaction}' has non-dict raw payload")

    adsorbate_keys = [key for key in raw if key.endswith("star") and key != "star"]
    if not adsorbate_keys:
        raise ValueError(f"No adsorbate '*star' key found in entry '{reaction}'")
    return adsorbate_keys[0]


def _extract_adslab_atoms(
    dataset: dict[str, Any],
    reaction: str,
    entry: dict[str, Any],
) -> Atoms:
    adsorbate_key = _adsorbate_structure_key(entry, reaction)
    structure_block = entry["raw"][adsorbate_key]
    if not isinstance(structure_block, dict):
        raise ValueError(
            f"Entry '{reaction}' key '{adsorbate_key}' is not a structure dict"
        )
    atoms_json = resolve_structure_atoms_json(
        dataset,
        structure_block,
        reaction=reaction,
        structure_key=adsorbate_key,
    )
    return atoms_from_atoms_json(atoms_json)


def _write_rows_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def _write_summary_json(
    *,
    dataset_path: str | Path,
    entry_count: int,
    adslab_formula_counts: Counter[str],
    slab_formula_counts: Counter[str],
    adsorbate_formula_counts: Counter[str],
    adslab_element_counts: Counter[str],
    slab_element_counts: Counter[str],
    adsorbate_element_counts: Counter[str],
    output_path: str | Path,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_path": str(dataset_path),
        "entry_count": entry_count,
        "adslab_formula_counts": dict(sorted(adslab_formula_counts.items())),
        "slab_formula_counts": dict(sorted(slab_formula_counts.items())),
        "adsorbate_formula_counts": dict(sorted(adsorbate_formula_counts.items())),
        "adslab_element_counts": dict(sorted(adslab_element_counts.items())),
        "slab_element_counts": dict(sorted(slab_element_counts.items())),
        "adsorbate_element_counts": dict(sorted(adsorbate_element_counts.items())),
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def _write_structures(structures: list[Atoms], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not structures:
        raise ValueError("No structures were extracted from the dataset")
    write(target, structures, format="extxyz")
    return target
