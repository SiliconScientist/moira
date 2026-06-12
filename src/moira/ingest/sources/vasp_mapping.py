from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

from moira.ingest.models import DatasetBundle, ReferenceSet, StructureRecord


def tag_to_adsorbate_label(tag: str, tag_map: dict[str, str]) -> str:
    return tag_map[tag].replace("*", "")


def read_oszicar_energy(file_path: Path) -> float:
    try:
        with file_path.open("r", encoding="utf-8") as file:
            lines = file.readlines()
            last_line = lines[-1]

        energy = None
        parts = last_line.split()
        for index, word in enumerate(parts):
            if word == "E0=":
                energy = float(parts[index + 1])
                break

        if energy is None:
            raise ValueError(f"Energy value not found in file: {file_path}")

        return energy
    except Exception as exc:
        raise RuntimeError(
            f"An error occurred while reading the file '{file_path}': {exc}"
        ) from exc


def resolve_mapping_root(source: Path) -> Path:
    if (source / "mapping.yaml").is_file():
        return source
    if (source.parent / "mapping.yaml").is_file():
        return source.parent
    raise FileNotFoundError(
        f"Could not find mapping.yaml in {source} or {source.parent}"
    )


def load_tag_map(mapping_root: Path) -> dict[str, str]:
    with (mapping_root / "mapping.yaml").open(encoding="utf-8") as file:
        tag_map = yaml.load(file, Loader=yaml.BaseLoader)
    if not isinstance(tag_map, dict):
        raise TypeError(f"Expected mapping.yaml to contain a mapping, got {type(tag_map)}")
    return {str(tag): str(formula) for tag, formula in tag_map.items()}


def load_vasp_mapping_bundle(
    source: Path,
    *,
    dataset_name: str | None = None,
) -> DatasetBundle:
    if not source.is_dir():
        raise FileNotFoundError(f"System source folder does not exist: {source}")

    mapping_root = resolve_mapping_root(source)
    tag_map = load_tag_map(mapping_root)
    gas_records = _load_gas_records(mapping_root, tag_map)
    system_records, reference_sets = _load_system_records(source, tag_map, gas_records)

    bundle = DatasetBundle(
        name=dataset_name or source.name,
        source=str(source),
        structures=[*gas_records.values(), *system_records],
        references=reference_sets,
        metadata={
            "loader": "vasp_mapping",
            "mapping_root": str(mapping_root),
            "tag_map": tag_map,
        },
    )
    return bundle


def _load_gas_records(
    mapping_root: Path,
    tag_map: dict[str, str],
) -> dict[str, StructureRecord]:
    gas_records: dict[str, StructureRecord] = {}
    gas_root = mapping_root / "gas"
    if not gas_root.is_dir():
        return gas_records

    for folder in sorted(gas_root.iterdir()):
        if not folder.is_dir():
            continue

        tag = folder.name[:4]
        if tag not in tag_map:
            continue

        adsorbate = tag_to_adsorbate_label(tag, tag_map)
        formula = tag_map[tag]
        gas_records[formula] = StructureRecord(
            id=f"gas:{adsorbate}",
            label=adsorbate,
            kind="gas",
            formula=formula,
            energy_ev=_maybe_read_oszicar(folder / "OSZICAR"),
            source_id=folder.name,
            source_path=str(folder),
            metadata={
                "adsorbate": adsorbate,
                "tag": tag,
                "catbench_relpath": f"gas/{adsorbate}gas",
            },
        )

    return gas_records


def _load_system_records(
    source: Path,
    tag_map: dict[str, str],
    gas_records: dict[str, StructureRecord],
) -> tuple[list[StructureRecord], list[ReferenceSet]]:
    records: list[StructureRecord] = []
    references: list[ReferenceSet] = []
    slab_by_system: dict[str, StructureRecord] = {}
    pending_entries: dict[str, list[tuple[str, str, float]]] = defaultdict(list)

    for folder in sorted(source.iterdir()):
        if not folder.is_dir():
            continue

        try:
            energy_ev = read_oszicar_energy(folder / "OSZICAR")
        except RuntimeError:
            continue

        parts = folder.name.split("-")
        if len(parts) < 3:
            continue

        tag = parts[-1][:4]
        system = "-".join(parts[:-1])

        if tag == "0000":
            slab_record = StructureRecord(
                id=f"{system}:slab",
                label=system,
                kind="slab",
                energy_ev=energy_ev,
                source_id=folder.name,
                source_path=str(folder),
                metadata={
                    "system": system,
                    "tag": tag,
                    "catbench_relpath": f"{system}/slab",
                },
            )
            slab_by_system[system] = slab_record
            records.append(slab_record)
            continue

        pending_entries[system].append((tag, folder.name, energy_ev))

    config_counters: dict[tuple[str, str], int] = defaultdict(int)
    for system, entries in pending_entries.items():
        for tag, folder_name, energy_ev in entries:
            if tag not in tag_map:
                continue

            adsorbate = tag_to_adsorbate_label(tag, tag_map)
            config_counters[(system, adsorbate)] += 1
            config_index = config_counters[(system, adsorbate)]
            formula = tag_map[tag]
            adslab_record = StructureRecord(
                id=f"{system}:{adsorbate}:{config_index}",
                label=adsorbate,
                kind="adslab",
                formula=formula,
                energy_ev=energy_ev,
                source_id=folder_name,
                source_path=str(source / folder_name),
                metadata={
                    "system": system,
                    "tag": tag,
                    "adsorbate": adsorbate,
                    "config_index": config_index,
                    "catbench_relpath": f"{system}/{adsorbate}/{config_index}",
                },
            )
            records.append(adslab_record)
            references.append(
                ReferenceSet(
                    id=adslab_record.id,
                    slab=slab_by_system.get(system),
                    adslab=adslab_record,
                    gas=[gas_records[formula]] if formula in gas_records else [],
                    metadata={
                        "system": system,
                        "tag": tag,
                        "adsorbate": adsorbate,
                    },
                )
            )

    return records, references


def _maybe_read_oszicar(file_path: Path) -> float | None:
    try:
        return read_oszicar_energy(file_path)
    except RuntimeError:
        return None
