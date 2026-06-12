from __future__ import annotations

from dataclasses import replace
from typing import Callable

from ase import Atoms
from ase.build import molecule

from moira.ingest.models import DatasetBundle, ReferenceSet, StructureRecord, Vector3
from moira.ingest.site_constraints import strip_adsorbate_from_adslab


def annotate_elemental_adsorption_bundle(
    bundle: DatasetBundle,
    *,
    adsorbate_symbol: str,
    structure_kind: str = "adslab",
) -> DatasetBundle:
    structures = [
        _annotated_adsorption_structure(
            structure,
            adsorbate_symbol=adsorbate_symbol,
            structure_kind=structure_kind,
        )
        for structure in bundle.structures
    ]
    return DatasetBundle(
        name=bundle.name,
        source=bundle.source,
        structures=structures,
        references=list(bundle.references),
        reactions=list(bundle.reactions),
        metadata=dict(bundle.metadata),
    )


def synthesize_adsorption_references(
    bundle: DatasetBundle,
    *,
    adsorbate_symbol: str,
    gas_record: StructureRecord,
    slab_key_fn: Callable[[StructureRecord], str] | None = None,
) -> DatasetBundle:
    synthesized_structures: list[StructureRecord] = []
    synthesized_references: list[ReferenceSet] = []
    slab_records: dict[str, StructureRecord] = {}
    slab_key_for = slab_key_fn or _default_slab_key

    for structure in bundle.structures:
        if structure.kind != "adslab":
            synthesized_structures.append(structure)
            continue

        adslab = _normalized_adslab(structure, adsorbate_symbol=adsorbate_symbol)
        synthesized_structures.append(adslab)

        slab_key = slab_key_for(adslab)
        slab_record = slab_records.get(slab_key)
        if slab_record is None:
            slab_record = _build_slab_record(adslab, slab_key=slab_key)
            slab_records[slab_key] = slab_record
            synthesized_structures.append(slab_record)

        synthesized_references.append(
            ReferenceSet(
                id=adslab.id,
                slab=slab_record,
                adslab=adslab,
                gas=[gas_record],
                metadata={
                    **adslab.metadata,
                    "reference_transform": "structural_references",
                    "adsorbate": adsorbate_symbol,
                },
            )
        )

    if synthesized_references and gas_record.id not in {record.id for record in synthesized_structures}:
        synthesized_structures.append(gas_record)

    return DatasetBundle(
        name=bundle.name,
        source=bundle.source,
        structures=synthesized_structures,
        references=synthesized_references or list(bundle.references),
        reactions=list(bundle.reactions),
        metadata={**bundle.metadata, "reference_transform": "structural_references"},
    )


def build_gas_reference_record(
    *,
    formula: str,
    cell_size: float = 12.0,
) -> StructureRecord:
    atoms = molecule(formula)
    atoms.set_cell((cell_size, cell_size, cell_size))
    atoms.center()
    atoms.pbc = (False, False, False)
    return StructureRecord(
        id=f"gas:{formula}",
        label=formula,
        kind="gas",
        formula=formula,
        symbols=atoms.get_chemical_symbols(),
        positions=_vectors_from_array(atoms.get_positions()),
        cell=_vectors_from_array(atoms.cell.array),
        pbc=tuple(bool(value) for value in atoms.pbc),
        energy_ev=None,
        metadata={
            "synthesized": True,
            "reference_species": formula,
            "reference_transform": "structural_references",
        },
    )


def _normalized_adslab(structure: StructureRecord, *, adsorbate_symbol: str) -> StructureRecord:
    _require_structure_geometry(structure)
    adsorbate_indices = _adsorbate_indices(structure)
    if len(adsorbate_indices) != 1:
        raise ValueError(
            f"Adslab '{structure.id}' must define exactly one adsorbate index"
        )
    adsorbate_index = adsorbate_indices[0]
    symbols = structure.symbols
    if symbols is None or symbols[adsorbate_index] != adsorbate_symbol:
        raise ValueError(
            f"Adslab '{structure.id}' adsorbate atom must be {adsorbate_symbol}"
        )

    metadata = structure.metadata.copy()
    metadata["adsorbate"] = adsorbate_symbol
    return replace(
        structure,
        label=structure.label or adsorbate_symbol,
        formula=structure.formula or f"*{adsorbate_symbol}",
        metadata=metadata,
    )


def _annotated_adsorption_structure(
    structure: StructureRecord,
    *,
    adsorbate_symbol: str,
    structure_kind: str,
) -> StructureRecord:
    symbols = structure.symbols
    if symbols is None:
        raise ValueError(f"Structure '{structure.id}' is missing symbols")
    adsorbate_indices = [
        index for index, symbol in enumerate(symbols) if symbol == adsorbate_symbol
    ]
    if len(adsorbate_indices) != 1:
        raise ValueError(
            f"Structure '{structure.id}' must contain exactly one {adsorbate_symbol} atom"
        )
    metadata = structure.metadata.copy()
    metadata["adsorbate_indices"] = adsorbate_indices
    return replace(
        structure,
        kind=structure_kind,
        metadata=metadata,
    )


def _build_slab_record(adslab: StructureRecord, *, slab_key: str) -> StructureRecord:
    slab_atoms = strip_adsorbate_from_adslab(
        _atoms_from_structure(adslab),
        _adsorbate_indices(adslab),
    )
    metadata = adslab.metadata.copy()
    metadata["synthesized_from"] = adslab.id
    return StructureRecord(
        id=f"{slab_key}:slab",
        label=slab_key,
        kind="slab",
        formula=None,
        symbols=slab_atoms.get_chemical_symbols(),
        positions=_vectors_from_array(slab_atoms.get_positions()),
        cell=_vectors_from_array(slab_atoms.cell.array),
        pbc=tuple(bool(value) for value in slab_atoms.pbc),
        energy_ev=None,
        metadata=metadata,
    )


def _default_slab_key(adslab: StructureRecord) -> str:
    system = adslab.metadata.get("system")
    if isinstance(system, str) and system:
        return system
    return adslab.id


def _adsorbate_indices(structure: StructureRecord) -> list[int]:
    indices = structure.metadata.get("adsorbate_indices")
    if not isinstance(indices, list) or not all(isinstance(index, int) for index in indices):
        raise ValueError(f"Adslab '{structure.id}' is missing metadata['adsorbate_indices']")
    return indices


def _require_structure_geometry(structure: StructureRecord) -> None:
    missing: list[str] = []
    if structure.symbols is None:
        missing.append("symbols")
    if structure.positions is None:
        missing.append("positions")
    if structure.cell is None:
        missing.append("cell")
    if structure.pbc is None:
        missing.append("pbc")
    if missing:
        raise ValueError(
            f"Adslab '{structure.id}' is missing inline geometry fields: {', '.join(missing)}"
        )


def _atoms_from_structure(structure: StructureRecord) -> Atoms:
    _require_structure_geometry(structure)
    assert structure.symbols is not None
    assert structure.positions is not None
    assert structure.cell is not None
    assert structure.pbc is not None
    return Atoms(
        symbols=structure.symbols,
        positions=structure.positions,
        cell=structure.cell,
        pbc=structure.pbc,
    )


def _vectors_from_array(values: object) -> list[Vector3]:
    return [tuple(float(component) for component in row) for row in values]  # type: ignore[arg-type]
