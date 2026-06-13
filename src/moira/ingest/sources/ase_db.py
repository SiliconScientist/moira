from __future__ import annotations

from pathlib import Path

from ase.db import connect

from moira.ingest.models import DatasetBundle, StructureRecord, Vector3


def load_ase_db_bundle(
    source: Path,
    *,
    dataset_name: str | None = None,
    row_limit: int | None = None,
) -> DatasetBundle:
    if not source.is_file():
        raise FileNotFoundError(f"ASE DB source does not exist: {source}")

    structures: list[StructureRecord] = []
    with connect(source) as db:
        for row in db.select(limit=row_limit):
            structures.append(_structure_from_row(row=row, source=source))

    return DatasetBundle(
        name=dataset_name or source.stem,
        source=str(source),
        structures=structures,
        metadata={
            "loader": "ase_db",
            "row_count": len(structures),
        },
    )


def _structure_from_row(*, row, source: Path) -> StructureRecord:
    atoms = row.toatoms()
    return StructureRecord(
        id=f"{source.stem}:{row.id}",
        label=row.formula,
        kind=None,
        formula=row.formula,
        symbols=list(row.symbols),
        positions=_vectors(row.positions.tolist()),
        cell=_vectors(row.cell.tolist()),
        pbc=tuple(bool(value) for value in row.pbc.tolist()),
        energy_ev=getattr(row, "energy", None),
        constraints=atoms.constraints or None,
        source_id=str(row.id),
        source_path=str(source),
        metadata={
            "db_id": row.id,
            "unique_id": getattr(row, "unique_id", None),
        },
    )


def _vectors(values: list[list[float]]) -> list[Vector3]:
    return [tuple(float(component) for component in row) for row in values]
