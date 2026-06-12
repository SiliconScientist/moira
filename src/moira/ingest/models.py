from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

StructureKind = Literal[
    "bulk",
    "slab",
    "adslab",
    "adsorbate",
    "gas",
    "reference",
]

Vector3 = tuple[float, float, float]


@dataclass
class StructureRecord:
    id: str
    label: str | None = None
    kind: StructureKind | None = None

    formula: str | None = None
    symbols: list[str] | None = None
    positions: list[Vector3] | None = None
    cell: list[Vector3] | None = None
    pbc: tuple[bool, bool, bool] | None = None

    energy_ev: float | None = None

    source_id: str | None = None
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_inline_geometry(self) -> bool:
        return (
            self.positions is not None
            and self.cell is not None
            and self.pbc is not None
        )

    def has_geometry(self) -> bool:
        return self.has_inline_geometry() or self.source_path is not None


@dataclass
class ReferenceSet:
    id: str
    slab: StructureRecord | None = None
    adslab: StructureRecord | None = None
    adsorbate: StructureRecord | None = None
    gas: list[StructureRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def energy_complete_components(self) -> list[tuple[str, StructureRecord]]:
        components: list[tuple[str, StructureRecord]] = []
        if self.slab is not None:
            components.append(("slab", self.slab))
        if self.adslab is not None:
            components.append(("adslab", self.adslab))
        for gas_record in self.gas:
            components.append(("gas", gas_record))
        return components

    def missing_energy_components(self) -> list[str]:
        missing: list[str] = []
        if self.slab is None:
            missing.append("slab")
        if self.adslab is None:
            missing.append("adslab")
        if not self.gas:
            missing.append("gas")
        return missing

    def is_energy_complete(self) -> bool:
        return not self.missing_energy_components()

    def missing_geometry_components(self) -> list[str]:
        missing = self.missing_energy_components()
        for role, structure in self.energy_complete_components():
            if not structure.has_geometry():
                missing.append(structure.id if role == "gas" else role)
        return missing

    def is_geometry_complete(self) -> bool:
        return not self.missing_geometry_components()

    def require_energy_complete(self) -> None:
        missing = self.missing_energy_components()
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"ReferenceSet '{self.id}' is missing energy components: {joined}")

    def require_geometry_complete(self) -> None:
        missing = self.missing_geometry_components()
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"ReferenceSet '{self.id}' is missing geometry components: {joined}")


@dataclass
class ReactionRecord:
    id: str
    adsorbate: str | None = None
    stoichiometry: dict[str, float] = field(default_factory=dict)
    energy_ev: float | None = None
    references: ReferenceSet | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    name: str
    source: str | None = None
    structures: list[StructureRecord] = field(default_factory=list)
    references: list[ReferenceSet] = field(default_factory=list)
    reactions: list[ReactionRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def structure_index(self) -> dict[str, StructureRecord]:
        return {structure.id: structure for structure in self.structures}

    def require_reference_structures_present(self) -> None:
        structure_ids = set(self.structure_index())
        missing: dict[str, list[str]] = {}
        for reference in self.references:
            for _, structure in reference.energy_complete_components():
                if structure.id not in structure_ids:
                    missing.setdefault(reference.id, []).append(structure.id)
        if missing:
            problems = ", ".join(
                f"{reference_id}: {', '.join(sorted(set(ids)))}"
                for reference_id, ids in sorted(missing.items())
            )
            raise ValueError(
                f"Referenced structures must be present in bundle.structures: {problems}"
            )

    def require_geometry_complete_references(self) -> None:
        self.require_reference_structures_present()
        failures: list[str] = []
        for reference in self.references:
            missing = reference.missing_geometry_components()
            if missing:
                failures.append(f"{reference.id}: {', '.join(missing)}")
        if failures:
            raise ValueError(
                "Referenced structures must include geometry: " + "; ".join(failures)
            )
