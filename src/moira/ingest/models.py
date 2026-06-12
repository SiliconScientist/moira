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


@dataclass
class ReferenceSet:
    id: str
    slab: StructureRecord | None = None
    adslab: StructureRecord | None = None
    adsorbate: StructureRecord | None = None
    gas: list[StructureRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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
