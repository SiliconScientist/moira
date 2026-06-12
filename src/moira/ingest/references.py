from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from moira.ingest.formulas import formula_to_composition
from moira.ingest.stoichiometry import solve_stoichiometry


@dataclass
class ReferenceBuildRecord:
    key: str
    formula: str | None = None
    composition: dict[str, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ReferenceStrategy(Protocol):
    def build(
        self,
        *,
        record: ReferenceBuildRecord,
        gas_terms: dict[str, int | float],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CatBenchReferenceStrategy:
    slab_key: str = "slab"
    slab_coefficient: int = -1
    adslab_key: str = "adslab"
    adslab_coefficient: int = 1
    gas_suffix: str = "gas"

    def build(
        self,
        *,
        record: ReferenceBuildRecord,
        gas_terms: dict[str, int | float],
    ) -> dict[str, Any]:
        return {
            self.slab_key: self.slab_coefficient,
            self.adslab_key: self.adslab_coefficient,
            **gas_terms,
        }


def clean_coefficient(value: float, *, eps: float = 1e-12) -> int | float:
    coefficient = float(value)
    if abs(coefficient) < eps:
        return 0

    rounded = round(coefficient)
    if abs(coefficient - rounded) < eps:
        return int(rounded)

    return coefficient


def build_references(
    records: Iterable[ReferenceBuildRecord],
    *,
    elements: list[str],
    basis_species: list[str],
    strategy: ReferenceStrategy,
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    allowed_elements = set(elements)
    gas_suffix = getattr(strategy, "gas_suffix", "gas")

    for record in records:
        composition = record.composition
        if composition is None:
            if record.formula is None:
                raise ValueError(
                    f"Reference record '{record.key}' must define formula or composition"
                )
            composition = formula_to_composition(record.formula)

        extra_elements = set(composition) - allowed_elements
        if extra_elements:
            raise ValueError(
                f"Reference record '{record.key}' includes elements "
                f"{sorted(extra_elements)} not in ingest stoich elements={elements}"
            )

        coefficients = solve_stoichiometry(
            elements=elements,
            basis_species=basis_species,
            target_composition=composition,
        )
        gas_terms = {}
        for species, coefficient in zip(basis_species, coefficients):
            cleaned = clean_coefficient(-coefficient)
            if cleaned != 0:
                gas_terms[f"{species}{gas_suffix}"] = cleaned

        references[record.key] = strategy.build(record=record, gas_terms=gas_terms)

    return references
