from __future__ import annotations

from typing import Any

from moira.ingest.models import DatasetBundle, ReferenceSet
from moira.ingest.references import (
    CatBenchReferenceStrategy,
    ReferenceBuildRecord,
    ReferenceStrategy,
    build_references,
)


def build_catbench_coefficients(
    bundle: DatasetBundle,
    *,
    elements: list[str],
    basis_species: list[str],
    strategy: ReferenceStrategy | None = None,
) -> dict[str, dict[str, Any]]:
    bundle.require_geometry_complete_references()
    return build_references(
        _reference_build_records(bundle.references),
        elements=elements,
        basis_species=basis_species,
        strategy=strategy or CatBenchReferenceStrategy(),
    )


def _reference_build_records(
    reference_sets: list[ReferenceSet],
) -> list[ReferenceBuildRecord]:
    records: list[ReferenceBuildRecord] = []
    for reference_set in reference_sets:
        formula = _reference_formula(reference_set)
        if formula is None:
            continue
        records.append(
            ReferenceBuildRecord(
                key=formula,
                formula=formula,
                metadata=reference_set.metadata.copy(),
            )
        )
    return records


def _reference_formula(reference_set: ReferenceSet) -> str | None:
    if reference_set.adslab is not None and reference_set.adslab.formula is not None:
        return reference_set.adslab.formula
    if reference_set.adsorbate is not None and reference_set.adsorbate.formula is not None:
        return reference_set.adsorbate.formula
    if reference_set.gas:
        return reference_set.gas[0].formula
    return None
