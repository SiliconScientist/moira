from __future__ import annotations

from typing import Any

from moira.ingest.references import (
    CatBenchReferenceStrategy,
    ReferenceBuildRecord,
    build_references,
    clean_coefficient,
)


def build_coeff_setting(
    *,
    tag_map: dict[str, str],
    elements: list[str],
    basis_species: list[str],
) -> dict[str, dict[str, Any]]:
    records = [
        ReferenceBuildRecord(
            key=formula,
            formula=formula,
            metadata={"tag": tag},
        )
        for tag, formula in tag_map.items()
    ]
    return build_references(
        records,
        elements=elements,
        basis_species=basis_species,
        strategy=CatBenchReferenceStrategy(),
    )
