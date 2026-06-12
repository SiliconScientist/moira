from __future__ import annotations

from typing import Any

from moira.ingest.formulas import formula_to_composition
from moira.ingest.stoichiometry import solve_stoichiometry


def clean_coefficient(value: float, *, eps: float = 1e-12) -> int | float:
    coefficient = float(value)
    if abs(coefficient) < eps:
        return 0

    rounded = round(coefficient)
    if abs(coefficient - rounded) < eps:
        return int(rounded)

    return coefficient


def build_coeff_setting(
    *,
    tag_map: dict[str, str],
    elements: list[str],
    basis_species: list[str],
) -> dict[str, dict[str, Any]]:
    coeff_setting: dict[str, dict[str, Any]] = {}

    for tag, formula in tag_map.items():
        composition = formula_to_composition(formula)
        extra_elements = set(composition) - set(elements)
        if extra_elements:
            raise ValueError(
                f"Tag {tag} -> '{formula}' includes elements {sorted(extra_elements)} "
                f"not in ingest stoich elements={elements}"
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
                gas_terms[f"{species}gas"] = cleaned

        coeff_setting[formula] = {
            "slab": -1,
            "adslab": 1,
            **gas_terms,
        }

    return coeff_setting
