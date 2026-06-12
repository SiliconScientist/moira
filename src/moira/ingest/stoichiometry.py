import sympy as sp
from moira.ingest.formulas import formula_to_composition


def build_basis_matrix(elements: list[str], basis_species: list[str]) -> sp.Matrix:
    rows = []
    for element in elements:
        row = []
        for species in basis_species:
            composition = formula_to_composition(species)
            row.append(composition.get(element, 0))
        rows.append(row)

    return sp.Matrix(rows)


def build_b_vector(
    elements: list[str], target_composition: dict[str, int]
) -> sp.Matrix:
    return sp.Matrix([int(target_composition.get(element, 0)) for element in elements])


def solve_stoichiometry(
    *,
    elements: list[str],
    basis_species: list[str],
    target_composition: dict[str, int],
) -> list[float]:
    basis_matrix = build_basis_matrix(elements, basis_species)
    target_vector = build_b_vector(elements, target_composition)
    solution = basis_matrix.LUsolve(target_vector)
    return [float(value) for value in list(solution)]
