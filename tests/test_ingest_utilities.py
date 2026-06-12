import unittest

from moira.ingest.catbench_coefficients import build_coeff_setting, clean_coefficient
from moira.ingest.formulas import formula_to_composition
from moira.ingest.stoichiometry import solve_stoichiometry


class FormulaUtilitiesTest(unittest.TestCase):
    def test_formula_to_composition_handles_parentheses_and_decorations(self) -> None:
        self.assertEqual(
            formula_to_composition("*CH3(CH2)2OH"),
            {"C": 3, "H": 8, "O": 1},
        )

    def test_formula_to_composition_rejects_unexpected_numbers(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected number"):
            formula_to_composition("2H")


class StoichiometryUtilitiesTest(unittest.TestCase):
    def test_solve_stoichiometry_uses_explicit_basis(self) -> None:
        coefficients = solve_stoichiometry(
            elements=["C", "H", "O"],
            basis_species=["CO2", "H2O", "H2"],
            target_composition={"C": 1, "H": 4, "O": 1},
        )

        self.assertEqual(coefficients, [1.0, -1.0, 3.0])

    def test_build_coeff_setting_formats_catbench_terms(self) -> None:
        coeff_setting = build_coeff_setting(
            tag_map={"0001": "*CH4"},
            elements=["C", "H"],
            basis_species=["CH4", "H2"],
        )

        self.assertEqual(
            coeff_setting,
            {"*CH4": {"slab": -1, "adslab": 1, "CH4gas": -1}},
        )

    def test_clean_coefficient_snaps_near_integers(self) -> None:
        self.assertEqual(clean_coefficient(-1.0e-13), 0)
        self.assertEqual(clean_coefficient(1.0000000000001), 1)
