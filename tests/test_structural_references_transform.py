import unittest

from moira.ingest.models import DatasetBundle, StructureRecord
from moira.ingest.transforms.catbench import build_catbench_coefficients
from moira.ingest.transforms.structural_references import (
    build_diatomic_gas_record,
    synthesize_adsorption_references,
)


class StructuralReferencesTransformTest(unittest.TestCase):
    def test_synthesizes_references_from_adslab_geometry(self) -> None:
        adslab = StructureRecord(
            id="alpha-beta:N:1",
            kind="adslab",
            symbols=["Pt", "Pt", "N"],
            positions=[
                (0.0, 0.0, 0.0),
                (1.5, 0.0, 0.0),
                (0.75, 0.75, 1.2),
            ],
            cell=[
                (5.0, 0.0, 0.0),
                (0.0, 5.0, 0.0),
                (0.0, 0.0, 15.0),
            ],
            pbc=(True, True, True),
            energy_ev=None,
            metadata={
                "system": "alpha-beta",
                "adsorbate_indices": [2],
            },
        )
        bundle = DatasetBundle(name="demo", structures=[adslab])
        gas_record = build_diatomic_gas_record(
            symbol="N",
            formula="N2",
            bond_length=1.10,
        )

        transformed = synthesize_adsorption_references(
            bundle,
            adsorbate_symbol="N",
            gas_record=gas_record,
        )

        records_by_id = {record.id: record for record in transformed.structures}
        self.assertEqual(
            sorted(records_by_id),
            ["alpha-beta:N:1", "alpha-beta:slab", "gas:N2"],
        )

        updated_adslab = records_by_id["alpha-beta:N:1"]
        slab = records_by_id["alpha-beta:slab"]
        gas = records_by_id["gas:N2"]

        self.assertEqual(updated_adslab.formula, "*N")
        self.assertEqual(updated_adslab.metadata["adsorbate"], "N")
        self.assertEqual(slab.kind, "slab")
        self.assertEqual(slab.symbols, ["Pt", "Pt"])
        self.assertIsNone(slab.energy_ev)
        self.assertEqual(gas.kind, "gas")
        self.assertEqual(gas.formula, "N2")
        self.assertEqual(gas.symbols, ["N", "N"])
        self.assertEqual(gas.pbc, (False, False, False))
        self.assertIsNone(gas.energy_ev)

        self.assertEqual(len(transformed.references), 1)
        reference = transformed.references[0]
        self.assertIs(reference.adslab, updated_adslab)
        self.assertIs(reference.slab, slab)
        self.assertEqual(reference.gas, [gas])

        coeff_setting = build_catbench_coefficients(
            transformed,
            elements=["N"],
            basis_species=["N2"],
        )
        self.assertEqual(
            coeff_setting,
            {"*N": {"slab": -1, "adslab": 1, "N2gas": -0.5}},
        )

    def test_rejects_missing_inline_geometry(self) -> None:
        adslab = StructureRecord(
            id="alpha-beta:N:1",
            kind="adslab",
            symbols=["Pt", "Pt", "N"],
            positions=[
                (0.0, 0.0, 0.0),
                (1.5, 0.0, 0.0),
                (0.75, 0.75, 1.2),
            ],
            pbc=(True, True, True),
            metadata={
                "system": "alpha-beta",
                "adsorbate_indices": [2],
            },
        )
        bundle = DatasetBundle(name="demo", structures=[adslab])

        with self.assertRaisesRegex(
            ValueError,
            "Adslab 'alpha-beta:N:1' is missing inline geometry fields: cell",
        ):
            synthesize_adsorption_references(
                bundle,
                adsorbate_symbol="N",
                gas_record=build_diatomic_gas_record(
                    symbol="N",
                    formula="N2",
                    bond_length=1.10,
                ),
            )
