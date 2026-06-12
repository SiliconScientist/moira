import unittest

from moira.ingest.models import DatasetBundle, ReferenceSet, ReactionRecord, StructureRecord


class IngestModelsTest(unittest.TestCase):
    def test_ingest_records_allow_sparse_data(self) -> None:
        structure = StructureRecord(id="adslab-1")
        reference_set = ReferenceSet(id="refs-1", adslab=structure)
        reaction = ReactionRecord(id="rxn-1")
        bundle = DatasetBundle(
            name="trimetallic_n",
            structures=[structure],
            references=[reference_set],
            reactions=[reaction],
        )

        self.assertIsNone(structure.kind)
        self.assertIsNone(structure.positions)
        self.assertIsNone(reference_set.slab)
        self.assertEqual(reference_set.gas, [])
        self.assertIsNone(reaction.references)
        self.assertEqual(reaction.stoichiometry, {})
        self.assertIsNone(bundle.source)

    def test_reference_completeness_distinguishes_energy_and_geometry(self) -> None:
        gas = StructureRecord(id="gas:CH4", source_path="/tmp/gas")
        slab = StructureRecord(id="alpha-beta:slab")
        adslab = StructureRecord(id="alpha-beta:CH4:1", source_path="/tmp/adslab")
        reference_set = ReferenceSet(
            id="refs-1",
            slab=slab,
            adslab=adslab,
            gas=[gas],
        )

        self.assertTrue(reference_set.is_energy_complete())
        self.assertFalse(reference_set.is_geometry_complete())
        self.assertEqual(reference_set.missing_geometry_components(), ["slab"])

        with self.assertRaisesRegex(
            ValueError,
            "ReferenceSet 'refs-1' is missing geometry components: slab",
        ):
            reference_set.require_geometry_complete()

    def test_energy_complete_references_do_not_require_energies(self) -> None:
        gas = StructureRecord(id="gas:CH4", energy_ev=None, source_path="/tmp/gas")
        slab = StructureRecord(id="alpha-beta:slab", energy_ev=None, source_path="/tmp/slab")
        adslab = StructureRecord(
            id="alpha-beta:CH4:1",
            energy_ev=None,
            source_path="/tmp/adslab",
        )
        reference_set = ReferenceSet(
            id="refs-1",
            slab=slab,
            adslab=adslab,
            gas=[gas],
        )

        self.assertTrue(reference_set.is_energy_complete())
        self.assertTrue(reference_set.is_geometry_complete())

    def test_bundle_validation_makes_missing_reference_geometry_explicit(self) -> None:
        gas = StructureRecord(id="gas:CH4", source_path="/tmp/gas")
        slab = StructureRecord(id="alpha-beta:slab")
        adslab = StructureRecord(id="alpha-beta:CH4:1", source_path="/tmp/adslab")
        reference_set = ReferenceSet(
            id="refs-1",
            slab=slab,
            adslab=adslab,
            gas=[gas],
        )
        bundle = DatasetBundle(
            name="demo",
            structures=[gas, slab, adslab],
            references=[reference_set],
        )

        with self.assertRaisesRegex(
            ValueError,
            "Referenced structures must include geometry: refs-1: slab",
        ):
            bundle.require_geometry_complete_references()

    def test_bundle_validation_rejects_missing_referenced_records(self) -> None:
        gas = StructureRecord(id="gas:CH4", source_path="/tmp/gas")
        slab = StructureRecord(id="alpha-beta:slab", source_path="/tmp/slab")
        adslab = StructureRecord(id="alpha-beta:CH4:1", source_path="/tmp/adslab")
        reference_set = ReferenceSet(
            id="refs-1",
            slab=slab,
            adslab=adslab,
            gas=[gas],
        )
        bundle = DatasetBundle(
            name="demo",
            structures=[slab, adslab],
            references=[reference_set],
        )

        with self.assertRaisesRegex(
            ValueError,
            "Referenced structures must be present in bundle.structures: refs-1: gas:CH4",
        ):
            bundle.require_geometry_complete_references()
