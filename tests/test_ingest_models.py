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
