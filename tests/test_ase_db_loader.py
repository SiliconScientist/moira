import tempfile
import unittest
from pathlib import Path

from ase import Atoms
from ase.db import connect

from moira.ingest.sources.ase_db import load_ase_db_bundle


class AseDbLoaderTest(unittest.TestCase):
    def test_loader_returns_source_structures_without_derived_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "structures.db"
            atoms = Atoms(
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
            )
            with connect(db_path) as db:
                db.write(atoms)

            bundle = load_ase_db_bundle(db_path, dataset_name="demo")

            self.assertEqual(bundle.name, "demo")
            self.assertEqual(bundle.source, str(db_path))
            self.assertEqual(bundle.metadata["loader"], "ase_db")
            self.assertEqual(bundle.metadata["row_count"], 1)
            self.assertEqual(bundle.references, [])
            self.assertEqual(bundle.reactions, [])
            self.assertEqual(len(bundle.structures), 1)

            structure = bundle.structures[0]
            self.assertEqual(structure.id, "structures:1")
            self.assertEqual(structure.label, "Pt2N")
            self.assertIsNone(structure.kind)
            self.assertEqual(structure.formula, "Pt2N")
            self.assertEqual(structure.symbols, ["Pt", "Pt", "N"])
            self.assertEqual(
                structure.positions,
                [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.75, 0.75, 1.2)],
            )
            self.assertEqual(
                structure.cell,
                [(5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 15.0)],
            )
            self.assertEqual(structure.pbc, (True, True, True))
            self.assertIsNone(structure.energy_ev)
            self.assertEqual(structure.source_id, "1")
            self.assertEqual(structure.source_path, str(db_path))
            self.assertEqual(structure.metadata["db_id"], 1)
            self.assertIn("unique_id", structure.metadata)
            self.assertNotIn("adsorbate_indices", structure.metadata)
            self.assertNotIn("catbench_relpath", structure.metadata)
