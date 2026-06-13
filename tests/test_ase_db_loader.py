import tempfile
import unittest
from pathlib import Path

from ase import Atoms
from ase.constraints import FixAtoms
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
            atoms.set_constraint(FixAtoms(indices=[0, 1]))
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
            self.assertEqual(len(structure.constraints), 1)
            self.assertIsInstance(structure.constraints[0], FixAtoms)
            self.assertEqual(structure.constraints[0].index.tolist(), [0, 1])
            self.assertEqual(structure.source_id, "1")
            self.assertEqual(structure.source_path, str(db_path))
            self.assertEqual(structure.metadata["db_id"], 1)
            self.assertIn("unique_id", structure.metadata)
            self.assertNotIn("adsorbate_indices", structure.metadata)
            self.assertNotIn("catbench_relpath", structure.metadata)

    def test_loader_preserves_test_n_db_source_metadata(self) -> None:
        db_path = Path("data/screening/test_n.db")
        self.assertTrue(db_path.is_file())

        bundle = load_ase_db_bundle(
            db_path,
            dataset_name="test-n",
            row_limit=1,
        )

        self.assertEqual(len(bundle.structures), 1)
        structure = bundle.structures[0]

        self.assertEqual(structure.metadata["adslab_id"], "adslab-000001")
        self.assertEqual(structure.metadata["parent_slab_id"], "slab-000004")
        self.assertEqual(structure.metadata["host_element"], "Pt")
        self.assertEqual(structure.metadata["surface_type"], "fcc111")
        self.assertEqual(structure.metadata["supercell_size"], [3, 3, 4])
        self.assertEqual(structure.metadata["swap_indices"], [0, 1])
        self.assertEqual(structure.metadata["swap_elements"], ["Cu", "Ag"])
        self.assertEqual(structure.metadata["top_layer_motif"], "heterodimer")
        self.assertEqual(structure.metadata["initial_site_label"], "top")
        self.assertEqual(
            structure.metadata["initial_site_coordinate"],
            [8.341095230486822, 4.8157335766578715, 18.810475736885053],
        )
        self.assertEqual(structure.metadata["adsorbate"], "N")
