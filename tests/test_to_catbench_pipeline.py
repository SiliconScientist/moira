import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ase import Atoms
from ase.db import connect

from moira.ingest.to_catbench import load_dataset, write_dataset


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class ToCatbenchPipelineTest(unittest.TestCase):
    def test_unified_pipeline_loads_elemental_n_ase_db_as_geometry_complete_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trimetallic_n.db"
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

            cfg = SimpleNamespace(
                ingest=SimpleNamespace(
                    source=db_path,
                    dataset_name="demo",
                    profile="elemental_n_ase_db",
                )
            )

            bundle = load_dataset(cfg)

            self.assertEqual(bundle.name, "demo")
            self.assertEqual(bundle.metadata["loader"], "ase_db")
            self.assertEqual(bundle.metadata["reference_transform"], "structural_references")
            self.assertEqual(len(bundle.references), 1)

            records_by_id = {record.id: record for record in bundle.structures}
            self.assertEqual(
                sorted(records_by_id),
                ["gas:N2", "trimetallic_n:1", "trimetallic_n:1:slab"],
            )
            self.assertEqual(records_by_id["trimetallic_n:1"].kind, "adslab")
            self.assertEqual(records_by_id["trimetallic_n:1:slab"].kind, "slab")
            self.assertEqual(records_by_id["gas:N2"].kind, "gas")

            reference = bundle.references[0]
            self.assertTrue(reference.is_geometry_complete())
            self.assertIs(reference.adslab, records_by_id["trimetallic_n:1"])
            self.assertIs(reference.slab, records_by_id["trimetallic_n:1:slab"])
            self.assertEqual(reference.gas, [records_by_id["gas:N2"]])
            self.assertEqual(
                records_by_id["trimetallic_n:1"].metadata["catbench_relpath"],
                "trimetallic_n_1/N/1",
            )
            self.assertEqual(
                records_by_id["trimetallic_n:1:slab"].metadata["catbench_relpath"],
                "trimetallic_n_1/slab",
            )
            self.assertEqual(
                records_by_id["gas:N2"].metadata["catbench_relpath"],
                "gas/N2gas",
            )

    def test_unified_pipeline_writes_elemental_n_layout_without_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db_path = root / "trimetallic_n.db"
            dest = root / "catbench"
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

            cfg = SimpleNamespace(
                ingest=SimpleNamespace(
                    source=db_path,
                    dataset_name="demo",
                    profile="elemental_n_ase_db",
                    catbench_folder=dest,
                )
            )
            bundle = load_dataset(cfg)

            with patch("moira.ingest.writers.catbench.catbench_vasp.vasp_preprocessing") as preprocess:
                output_path = write_dataset(
                    cfg,
                    bundle=bundle,
                    coeff_setting={"*N": {"slab": -1, "adslab": 1, "N2gas": -0.5}},
                )

            self.assertEqual(output_path.name, "demo_adsorption.json")
            self.assertTrue((dest / "trimetallic_n_1" / "slab" / "CONTCAR").is_file())
            self.assertTrue((dest / "trimetallic_n_1" / "N" / "1" / "CONTCAR").is_file())
            self.assertTrue((dest / "gas" / "N2gas" / "CONTCAR").is_file())
            self.assertFalse((dest / "trimetallic_n_1" / "slab" / "OSZICAR").exists())
            self.assertFalse((dest / "trimetallic_n_1" / "N" / "1" / "OSZICAR").exists())
            self.assertFalse((dest / "gas" / "N2gas" / "OSZICAR").exists())
            preprocess.assert_not_called()

    def test_unified_pipeline_dispatches_vasp_mapping_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "systems"
            source.mkdir()

            _write_text(root / "mapping.yaml", '0001: "*CH4"\n')
            _write_text(root / "gas" / "0001-gas" / "OSZICAR", "1 F= 0 E0= -1.5 d E =0\n")
            _write_text(root / "gas" / "0001-gas" / "CONTCAR", "gas\n")
            _write_text(source / "alpha-beta-0000" / "OSZICAR", "1 F= 0 E0= -10.0 d E =0\n")
            _write_text(source / "alpha-beta-0000" / "CONTCAR", "slab\n")
            _write_text(source / "alpha-beta-0001" / "OSZICAR", "1 F= 0 E0= -12.5 d E =0\n")
            _write_text(source / "alpha-beta-0001" / "CONTCAR", "adslab\n")

            cfg = SimpleNamespace(
                ingest=SimpleNamespace(
                    source=source,
                    dataset_name="demo",
                    profile="vasp_mapping",
                )
            )

            bundle = load_dataset(cfg)

            self.assertEqual(bundle.name, "demo")
            self.assertEqual(bundle.metadata["loader"], "vasp_mapping")
            self.assertEqual(sorted(record.id for record in bundle.structures), [
                "alpha-beta:CH4:1",
                "alpha-beta:slab",
                "gas:CH4",
            ])

    def test_unified_pipeline_rejects_unknown_profile(self) -> None:
        cfg = SimpleNamespace(
            ingest=SimpleNamespace(
                source=Path("ignored"),
                dataset_name="demo",
                profile="unknown",
            )
        )

        with self.assertRaisesRegex(ValueError, "Unsupported ingest.profile: unknown"):
            load_dataset(cfg)
