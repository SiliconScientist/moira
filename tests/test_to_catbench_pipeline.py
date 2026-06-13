import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ase import Atoms
from ase.db import connect

from moira.ingest.models import StructureRecord
from moira.ingest.site_constraints import atoms_from_atoms_json
from moira.ingest.to_catbench import (
    build_coefficients,
    load_dataset,
    load_elemental_ase_db_dataset,
    write_dataset,
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class ToCatbenchPipelineTest(unittest.TestCase):
    def test_elemental_ase_db_dataset_preserves_source_metadata_through_reference_synthesis(self) -> None:
        db_path = Path("data/screening/test_n.db")
        self.assertTrue(db_path.is_file())

        bundle = load_elemental_ase_db_dataset(
            db_path,
            adsorbate_symbol="N",
            dataset_name="test-n",
            row_limit=1,
        )

        self.assertEqual(len(bundle.references), 1)
        reference = bundle.references[0]
        adslab = reference.adslab
        slab = reference.slab

        self.assertIsNotNone(adslab)
        self.assertIsNotNone(slab)
        assert adslab is not None
        assert slab is not None

        for metadata in (reference.metadata, adslab.metadata, slab.metadata):
            self.assertEqual(metadata["adslab_id"], "adslab-000001")
            self.assertEqual(metadata["parent_slab_id"], "slab-000004")
            self.assertEqual(metadata["host_element"], "Pt")
            self.assertEqual(metadata["surface_type"], "fcc111")
            self.assertEqual(metadata["supercell_size"], [3, 3, 4])
            self.assertEqual(metadata["swap_indices"], [0, 1])
            self.assertEqual(metadata["swap_elements"], ["Cu", "Ag"])
            self.assertEqual(metadata["top_layer_motif"], "heterodimer")
            self.assertEqual(metadata["initial_site_label"], "top")
            self.assertEqual(
                metadata["initial_site_coordinate"],
                [8.341095230486822, 4.8157335766578715, 18.810475736885053],
            )
            self.assertEqual(metadata["adsorbate"], "N")

        self.assertEqual(adslab.metadata["source_formula"], "AgCuPt34N")
        self.assertEqual(adslab.metadata["adsorbate_symbol"], "N")
        self.assertEqual(slab.metadata["synthesized_from"], adslab.id)
        self.assertEqual(reference.metadata["reference_transform"], "structural_references")

    def test_end_to_end_elemental_n_ase_db_to_catbench_layout_uses_real_db(self) -> None:
        db_path = Path("data/screening/trimetallic_n.db")
        self.assertTrue(db_path.is_file())

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dest = root / "catbench"
            bundle = load_elemental_ase_db_dataset(
                db_path,
                adsorbate_symbol="N",
                dataset_name="demo",
                row_limit=1,
            )
            coeff_setting = build_coefficients(
                SimpleNamespace(
                    ingest=SimpleNamespace(
                        stoich=SimpleNamespace(
                            elements=["N"],
                            basis_species=["N2"],
                        )
                    )
                ),
                bundle,
            )
            cfg = SimpleNamespace(
                ingest=SimpleNamespace(
                    dataset_name="demo",
                    catbench_folder=dest,
                )
            )

            self.assertEqual(bundle.metadata["loader"], "ase_db")
            self.assertEqual(bundle.metadata["reference_transform"], "structural_references")
            self.assertEqual(len(bundle.references), 1)
            self.assertEqual(coeff_setting, {"*N": {"slab": -1, "adslab": 1, "N2gas": -0.5}})

            records_by_id = {record.id: record for record in bundle.structures}
            self.assertIn("gas:N2", records_by_id)
            self.assertIn("trimetallic_n:1", records_by_id)
            self.assertIn("trimetallic_n:1:slab", records_by_id)
            self.assertIsNone(records_by_id["gas:N2"].energy_ev)
            self.assertIsNone(records_by_id["trimetallic_n:1"].energy_ev)
            self.assertIsNone(records_by_id["trimetallic_n:1:slab"].energy_ev)

            with patch("moira.ingest.writers.catbench.catbench_vasp.vasp_preprocessing") as preprocess:
                output_path = write_dataset(
                    cfg,
                    bundle=bundle,
                    coeff_setting=coeff_setting,
                )

            self.assertEqual(output_path.name, "demo_adsorption.json")
            self.assertTrue((dest / "trimetallic_n_1" / "slab" / "CONTCAR").is_file())
            self.assertTrue((dest / "trimetallic_n_1" / "N" / "1" / "CONTCAR").is_file())
            self.assertTrue((dest / "gas" / "N2gas" / "CONTCAR").is_file())
            self.assertFalse((dest / "trimetallic_n_1" / "slab" / "OSZICAR").exists())
            self.assertFalse((dest / "trimetallic_n_1" / "N" / "1" / "OSZICAR").exists())
            self.assertFalse((dest / "gas" / "N2gas" / "OSZICAR").exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(payload), ["trimetallic_n_1_N"])
            entry = payload["trimetallic_n_1_N"]
            self.assertIsNone(entry["ref_ads_eng"])
            self.assertEqual(entry["adsorbate_indices"], [36])
            self.assertEqual(sorted(entry["raw"]), ["N2gas", "Nstar", "star"])
            self.assertIsNone(entry["raw"]["star"]["energy_ref"])
            self.assertIsNone(entry["raw"]["Nstar"]["energy_ref"])
            self.assertIsNone(entry["raw"]["N2gas"]["energy_ref"])
            slab_atoms = atoms_from_atoms_json(entry["raw"]["star"]["atoms_json"])
            adslab_atoms = atoms_from_atoms_json(entry["raw"]["Nstar"]["atoms_json"])
            self.assertEqual(len(slab_atoms.constraints), 1)
            self.assertEqual(
                slab_atoms.constraints[0].index.tolist(),
                list(range(18, 36)),
            )
            self.assertEqual(len(adslab_atoms.constraints), 1)
            self.assertEqual(
                adslab_atoms.constraints[0].index.tolist(),
                list(range(18, 36)),
            )
            preprocess.assert_not_called()

    def test_elemental_ase_db_json_output_includes_reference_metadata(self) -> None:
        db_path = Path("data/screening/test_n.db")
        self.assertTrue(db_path.is_file())

        bundle = load_elemental_ase_db_dataset(
            db_path,
            adsorbate_symbol="N",
            dataset_name="test-n",
            row_limit=1,
        )
        coeff_setting = build_coefficients(
            SimpleNamespace(
                ingest=SimpleNamespace(
                    stoich=SimpleNamespace(
                        elements=["N"],
                        basis_species=["N2"],
                    )
                )
            ),
            bundle,
        )
        cfg = SimpleNamespace(
            ingest=SimpleNamespace(
                dataset_name="test-n",
                catbench_folder=None,
            )
        )

        with patch("moira.ingest.writers.catbench.catbench_vasp.vasp_preprocessing") as preprocess:
            output_path = write_dataset(
                cfg,
                bundle=bundle,
                coeff_setting=coeff_setting,
            )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        entry = payload["test_n_1_N"]

        self.assertEqual(
            entry["metadata"]["reference"]["adslab_id"],
            "adslab-000001",
        )
        self.assertEqual(
            entry["metadata"]["reference"]["initial_site_coordinate"],
            [8.341095230486822, 4.8157335766578715, 18.810475736885053],
        )
        self.assertEqual(
            entry["metadata"]["structures"]["slab"]["synthesized_from"],
            "test_n:1",
        )
        self.assertEqual(
            entry["metadata"]["structures"]["adslab"]["source_formula"],
            "AgCuPt34N",
        )
        self.assertEqual(
            entry["metadata"]["structures"]["gas"]["gas:N2"]["reference_species"],
            "N2",
        )
        preprocess.assert_not_called()

    def test_end_to_end_elemental_n_ase_db_writes_json_without_catbench_folder(self) -> None:
        db_path = Path("data/screening/trimetallic_n.db")
        self.assertTrue(db_path.is_file())

        bundle = load_elemental_ase_db_dataset(
            db_path,
            adsorbate_symbol="N",
            dataset_name="demo",
            row_limit=1,
        )
        coeff_setting = build_coefficients(
            SimpleNamespace(
                ingest=SimpleNamespace(
                    stoich=SimpleNamespace(
                        elements=["N"],
                        basis_species=["N2"],
                    )
                )
            ),
            bundle,
        )
        cfg = SimpleNamespace(
            ingest=SimpleNamespace(
                dataset_name="demo",
                catbench_folder=None,
            )
        )

        with patch("moira.ingest.writers.catbench.catbench_vasp.vasp_preprocessing") as preprocess:
            output_path = write_dataset(
                cfg,
                bundle=bundle,
                coeff_setting=coeff_setting,
            )

        self.assertEqual(output_path.name, "demo_adsorption.json")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(sorted(payload), ["trimetallic_n_1_N"])
        preprocess.assert_not_called()

    def test_generic_elemental_adsorption_profile_supports_mixed_adsorbates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "mixed.db"
            with connect(db_path) as db:
                db.write(
                    Atoms(
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
                )
                db.write(
                    Atoms(
                        symbols=["Pt", "Pt", "O"],
                        positions=[
                            (0.0, 0.0, 0.0),
                            (1.5, 0.0, 0.0),
                            (0.75, 0.75, 1.1),
                        ],
                        cell=[
                            (5.0, 0.0, 0.0),
                            (0.0, 5.0, 0.0),
                            (0.0, 0.0, 15.0),
                        ],
                        pbc=(True, True, True),
                    )
                )

            cfg = SimpleNamespace(
                ingest=SimpleNamespace(
                    source=db_path,
                    dataset_name="demo",
                    profile="elemental_adsorption_ase_db",
                )
            )
            bundle = load_dataset(cfg)

            records_by_id = {record.id: record for record in bundle.structures}
            self.assertIn("gas:N2", records_by_id)
            self.assertIn("gas:O2", records_by_id)
            self.assertEqual(
                sorted(reference.metadata["adsorbate"] for reference in bundle.references),
                ["N", "O"],
            )

    def test_end_to_end_elemental_n_ase_db_fails_when_geometry_is_missing(self) -> None:
        db_path = Path("data/screening/trimetallic_n.db")
        self.assertTrue(db_path.is_file())

        bundle = load_elemental_ase_db_dataset(
            db_path,
            adsorbate_symbol="N",
            dataset_name="demo",
            row_limit=1,
        )
        bundle.references[0].slab = StructureRecord(
            id=bundle.references[0].slab.id,  # type: ignore[union-attr]
            kind="slab",
            metadata=bundle.references[0].slab.metadata.copy(),  # type: ignore[union-attr]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimpleNamespace(
                ingest=SimpleNamespace(
                    dataset_name="demo",
                    catbench_folder=Path(tmpdir) / "catbench",
                )
            )
            with self.assertRaisesRegex(
                ValueError,
                "Referenced structures must include geometry: trimetallic_n:1: slab",
            ):
                write_dataset(
                    cfg,
                    bundle=bundle,
                    coeff_setting={"*N": {"slab": -1, "adslab": 1, "N2gas": -0.5}},
                )

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
