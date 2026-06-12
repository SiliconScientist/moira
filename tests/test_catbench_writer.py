import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from moira.ingest.models import DatasetBundle, ReferenceSet, StructureRecord
from moira.ingest.writers.catbench import materialize_catbench_layout, write_catbench_dataset


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class CatbenchWriterTest(unittest.TestCase):
    def test_write_catbench_dataset_materializes_inline_geometries_without_oszicars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dest = root / "dest"
            output_dir = root / "raw"

            gas = StructureRecord(
                id="gas:N2",
                kind="gas",
                formula="N2",
                symbols=["N", "N"],
                positions=[(-0.55, 0.0, 0.0), (0.55, 0.0, 0.0)],
                cell=[(12.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 12.0)],
                pbc=(False, False, False),
                energy_ev=None,
                metadata={"catbench_relpath": "gas/N2gas"},
            )
            slab = StructureRecord(
                id="surface-1:slab",
                kind="slab",
                symbols=["Pt", "Pt"],
                positions=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
                cell=[(5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 15.0)],
                pbc=(True, True, True),
                energy_ev=None,
                metadata={"catbench_relpath": "surface-1/slab"},
            )
            adslab = StructureRecord(
                id="surface-1:N:1",
                kind="adslab",
                formula="*N",
                symbols=["Pt", "Pt", "N"],
                positions=[
                    (0.0, 0.0, 0.0),
                    (1.5, 0.0, 0.0),
                    (0.75, 0.75, 1.2),
                ],
                cell=[(5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 15.0)],
                pbc=(True, True, True),
                energy_ev=None,
                metadata={"catbench_relpath": "surface-1/N/1"},
            )
            bundle = DatasetBundle(
                name="demo",
                structures=[gas, slab, adslab],
                references=[
                    ReferenceSet(
                        id=adslab.id,
                        slab=slab,
                        adslab=adslab,
                        gas=[gas],
                    )
                ],
            )

            with patch("moira.ingest.writers.catbench.catbench_vasp.vasp_preprocessing") as preprocess:
                output_path = write_catbench_dataset(
                    bundle=bundle,
                    dest=dest,
                    coeff_setting={"*N": {"slab": -1, "adslab": 1, "N2gas": -0.5}},
                    output_dir=output_dir,
                    output_name="demo",
                )

            self.assertEqual(output_path, output_dir / "demo_adsorption.json")
            self.assertTrue((dest / "gas" / "N2gas" / "CONTCAR").is_file())
            self.assertTrue((dest / "surface-1" / "slab" / "CONTCAR").is_file())
            self.assertTrue((dest / "surface-1" / "N" / "1" / "CONTCAR").is_file())
            self.assertFalse((dest / "gas" / "N2gas" / "OSZICAR").exists())
            self.assertFalse((dest / "surface-1" / "slab" / "OSZICAR").exists())
            self.assertFalse((dest / "surface-1" / "N" / "1" / "OSZICAR").exists())
            preprocess.assert_not_called()

    def test_write_catbench_dataset_allows_referenced_geometries_without_energies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gas_source = root / "source" / "gas" / "0001-gas"
            slab_source = root / "source" / "alpha-beta-0000"
            adslab_source = root / "source" / "alpha-beta-0001"
            dest = root / "dest"
            output_dir = root / "raw"

            _write_text(gas_source / "CONTCAR", "gas contcar\n")
            _write_text(gas_source / "OSZICAR", "gas oszicar\n")
            _write_text(slab_source / "CONTCAR", "slab contcar\n")
            _write_text(slab_source / "OSZICAR", "slab oszicar\n")
            _write_text(adslab_source / "CONTCAR", "adslab contcar\n")
            _write_text(adslab_source / "OSZICAR", "adslab oszicar\n")

            gas = StructureRecord(
                id="gas:CH4",
                kind="gas",
                formula="*CH4",
                energy_ev=None,
                source_path=str(gas_source),
                metadata={"catbench_relpath": "gas/CH4gas"},
            )
            slab = StructureRecord(
                id="alpha-beta:slab",
                kind="slab",
                energy_ev=None,
                source_path=str(slab_source),
                metadata={"catbench_relpath": "alpha-beta/slab"},
            )
            adslab = StructureRecord(
                id="alpha-beta:CH4:1",
                kind="adslab",
                formula="*CH4",
                energy_ev=None,
                source_path=str(adslab_source),
                metadata={"catbench_relpath": "alpha-beta/CH4/1"},
            )
            bundle = DatasetBundle(
                name="demo",
                structures=[gas, slab, adslab],
                references=[
                    ReferenceSet(
                        id=adslab.id,
                        slab=slab,
                        adslab=adslab,
                        gas=[gas],
                    )
                ],
            )

            with patch("moira.ingest.writers.catbench.catbench_vasp.vasp_preprocessing") as preprocess:
                output_path = write_catbench_dataset(
                    bundle=bundle,
                    dest=dest,
                    coeff_setting={"*CH4": {"slab": -1, "adslab": 1, "CH4gas": -1}},
                    output_dir=output_dir,
                    output_name="demo",
                )

            self.assertEqual(output_path, output_dir / "demo_adsorption.json")
            self.assertEqual(
                (dest / "gas" / "CH4gas" / "CONTCAR").read_text(encoding="utf-8"),
                "gas contcar\n",
            )
            self.assertEqual(
                (dest / "alpha-beta" / "slab" / "CONTCAR").read_text(encoding="utf-8"),
                "slab contcar\n",
            )
            self.assertEqual(
                (dest / "alpha-beta" / "CH4" / "1" / "CONTCAR").read_text(encoding="utf-8"),
                "adslab contcar\n",
            )
            preprocess.assert_called_once_with(
                dataset_name=str(dest),
                coeff_setting={"*CH4": {"slab": -1, "adslab": 1, "CH4gas": -1}},
            )

    def test_materialize_catbench_layout_copies_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source" / "alpha-beta-0001"
            dest = root / "dest"
            _write_text(source / "CONTCAR", "contcar\n")
            _write_text(source / "OSZICAR", "oszicar\n")
            _write_text(source / "IGNORED", "ignored\n")

            bundle = DatasetBundle(
                name="demo",
                structures=[
                    StructureRecord(
                        id="alpha-beta:CH4:1",
                        source_path=str(source),
                        metadata={"catbench_relpath": "alpha-beta/CH4/1"},
                    )
                ],
            )

            materialize_catbench_layout(bundle, dest)

            self.assertEqual(
                (dest / "alpha-beta" / "CH4" / "1" / "CONTCAR").read_text(encoding="utf-8"),
                "contcar\n",
            )
            self.assertEqual(
                (dest / "alpha-beta" / "CH4" / "1" / "OSZICAR").read_text(encoding="utf-8"),
                "oszicar\n",
            )
            self.assertFalse((dest / "alpha-beta" / "CH4" / "1" / "IGNORED").exists())

    def test_write_catbench_dataset_runs_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source" / "alpha-beta-0001"
            dest = root / "dest"
            output_dir = root / "raw"
            _write_text(source / "CONTCAR", "contcar\n")
            _write_text(source / "OSZICAR", "oszicar\n")

            bundle = DatasetBundle(
                name="demo",
                structures=[
                    StructureRecord(
                        id="alpha-beta:CH4:1",
                        source_path=str(source),
                        metadata={"catbench_relpath": "alpha-beta/CH4/1"},
                    )
                ],
            )

            with patch("moira.ingest.writers.catbench.catbench_vasp.vasp_preprocessing") as preprocess:
                output_path = write_catbench_dataset(
                    bundle=bundle,
                    dest=dest,
                    coeff_setting={"*CH4": {"slab": -1, "adslab": 1, "CH4gas": -1}},
                    output_dir=output_dir,
                    output_name="demo",
                )

            self.assertEqual(output_path, output_dir / "demo_adsorption.json")
            self.assertTrue((dest / "alpha-beta" / "CH4" / "1" / "CONTCAR").exists())
            preprocess.assert_called_once_with(
                dataset_name=str(dest),
                coeff_setting={"*CH4": {"slab": -1, "adslab": 1, "CH4gas": -1}},
            )

    def test_write_catbench_dataset_rejects_missing_geometry_for_referenced_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dest = root / "dest"
            output_dir = root / "raw"

            referenced_gas = StructureRecord(
                id="gas:CH4",
                kind="gas",
                formula="*CH4",
                energy_ev=None,
                metadata={"catbench_relpath": "gas/CH4gas"},
            )
            slab = StructureRecord(
                id="alpha-beta:slab",
                kind="slab",
                energy_ev=None,
                source_path=str(root / "source" / "alpha-beta-0000"),
                metadata={"catbench_relpath": "alpha-beta/slab"},
            )
            adslab = StructureRecord(
                id="alpha-beta:CH4:1",
                kind="adslab",
                formula="*CH4",
                energy_ev=None,
                source_path=str(root / "source" / "alpha-beta-0001"),
                metadata={"catbench_relpath": "alpha-beta/CH4/1"},
            )
            bundle = DatasetBundle(
                name="demo",
                structures=[slab, adslab],
                references=[
                    ReferenceSet(
                        id=adslab.id,
                        slab=slab,
                        adslab=adslab,
                        gas=[referenced_gas],
                    )
                ],
            )

            with self.assertRaisesRegex(
                ValueError,
                "Referenced structures must be present in bundle.structures: alpha-beta:CH4:1: gas:CH4",
            ):
                write_catbench_dataset(
                    bundle=bundle,
                    dest=dest,
                    coeff_setting={"*CH4": {"slab": -1, "adslab": 1, "CH4gas": -1}},
                    output_dir=output_dir,
                    output_name="demo",
                )

    def test_write_catbench_dataset_rejects_referenced_records_without_source_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gas_source = root / "source" / "gas" / "0001-gas"
            slab_source = root / "source" / "alpha-beta-0000"
            dest = root / "dest"
            output_dir = root / "raw"

            _write_text(gas_source / "CONTCAR", "gas contcar\n")
            _write_text(gas_source / "OSZICAR", "gas oszicar\n")
            _write_text(slab_source / "CONTCAR", "slab contcar\n")
            _write_text(slab_source / "OSZICAR", "slab oszicar\n")

            gas = StructureRecord(
                id="gas:CH4",
                kind="gas",
                formula="*CH4",
                energy_ev=None,
                source_path=str(gas_source),
                metadata={"catbench_relpath": "gas/CH4gas"},
            )
            slab = StructureRecord(
                id="alpha-beta:slab",
                kind="slab",
                energy_ev=None,
                source_path=str(slab_source),
                metadata={"catbench_relpath": "alpha-beta/slab"},
            )
            adslab = StructureRecord(
                id="alpha-beta:CH4:1",
                kind="adslab",
                formula="*CH4",
                energy_ev=None,
            )
            bundle = DatasetBundle(
                name="demo",
                structures=[gas, slab, adslab],
                references=[
                    ReferenceSet(
                        id=adslab.id,
                        slab=slab,
                        adslab=adslab,
                        gas=[gas],
                    )
                ],
            )

            with self.assertRaisesRegex(
                ValueError,
                "Referenced structures must include geometry: alpha-beta:CH4:1: adslab",
            ):
                write_catbench_dataset(
                    bundle=bundle,
                    dest=dest,
                    coeff_setting={"*CH4": {"slab": -1, "adslab": 1, "CH4gas": -1}},
                    output_dir=output_dir,
                    output_name="demo",
                )

    def test_write_catbench_dataset_rejects_missing_catbench_relpaths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gas_source = root / "source" / "gas" / "0001-gas"
            slab_source = root / "source" / "alpha-beta-0000"
            adslab_source = root / "source" / "alpha-beta-0001"
            dest = root / "dest"
            output_dir = root / "raw"

            _write_text(gas_source / "CONTCAR", "gas contcar\n")
            _write_text(gas_source / "OSZICAR", "gas oszicar\n")
            _write_text(slab_source / "CONTCAR", "slab contcar\n")
            _write_text(slab_source / "OSZICAR", "slab oszicar\n")
            _write_text(adslab_source / "CONTCAR", "adslab contcar\n")
            _write_text(adslab_source / "OSZICAR", "adslab oszicar\n")

            gas = StructureRecord(
                id="gas:CH4",
                kind="gas",
                formula="*CH4",
                source_path=str(gas_source),
                metadata={"catbench_relpath": "gas/CH4gas"},
            )
            slab = StructureRecord(
                id="alpha-beta:slab",
                kind="slab",
                source_path=str(slab_source),
                metadata={"catbench_relpath": "alpha-beta/slab"},
            )
            adslab = StructureRecord(
                id="alpha-beta:CH4:1",
                kind="adslab",
                formula="*CH4",
                source_path=str(adslab_source),
            )
            bundle = DatasetBundle(
                name="demo",
                structures=[gas, slab, adslab],
                references=[
                    ReferenceSet(
                        id=adslab.id,
                        slab=slab,
                        adslab=adslab,
                        gas=[gas],
                    )
                ],
            )

            with self.assertRaisesRegex(
                ValueError,
                "Referenced structures must include CatBench relpaths: alpha-beta:CH4:1: alpha-beta:CH4:1",
            ):
                write_catbench_dataset(
                    bundle=bundle,
                    dest=dest,
                    coeff_setting={"*CH4": {"slab": -1, "adslab": 1, "CH4gas": -1}},
                    output_dir=output_dir,
                    output_name="demo",
                )
