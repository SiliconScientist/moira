import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from moira.ingest.models import DatasetBundle, StructureRecord
from moira.ingest.writers.catbench import materialize_catbench_layout, write_catbench_dataset


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class CatbenchWriterTest(unittest.TestCase):
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
