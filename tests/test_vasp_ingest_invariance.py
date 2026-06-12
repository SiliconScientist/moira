import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from moira.ingest.sources.vasp_mapping import load_vasp_mapping_bundle
from moira.ingest.transforms.catbench import build_catbench_coefficients
from moira.ingest.writers.catbench import write_catbench_dataset


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_sample_vasp_tree(root: Path) -> Path:
    source = root / "systems"
    source.mkdir()

    _write_text(
        root / "mapping.yaml",
        '0001: "*CH4"\n0002: "*OH"\n',
    )

    _write_text(root / "gas" / "0001-gas" / "OSZICAR", "1 F= 0 E0= -1.5 d E =0\n")
    _write_text(root / "gas" / "0001-gas" / "CONTCAR", "ch4 gas\n")
    _write_text(root / "gas" / "0002-gas" / "OSZICAR", "1 F= 0 E0= -2.5 d E =0\n")
    _write_text(root / "gas" / "0002-gas" / "CONTCAR", "oh gas\n")

    _write_text(source / "alpha-beta-0000" / "OSZICAR", "1 F= 0 E0= -10.0 d E =0\n")
    _write_text(source / "alpha-beta-0000" / "CONTCAR", "slab\n")
    _write_text(source / "alpha-beta-0001" / "OSZICAR", "1 F= 0 E0= -12.5 d E =0\n")
    _write_text(source / "alpha-beta-0001" / "CONTCAR", "ch4 adslab 1\n")
    _write_text(source / "alpha-beta-0001b" / "OSZICAR", "1 F= 0 E0= -12.0 d E =0\n")
    _write_text(source / "alpha-beta-0001b" / "CONTCAR", "ch4 adslab 2\n")
    _write_text(source / "alpha-beta-0002" / "OSZICAR", "1 F= 0 E0= -11.0 d E =0\n")
    _write_text(source / "alpha-beta-0002" / "CONTCAR", "oh adslab\n")

    _write_text(source / "alpha-beta-0003" / "OSZICAR", "1 F= 0 E0= -9.0 d E =0\n")
    _write_text(source / "too-short" / "OSZICAR", "1 F= 0 E0= -3.0 d E =0\n")
    _write_text(source / "alpha-beta-0004" / "OSZICAR", "not valid\n")

    return source


class VaspIngestInvarianceTest(unittest.TestCase):
    def test_loader_preserves_current_mapping_and_grouping_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = _build_sample_vasp_tree(Path(tmpdir))

            bundle = load_vasp_mapping_bundle(source, dataset_name="demo")

            self.assertEqual(bundle.name, "demo")
            self.assertEqual(bundle.metadata["loader"], "vasp_mapping")
            self.assertEqual(bundle.metadata["tag_map"], {"0001": "*CH4", "0002": "*OH"})
            self.assertTrue(str(source.parent) in bundle.metadata["mapping_root"])

            records_by_id = {record.id: record for record in bundle.structures}
            self.assertEqual(
                sorted(records_by_id),
                [
                    "alpha-beta:CH4:1",
                    "alpha-beta:CH4:2",
                    "alpha-beta:OH:1",
                    "alpha-beta:slab",
                    "gas:CH4",
                    "gas:OH",
                ],
            )

            self.assertEqual(records_by_id["gas:CH4"].metadata["tag"], "0001")
            self.assertEqual(records_by_id["gas:OH"].metadata["tag"], "0002")
            self.assertEqual(records_by_id["alpha-beta:slab"].kind, "slab")
            self.assertEqual(records_by_id["alpha-beta:CH4:1"].metadata["config_index"], 1)
            self.assertEqual(records_by_id["alpha-beta:CH4:2"].metadata["config_index"], 2)
            self.assertEqual(records_by_id["alpha-beta:OH:1"].label, "OH")

            self.assertEqual(len(bundle.references), 3)
            grouped = {
                ref.id: (
                    ref.slab.id if ref.slab is not None else None,
                    ref.adslab.id if ref.adslab is not None else None,
                    [gas.id for gas in ref.gas],
                )
                for ref in bundle.references
            }
            self.assertEqual(
                grouped,
                {
                    "alpha-beta:CH4:1": (
                        "alpha-beta:slab",
                        "alpha-beta:CH4:1",
                        ["gas:CH4"],
                    ),
                    "alpha-beta:CH4:2": (
                        "alpha-beta:slab",
                        "alpha-beta:CH4:2",
                        ["gas:CH4"],
                    ),
                    "alpha-beta:OH:1": (
                        "alpha-beta:slab",
                        "alpha-beta:OH:1",
                        ["gas:OH"],
                    ),
                },
            )

    def test_transform_preserves_current_gas_coefficient_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = _build_sample_vasp_tree(Path(tmpdir))
            bundle = load_vasp_mapping_bundle(source, dataset_name="demo")

            coeff_setting = build_catbench_coefficients(
                bundle,
                elements=["C", "H", "O"],
                basis_species=["CH4", "H2O", "H2"],
            )

            self.assertEqual(
                coeff_setting,
                {
                    "*CH4": {"slab": -1, "adslab": 1, "CH4gas": -1},
                    "*OH": {"slab": -1, "adslab": 1, "H2Ogas": -1, "H2gas": 0.5},
                },
            )

    def test_loader_ignores_malformed_or_unmapped_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = _build_sample_vasp_tree(Path(tmpdir))

            bundle = load_vasp_mapping_bundle(source, dataset_name="demo")
            ids = {record.id for record in bundle.structures}

            self.assertNotIn("alpha-beta:0003:1", ids)
            self.assertNotIn("too-short:slab", ids)
            self.assertEqual(len(bundle.structures), 6)

    def test_writer_emits_current_layout_and_preprocessing_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = _build_sample_vasp_tree(root)
            bundle = load_vasp_mapping_bundle(source, dataset_name="demo")
            coeff_setting = build_catbench_coefficients(
                bundle,
                elements=["C", "H", "O"],
                basis_species=["CH4", "H2O", "H2"],
            )
            dest = root / "catbench"
            output_dir = root / "raw"

            with patch("moira.ingest.writers.catbench.catbench_vasp.vasp_preprocessing") as preprocess:
                output_path = write_catbench_dataset(
                    bundle=bundle,
                    dest=dest,
                    coeff_setting=coeff_setting,
                    output_dir=output_dir,
                    output_name="demo",
                )

            self.assertEqual(output_path, output_dir / "demo_adsorption.json")
            self.assertEqual(
                (dest / "gas" / "CH4gas" / "CONTCAR").read_text(encoding="utf-8"),
                "ch4 gas\n",
            )
            self.assertEqual(
                (dest / "alpha-beta" / "slab" / "CONTCAR").read_text(encoding="utf-8"),
                "slab\n",
            )
            self.assertEqual(
                (dest / "alpha-beta" / "CH4" / "2" / "CONTCAR").read_text(encoding="utf-8"),
                "ch4 adslab 2\n",
            )
            self.assertEqual(
                (dest / "alpha-beta" / "OH" / "1" / "CONTCAR").read_text(encoding="utf-8"),
                "oh adslab\n",
            )
            self.assertFalse((dest / "alpha-beta" / "0003").exists())
            preprocess.assert_called_once_with(
                dataset_name=str(dest),
                coeff_setting={
                    "*CH4": {"slab": -1, "adslab": 1, "CH4gas": -1},
                    "*OH": {"slab": -1, "adslab": 1, "H2Ogas": -1, "H2gas": 0.5},
                },
            )
