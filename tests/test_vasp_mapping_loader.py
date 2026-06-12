import tempfile
import unittest
from pathlib import Path

from moira.ingest.sources.vasp_mapping import load_vasp_mapping_bundle


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class VaspMappingLoaderTest(unittest.TestCase):
    def test_loader_returns_intermediate_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "systems"
            source.mkdir()

            _write_text(root / "mapping.yaml", "0001: \"*CH4\"\n")
            _write_text(root / "gas" / "0001-gas" / "OSZICAR", "1 F= 0 E0= -1.5 d E =0\n")
            _write_text(root / "gas" / "0001-gas" / "CONTCAR", "gas\n")
            _write_text(source / "alpha-beta-0000" / "OSZICAR", "1 F= 0 E0= -10.0 d E =0\n")
            _write_text(source / "alpha-beta-0000" / "CONTCAR", "slab\n")
            _write_text(source / "alpha-beta-0001" / "OSZICAR", "1 F= 0 E0= -12.5 d E =0\n")
            _write_text(source / "alpha-beta-0001" / "CONTCAR", "adslab\n")
            _write_text(source / "alpha-beta-0002" / "OSZICAR", "not valid\n")

            bundle = load_vasp_mapping_bundle(source, dataset_name="demo")

            self.assertEqual(bundle.name, "demo")
            self.assertEqual(bundle.metadata["loader"], "vasp_mapping")
            self.assertEqual(bundle.metadata["tag_map"], {"0001": "*CH4"})
            self.assertEqual(len(bundle.structures), 3)
            self.assertEqual(len(bundle.references), 1)

            records_by_id = {record.id: record for record in bundle.structures}
            gas_record = records_by_id["gas:CH4"]
            slab_record = records_by_id["alpha-beta:slab"]
            adslab_record = records_by_id["alpha-beta:CH4:1"]

            self.assertEqual(gas_record.kind, "gas")
            self.assertEqual(gas_record.metadata["catbench_relpath"], "gas/CH4gas")
            self.assertEqual(slab_record.energy_ev, -10.0)
            self.assertEqual(adslab_record.kind, "adslab")
            self.assertEqual(adslab_record.formula, "*CH4")
            self.assertEqual(
                adslab_record.metadata["catbench_relpath"],
                "alpha-beta/CH4/1",
            )

            reference = bundle.references[0]
            self.assertIs(reference.slab, slab_record)
            self.assertIs(reference.adslab, adslab_record)
            self.assertEqual(reference.gas, [gas_record])
