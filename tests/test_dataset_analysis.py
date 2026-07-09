from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ase import Atoms
from ase.io import read
from ase.io.jsonio import encode

from moira.mlip.dataset_analysis import analyze_adsorption_dataset
from moira.mlip.cli import main as mlip_main


def _wrapped_atoms_json(atoms: Atoms) -> str:
    row = json.loads(encode(atoms))
    return json.dumps({"1": row, "ids": [1], "nextid": 2})


class AdsorptionDatasetAnalysisTests(unittest.TestCase):
    def test_analyze_adsorption_dataset_writes_outputs_for_inline_atoms_json(self) -> None:
        dataset_path = Path("data/raw_data/test-n_adsorption.json")
        self.assertTrue(dataset_path.is_file())

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "analysis.csv"
            summary_path = tmp / "summary.json"
            structures_path = tmp / "adslabs.extxyz"

            result = analyze_adsorption_dataset(
                dataset_path,
                csv_output_path=csv_path,
                summary_output_path=summary_path,
                structures_output_path=structures_path,
            )

            self.assertEqual(result["entry_count"], 1)
            rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["reaction"], "adslab-000001")
            self.assertEqual(rows[0]["adsorbate_formula"], "N")
            self.assertEqual(rows[0]["adslab_formula"], "AgCuNPt34")
            self.assertEqual(rows[0]["slab_formula"], "AgCuPt34")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["entry_count"], 1)
            self.assertEqual(summary["adsorbate_formula_counts"], {"N": 1})
            self.assertEqual(summary["slab_formula_counts"], {"AgCuPt34": 1})
            self.assertEqual(summary["adslab_formula_counts"], {"AgCuNPt34": 1})

            frames = read(structures_path, index=":")
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].info["reaction"], "adslab-000001")
            self.assertEqual(frames[0].info["adsorbate_formula"], "N")

    def test_analyze_adsorption_dataset_resolves_bm_style_structure_refs(self) -> None:
        slab = Atoms(
            "Pt2",
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        adslab = slab.copy()
        adslab.extend(
            Atoms(
                "O",
                positions=[(0.5, 0.5, 1.5)],
                cell=slab.cell,
                pbc=slab.pbc,
            )
        )
        dataset = {
            "pt_o": {
                "raw": {
                    "star": {"ref": "slab-ref", "stoi": -1},
                    "Ostar": {"ref": "adslab-ref", "stoi": 1},
                },
                "adsorbate_indices": [2],
            },
            "_structures": {
                "slab-ref": _wrapped_atoms_json(slab),
                "adslab-ref": _wrapped_atoms_json(adslab),
            },
        }

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "bm_adsorption.json"
            csv_path = tmp / "analysis.csv"
            summary_path = tmp / "summary.json"
            structures_path = tmp / "adslabs.extxyz"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

            analyze_adsorption_dataset(
                dataset_path,
                csv_output_path=csv_path,
                summary_output_path=summary_path,
                structures_output_path=structures_path,
            )

            rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["reaction"], "pt_o")
            self.assertEqual(rows[0]["adsorbate_formula"], "O")
            self.assertEqual(rows[0]["slab_formula"], "Pt2")
            self.assertEqual(rows[0]["adslab_formula"], "OPt2")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["adsorbate_element_counts"], {"O": 1})

            frames = read(structures_path, index=":")
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].get_chemical_formula(mode="hill"), "OPt2")

    def test_cli_subcommand_writes_requested_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "analysis.csv"
            summary_path = tmp / "summary.json"
            structures_path = tmp / "adslabs.extxyz"

            mlip_main(
                [
                    "analyze-adsorption-dataset",
                    "--input",
                    "data/raw_data/test-n_adsorption.json",
                    "--csv-out",
                    str(csv_path),
                    "--summary-out",
                    str(summary_path),
                    "--structures-out",
                    str(structures_path),
                ]
            )

            self.assertTrue(csv_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertTrue(structures_path.is_file())
