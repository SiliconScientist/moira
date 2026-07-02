from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from moira.probe import (
    build_probe_dataset,
    load_adsorbate_template_atoms,
    probe_template,
    raw_adsorbate_structure_key,
    unique_probe_output_path,
    updated_dataset_output_path,
)
from moira.ingest.site_constraints import extract_adsorbed_atom


FIXTURE_DIR = Path(__file__).with_name("fixtures") / "probe"


def _strip_non_probe_fields(payload: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    trimmed: dict[str, dict[str, object]] = {}
    for reaction, entry in payload.items():
        trimmed[reaction] = {
            key: value
            for key, value in entry.items()
            if key != "mlip_feature_matrix"
        }
    return trimmed


class ProbeCompatibilityTest(unittest.TestCase):
    def test_probe_template_uses_methyl_for_carbon(self) -> None:
        template = probe_template("C")

        self.assertEqual(template.symbols, ["C", "H", "H", "H"])
        self.assertEqual(template.raw_star_key, "ch3star")
        self.assertEqual(template.gas_refs, (("ch4gas", -1.0, "CH4"), ("h2gas", 0.5, "H2")))
        self.assertEqual(template.dedup_atom_indices, (0,))

    def test_probe_template_uses_hydroxyl_for_oxygen(self) -> None:
        template = probe_template("O")

        self.assertEqual(template.symbols, ["O", "H"])
        self.assertEqual(template.positions.shape, (2, 3))
        self.assertEqual(template.positions[0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(template.raw_star_key, "OHstar")
        self.assertEqual(template.gas_refs, (("O2gas", -0.5, "O2"), ("h2gas", -0.5, "H2")))
        self.assertEqual(template.dedup_atom_indices, (0,))

    def test_raw_adsorbate_structure_key_supports_non_tolstar_entries(self) -> None:
        entry = {
            "raw": {
                "star": {"atoms_json": "bare"},
                "OHstar": {"atoms_json": "ads"},
                "O2gas": {"atoms_json": "gas"},
                "H2gas": {"atoms_json": "gas"},
            }
        }

        self.assertEqual(raw_adsorbate_structure_key(entry, "surface_OH"), "OHstar")

    def test_bm_style_ref_indirection_is_resolved(self) -> None:
        atoms_json = "{\"1\": {\"numbers\": {\"__ndarray__\": [[1], \"int64\", [1]]}, \"positions\": {\"__ndarray__\": [[1, 3], \"float64\", [0.0, 0.0, 0.0]]}, \"cell\": {\"__ndarray__\": [[3, 3], \"float64\", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]}, \"pbc\": {\"__ndarray__\": [[3], \"bool\", [true, true, true]]}, \"__ase_objtype__\": \"atoms\"}, \"ids\": [1], \"nextid\": 2}"
        payload = {
            "rxn": {
                "raw": {
                    "star": {"ref": "slab-ref"},
                    "Hstar": {"ref": "ads-ref"},
                }
            },
            "_structures": {
                "slab-ref": atoms_json,
                "ads-ref": atoms_json,
            },
        }

        adsorbed = extract_adsorbed_atom(payload["rxn"], "rxn", dataset=payload)
        self.assertEqual(len(adsorbed), 1)

        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "bm_adsorption.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            atoms_list = load_adsorbate_template_atoms(dataset_path)

        self.assertEqual(len(atoms_list), 1)
        self.assertEqual(len(atoms_list[0]), 1)

    def test_output_path_contract_matches_oasis(self) -> None:
        self.assertEqual(
            unique_probe_output_path(Path("KHLOHC_origin_tolstar_adsorption.json")),
            Path("KHLOHC_origin_unique_probe_adsorption.json"),
        )
        self.assertEqual(
            updated_dataset_output_path(Path("KHLOHC_origin_tolstar_adsorption.json")),
            Path("KHLOHC_origin_tolstar_adsorption_with_probe_ids.json"),
        )

    def test_probe_generation_matches_small_oasis_fixture(self) -> None:
        fixture_input = json.loads(
            (FIXTURE_DIR / "fixture_tolstar_adsorption.json").read_text(
                encoding="utf-8"
            )
        )
        expected_updated = json.loads(
            (FIXTURE_DIR / "fixture_tolstar_adsorption_with_probe_ids.json").read_text(
                encoding="utf-8"
            )
        )
        expected_unique = json.loads(
            (FIXTURE_DIR / "fixture_unique_probe_adsorption.json").read_text(
                encoding="utf-8"
            )
        )

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "fixture_tolstar_adsorption.json"
            dataset_path.write_text(
                json.dumps(fixture_input, indent=2) + "\n",
                encoding="utf-8",
            )

            cfg = SimpleNamespace(
                mlip=SimpleNamespace(
                    dataset=str(dataset_path),
                    dev_run=False,
                )
            )
            build_probe_dataset(cfg)

            actual_updated = json.loads(
                updated_dataset_output_path(dataset_path).read_text(encoding="utf-8")
            )
            actual_unique = json.loads(
                unique_probe_output_path(dataset_path).read_text(encoding="utf-8")
            )

        self.assertEqual(
            _strip_non_probe_fields(actual_updated),
            _strip_non_probe_fields(expected_updated),
        )
        self.assertEqual(actual_unique, expected_unique)
