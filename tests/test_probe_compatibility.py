from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from moira.probe import (
    build_probe_dataset,
    probe_template,
    unique_probe_output_path,
    updated_dataset_output_path,
)


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
