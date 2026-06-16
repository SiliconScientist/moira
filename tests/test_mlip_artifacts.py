from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

from moira.mlip.artifacts import (
    find_result_files,
    load_result_json,
    load_result_analysis,
    load_wide_predictions,
    merge_result_jsons,
    mlip_detail_column_name,
    mlip_energy_column_name,
    mlip_label_column_name,
    model_name_from_result_path,
    result_file_name,
)
from moira.mlip.result_metadata import (
    attach_dataset_metadata_to_result_file,
    enrich_result_file,
    write_efficiency_summary,
)
from moira.mlip.result_parsing import (
    RESULT_ANALYSIS_KEY,
    detect_anomalies_from_result_dict,
    extract_adsorbate,
)


class MlipResultParsingTests(unittest.TestCase):
    def test_extract_adsorbate_parses_product_side(self) -> None:
        self.assertEqual(extract_adsorbate("COgas+*->OH*"), "OH")
        self.assertIsNone(extract_adsorbate("not-a-reaction"))

    def test_detect_anomalies_from_result_dict_returns_expected_fields(self) -> None:
        result = detect_anomalies_from_result_dict(
            {
                "calculation_settings": {
                    "chemical_bond_cutoff": 1.25,
                    "n_crit_relax": 200,
                },
                "rxn-1->OH*": {
                    "reference": {"ads_eng": 1.0},
                    "final": {
                        "median_num": 0,
                        "ads_eng_median": 1.1,
                        "ads_seed_range": 0.0,
                        "ads_eng_seed_range": 0.0,
                    },
                    "0": {
                        "adslab_steps": 50,
                        "substrate_displacement": 0.1,
                        "max_bond_change": 5.0,
                    },
                    "single_calculation": {"ads_eng": 1.15},
                },
            }
        )

        self.assertEqual(
            result["rxn-1->OH*"],
            {
                "dft_ads_eng": 1.0,
                "mlip_ads_eng_median": 1.1,
                "mlip_ads_eng_single": 1.15,
                "metadata": None,
                "metadata_json": None,
                "label": "normal",
                "labels": [],
                "details": {
                    "slab_conv": 0,
                    "ads_conv": 0,
                    "slab_move": 0,
                    "ads_move": 0,
                    "slab_seed": 0,
                    "ads_seed": 0,
                    "ads_eng_seed": 0,
                    "adsorbate_migration": 0,
                    "energy_anomaly": 0,
                },
            },
        )


class MlipArtifactTests(unittest.TestCase):
    def test_naming_helpers_match_artifact_conventions(self) -> None:
        self.assertEqual(result_file_name("mace"), "mace_result.json")
        self.assertEqual(model_name_from_result_path(Path("x/mace_result.json")), "mace")
        self.assertEqual(mlip_energy_column_name("mace"), "mace_mlip_ads_eng_median")
        self.assertEqual(mlip_label_column_name("mace"), "mace_label")
        self.assertEqual(
            mlip_detail_column_name("mace", "ads_move"),
            "mace_ads_move",
        )

    def test_find_and_load_result_json(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            model_dir = base_dir / "mace"
            model_dir.mkdir()
            result_path = model_dir / "mace_result.json"
            result_path.write_text(json.dumps({"rxn-1": {"final": {}}}), encoding="utf-8")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                files = find_result_files(base_dir)
            payload = load_result_json(result_path)

        self.assertEqual(files, [result_path])
        self.assertEqual(payload, {"rxn-1": {"final": {}}})
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, DeprecationWarning)

    def test_load_wide_predictions_builds_expected_columns(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            result_path = base_dir / "mace_result.json"
            result_path.write_text(
                json.dumps(
            {
                "calculation_settings": {
                    "chemical_bond_cutoff": 1.25,
                    "n_crit_relax": 200,
                },
                "rxn-1->OH*": {
                    "metadata": {
                        "adslab_id": "adslab-1",
                        "surface_type": "fcc111",
                    },
                    "reference": {"ads_eng": 1.0},
                    "final": {
                        "median_num": 0,
                                "ads_eng_median": 1.1,
                                "ads_seed_range": 0.0,
                                "ads_eng_seed_range": 0.0,
                            },
                            "0": {
                                "adslab_steps": 50,
                                "substrate_displacement": 0.1,
                                "max_bond_change": 5.0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            enrich_result_file(dataset_path=None, result_path=result_path)

            wide_df = load_wide_predictions([str(result_path)])

        self.assertEqual(wide_df.get_column("reaction").to_list(), ["rxn-1->OH*"])
        self.assertEqual(wide_df.get_column("adsorbate").to_list(), ["OH"])
        self.assertEqual(wide_df.get_column("reference_ads_eng").to_list(), [1.0])
        self.assertEqual(
            wide_df.get_column("reaction_metadata_json").to_list(),
            ['{"adslab_id": "adslab-1", "surface_type": "fcc111"}'],
        )
        self.assertEqual(wide_df.get_column("mace_mlip_ads_eng_median").to_list(), [1.1])
        self.assertEqual(wide_df.get_column("mace_label").to_list(), ["normal"])

    def test_load_result_analysis_falls_back_for_unenriched_results(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "mace_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "calculation_settings": {
                            "chemical_bond_cutoff": 1.25,
                            "n_crit_relax": 200,
                        },
                        "rxn-1->OH*": {
                            "reference": {"ads_eng": 1.0},
                            "final": {
                                "median_num": 0,
                                "ads_eng_median": 1.1,
                                "ads_seed_range": 0.0,
                                "ads_eng_seed_range": 0.0,
                            },
                            "0": {
                                "adslab_steps": 50,
                                "substrate_displacement": 0.1,
                                "max_bond_change": 5.0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            analysis = load_result_analysis(result_path)

        self.assertEqual(analysis["rxn-1->OH*"]["label"], "normal")

    def test_merge_result_jsons_combines_shard_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard_a = root / "mace-shard-a.json"
            shard_b = root / "mace-shard-b.json"
            merged_path = root / "merged" / "mace_result.json"
            shard_a.write_text(
                json.dumps(
                    {
                        "calculation_settings": {"n_crit_relax": 200},
                        "rxn-1->OH*": {"final": {"ads_eng_median": 1.1}},
                    }
                ),
                encoding="utf-8",
            )
            shard_b.write_text(
                json.dumps(
                    {
                        "calculation_settings": {"n_crit_relax": 200},
                        "rxn-2->NH*": {"final": {"ads_eng_median": 2.2}},
                    }
                ),
                encoding="utf-8",
            )

            merged = merge_result_jsons([shard_a, shard_b], output_path=merged_path)
            written = json.loads(merged_path.read_text(encoding="utf-8"))

        self.assertEqual(
            merged,
            {
                "calculation_settings": {"n_crit_relax": 200},
                "rxn-1->OH*": {"final": {"ads_eng_median": 1.1}},
                "rxn-2->NH*": {"final": {"ads_eng_median": 2.2}},
            },
        )
        self.assertEqual(written, merged)

    def test_merge_result_jsons_rejects_duplicate_reactions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard_a = root / "mace-shard-a.json"
            shard_b = root / "mace-shard-b.json"
            payload = {
                "calculation_settings": {"n_crit_relax": 200},
                "rxn-1->OH*": {"final": {"ads_eng_median": 1.1}},
            }
            shard_a.write_text(json.dumps(payload), encoding="utf-8")
            shard_b.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "Duplicate reaction key across shard results",
            ):
                merge_result_jsons([shard_a, shard_b])

    def test_enrich_result_file_copies_reaction_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_path = root / "dataset.json"
            result_path = root / "mace_result.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "rxn-1->OH*": {
                            "metadata": {
                                "adslab_id": "adslab-1",
                                "surface_type": "fcc111",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            result_path.write_text(
                json.dumps(
                    {
                        "calculation_settings": {"chemical_bond_cutoff": 1.25},
                        "rxn-1->OH*": {
                            "reference": {"ads_eng": 1.0},
                            "final": {"ads_eng_median": 1.1, "median_num": 0},
                            "0": {
                                "adslab_steps": 5,
                                "substrate_displacement": 0.1,
                                "max_bond_change": 5.0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            enrich_result_file(
                dataset_path=dataset_path,
                result_path=result_path,
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["rxn-1->OH*"]["metadata"],
            {"adslab_id": "adslab-1", "surface_type": "fcc111"},
        )
        self.assertEqual(
            payload["rxn-1->OH*"][RESULT_ANALYSIS_KEY],
            {
                "dft_ads_eng": 1.0,
                "mlip_ads_eng_median": 1.1,
                "mlip_ads_eng_single": None,
                "metadata": {
                    "adslab_id": "adslab-1",
                    "surface_type": "fcc111",
                },
                "metadata_json": (
                    '{"adslab_id": "adslab-1", "surface_type": "fcc111"}'
                ),
                "label": "normal",
                "labels": [],
                "details": {
                    "slab_conv": 0,
                    "ads_conv": 0,
                    "slab_move": 0,
                    "ads_move": 0,
                    "slab_seed": 0,
                    "ads_seed": 0,
                    "ads_eng_seed": 0,
                    "adsorbate_migration": 0,
                    "energy_anomaly": 0,
                },
            },
        )

    def test_enrich_result_file_falls_back_to_dataset_name_and_flat_result_path(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "data" / "raw_data"
            raw_dir.mkdir(parents=True)
            dataset_path = raw_dir / "test_n_adsorption.json"
            result_dir = root / "data" / "results" / "7net-omni"
            result_dir.mkdir(parents=True)
            flat_result_path = result_dir.parent / "7net-omni_result.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "adslab-000001": {
                            "metadata": {
                                "adslab_id": "adslab-000001",
                                "surface_type": "fcc111",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            flat_result_path.write_text(
                json.dumps(
                    {
                        "calculation_settings": {"chemical_bond_cutoff": 1.25},
                        "adslab-000001": {
                            "reference": {"ads_eng": 1.0},
                            "final": {"ads_eng_median": 1.1, "median_num": 0},
                            "0": {
                                "adslab_steps": 5,
                                "substrate_displacement": 0.1,
                                "max_bond_change": 5.0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch("pathlib.Path.cwd", return_value=root):
                enrich_result_file(
                    dataset_path=None,
                    dataset_name="test_n",
                    result_path=result_dir / "7net-omni_result.json",
                    mlip_name="7net-omni",
                )
                payload = json.loads(flat_result_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["adslab-000001"]["metadata"],
            {"adslab_id": "adslab-000001", "surface_type": "fcc111"},
        )
        self.assertIn(RESULT_ANALYSIS_KEY, payload["adslab-000001"])

    def test_enrich_result_file_persists_analysis_without_dataset(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            result_path = root / "mace_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "calculation_settings": {"chemical_bond_cutoff": 1.25},
                        "rxn-1->OH*": {
                            "reference": {"ads_eng": 1.0},
                            "final": {"ads_eng_median": 3.5, "median_num": 0},
                            "0": {
                                "adslab_steps": 5,
                                "substrate_displacement": 0.1,
                                "max_bond_change": 5.0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            enrich_result_file(
                dataset_path=None,
                result_path=result_path,
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["rxn-1->OH*"][RESULT_ANALYSIS_KEY]["label"],
            "energy_anomaly",
        )

    def test_write_efficiency_summary_captures_runtime_quantities(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_path = root / "dataset.json"
            result_path = root / "mace_result.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "rxn-1->OH*": {"metadata": {"id": 1}},
                        "rxn-2->NH*": {"metadata": {"id": 2}},
                    }
                ),
                encoding="utf-8",
            )
            result_path.write_text(
                json.dumps(
                    {
                        "calculation_settings": {"chemical_bond_cutoff": 1.25},
                        "rxn-1->OH*": {
                            "final": {
                                "time_total_slab": 2.0,
                                "time_total_adslab": 3.0,
                                "steps_total_slab": 4,
                                "steps_total_adslab": 6,
                                "step_weighted_atoms": 10.0,
                                "time_per_step_per_atom": 0.05,
                            }
                        },
                        "rxn-2->NH*": {
                            "final": {
                                "time_total_slab": 1.0,
                                "time_total_adslab": 5.0,
                                "steps_total_slab": 2,
                                "steps_total_adslab": 8,
                                "step_weighted_atoms": 12.0,
                                "time_per_step_per_atom": 0.04,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary_path = write_efficiency_summary(
                dataset_path=dataset_path,
                dataset_name="example",
                result_path=result_path,
                mlip_name="mace",
                model_name="mace",
                task_wall_seconds=20.0,
                shard_index=1,
                shard_count=5,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["dataset_reaction_count"], 2)
        self.assertEqual(summary["completed_reaction_count"], 2)
        self.assertEqual(summary["task_wall_seconds"], 20.0)
        self.assertEqual(summary["reactions_per_hour_wall"], 360.0)
        self.assertEqual(summary["total_relaxation_time_seconds"], 11.0)
        self.assertEqual(summary["total_relaxation_steps"], 20)
        self.assertEqual(summary["total_atom_steps"], 220.0)
        self.assertEqual(summary["mean_relaxation_steps_per_reaction"], 10.0)
        self.assertEqual(summary["shard_index"], 1)
        self.assertEqual(summary["shard_count"], 5)

    def test_attach_dataset_metadata_to_result_file_remains_alias(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            result_path = root / "mace_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "calculation_settings": {"chemical_bond_cutoff": 1.25},
                        "rxn-1->OH*": {
                            "reference": {"ads_eng": 1.0},
                            "final": {"ads_eng_median": 1.1, "median_num": 0},
                            "0": {
                                "adslab_steps": 5,
                                "substrate_displacement": 0.1,
                                "max_bond_change": 5.0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            attach_dataset_metadata_to_result_file(
                dataset_path=None,
                result_path=result_path,
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertIn(RESULT_ANALYSIS_KEY, payload["rxn-1->OH*"])


class DependencyBoundaryTests(unittest.TestCase):
    def test_importing_artifacts_does_not_import_runtime_modules(self) -> None:
        for module_name in (
            "moira.mlip.artifacts",
            "moira.mlip.runner",
            "moira.mlip.tasks",
            "moira.adapters.rootstock_adapter",
        ):
            sys.modules.pop(module_name, None)
        before_import = set(sys.modules)

        importlib.import_module("moira.mlip.artifacts")

        imported = set(sys.modules) - before_import
        self.assertNotIn("moira.mlip.runner", imported)
        self.assertNotIn("moira.mlip.tasks", imported)
        self.assertNotIn("moira.adapters.rootstock_adapter", imported)

    def test_importing_cli_does_not_import_adapter_runtime_modules(self) -> None:
        for module_name in (
            "moira.mlip.cli",
            "moira.mlip.runner",
            "moira.adapters.rootstock_adapter",
        ):
            sys.modules.pop(module_name, None)
        before_import = set(sys.modules)

        importlib.import_module("moira.mlip.cli")

        imported = set(sys.modules) - before_import
        self.assertNotIn("moira.mlip.runner", imported)
        self.assertNotIn("moira.adapters.rootstock_adapter", imported)
