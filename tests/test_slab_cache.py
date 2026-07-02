from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
from json import JSONDecodeError
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

from ase import Atoms
from ase.io import write
import numpy as np

from moira.mlip.slab_cache import (
    SlabCacheEntry,
    load_slab_cache_entry,
    slab_cache_entry_path,
    write_slab_cache_entry,
)

VENDOR_CATBENCH = Path(__file__).resolve().parents[1] / "vendor" / "catbench"
if str(VENDOR_CATBENCH) not in sys.path:
    sys.path.insert(0, str(VENDOR_CATBENCH))

for module_name in (
    "catbench",
    "catbench.adsorption",
    "catbench.adsorption.calculation",
    "catbench.adsorption.calculation.calculation",
    "catbench.utils",
    "catbench.utils.data_utils",
):
    sys.modules.pop(module_name, None)

calculation_module = importlib.import_module(
    "catbench.adsorption.calculation.calculation"
)
AdsorptionCalculation = calculation_module.AdsorptionCalculation
load_catbench_json = importlib.import_module("catbench.utils.data_utils").load_catbench_json


class SlabCacheTests(unittest.TestCase):
    @staticmethod
    def _without_cache_diagnostics(payload):
        if isinstance(payload, dict):
            filtered = {}
            for key, value in payload.items():
                if key in {
                    "slab_cache_hit",
                    "slab_cache_hit_count",
                    "saved_slab_time_estimate_seconds",
                    "slab_energy_change",
                }:
                    continue
                filtered[key] = SlabCacheTests._without_cache_diagnostics(value)
            return filtered
        if isinstance(payload, list):
            return [SlabCacheTests._without_cache_diagnostics(item) for item in payload]
        return payload

    def test_slab_cache_entry_round_trips_cleanly(self) -> None:
        slab = Atoms(
            "Cu2",
            positions=[(0.0, 0.0, 0.0), (1.8, 1.8, 0.2)],
            cell=[(3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, False),
        )
        entry = SlabCacheEntry(
            cache_key="abc123",
            slab_energy_ev=-12.5,
            slab_geometry=slab,
            relaxation_steps=24,
            relaxation_time_seconds=3.75,
            displacement_metrics={
                "max_disp": 0.12,
                "mae_mobile": 0.05,
                "rmsd_mobile": 0.06,
            },
        )

        with TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            path = write_slab_cache_entry(cache_dir, entry)
            loaded = load_slab_cache_entry(path)

        self.assertEqual(path, slab_cache_entry_path(cache_dir, "abc123"))
        self.assertEqual(loaded.cache_key, entry.cache_key)
        self.assertEqual(loaded.slab_energy_ev, entry.slab_energy_ev)
        self.assertEqual(loaded.relaxation_steps, entry.relaxation_steps)
        self.assertEqual(
            loaded.relaxation_time_seconds,
            entry.relaxation_time_seconds,
        )
        self.assertEqual(loaded.displacement_metrics, entry.displacement_metrics)
        self.assertEqual(loaded.slab_geometry.get_chemical_symbols(), ["Cu", "Cu"])
        self.assertEqual(
            loaded.slab_geometry.get_positions().tolist(),
            slab.get_positions().tolist(),
        )
        self.assertEqual(
            loaded.slab_geometry.cell.tolist(),
            slab.cell.tolist(),
        )
        self.assertEqual(
            tuple(bool(value) for value in loaded.slab_geometry.pbc),
            (True, True, False),
        )

    def test_slab_cache_entry_round_trips_without_displacement_metrics(self) -> None:
        entry = SlabCacheEntry(
            cache_key="no-metrics",
            slab_energy_ev=-1.0,
            slab_geometry=Atoms("H", positions=[(0.0, 0.0, 0.0)]),
            relaxation_steps=1,
            relaxation_time_seconds=0.25,
        )

        restored = SlabCacheEntry.from_dict(entry.to_dict())

        self.assertIsNone(restored.displacement_metrics)
        self.assertEqual(
            restored.slab_geometry.get_chemical_symbols(),
            entry.slab_geometry.get_chemical_symbols(),
        )

    def test_concurrent_writes_do_not_leave_partial_cache_state(self) -> None:
        slab = Atoms(
            "Cu2",
            positions=[(0.0, 0.0, 0.0), (1.8, 1.8, 0.2)],
            cell=[(3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, False),
        )

        with TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            path = slab_cache_entry_path(cache_dir, "shared")

            def write_entry(index: int) -> None:
                entry = SlabCacheEntry(
                    cache_key="shared",
                    slab_energy_ev=-10.0 - index,
                    slab_geometry=slab.copy(),
                    relaxation_steps=index + 1,
                    relaxation_time_seconds=0.5 + index,
                    displacement_metrics={"max_disp": 0.1 + index},
                )
                write_slab_cache_entry(cache_dir, entry)

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(write_entry, index) for index in range(16)]
                for future in futures:
                    future.result()

            try:
                loaded = load_slab_cache_entry(path)
            except JSONDecodeError as exc:  # pragma: no cover
                self.fail(f"Concurrent slab cache write left partial JSON: {exc}")

            self.assertEqual(loaded.cache_key, "shared")
            self.assertEqual(loaded.slab_geometry.get_chemical_symbols(), ["Cu", "Cu"])
            self.assertTrue(path.is_file())
            self.assertEqual(list(cache_dir.glob("shared.json.*.tmp")), [])

    def test_process_reaction_basic_preserves_behavior_when_cache_disabled(self) -> None:
        slab = Atoms(
            "Cu2",
            positions=[(0.0, 0.0, 0.0), (1.8, 1.8, 0.0)],
            cell=[(3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, False),
        )
        adslab = Atoms(
            "Cu2N",
            positions=[(0.0, 0.0, 0.0), (1.8, 1.8, 0.0), (0.9, 0.9, 1.2)],
            cell=[(3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, False),
        )
        gas = Atoms("N", positions=[(0.0, 0.0, 0.0)])

        reaction_data = {
            "ref_ads_eng": -0.75,
            "adsorbate_indices": [2],
            "raw": {
                "star": {"atoms": slab, "stoi": 1.0, "energy_ref": -11.0},
                "Nstar": {"atoms": adslab, "stoi": 1.0, "energy_ref": -7.5},
                "Ngas": {"atoms": gas, "stoi": 1.0},
            },
        }
        calculation = AdsorptionCalculation(
            calculators=["fake-calculator"],
            mlip_name="7net-omni",
            benchmark="test_n",
            save_files=False,
            use_slab_cache=False,
            f_crit_relax=0.05,
            n_crit_relax=50,
            damping=1.0,
            optimizer="LBFGS",
            rate=0.5,
        )

        def fake_energy_cal_single(_calculator, atoms):
            if len(atoms) == 2:
                return 1.0
            if len(atoms) == 3:
                return 5.0
            return 0.25

        def fake_energy_cal(_calculator, atoms, *_args, **_kwargs):
            if len(atoms) == 2:
                relaxed = atoms.copy()
                relaxed.positions[1, 2] = 0.1
                return (-10.0, 3, relaxed, 1.5, -0.2)
            relaxed = atoms.copy()
            relaxed.positions[2, 2] = 1.4
            return (-7.0, 4, relaxed, 2.5, -0.3)

        def fake_calc_displacement(initial, _final, _z_target=None):
            if len(initial) == 2:
                return {"max_disp": 0.11, "mae_mobile": 0.04, "rmsd_mobile": 0.05}
            return {"max_disp": 0.22, "mae_mobile": 0.06, "rmsd_mobile": 0.07}

        with (
            patch.object(calculation_module, "energy_cal_single", side_effect=fake_energy_cal_single),
            patch.object(calculation_module, "energy_cal", side_effect=fake_energy_cal),
            patch.object(calculation_module, "energy_cal_gas", return_value=(gas.copy(), 0.5)),
            patch.object(calculation_module, "calc_displacement", side_effect=fake_calc_displacement),
            patch.object(calculation_module, "fix_z", return_value=0.5),
            patch.object(calculation, "_calculate_max_bond_change", return_value=0.8),
            patch.object(calculation, "_calculate_substrate_displacement", return_value=0.3),
        ):
            result = calculation._process_reaction_basic(
                "rxn-1->N*",
                reaction_data,
                save_directory="/tmp/unused",
                gas_energies={},
                gas_energies_single={},
            )

        self.assertEqual(result["time_consumed"], 4.0)
        self.assertEqual(result["reaction_result"]["final"]["slab_cache_hit_count"], 0)
        self.assertFalse(result["reaction_result"]["0"]["slab_cache_hit"])

    def test_vendor_loader_resolves_bm_style_structure_refs(self) -> None:
        slab = Atoms(
            "Cu",
            positions=[(0.0, 0.0, 0.0)],
            cell=np.eye(3),
            pbc=(True, True, True),
        )
        adslab = Atoms(
            "CuH",
            positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
            cell=np.eye(3),
            pbc=(True, True, True),
        )

        def atoms_json(atoms: Atoms) -> str:
            buffer = io.StringIO()
            write(buffer, atoms, format="json")
            return buffer.getvalue()

        payload = {
            "rxn": {
                "raw": {
                    "star": {"ref": "slab-ref", "stoi": -1, "energy_ref": -1.0},
                    "Hstar": {"ref": "ads-ref", "stoi": 1, "energy_ref": -0.5},
                },
                "ref_ads_eng": 0.5,
                "adsorbate_indices": [1],
            },
            "_structures": {
                "slab-ref": atoms_json(slab),
                "ads-ref": atoms_json(adslab),
            },
        }

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bm_adsorption.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_catbench_json(str(path))

        self.assertEqual(loaded["rxn"]["raw"]["star"]["atoms"].get_chemical_symbols(), ["Cu"])
        self.assertEqual(loaded["rxn"]["raw"]["Hstar"]["atoms"].get_chemical_symbols(), ["Cu", "H"])

    def test_two_shards_with_same_parent_slab_id_reuse_shared_slab_cache(self) -> None:
        slab = Atoms(
            "Cu2",
            positions=[(0.0, 0.0, 0.0), (1.8, 1.8, 0.0)],
            cell=[(3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, False),
        )
        adslab_a = Atoms(
            "Cu2N",
            positions=[(0.0, 0.0, 0.0), (1.8, 1.8, 0.0), (0.9, 0.9, 1.2)],
            cell=[(3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, False),
        )
        adslab_b = Atoms(
            "Cu2N",
            positions=[(0.0, 0.0, 0.0), (1.8, 1.8, 0.0), (1.2, 0.9, 1.2)],
            cell=[(3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, False),
        )
        gas = Atoms("N", positions=[(0.0, 0.0, 0.0)])

        def reaction(adslab_atoms):
            return {
                "ref_ads_eng": -0.75,
                "adsorbate_indices": [2],
                "metadata": {"reference": {"parent_slab_id": "slab-000004"}},
                "raw": {
                    "star": {"atoms": slab.copy(), "stoi": 1.0, "energy_ref": -11.0},
                    "Nstar": {"atoms": adslab_atoms.copy(), "stoi": 1.0, "energy_ref": -7.5},
                    "Ngas": {"atoms": gas.copy(), "stoi": 1.0},
                },
            }

        slab_relax_calls = 0
        adslab_relax_calls = 0

        def fake_energy_cal_single(_calculator, atoms):
            if len(atoms) == 2:
                return 1.0
            if len(atoms) == 3:
                return 5.0
            return 0.25

        def fake_energy_cal(_calculator, atoms, *_args, **_kwargs):
            nonlocal slab_relax_calls, adslab_relax_calls
            if len(atoms) == 2:
                slab_relax_calls += 1
                relaxed = atoms.copy()
                relaxed.positions[1, 2] = 0.1
                return (-10.0, 3, relaxed, 1.5, -0.2)
            adslab_relax_calls += 1
            relaxed = atoms.copy()
            relaxed.positions[2, 2] += 0.2
            return (-7.0, 4, relaxed, 2.5, -0.3)

        def fake_calc_displacement(initial, _final, _z_target=None):
            if len(initial) == 2:
                return {"max_disp": 0.11, "mae_mobile": 0.04, "rmsd_mobile": 0.05}
            return {"max_disp": 0.22, "mae_mobile": 0.06, "rmsd_mobile": 0.07}

        with TemporaryDirectory() as tmp_dir:
            shared_cache_dir = Path(tmp_dir) / "example" / "_shared" / "slab_cache"
            shard_a_dir = Path(tmp_dir) / "example_shard_00_of_02"
            shard_b_dir = Path(tmp_dir) / "example_shard_01_of_02"
            calculation = AdsorptionCalculation(
                calculators=["fake-calculator"],
                mlip_name="7net-omni",
                benchmark="test_n",
                save_files=False,
                use_slab_cache=True,
                model_name="sevennet",
                slab_cache_dir=str(shared_cache_dir),
                f_crit_relax=0.05,
                n_crit_relax=50,
                damping=1.0,
                optimizer="LBFGS",
                rate=0.5,
            )
            with patch.object(
                calculation_module,
                "energy_cal_single",
                side_effect=fake_energy_cal_single,
            ), patch.object(
                calculation_module,
                "energy_cal",
                side_effect=fake_energy_cal,
            ), patch.object(
                calculation_module,
                "energy_cal_gas",
                return_value=(gas.copy(), 0.5),
            ), patch.object(
                calculation_module,
                "calc_displacement",
                side_effect=fake_calc_displacement,
            ), patch.object(
                calculation_module,
                "fix_z",
                return_value=0.5,
            ), patch.object(
                calculation,
                "_calculate_max_bond_change",
                return_value=0.8,
            ), patch.object(
                calculation,
                "_calculate_substrate_displacement",
                return_value=0.3,
            ):
                result_a = calculation._process_reaction_basic(
                    "rxn-1->N*",
                    reaction(adslab_a),
                    save_directory=str(shard_a_dir),
                    gas_energies={},
                    gas_energies_single={},
                )
                result_b = calculation._process_reaction_basic(
                    "rxn-2->N*",
                    reaction(adslab_b),
                    save_directory=str(shard_b_dir),
                    gas_energies={},
                    gas_energies_single={},
                )

        self.assertEqual(slab_relax_calls, 1)
        self.assertEqual(adslab_relax_calls, 2)
        self.assertFalse(result_a["reaction_result"]["final"]["slab_cache_hit"])
        self.assertTrue(result_b["reaction_result"]["final"]["slab_cache_hit"])

    def test_first_run_misses_and_writes_cache_then_second_run_hits(self) -> None:
        slab = Atoms(
            "Cu2",
            positions=[(0.0, 0.0, 0.0), (1.8, 1.8, 0.0)],
            cell=[(3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, False),
        )
        reaction_data = {"metadata": {"reference": {"parent_slab_id": "slab-000004"}}}
        calculation = AdsorptionCalculation(
            calculators=["fake-calculator"],
            mlip_name="7net-omni",
            benchmark="test_n",
            use_slab_cache=True,
            model_name="sevennet",
            f_crit_relax=0.05,
            n_crit_relax=50,
            damping=1.0,
            optimizer="LBFGS",
            rate=0.5,
        )
        slab_relax_calls = 0

        def fake_energy_cal(_calculator, atoms, *_args, **_kwargs):
            nonlocal slab_relax_calls
            slab_relax_calls += 1
            relaxed = atoms.copy()
            relaxed.positions[1, 2] = 0.1
            return (-10.0, 3, relaxed, 1.5, -0.2)

        def fake_energy_cal_single(_calculator, _atoms):
            return 1.0

        with TemporaryDirectory() as tmp_dir:
            with patch.object(
                calculation_module, "energy_cal", side_effect=fake_energy_cal
            ), patch.object(
                calculation_module,
                "energy_cal_single",
                side_effect=fake_energy_cal_single,
            ), patch.object(
                calculation_module,
                "calc_displacement",
                return_value={"max_disp": 0.11, "mae_mobile": 0.04, "rmsd_mobile": 0.05},
            ):
                first = calculation._relax_slab_structure(
                    calculator="fake-calculator",
                    slab_atoms=slab,
                    reaction_data=reaction_data,
                    save_directory=tmp_dir,
                    z_target=0.5,
                    log_path=None,
                    traj_path=None,
                    calculation_index=0,
                )
                cache_key = calculation._slab_cache_key(reaction_data)
                cache_path = calculation._slab_cache_path(tmp_dir, cache_key)
                cached_entry = load_slab_cache_entry(cache_path)
                second = calculation._relax_slab_structure(
                    calculator="fake-calculator",
                    slab_atoms=slab,
                    reaction_data=reaction_data,
                    save_directory=tmp_dir,
                    z_target=0.5,
                    log_path=None,
                    traj_path=None,
                    calculation_index=1,
                )

        self.assertEqual(slab_relax_calls, 1)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(cached_entry.relaxation_time_seconds, 1.5)

    def test_end_to_end_slab_cache_reuse_preserves_scientific_output(self) -> None:
        slab = Atoms(
            "Cu2",
            positions=[(0.0, 0.0, 0.0), (1.8, 1.8, 0.0)],
            cell=[(3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, False),
        )
        gas = Atoms("N", positions=[(0.0, 0.0, 0.0)])
        adslab_a = Atoms(
            "Cu2N",
            positions=[(0.0, 0.0, 0.0), (1.8, 1.8, 0.0), (0.9, 0.9, 1.2)],
            cell=[(3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, False),
        )
        adslab_b = Atoms(
            "Cu2N",
            positions=[(0.0, 0.0, 0.0), (1.8, 1.8, 0.0), (1.2, 0.9, 1.2)],
            cell=[(3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, False),
        )
        fixture = {
            "rxn-1->N*": {
                "ref_ads_eng": -0.75,
                "adsorbate_indices": [2],
                "metadata": {"reference": {"parent_slab_id": "slab-000004"}},
                "raw": {
                    "star": {"atoms": slab.copy(), "stoi": 1.0, "energy_ref": -11.0},
                    "Nstar": {"atoms": adslab_a.copy(), "stoi": 1.0, "energy_ref": -7.5},
                    "Ngas": {"atoms": gas.copy(), "stoi": 1.0},
                },
            },
            "rxn-2->N*": {
                "ref_ads_eng": -0.75,
                "adsorbate_indices": [2],
                "metadata": {"reference": {"parent_slab_id": "slab-000004"}},
                "raw": {
                    "star": {"atoms": slab.copy(), "stoi": 1.0, "energy_ref": -11.0},
                    "Nstar": {"atoms": adslab_b.copy(), "stoi": 1.0, "energy_ref": -7.5},
                    "Ngas": {"atoms": gas.copy(), "stoi": 1.0},
                },
            },
        }

        def make_calculation(*, use_slab_cache: bool, cache_dir: str | None):
            return AdsorptionCalculation(
                calculators=["fake-calculator"],
                mlip_name="7net-omni",
                benchmark="test_n",
                save_files=False,
                save_step=50,
                use_slab_cache=use_slab_cache,
                model_name="sevennet",
                slab_cache_dir=cache_dir,
                f_crit_relax=0.05,
                n_crit_relax=50,
                damping=1.0,
                optimizer="LBFGS",
                rate=0.5,
            )

        uncached_counts = {"slab": 0, "adslab": 0}
        cached_counts = {"slab": 0, "adslab": 0}

        def fake_energy_cal_single(_calculator, atoms):
            if len(atoms) == 2:
                return 1.0
            if len(atoms) == 3:
                return 5.0
            return 0.25

        def make_fake_energy_cal(counter):
            def fake_energy_cal(_calculator, atoms, *_args, **_kwargs):
                if len(atoms) == 2:
                    counter["slab"] += 1
                    relaxed = atoms.copy()
                    relaxed.positions[1, 2] = 0.1
                    return (-10.0, 3, relaxed, 1.5, -0.2)
                counter["adslab"] += 1
                relaxed = atoms.copy()
                relaxed.positions[2, 2] += 0.2
                return (-7.0, 4, relaxed, 2.5, -0.3)

            return fake_energy_cal

        def fake_calc_displacement(initial, _final, _z_target=None):
            if len(initial) == 2:
                return {"max_disp": 0.11, "mae_mobile": 0.04, "rmsd_mobile": 0.05}
            return {"max_disp": 0.22, "mae_mobile": 0.06, "rmsd_mobile": 0.07}

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            uncached_dir = root / "uncached"
            cached_dir = root / "cached"
            shared_cache_dir = root / "shared" / "slab_cache"
            uncached_dir.mkdir(parents=True, exist_ok=True)
            cached_dir.mkdir(parents=True, exist_ok=True)

            uncached = make_calculation(use_slab_cache=False, cache_dir=None)
            cached = make_calculation(use_slab_cache=True, cache_dir=str(shared_cache_dir))

            with patch.object(uncached, "_load_data", return_value=fixture), patch.object(
                cached, "_load_data", return_value=fixture
            ), patch.object(
                uncached, "_setup_directories", return_value=str(uncached_dir)
            ), patch.object(
                cached, "_setup_directories", return_value=str(cached_dir)
            ), patch.object(
                calculation_module,
                "energy_cal_single",
                side_effect=fake_energy_cal_single,
            ), patch.object(
                calculation_module, "energy_cal_gas", return_value=(gas.copy(), 0.5)
            ), patch.object(
                calculation_module,
                "calc_displacement",
                side_effect=fake_calc_displacement,
            ), patch.object(
                calculation_module, "fix_z", return_value=0.5
            ), patch.object(
                uncached, "_calculate_max_bond_change", return_value=0.8
            ), patch.object(
                cached, "_calculate_max_bond_change", return_value=0.8
            ), patch.object(
                uncached, "_calculate_substrate_displacement", return_value=0.3
            ), patch.object(
                cached, "_calculate_substrate_displacement", return_value=0.3
            ):
                with patch.object(
                    calculation_module,
                    "energy_cal",
                    side_effect=make_fake_energy_cal(uncached_counts),
                ):
                    uncached_run_dir = Path(uncached.run())

                with patch.object(
                    calculation_module,
                    "energy_cal",
                    side_effect=make_fake_energy_cal(cached_counts),
                ):
                    cached_run_dir = Path(cached.run())

            uncached_result = json.loads(
                (uncached_run_dir / "7net-omni_result.json").read_text(encoding="utf-8")
            )
            cached_result = json.loads(
                (cached_run_dir / "7net-omni_result.json").read_text(encoding="utf-8")
            )

        self.assertEqual(uncached_counts["slab"], 2)
        self.assertEqual(cached_counts["slab"], 1)
        self.assertEqual(uncached_counts["adslab"], 2)
        self.assertEqual(cached_counts["adslab"], 2)
        self.assertEqual(
            self._without_cache_diagnostics(cached_result),
            self._without_cache_diagnostics(uncached_result),
        )
