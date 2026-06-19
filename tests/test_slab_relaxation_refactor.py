from __future__ import annotations

import importlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ase import Atoms

from moira.mlip.slab_cache import load_slab_cache_entry


VENDOR_CATBENCH = Path(__file__).resolve().parents[1] / "vendor" / "catbench"
if str(VENDOR_CATBENCH) not in sys.path:
    sys.path.insert(0, str(VENDOR_CATBENCH))

calculation_module = importlib.import_module(
    "catbench.adsorption.calculation.calculation"
)
AdsorptionCalculation = calculation_module.AdsorptionCalculation


class SlabRelaxationRefactorTests(unittest.TestCase):
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
        self.assertEqual(
            result["reaction_result"],
            {
                "reference": {
                    "ads_eng": -0.75,
                    "star_tot_eng": -11.0,
                    "Nstar_tot_eng": -7.5,
                },
                "adsorbate_indices": [2],
                "single_calculation": {
                    "ads_eng": 6.25,
                    "slab_tot_eng": 1.0,
                    "adslab_tot_eng": 5.0,
                },
                "0": {
                    "ads_eng": -16.5,
                    "slab_tot_eng": -10.0,
                    "adslab_tot_eng": -7.0,
                    "slab_max_disp": 0.11,
                    "slab_pos_mae": 0.04,
                    "slab_pos_rmsd": 0.05,
                    "adslab_max_disp": 0.22,
                    "adslab_pos_mae": 0.06,
                    "adslab_pos_rmsd": 0.07,
                    "max_bond_change": 0.8,
                    "substrate_displacement": 0.3,
                    "slab_energy_change": -0.2,
                    "adslab_energy_change": -0.3,
                    "slab_time": 1.5,
                    "adslab_time": 2.5,
                    "slab_steps": 3,
                    "adslab_steps": 4,
                },
                "final": {
                    "ads_eng_median": -16.5,
                    "median_num": 0,
                    "slab_max_disp": 0.11,
                    "adslab_max_disp": 0.22,
                    "slab_seed_range": 0.0,
                    "ads_seed_range": 0.0,
                    "ads_eng_seed_range": 0.0,
                    "time_total_slab": 1.5,
                    "time_total_adslab": 2.5,
                    "steps_total_slab": 3,
                    "steps_total_adslab": 4,
                    "step_weighted_atoms": 18.0 / 7.0,
                    "time_per_step": 4.0 / 7.0,
                    "time_per_step_per_atom": 4.0 / 18.0,
                },
            },
        )

    def test_repeated_reactions_with_same_parent_slab_id_reuse_cached_slab(self) -> None:
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
                "metadata": {
                    "reference": {
                        "parent_slab_id": "slab-000004",
                    }
                },
                "raw": {
                    "star": {"atoms": slab.copy(), "stoi": 1.0, "energy_ref": -11.0},
                    "Nstar": {"atoms": adslab_atoms.copy(), "stoi": 1.0, "energy_ref": -7.5},
                    "Ngas": {"atoms": gas.copy(), "stoi": 1.0},
                },
            }

        calculation = AdsorptionCalculation(
            calculators=["fake-calculator"],
            mlip_name="7net-omni",
            benchmark="test_n",
            save_files=False,
            use_slab_cache=True,
            model_name="sevennet",
            f_crit_relax=0.05,
            n_crit_relax=50,
            damping=1.0,
            optimizer="LBFGS",
            rate=0.5,
        )
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
                    save_directory=tmp_dir,
                    gas_energies={},
                    gas_energies_single={},
                )
                result_b = calculation._process_reaction_basic(
                    "rxn-2->N*",
                    reaction(adslab_b),
                    save_directory=tmp_dir,
                    gas_energies={},
                    gas_energies_single={},
                )

        self.assertEqual(slab_relax_calls, 1)
        self.assertEqual(adslab_relax_calls, 2)
        self.assertEqual(result_a["reaction_result"]["0"]["slab_tot_eng"], -10.0)
        self.assertEqual(result_b["reaction_result"]["0"]["slab_tot_eng"], -10.0)

    def test_first_run_misses_and_writes_cache_then_second_run_hits(self) -> None:
        slab = Atoms(
            "Cu2",
            positions=[(0.0, 0.0, 0.0), (1.8, 1.8, 0.0)],
            cell=[(3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, False),
        )
        reaction_data = {
            "metadata": {
                "reference": {
                    "parent_slab_id": "slab-000004",
                }
            }
        }
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
                calculation_module,
                "energy_cal",
                side_effect=fake_energy_cal,
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
        self.assertEqual(first["energy"], -10.0)
        self.assertEqual(second["energy"], -10.0)
        self.assertEqual(cached_entry.slab_energy_ev, -10.0)
        self.assertEqual(cached_entry.relaxation_steps, 3)
        self.assertEqual(cached_entry.relaxation_time_seconds, 1.5)
