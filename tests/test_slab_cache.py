from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ase import Atoms

from moira.mlip.slab_cache import (
    SlabCacheEntry,
    load_slab_cache_entry,
    slab_cache_entry_path,
    write_slab_cache_entry,
)


class SlabCacheTests(unittest.TestCase):
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
