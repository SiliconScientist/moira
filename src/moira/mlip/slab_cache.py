from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read, write

from moira.mlip.schema import atomic_write_json


@dataclass(frozen=True)
class SlabCacheEntry:
    cache_key: str
    slab_energy_ev: float
    slab_geometry: Atoms
    relaxation_steps: int
    relaxation_time_seconds: float
    displacement_metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cache_key": self.cache_key,
            "slab_energy_ev": self.slab_energy_ev,
            "slab_geometry_json": atoms_to_json(self.slab_geometry),
            "relaxation_steps": self.relaxation_steps,
            "relaxation_time_seconds": self.relaxation_time_seconds,
        }
        if self.displacement_metrics is not None:
            payload["displacement_metrics"] = dict(self.displacement_metrics)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SlabCacheEntry:
        displacement_metrics = payload.get("displacement_metrics")
        if displacement_metrics is not None and not isinstance(displacement_metrics, dict):
            raise TypeError(
                "Expected displacement_metrics to be a dict when present, "
                f"got {type(displacement_metrics).__name__}"
            )
        return cls(
            cache_key=str(payload["cache_key"]),
            slab_energy_ev=float(payload["slab_energy_ev"]),
            slab_geometry=atoms_from_json(str(payload["slab_geometry_json"])),
            relaxation_steps=int(payload["relaxation_steps"]),
            relaxation_time_seconds=float(payload["relaxation_time_seconds"]),
            displacement_metrics=(
                {str(key): float(value) for key, value in displacement_metrics.items()}
                if displacement_metrics is not None
                else None
            ),
        )


def atoms_to_json(atoms: Atoms) -> str:
    buffer = io.StringIO()
    write(buffer, atoms, format="json")
    return buffer.getvalue()


def atoms_from_json(payload: str) -> Atoms:
    return read(io.StringIO(payload), format="json")


def slab_cache_entry_path(cache_dir: str | Path, cache_key: str) -> Path:
    return Path(cache_dir) / f"{cache_key}.json"


def write_slab_cache_entry(cache_dir: str | Path, entry: SlabCacheEntry) -> Path:
    path = slab_cache_entry_path(cache_dir, entry.cache_key)
    atomic_write_json(path, entry.to_dict())
    return path


def load_slab_cache_entry(path: str | Path) -> SlabCacheEntry:
    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        decoded = json.load(handle)
    if not isinstance(decoded, dict):
        raise TypeError(
            f"Expected slab cache payload at {resolved_path} to be an object, "
            f"got {type(decoded).__name__}"
        )
    return SlabCacheEntry.from_dict(decoded)
