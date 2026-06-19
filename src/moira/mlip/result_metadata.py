from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RELAXATION_SETTING_KEYS = (
    "optimizer",
    "f_crit_relax",
    "n_crit_relax",
    "rate",
    "damping",
    "chemical_bond_cutoff",
)


def enrich_result_file(
    *,
    dataset_path: str | Path | None,
    dataset_name: str | None = None,
    result_path: str | Path,
    mlip_name: str | None = None,
    model_name: str | None = None,
) -> None:
    from moira.mlip.result_parsing import (
        RESULT_ANALYSIS_KEY,
        detect_anomalies_from_result_dict,
    )

    resolved_result_path = _resolve_result_path(
        result_path=Path(result_path),
        mlip_name=mlip_name,
    )
    if resolved_result_path is None:
        return
    resolved_dataset_path = _resolve_dataset_path(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
    )

    result = _load_json_object(resolved_result_path)
    dataset = (
        _load_json_object(resolved_dataset_path)
        if resolved_dataset_path is not None
        else None
    )
    updated = False

    if dataset is not None:
        for reaction, reaction_data in result.items():
            if reaction == "calculation_settings" or not isinstance(reaction_data, dict):
                continue
            dataset_entry = dataset.get(reaction)
            if not isinstance(dataset_entry, dict):
                continue
            metadata = dataset_entry.get("metadata")
            if metadata is None or "metadata" in reaction_data:
                continue
            reaction_data["metadata"] = metadata
            updated = True

    calculation_settings = result.get("calculation_settings")
    if not isinstance(calculation_settings, dict):
        calculation_settings = {}

    for reaction, reaction_data in result.items():
        if reaction == "calculation_settings" or not isinstance(reaction_data, dict):
            continue
        metadata = reaction_data.get("metadata")
        if not isinstance(metadata, dict):
            continue
        slab_cache_key = build_slab_cache_key(
            metadata=metadata,
            model_name=model_name,
            mlip_name=mlip_name,
            calculation_settings=calculation_settings,
        )
        if slab_cache_key is None or metadata.get("slab_cache_key") == slab_cache_key:
            continue
        metadata["slab_cache_key"] = slab_cache_key
        updated = True

    analysis_by_reaction = detect_anomalies_from_result_dict(result)
    for reaction, analysis_payload in analysis_by_reaction.items():
        reaction_data = result.get(reaction)
        if not isinstance(reaction_data, dict):
            continue
        if reaction_data.get(RESULT_ANALYSIS_KEY) == analysis_payload:
            continue
        reaction_data[RESULT_ANALYSIS_KEY] = analysis_payload
        updated = True

    if updated:
        resolved_result_path.write_text(
            json.dumps(result, indent=4) + "\n",
            encoding="utf-8",
        )


def attach_dataset_metadata_to_result_file(
    *,
    dataset_path: str | Path | None,
    dataset_name: str | None = None,
    result_path: str | Path,
    mlip_name: str | None = None,
    model_name: str | None = None,
) -> None:
    enrich_result_file(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        result_path=result_path,
        mlip_name=mlip_name,
        model_name=model_name,
    )


def build_slab_cache_key(
    *,
    metadata: dict[str, Any],
    model_name: str | None = None,
    mlip_name: str | None = None,
    calculation_settings: dict[str, Any] | None = None,
) -> str | None:
    parent_slab_id = metadata.get("parent_slab_id")
    if parent_slab_id is None:
        return None

    identity = {
        "parent_slab_id": str(parent_slab_id),
        "model_name": model_name,
        "mlip_name": mlip_name,
        "relaxation_settings": {
            key: calculation_settings[key]
            for key in RELAXATION_SETTING_KEYS
            if calculation_settings is not None and key in calculation_settings
        },
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_efficiency_summary(
    *,
    dataset_path: str | Path | None,
    dataset_name: str | None = None,
    result_path: str | Path,
    mlip_name: str | None = None,
    model_name: str | None = None,
    task_wall_seconds: float | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> Path | None:
    resolved_result_path = _resolve_result_path(
        result_path=Path(result_path),
        mlip_name=mlip_name,
    )
    if resolved_result_path is None:
        return None
    resolved_dataset_path = _resolve_dataset_path(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
    )
    result = _load_json_object(resolved_result_path)
    dataset = (
        _load_json_object(resolved_dataset_path)
        if resolved_dataset_path is not None
        else None
    )

    reactions = [
        (reaction, reaction_data)
        for reaction, reaction_data in result.items()
        if reaction != "calculation_settings" and isinstance(reaction_data, dict)
    ]
    reaction_count = len(reactions)
    total_slab_time = 0.0
    total_adslab_time = 0.0
    total_slab_steps = 0
    total_adslab_steps = 0
    total_atom_steps = 0.0
    time_per_step_per_atom_values: list[float] = []

    for _reaction, reaction_data in reactions:
        final = reaction_data.get("final", {})
        if not isinstance(final, dict):
            continue
        total_slab_time += float(final.get("time_total_slab", 0.0) or 0.0)
        total_adslab_time += float(final.get("time_total_adslab", 0.0) or 0.0)
        total_slab_steps += int(final.get("steps_total_slab", 0) or 0)
        total_adslab_steps += int(final.get("steps_total_adslab", 0) or 0)
        total_steps = float(final.get("steps_total_slab", 0) or 0) + float(
            final.get("steps_total_adslab", 0) or 0
        )
        step_weighted_atoms = float(final.get("step_weighted_atoms", 0.0) or 0.0)
        total_atom_steps += total_steps * step_weighted_atoms
        time_per_step_per_atom = final.get("time_per_step_per_atom")
        if time_per_step_per_atom is not None:
            time_per_step_per_atom_values.append(float(time_per_step_per_atom))

    total_relaxation_time = total_slab_time + total_adslab_time
    total_relaxation_steps = total_slab_steps + total_adslab_steps
    dataset_reaction_count = len(dataset) if dataset is not None else None

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        "mlip_name": mlip_name,
        "dataset_name": dataset_name,
        "dataset_path": str(resolved_dataset_path) if resolved_dataset_path is not None else None,
        "result_path": str(resolved_result_path),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "dataset_reaction_count": dataset_reaction_count,
        "completed_reaction_count": reaction_count,
        "task_wall_seconds": task_wall_seconds,
        "task_wall_hours": (
            task_wall_seconds / 3600.0 if task_wall_seconds is not None else None
        ),
        "reactions_per_hour_wall": (
            (reaction_count * 3600.0 / task_wall_seconds)
            if task_wall_seconds and reaction_count > 0
            else None
        ),
        "total_relaxation_time_seconds": total_relaxation_time,
        "total_slab_relaxation_time_seconds": total_slab_time,
        "total_adslab_relaxation_time_seconds": total_adslab_time,
        "total_relaxation_steps": total_relaxation_steps,
        "total_slab_relaxation_steps": total_slab_steps,
        "total_adslab_relaxation_steps": total_adslab_steps,
        "total_atom_steps": total_atom_steps,
        "mean_relaxation_time_seconds_per_reaction": (
            total_relaxation_time / reaction_count if reaction_count > 0 else None
        ),
        "mean_relaxation_steps_per_reaction": (
            total_relaxation_steps / reaction_count if reaction_count > 0 else None
        ),
        "mean_time_per_step_per_atom_seconds": (
            sum(time_per_step_per_atom_values) / len(time_per_step_per_atom_values)
            if time_per_step_per_atom_values
            else None
        ),
    }

    summary_path = resolved_result_path.with_name(
        resolved_result_path.name.removesuffix("_result.json") + "_efficiency.json"
    )
    summary_path.write_text(json.dumps(summary, indent=4) + "\n", encoding="utf-8")
    return summary_path


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected JSON object at {path}, got {type(payload).__name__}"
        )
    return payload


def _resolve_dataset_path(
    *,
    dataset_path: str | Path | None,
    dataset_name: str | None,
) -> Path | None:
    candidates: list[Path] = []
    if dataset_path is not None:
        candidates.append(Path(dataset_path))
    if dataset_name:
        candidates.extend(
            [
                Path.cwd() / "raw_data" / f"{dataset_name}_adsorption.json",
                Path.cwd() / "data" / "raw_data" / f"{dataset_name}_adsorption.json",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _resolve_result_path(
    *,
    result_path: Path,
    mlip_name: str | None,
) -> Path | None:
    candidates = [result_path]
    if mlip_name:
        candidates.extend(
            [
                result_path.parent / f"{mlip_name}_result.json",
                result_path.parent.parent / f"{mlip_name}_result.json",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
