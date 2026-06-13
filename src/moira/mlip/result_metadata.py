from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def attach_dataset_metadata_to_result_file(
    *,
    dataset_path: str | Path | None,
    dataset_name: str | None = None,
    result_path: str | Path,
    mlip_name: str | None = None,
) -> None:
    resolved_dataset_path = _resolve_dataset_path(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
    )
    if resolved_dataset_path is None:
        return
    resolved_result_path = _resolve_result_path(
        result_path=Path(result_path),
        mlip_name=mlip_name,
    )
    if resolved_result_path is None:
        return

    dataset = _load_json_object(resolved_dataset_path)
    result = _load_json_object(resolved_result_path)
    updated = False

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

    if updated:
        resolved_result_path.write_text(
            json.dumps(result, indent=4) + "\n",
            encoding="utf-8",
        )


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
