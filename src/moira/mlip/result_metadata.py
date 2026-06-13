from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def attach_dataset_metadata_to_result_file(
    *,
    dataset_path: str | Path | None,
    result_path: str | Path,
) -> None:
    if dataset_path is None:
        return

    dataset = _load_json_object(Path(dataset_path))
    result = _load_json_object(Path(result_path))
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
        Path(result_path).write_text(json.dumps(result, indent=4) + "\n", encoding="utf-8")


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected JSON object at {path}, got {type(payload).__name__}"
        )
    return payload
