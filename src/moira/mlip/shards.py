from __future__ import annotations

from math import ceil
from typing import Any


def shard_json_obj(
    obj: Any,
    *,
    shard_size: int | None = None,
    num_shards: int | None = None,
    shard_index: int,
) -> Any:
    shard_count = infer_shard_count(
        obj,
        shard_size=shard_size,
        num_shards=num_shards,
    )
    if shard_index < 0 or shard_index >= shard_count:
        raise IndexError(
            f"Shard index {shard_index} out of range for {shard_count} shard(s)"
        )

    start, stop = shard_bounds(
        obj,
        shard_size=shard_size,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    return slice_json_obj(obj, start=start, stop=stop)


def infer_shard_count(
    obj: Any,
    *,
    shard_size: int | None = None,
    num_shards: int | None = None,
) -> int:
    if shard_size is not None and num_shards is not None:
        raise ValueError("Provide only one of shard_size or num_shards")
    if shard_size is None and num_shards is None:
        raise ValueError("Provide shard_size or num_shards")

    size = json_obj_size(obj)
    if size == 0:
        return 1
    if num_shards is not None:
        return num_shards
    return ceil(size / shard_size)


def shard_bounds(
    obj: Any,
    *,
    shard_size: int | None = None,
    num_shards: int | None = None,
    shard_index: int,
) -> tuple[int, int]:
    size = json_obj_size(obj)
    shard_count = infer_shard_count(
        obj,
        shard_size=shard_size,
        num_shards=num_shards,
    )
    if shard_index < 0 or shard_index >= shard_count:
        raise IndexError(
            f"Shard index {shard_index} out of range for {shard_count} shard(s)"
        )

    if shard_size is not None:
        start = shard_index * shard_size
        stop = min(size, start + shard_size)
        return start, stop

    start = (size * shard_index) // shard_count
    stop = (size * (shard_index + 1)) // shard_count
    return start, stop


def slice_json_obj(obj: Any, *, start: int, stop: int) -> Any:
    if start < 0 or stop < start:
        raise ValueError(f"Invalid slice bounds: start={start}, stop={stop}")

    if isinstance(obj, list):
        return obj[start:stop]

    if isinstance(obj, dict):
        items = list(obj.items())[start:stop]
        return dict(items)

    raise TypeError(f"Expected top-level JSON list or dict, got {type(obj).__name__}")


def json_obj_size(obj: Any) -> int:
    if isinstance(obj, (list, dict)):
        return len(obj)
    raise TypeError(f"Expected top-level JSON list or dict, got {type(obj).__name__}")
