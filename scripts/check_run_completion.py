#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


METADATA_KEYS = {"calculation_settings"}


def classify_file(path: Path, expected_samples: int | None) -> tuple[str, int, int | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return ("invalid_json", 0, None)

    if not isinstance(data, dict):
        return ("unexpected_top_level", 0, None)

    sample_count = sum(1 for key in data if key not in METADATA_KEYS)
    metadata_count = len(data) - sample_count

    if expected_samples is None:
        return ("counted", sample_count, metadata_count)
    if sample_count >= expected_samples:
        return ("full", sample_count, metadata_count)
    if sample_count == 0:
        return ("empty", sample_count, metadata_count)
    return ("partial", sample_count, metadata_count)


def expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        candidate = Path(item)
        if any(ch in item for ch in "*?[]"):
            paths.extend(sorted(Path().glob(item)))
        elif candidate.is_dir():
            direct_results = sorted(candidate.glob("*_result.json"))
            if direct_results:
                paths.extend(direct_results)
            else:
                paths.extend(sorted(candidate.glob("*/*_result.json")))
        else:
            paths.append(candidate)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether result JSON files reached an expected sample count."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["data/results"],
        help="JSON files, globs, or result directories to inspect",
    )
    parser.add_argument(
        "--expected",
        type=int,
        default=None,
        help="Expected number of sample entries per run",
    )
    args = parser.parse_args(argv)

    paths = expand_inputs(args.inputs)
    if not paths:
        print("No files matched.", file=sys.stderr)
        return 1

    any_problem = False
    for path in paths:
        if not path.exists():
            print(f"{path}: missing")
            any_problem = True
            continue

        status, sample_count, metadata_count = classify_file(path, args.expected)
        if metadata_count is None:
            print(f"{path}: {status}")
            any_problem = True
            continue

        if args.expected is None:
            print(
                f"{path}: {status} "
                f"({sample_count} samples, {metadata_count} metadata entries)"
            )
            continue

        print(
            f"{path}: {status} "
            f"({sample_count}/{args.expected} samples, {metadata_count} metadata entries)"
        )
        if status != "full":
            any_problem = True

    return 1 if any_problem else 0


if __name__ == "__main__":
    raise SystemExit(main())
