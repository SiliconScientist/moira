from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import warnings

import polars as pl

from moira.mlip.result_parsing import (
    RESULT_ANALYSIS_KEY,
    detect_anomalies_from_result_json,
    extract_adsorbate,
)

INFERENCE_DETAIL_COLUMNS = (
    "slab_conv",
    "ads_conv",
    "slab_move",
    "ads_move",
    "slab_seed",
    "ads_seed",
    "ads_eng_seed",
    "adsorbate_migration",
)


def find_result_files(
    base_dir: Path,
    *,
    pattern: str = "*/*_result.json",
    exclude_processed: bool = True,
) -> list[Path]:
    warnings.warn(
        "Directory-based result discovery is a compatibility helper; prefer passing "
        "explicit result JSON paths to load_wide_predictions().",
        DeprecationWarning,
        stacklevel=2,
    )
    candidates = sorted(base_dir.glob(pattern))
    if exclude_processed:
        candidates = [
            path
            for path in candidates
            if not path.name.endswith("_processed_result.json")
        ]
    if candidates:
        return candidates

    raise FileNotFoundError(
        f"No result files found under {base_dir} (expected pattern: {pattern})"
    )


def load_result_json(result_path: str | Path) -> dict[str, Any]:
    path = Path(result_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected result JSON top-level to be an object/dict, got {type(payload).__name__}"
        )
    return payload


def merge_result_jsons(
    result_files: list[str | Path],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_result_files = _resolve_result_files(result_files)
    merged: dict[str, Any] = {}
    merged_settings: dict[str, Any] | None = None

    for path in resolved_result_files:
        payload = load_result_json(path)
        settings = payload.get("calculation_settings")
        if settings is not None:
            if not isinstance(settings, dict):
                raise TypeError(
                    f"Expected calculation_settings in {path} to be a dict, "
                    f"got {type(settings).__name__}"
                )
            if merged_settings is None:
                merged_settings = settings
            elif merged_settings != settings:
                raise ValueError(
                    f"calculation_settings differ in shard result {path}"
                )

        for reaction, reaction_data in payload.items():
            if reaction == "calculation_settings":
                continue
            if reaction in merged:
                raise ValueError(f"Duplicate reaction key across shard results: {reaction}")
            merged[reaction] = reaction_data

    result: dict[str, Any] = {}
    if merged_settings is not None:
        result["calculation_settings"] = merged_settings
    result.update(dict(sorted(merged.items())))

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=4) + "\n", encoding="utf-8")

    return result


def result_file_name(model_name: str) -> str:
    return f"{model_name}_result.json"


def model_name_from_result_path(result_path: str | Path) -> str:
    return Path(result_path).name.removesuffix("_result.json")


def mlip_energy_column_name(model_name: str) -> str:
    return f"{model_name}_mlip_ads_eng_median"


def mlip_label_column_name(model_name: str) -> str:
    return f"{model_name}_label"


def mlip_detail_column_name(model_name: str, detail_name: str) -> str:
    return f"{model_name}_{detail_name}"


def load_wide_predictions(result_files: list[str | Path]) -> pl.DataFrame:
    resolved_result_files = _resolve_result_files(result_files)
    reference_df: pl.DataFrame | None = None
    wide_parts: list[pl.DataFrame] = []
    mlip_cols: list[str] = []
    label_cols: list[str] = []
    detail_cols: list[str] = []

    for path in resolved_result_files:
        model_name = model_name_from_result_path(path)
        per_reaction = load_result_analysis(path)
        rows = [
            {
                "reaction": reaction,
                "adsorbate": extract_adsorbate(reaction),
                **{
                    detail_name: int(payload.get("details", {}).get(detail_name, 0))
                    for detail_name in INFERENCE_DETAIL_COLUMNS
                },
                **payload,
            }
            for reaction, payload in per_reaction.items()
        ]
        df = pl.from_dicts(rows)

        reaction_col = "id" if "id" in df.columns else "reaction"
        required = {
            reaction_col,
            "adsorbate",
            "dft_ads_eng",
            "mlip_ads_eng_median",
            "metadata_json",
            "label",
            *INFERENCE_DETAIL_COLUMNS,
        }
        missing = required.difference(set(df.columns))
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in {path}: {missing_cols}")

        part = df.select(
            [
                reaction_col,
                "adsorbate",
                "dft_ads_eng",
                "mlip_ads_eng_median",
                "metadata_json",
                "label",
                *INFERENCE_DETAIL_COLUMNS,
            ]
        ).rename(
            {
                reaction_col: "reaction",
                "dft_ads_eng": "reference_ads_eng",
                "metadata_json": "reaction_metadata_json",
                "mlip_ads_eng_median": mlip_energy_column_name(model_name),
                "label": mlip_label_column_name(model_name),
                **{
                    detail_name: mlip_detail_column_name(model_name, detail_name)
                    for detail_name in INFERENCE_DETAIL_COLUMNS
                },
            }
        )
        part = part.drop_nulls(
            subset=["reaction", mlip_energy_column_name(model_name)]
        )

        if reference_df is None:
            reference_df = part.select(
                ["reaction", "adsorbate", "reference_ads_eng", "reaction_metadata_json"]
            )
        else:
            ref_part = part.select(
                ["reaction", "adsorbate", "reference_ads_eng", "reaction_metadata_json"]
            )
            overlap = reference_df.join(
                ref_part, on="reaction", how="inner", suffix="_incoming"
            )
            energy_mismatch = overlap.filter(
                pl.col("reference_ads_eng") != pl.col("reference_ads_eng_incoming")
            )
            if energy_mismatch.height > 0:
                raise ValueError(
                    f"Reference energies differ for overlapping reactions in {path}"
                )
            adsorbate_mismatch = overlap.filter(
                pl.col("adsorbate").fill_null("")
                != pl.col("adsorbate_incoming").fill_null("")
            )
            if adsorbate_mismatch.height > 0:
                raise ValueError(
                    f"Adsorbates differ for overlapping reactions in {path}"
                )
            metadata_mismatch = overlap.filter(
                pl.col("reaction_metadata_json").fill_null("")
                != pl.col("reaction_metadata_json_incoming").fill_null("")
            )
            if metadata_mismatch.height > 0:
                raise ValueError(
                    f"Reaction metadata differ for overlapping reactions in {path}"
                )
            reference_df = (
                pl.concat([reference_df, ref_part])
                .unique(subset="reaction", keep="first")
                .sort("reaction")
            )

        mlip_col = mlip_energy_column_name(model_name)
        label_col = mlip_label_column_name(model_name)
        mlip_cols.append(mlip_col)
        label_cols.append(label_col)
        model_detail_cols = [
            mlip_detail_column_name(model_name, detail_name)
            for detail_name in INFERENCE_DETAIL_COLUMNS
        ]
        detail_cols.extend(model_detail_cols)
        wide_parts.append(
            part.select(["reaction", mlip_col, label_col, *model_detail_cols])
        )

    if reference_df is None:
        raise RuntimeError("No MLIP result rows were loaded.")

    wide_df = reference_df.clone()
    for part in wide_parts:
        wide_df = wide_df.join(part, on="reaction", how="inner")

    return wide_df.drop_nulls(
        subset=["reference_ads_eng", *mlip_cols, *label_cols, *detail_cols]
    ).sort("reaction")


def load_result_analysis(result_path: str | Path) -> dict[str, dict[str, Any]]:
    payload = load_result_json(result_path)
    persisted = {
        reaction: reaction_data[RESULT_ANALYSIS_KEY]
        for reaction, reaction_data in payload.items()
        if reaction != "calculation_settings"
        and isinstance(reaction_data, dict)
        and isinstance(reaction_data.get(RESULT_ANALYSIS_KEY), dict)
    }
    if persisted:
        return persisted
    return detect_anomalies_from_result_json(result_path)


def _resolve_result_files(result_files: list[str | Path]) -> list[Path]:
    resolved = [Path(path) for path in result_files]
    if not resolved:
        raise ValueError("result_files must contain at least one explicit result path")
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        missing_paths = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Result JSON paths not found: {missing_paths}")
    return resolved
