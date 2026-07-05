from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from moira.pathing import resolve_project_path


def resolve_results_dir(
    results_dir: str | None,
    *,
    config_path: str | Path,
    dev_run: bool = False,
) -> Path | None:
    if results_dir is None:
        return None

    resolved_config_path = Path(config_path).resolve()
    path = Path(results_dir)
    if not path.is_absolute():
        path = resolve_project_path(path, config_path=resolved_config_path)
    if dev_run and not path.name.endswith("_dev"):
        path = path.with_name(f"{path.name}_dev")
    return path


@contextmanager
def patch_adsorption_paths(
    *,
    dataset_path: str | None = None,
    results_dir: str | Path | None = None,
):
    if dataset_path is None and results_dir is None:
        yield
        return

    resolved_dataset_path = None
    if dataset_path is not None:
        resolved_dataset_path = str(Path(dataset_path).resolve())

    resolved_results_dir = None
    if results_dir is not None:
        resolved_results_dir = str(Path(results_dir).resolve())

    import catbench.adsorption.calculation.calculation as adsorption_calculation
    import catbench.utils.io_utils as io_utils

    def get_result_directory(mlip_name: str, mode_suffix: str = "") -> str:
        if resolved_results_dir is None:
            return io_utils.get_result_directory(mlip_name, mode_suffix=mode_suffix)

        base_path = Path(resolved_results_dir)
        if mode_suffix:
            base_path = base_path.parent / f"{base_path.name}_{mode_suffix}"
        return str(base_path / mlip_name)

    with ExitStack() as stack:
        if resolved_dataset_path is not None:
            stack.enter_context(
                patch.object(
                    io_utils,
                    "get_raw_data_path",
                    lambda _benchmark: resolved_dataset_path,
                )
            )
            stack.enter_context(
                patch.object(
                    adsorption_calculation,
                    "get_raw_data_path",
                    lambda _benchmark: resolved_dataset_path,
                )
            )
        if resolved_results_dir is not None:
            stack.enter_context(
                patch.object(io_utils, "get_result_directory", get_result_directory)
            )
            stack.enter_context(
                patch.object(
                    adsorption_calculation,
                    "get_result_directory",
                    get_result_directory,
                )
            )
        yield
