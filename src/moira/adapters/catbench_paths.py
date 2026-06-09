from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch


@contextmanager
def patch_adsorption_dataset_path(dataset_path: str | None):
    if dataset_path is None:
        yield
        return

    resolved_path = str(Path(dataset_path).resolve())

    import catbench.adsorption.calculation.calculation as adsorption_calculation
    import catbench.utils.io_utils as io_utils

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(io_utils, "get_raw_data_path", lambda _benchmark: resolved_path)
        )
        stack.enter_context(
            patch.object(
                adsorption_calculation,
                "get_raw_data_path",
                lambda _benchmark: resolved_path,
            )
        )
        yield
