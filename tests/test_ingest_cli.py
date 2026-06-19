from __future__ import annotations

import runpy
import sys
import unittest
from unittest.mock import patch

from moira.ingest.to_catbench import main as ingest_main


class IngestCliTests(unittest.TestCase):
    def test_python_dash_m_moira_ingest_delegates_to_to_catbench_main(self) -> None:
        with patch("moira.ingest.to_catbench.main") as mock_main:
            with patch.object(sys, "argv", ["python", "--config", "config.toml"]):
                runpy.run_module("moira.ingest", run_name="__main__")

        mock_main.assert_called_once_with()

    def test_default_invocation_uses_config_default(self) -> None:
        with patch("moira.ingest.to_catbench.get_config") as mock_get_config:
            with patch("moira.ingest.to_catbench.load_dataset") as mock_load_dataset:
                with patch(
                    "moira.ingest.to_catbench.build_coefficients",
                    return_value={},
                ):
                    with patch("moira.ingest.to_catbench.write_dataset"):
                        mock_load_dataset.return_value = object()
                        ingest_main([])

        mock_get_config.assert_called_once_with("config.toml")

    def test_config_flag_is_forwarded(self) -> None:
        with patch("moira.ingest.to_catbench.get_config") as mock_get_config:
            with patch("moira.ingest.to_catbench.load_dataset") as mock_load_dataset:
                with patch(
                    "moira.ingest.to_catbench.build_coefficients",
                    return_value={},
                ):
                    with patch("moira.ingest.to_catbench.write_dataset"):
                        mock_load_dataset.return_value = object()
                        ingest_main(["--config", "custom.toml"])

        mock_get_config.assert_called_once_with("custom.toml")
