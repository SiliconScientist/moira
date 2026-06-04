from __future__ import annotations

import importlib
import runpy
import sys
import unittest
from unittest.mock import patch

from moira.__main__ import main
from moira.mlip.cli import main as mlip_main


class MainDispatchTests(unittest.TestCase):
    def test_python_dash_m_moira_forwards_to_mlip_cli(self) -> None:
        with patch("moira.mlip.cli.main") as mock_mlip_main:
            runpy.run_module("moira", run_name="__main__")

        mock_mlip_main.assert_called_once_with()

    def test_importing_main_does_not_load_non_mlip_modules(self) -> None:
        sys.modules.pop("moira.__main__", None)
        before_import = set(sys.modules)

        importlib.import_module("moira.__main__")

        for name in (
            "moira.adapters.rootstock_adapter",
            "moira.mlip.runner",
            "moira.mlip.submit",
            "moira.mlip.tasks",
        ):
            self.assertNotIn(name, set(sys.modules) - before_import)


class MlipCliTests(unittest.TestCase):
    def test_importing_mlip_cli_does_not_load_experiment_modules(self) -> None:
        sys.modules.pop("moira.mlip.cli", None)
        before_import = set(sys.modules)

        importlib.import_module("moira.mlip.cli")

        for name in (
            "moira.analysis",
            "moira.exp",
            "moira.experiment_runner",
            "moira.graphs",
            "moira.learning_curve.registry",
            "moira.learning_curve.results_io",
            "moira.learning_curve.runners",
            "moira.plot",
            "moira.probe",
            "moira.probe_features",
        ):
            self.assertNotIn(name, set(sys.modules) - before_import)

    def test_submit_delegates_to_submit_jobs(self) -> None:
        with patch("moira.mlip.submit.submit_jobs") as mock_submit_jobs:
            mlip_main(
                ["submit", "--config", "mlip.toml", "--run-tag", "tag", "dataset.json"]
            )

        mock_submit_jobs.assert_called_once_with(
            config_path="mlip.toml",
            run_tag="tag",
            datasets=["dataset.json"],
        )

    def test_run_one_delegates_to_runner(self) -> None:
        with patch("moira.mlip.runner.run_one_task") as mock_run_one_task:
            mlip_main(["run-one", "--line", "task-line", "--config", "mlip.toml"])

        mock_run_one_task.assert_called_once_with(
            line="task-line",
            config_path="mlip.toml",
        )

    def test_make_tasks_delegates_to_task_writer(self) -> None:
        with patch("moira.mlip.tasks.make_tasks") as mock_make_tasks:
            mlip_main(
                [
                    "make-tasks",
                    "--config",
                    "mlip.toml",
                    "--run-tag",
                    "tag",
                    "--out",
                    "tasks.txt",
                    "dataset.json",
                ]
            )

        mock_make_tasks.assert_called_once_with(
            config_path="mlip.toml",
            run_tag="tag",
            out_path="tasks.txt",
            datasets=["dataset.json"],
        )

    def test_python_dash_m_moira_mlip_delegates_to_cli_main(self) -> None:
        with patch("moira.mlip.cli.main") as mock_mlip_main:
            with patch.object(sys, "argv", ["python", "submit", "--config", "mlip.toml"]):
                runpy.run_module("moira.mlip", run_name="__main__")

        mock_mlip_main.assert_called_once_with()
