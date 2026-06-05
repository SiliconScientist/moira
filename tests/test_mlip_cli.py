from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from moira.__main__ import main
from moira.mlip.cli import main as mlip_main
from moira.mlip.registry import get_model_specs
from moira.mlip.runner import run_one_task
from moira.mlip.tasks import make_task_lines


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


class MlipTaskTests(unittest.TestCase):
    def test_make_task_lines_use_model_work_path_not_model_json(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            dataset_path = tmp / "example_adsorption.json"
            dataset_path.write_text("{}", encoding="utf-8")
            config_path = tmp / "mlip.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[mlip]",
                        "dev_n = 2",
                        "dev_run = false",
                        "",
                        "[mlip.models]",
                        'enabled = ["mace"]',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            lines = make_task_lines(
                config_path=config_path,
                run_tag="dev",
                datasets=[str(dataset_path)],
            )

        self.assertEqual(
            lines,
            [
                "mace example "
                f"{dataset_path.as_posix()} data/results/mlips/dev/example/mace"
            ],
        )


class MlipRegistryTests(unittest.TestCase):
    def test_model_specs_expose_importable_adapter_callable(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "mlip.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[mlip]",
                        "dev_n = 2",
                        "dev_run = false",
                        "",
                        "[mlip.models]",
                        'enabled = ["mace"]',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            specs = get_model_specs(config_path)

        self.assertEqual(specs["mace"].adapter_module, "moira.adapters.rootstock_adapter")
        self.assertEqual(specs["mace"].adapter_function, "run")


class MlipRunnerTests(unittest.TestCase):
    def test_run_one_task_dispatches_adapter_in_process(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "mlip.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[mlip]",
                        "dev_n = 2",
                        "dev_run = false",
                        "",
                        "[mlip.models]",
                        'enabled = ["mace"]',
                        "",
                        "[mlip.rootstock.models.mace]",
                        'model = "mace"',
                        'mlip_name = "mace-mh-1"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            mock_runner = Mock()
            with patch(
                "moira.mlip.runner.importlib.import_module",
                return_value=SimpleNamespace(run=mock_runner),
            ) as mock_import_module:

                run_one_task(
                    "mace example data/raw_data/example.json data/results/mlips/dev/example/mace",
                    str(config_path),
                )

        mock_import_module.assert_called_once_with("moira.adapters.rootstock_adapter")
        mock_runner.assert_called_once_with(
            model="mace",
            input_path="data/raw_data/example.json",
            output_path="data/results/mlips/dev/example/mace",
            dataset_name="example",
            config_path=str(config_path),
        )
