from __future__ import annotations

import importlib
import json
import os
import runpy
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch

from moira.__main__ import main
from moira.mlip.cli import main as mlip_main
from moira.mlip.preflight import validate_model_envs
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

    def test_default_invocation_uses_config_defaults(self) -> None:
        with patch("moira.mlip.submit.submit_jobs") as mock_submit_jobs:
            mlip_main([])

        mock_submit_jobs.assert_called_once_with(
            config_path="mlip.toml",
            run_tag=None,
            datasets=[],
        )

    def test_default_invocation_delegates_to_submit_jobs(self) -> None:
        with patch("moira.mlip.submit.submit_jobs") as mock_submit_jobs:
            mlip_main(["--config", "mlip.toml", "--run-tag", "tag", "dataset.json"])

        mock_submit_jobs.assert_called_once_with(
            config_path="mlip.toml",
            run_tag="tag",
            datasets=["dataset.json"],
        )

    def test_python_dash_m_moira_mlip_delegates_to_cli_main(self) -> None:
        with patch("moira.mlip.cli.main") as mock_mlip_main:
            with patch.object(sys, "argv", ["python", "--config", "mlip.toml"]):
                runpy.run_module("moira.mlip", run_name="__main__")

        mock_mlip_main.assert_called_once_with()

    def test_python_dash_m_moira_mlip_run_one_supports_line_parameter(self) -> None:
        with patch("moira.mlip.runner.run_one_task") as mock_run_one_task:
            with patch.object(
                sys,
                "argv",
                [
                    "python",
                    "run-one",
                    "--line",
                    '{"model": "mace", "dataset_name": "example"}',
                    "--config",
                    "mlip.toml",
                ],
            ):
                runpy.run_module("moira.mlip", run_name="__main__")

        mock_run_one_task.assert_called_once_with(
            '{"model": "mace", "dataset_name": "example"}',
            "mlip.toml",
        )


class MlipTaskTests(unittest.TestCase):
    def test_make_task_lines_emit_json_records(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            dataset_path = tmp / "example_adsorption.json"
            dataset_path.write_text("{}", encoding="utf-8")
            config_path = tmp / "mlip.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[mlip]",
                        'device = "cpu"',
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

        self.assertEqual(len(lines), 1)
        self.assertEqual(
            json.loads(lines[0]),
            {
                "model": "mace",
                "dataset_name": "example",
            },
        )

    def test_make_task_lines_use_generated_dev_dataset_name(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            dataset_path = tmp / "example_adsorption.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "first": {"value": 1},
                        "second": {"value": 2},
                        "third": {"value": 3},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = tmp / "mlip.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[mlip]",
                        'device = "cpu"',
                        "dev_n = 2",
                        "dev_run = true",
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
            dev_dataset_path = tmp / "example_dev_adsorption.json"
            self.assertTrue(dev_dataset_path.exists())
            self.assertEqual(
                json.loads(dev_dataset_path.read_text(encoding="utf-8")),
                {
                    "first": {"value": 1},
                    "second": {"value": 2},
                },
            )
            self.assertEqual(len(lines), 1)
            self.assertEqual(
                json.loads(lines[0]),
                {
                    "model": "mace",
                    "dataset_name": "example_dev",
                },
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

        self.assertEqual(
            specs["mace"].adapter_module, "moira.adapters.rootstock_adapter"
        )
        self.assertEqual(specs["mace"].adapter_function, "run")

    def test_model_specs_can_select_legacy_adapters(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            legacy_python = tmp_path / "envs" / "mace" / ".venv" / "bin" / "python"
            legacy_python.parent.mkdir(parents=True)
            legacy_python.write_text("", encoding="utf-8")
            config_path = tmp_path / "mlip.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[mlip]",
                        'adapter_backend = "legacy"',
                        'device = "cpu"',
                        "dev_n = 2",
                        "dev_run = false",
                        "",
                        "[mlip.models]",
                        'enabled = ["mace", "uma"]',
                        "",
                        "[mlip.rootstock.models.mace]",
                        'model = "mace"',
                        'mlip_name = "mace-mh-1"',
                        "",
                        "[mlip.rootstock.models.uma]",
                        'model = "uma"',
                        'mlip_name = "uma-s-1p1"',
                        'checkpoint = "uma.pt"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            specs = get_model_specs(config_path)

        self.assertEqual(
            specs["mace"].adapter_module, "moira.adapters.legacy.mace_adapter"
        )
        self.assertEqual(
            specs["uma"].adapter_module, "moira.adapters.legacy.uma_adapter"
        )
        self.assertEqual(specs["mace"].python, str(legacy_python.resolve()))

    def test_run_one_task_reexecs_into_legacy_model_python(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            legacy_python = tmp_path / "envs" / "mace" / ".venv" / "bin" / "python"
            legacy_python.parent.mkdir(parents=True)
            legacy_python.write_text("", encoding="utf-8")
            config_path = tmp_path / "mlip.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[mlip]",
                        'adapter_backend = "legacy"',
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

            with (
                patch(
                    "moira.mlip.runner.os.execve",
                    side_effect=SystemExit(0),
                ) as mock_execve,
                patch("moira.mlip.runner.sys.executable", str(tmp_path / "python")),
            ):
                with self.assertRaises(SystemExit):
                    run_one_task(
                        '{"model": "mace", "dataset_name": "example"}',
                        str(config_path),
                    )

                mock_execve.assert_called_once()
                exec_path, exec_argv, exec_env = mock_execve.call_args.args

        self.assertEqual(exec_path, str(legacy_python.resolve()))
        self.assertEqual(
            exec_argv,
            [
                str(legacy_python.resolve()),
                "-m",
                "moira.mlip",
                "run-one",
                "--line",
                '{"model": "mace", "dataset_name": "example"}',
                "--config",
                str(config_path.resolve()),
            ],
        )
        self.assertEqual(
            exec_env["MOIRA_ACTIVE_MODEL_PYTHON"],
            str(legacy_python.resolve()),
        )
        self.assertIn(
            str(Path(__file__).resolve().parents[1] / "src"),
            exec_env["PYTHONPATH"].split(os.pathsep),
        )

    def test_run_one_task_reexecs_when_venvs_share_base_interpreter(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            legacy_python = tmp_path / "envs" / "mace" / ".venv" / "bin" / "python"
            legacy_python.parent.mkdir(parents=True)
            legacy_python.write_text("", encoding="utf-8")
            config_path = tmp_path / "mlip.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[mlip]",
                        'adapter_backend = "legacy"',
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
            shared_base = tmp_path / "shared-python" / "python3.13"
            shared_base.parent.mkdir(parents=True)
            shared_base.write_text("", encoding="utf-8")
            main_python = tmp_path / ".venv" / "bin" / "python"
            main_python.parent.mkdir(parents=True)
            main_python.symlink_to(shared_base)
            legacy_python.unlink()
            legacy_python.symlink_to(shared_base)

            with (
                patch(
                    "moira.mlip.runner.os.execve",
                    side_effect=SystemExit(0),
                ) as mock_execve,
                patch("moira.mlip.runner.sys.executable", str(main_python)),
            ):
                with self.assertRaises(SystemExit):
                    run_one_task(
                        '{"model": "mace", "dataset_name": "example"}',
                        str(config_path),
                    )

                mock_execve.assert_called_once()


class MlipPreflightTests(unittest.TestCase):
    def test_validate_model_envs_checks_legacy_imports(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            legacy_python = tmp_path / "envs" / "mace" / ".venv" / "bin" / "python"
            legacy_python.parent.mkdir(parents=True)
            legacy_python.write_text("", encoding="utf-8")
            config_path = tmp_path / "mlip.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[mlip]",
                        'adapter_backend = "legacy"',
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

            with patch("moira.mlip.preflight.subprocess.run") as mock_run:
                mock_run.return_value = SimpleNamespace(
                    returncode=1,
                    stderr="ModuleNotFoundError: No module named 'mace'",
                    stdout="",
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Model 'mace' failed preflight import",
                ):
                    validate_model_envs(config_path)

                mock_run.assert_called_once_with(
                    [str(legacy_python.resolve()), "-c", "import mace"],
                    capture_output=True,
                    text=True,
                )

    def test_submit_jobs_runs_preflight_before_sbatch(self) -> None:
        with patch("moira.mlip.submit.validate_model_envs") as mock_validate:
            with patch("moira.mlip.submit.make_tasks") as mock_make_tasks:
                with patch("pathlib.Path.open", mock_open(read_data='{"model":"mace"}\n')):
                    with patch("moira.mlip.submit.subprocess.run") as mock_run:
                        from moira.mlip.submit import submit_jobs

                        submit_jobs(
                            config_path="mlip.toml",
                            run_tag="run",
                            datasets=[],
                        )

        mock_validate.assert_called_once()
        mock_run.assert_called_once()


class MlipRunnerTests(unittest.TestCase):
    def test_run_one_task_dispatches_adapter_in_process(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "mlip.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[mlip]",
                        'device = "cpu"',
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
                    (
                        '{"model": "mace", "dataset_name": "example", '
                        '"input_path": "data/raw_data/example.json"}'
                    ),
                    str(config_path),
                )

        mock_import_module.assert_called_once_with("moira.adapters.rootstock_adapter")
        mock_runner.assert_called_once_with(
            model="mace",
            dataset_name="example",
            device="cpu",
            config_path=str(config_path.resolve()),
        )

    def test_run_one_task_dispatches_legacy_adapter_when_selected(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "mlip.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[mlip]",
                        'adapter_backend = "legacy"',
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
                    (
                        '{"model": "mace", "dataset_name": "example", '
                        '"input_path": "data/raw_data/example.json"}'
                    ),
                    str(config_path),
                )

        mock_import_module.assert_called_once_with("moira.adapters.legacy.mace_adapter")
        mock_runner.assert_called_once_with(
            model="mace",
            dataset_name="example",
            device="cpu",
            config_path=str(config_path.resolve()),
        )

    def test_run_one_task_accepts_legacy_task_lines(self) -> None:
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
            ):
                run_one_task(
                    "mace example data/raw_data/example.json data/results/mlips/dev/example/mace",
                    str(config_path),
                )

        mock_runner.assert_called_once_with(
            model="mace",
            dataset_name="example",
            device="cpu",
            config_path=str(config_path.resolve()),
        )

    def test_run_one_task_uses_configured_cpu_device(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "mlip.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[mlip]",
                        'device = "cpu"',
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
            ):
                run_one_task(
                    '{"model": "mace", "dataset_name": "example"}',
                    str(config_path),
                )

        mock_runner.assert_called_once_with(
            model="mace",
            dataset_name="example",
            device="cpu",
            config_path=str(config_path.resolve()),
        )

    def test_run_one_task_falls_back_to_cpu_when_cuda_unavailable(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "mlip.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[mlip]",
                        'device = "cuda"',
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
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False)
            )
            with (
                patch.dict(sys.modules, {"torch": fake_torch}),
                patch(
                    "moira.mlip.runner.importlib.import_module",
                    return_value=SimpleNamespace(run=mock_runner),
                ),
            ):
                run_one_task(
                    '{"model": "mace", "dataset_name": "example"}',
                    str(config_path),
                )

        mock_runner.assert_called_once_with(
            model="mace",
            dataset_name="example",
            device="cpu",
            config_path=str(config_path.resolve()),
        )
