from __future__ import annotations

from datetime import datetime
import shutil
import subprocess
from pathlib import Path

from moira.mlip.preflight import validate_model_envs
from moira.mlip.tasks import make_tasks
from moira.pathing import get_project_root


def submission_runs_dir(config_path: str | Path) -> Path:
    return get_project_root(config_path) / "slurm_output" / "runs"


def create_submission_run_dir(config_path: str | Path, *, run_tag: str) -> Path:
    runs_dir = submission_runs_dir(config_path)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    run_dir = runs_dir / f"{run_tag}.{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def freeze_config_snapshot(config_path: str | Path, *, run_dir: str | Path) -> Path:
    source_path = Path(config_path).resolve()
    snapshot_path = Path(run_dir).resolve() / source_path.name
    shutil.copy2(source_path, snapshot_path)
    return snapshot_path


def submit_jobs(
    *,
    config_path: str | Path,
    run_tag: str | None,
    datasets: list[str] | None,
    skip_preflight: bool = False,
) -> None:
    resolved_config_path = Path(config_path).resolve()
    if skip_preflight:
        print("Skipping MLIP preflight checks.")
    else:
        print("Running MLIP preflight checks. To skip them, rerun with --skip-preflight.")
        validate_model_envs(resolved_config_path, show_progress=True)

    # Decide run tag
    if run_tag is None:
        run_tag = "run"

    run_dir = create_submission_run_dir(resolved_config_path, run_tag=run_tag)
    frozen_config_path = freeze_config_snapshot(
        resolved_config_path,
        run_dir=run_dir,
    )

    taskfile = run_dir / "mlip_tasks.jsonl"

    # Generate task file
    make_tasks(
        config_path=frozen_config_path,
        run_tag=run_tag,
        out_path=taskfile,
        datasets=datasets,
    )

    # Count tasks
    n_tasks = sum(1 for _ in taskfile.open())
    if n_tasks == 0:
        raise RuntimeError("No MLIP tasks generated; nothing to submit.")

    # Submit Slurm array
    cmd = [
        "sbatch",
        f"--array=0-{n_tasks - 1}",
        "slurm/mlip_one.sbatch",
        str(taskfile),
        str(frozen_config_path),
    ]

    print("Submitting Slurm array:")
    print(" ", " ".join(cmd))

    subprocess.run(cmd, check=True)
