from __future__ import annotations

from datetime import datetime
import shutil
import subprocess
from pathlib import Path

from moira.mlip.preflight import validate_model_envs
from moira.mlip.tasks import make_tasks
from moira.pathing import get_project_root


def config_snapshot_dir(config_path: str | Path) -> Path:
    return get_project_root(config_path) / "slurm_output" / "config_snapshots"


def freeze_config_snapshot(config_path: str | Path, *, run_tag: str) -> Path:
    source_path = Path(config_path).resolve()
    snapshot_dir = config_snapshot_dir(source_path)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    snapshot_name = f"{source_path.stem}.{run_tag}.{timestamp}{source_path.suffix}"
    snapshot_path = snapshot_dir / snapshot_name
    shutil.copy2(source_path, snapshot_path)
    return snapshot_path


def submit_jobs(
    *,
    config_path: str | Path,
    run_tag: str | None,
    datasets: list[str] | None,
) -> None:
    resolved_config_path = Path(config_path).resolve()
    validate_model_envs(resolved_config_path)

    # Decide run tag
    if run_tag is None:
        run_tag = "run"

    frozen_config_path = freeze_config_snapshot(resolved_config_path, run_tag=run_tag)

    taskfile = Path("slurm_output") / f"mlip_tasks_{run_tag}.jsonl"
    taskfile.parent.mkdir(parents=True, exist_ok=True)

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
