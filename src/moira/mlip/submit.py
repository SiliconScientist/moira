from __future__ import annotations

from datetime import datetime
import re
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


def _parse_sbatch_job_id(output: str) -> str | None:
    match = re.search(r"\bSubmitted batch job (\d+)\b", output)
    if match is None:
        return None
    return match.group(1)


def write_submission_job_id(*, run_dir: str | Path, job_id: str) -> Path:
    job_id_path = Path(run_dir).resolve() / "slurm_job_id.txt"
    job_id_path.write_text(f"{job_id}\n", encoding="utf-8")
    return job_id_path


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

    stdout_path = run_dir / "slurm_%x_%A_%a.out"
    stderr_path = run_dir / "slurm_%x_%A_%a.err"

    # Submit Slurm array
    cmd = [
        "sbatch",
        f"--array=0-{n_tasks - 1}",
        f"--output={stdout_path}",
        f"--error={stderr_path}",
        "slurm/mlip_one.sbatch",
        str(taskfile),
        str(frozen_config_path),
    ]

    print("Submitting Slurm array:")
    print(" ", " ".join(cmd))

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)

    job_id = _parse_sbatch_job_id(result.stdout)
    if job_id is not None:
        write_submission_job_id(run_dir=run_dir, job_id=job_id)
