from __future__ import annotations

import subprocess
from pathlib import Path

from moira.mlip.preflight import validate_model_envs
from moira.mlip.tasks import make_tasks


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

    taskfile = Path("slurm_output") / f"mlip_tasks_{run_tag}.jsonl"
    taskfile.parent.mkdir(parents=True, exist_ok=True)

    # Generate task file
    make_tasks(
        config_path=resolved_config_path,
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
        str(resolved_config_path),
    ]

    print("Submitting Slurm array:")
    print(" ", " ".join(cmd))

    subprocess.run(cmd, check=True)
