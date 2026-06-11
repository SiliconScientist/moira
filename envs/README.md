All virtual environments created with Python 3.11.

Create or rebuild the per-model environments with:

```bash
./envs/setup_mlip_envs.sh
```

This skips any existing `envs/<model>/.venv` by default. To force a full rebuild:

```bash
./envs/setup_mlip_envs.sh --force
```

Only environments with a completed install are skipped. If a prior run failed
mid-install, rerunning the script will rebuild that environment automatically.

By default, all environments use Python 3.11. You can override that default:

```bash
PYTHON_VERSION=3.11 ./envs/setup_mlip_envs.sh
```

Per-model environments should contain only model-specific dependencies.
Rootstock is installed in the main project environment and dispatches to
its own prebuilt MLIP environments. CatBench is provided from the shared
vendored source at `vendor/catbench` via `mlip.catbench_source` and
injected into the Rootstock adapter subprocess with `PYTHONPATH`.
