All virtual environments created with Python 3.13 by default.

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

By default, all environments use `PYTHON_VERSION` except `orb_v3`, which uses
`ORB_V3_PYTHON_VERSION=3.12`:

```bash
PYTHON_VERSION=3.13 ORB_V3_PYTHON_VERSION=3.12 ./envs/setup_mlip_envs.sh
```

Per-model environments should contain only model-specific dependencies.
Rootstock is installed in the main project environment and dispatches to
its own prebuilt MLIP environments. CatBench is provided from the shared
vendored source at `vendor/catbench` via `mlip.catbench_source` and
injected into the Rootstock adapter subprocess with `PYTHONPATH`.

`alphanet` is the one exception that needs extra care for accelerator support:
its requirements file now follows AlphaNet's simpler CUDA torch stack instead
of the optional JAX/Haiku path.
