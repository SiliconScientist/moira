All virtual environments created with Python 3.13.0.

Create or rebuild the per-model environments with:

```bash
./envs/setup_mlip_envs.sh
```

Per-model environments should contain only model-specific dependencies.
Rootstock is installed in the main project environment and dispatches to
its own prebuilt MLIP environments. CatBench is provided from the shared
vendored source at `vendor/catbench` via `mlip.catbench_source` and
injected into the Rootstock adapter subprocess with `PYTHONPATH`.
