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
its requirements file follows the simpler torch path instead of the optional
JAX/Haiku path, but uses a newer PyTorch Geometric wheel set than AlphaNet
upstream documents so it can stay on the project-wide Python 3.13 default.

`grace` now has a tested Python 3.13 path, but it still needs a packaging
workaround: TensorFlow 2.21 and `tf_keras` install cleanly on Python 3.13,
while the published `tensorpotential` metadata still requires `tensorflow<2.20`.
The setup script therefore installs GRACE's base dependencies from
`envs/grace/requirements.txt` and then installs `grace-tensorpotential` from a
pinned GitHub commit with `--no-deps`. Override that source with
`GRACE_GIT_REF=<tag-or-commit>` if needed.

`aqcat25` also needs a packaging workaround. The AQCat25 model card documents a
Python 3.10 / `torch==2.4.0` path, but the fairchem tag it depends on declares
`requires-python <3.13`. This repo instead uses a Python 3.13-compatible torch
stack in `envs/aqcat25/requirements.txt`, then installs the base
`fairchem-core` tag with `--no-deps`. After that, you still need to accept the
gated Hugging Face terms, download the AQCat25 files locally, and run:

```bash
./envs/aqcat25/patch_fairchem.sh /path/to/aqcat25-ev2
```

That helper copies the gated `equiformer_v2_film.py` and patched
`ase_utils.py` files into the installed `fairchem-core` package inside
`envs/aqcat25/.venv`.
