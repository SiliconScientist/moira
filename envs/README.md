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

`chgnet` is no longer a pure one-line dependency file. In practice, GPU
execution depends on which torch wheel gets resolved, so this repo now pins
CHGNet to PyTorch's official `cu128` wheel line:

```text
--index-url https://download.pytorch.org/whl/cu128
--extra-index-url https://pypi.org/simple
chgnet==0.4.2
torch==2.10.0+cu128
```

That avoids silently resolving a CPU-only torch wheel or a CUDA wheel newer
than the target cluster driver. Rebuild `envs/chgnet/.venv` after pulling this
change if you created it before the pin was added.

`allegro` now has a verified Python 3.13 path using the current upstream PyPI
packages `nequip-allegro==0.8.3` and `nequip==0.18.0` with `torch==2.10.0`.
That keeps the env definition minimal while still pinning the exact
post-`nequip` v0.7 compatibility line that was tested locally with `uv`.

Allegro's runtime artifact is not a raw training checkpoint. Moira's legacy
adapter expects a compiled ASE model file produced with upstream's
`nequip-compile --target ase`, typically a `.nequip.pt2` file. If your model's
type names already match chemical symbols, you can set
`chemical_species_to_atom_type_map = true` in the model metadata to suppress
NequIP's default warning; otherwise provide an explicit mapping table. See the
repo README's Allegro Setup section for the tested Allegro-OAM-L download,
HPC upload, GPU compilation, and configuration process.

`grace` now has a tested Python 3.13 path, but it still needs a packaging
workaround: TensorFlow 2.21 and `tf_keras` install cleanly on Python 3.13,
while the published `tensorpotential` metadata still requires `tensorflow<2.20`.
The setup script therefore installs GRACE's base dependencies from
`envs/grace/requirements.txt` and then installs `grace-tensorpotential` from a
pinned GitHub commit with `--no-deps`. That requirements file now uses
`tensorflow[and-cuda]==2.21.0` so the GRACE virtualenv carries NVIDIA's
user-space CUDA/cuDNN wheels on clusters that do not provide cuDNN modules.
Override the tensorpotential source with `GRACE_GIT_REF=<tag-or-commit>` if
needed.

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

AQCat25 is sensitive to mixed CUDA builds. The requirements file intentionally
pins the exact `torch==2.12.0+cu126`, `torch_scatter==2.1.2+pt212cu126`, and
`torch_sparse==0.6.18+pt212cu126` line so the compiled PyG extensions match the
PyTorch runtime. If you created `envs/aqcat25/.venv` before those pins were in
place, rebuild just that environment or rerun the full script with `--force`.

AQCat25's patched fairchem dependency also still imports
`scipy.special.sph_harm`, so SciPy is pinned to `1.15.3`. Newer SciPy releases
remove that symbol and will fail at calculator startup.
