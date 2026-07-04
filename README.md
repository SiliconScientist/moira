# Moira
MLIP ingestion and prediction workflows extracted from Oasis.

## Naming

- Project name: `Moira`
- Repository slug: `moira`
- Python package name: `moira`
- CLI: `moira`
- Python import namespace: `moira`

## Environment

Install the project and test dependencies with:

```bash
pip install -e .[dev]
```

If you use `uv`, the equivalent is:

```bash
uv pip install -e ".[dev]"
```

Run the test suite with:

```bash
PYTHONPATH=src python -m unittest
```

## AQCat25 Setup

AQCat25 needs one extra install step beyond creating the model environment.

1. Create the MLIP environments:

```bash
./envs/setup_mlip_envs.sh
```

AQCat25's environment is pinned to a specific `torch==2.12.0+cu126` /
`torch_scatter==2.1.2+pt212cu126` / `torch_sparse==0.6.18+pt212cu126` /
`scipy==1.15.3` stack.
If you already created `envs/aqcat25/.venv` before this change, rebuild it:

```bash
./envs/setup_mlip_envs.sh --force
```

2. Request access to the gated Hugging Face repo `SandboxAQ/aqcat25-ev2` and
download it locally. The adapter needs both the pretrained checkpoint and the
patched fairchem source files from that repo.

Example:

```bash
hf download SandboxAQ/aqcat25-ev2 --repo-type model --local-dir ~/Downloads/aqcat25-ev2
```

3. Patch the installed `fairchem-core` inside `envs/aqcat25/.venv`:

```bash
./envs/aqcat25/patch_fairchem.sh ~/Downloads/aqcat25-ev2
```

Before patching, you can verify the torch stack resolves to the expected CUDA
line:

```bash
./envs/aqcat25/.venv/bin/python -c "import torch, torch_scatter, torch_sparse; print(torch.__version__); print(torch.version.cuda)"
```

Expected output includes `2.12.0+cu126` and `12.6`.

4. Point `mlip.rootstock.models.aqcat25.checkpoint` at the downloaded AQCat25
checkpoint `.pt` file in your `config.toml`.

AQCat25 optionally reads `[mlip.rootstock.models.aqcat25.metadata]`. The
supported override today is `is_spin_off`; set `is_spin_off = false` to force
spin-polarized mode instead of the default element-based heuristic. The
downloaded `aqcat25-ev2` directory is only needed for the patch step above, not
at runtime.

## CHGNet Setup

CHGNet is the simplest legacy environment in this repo.

1. Create the MLIP environments:

```bash
./envs/setup_mlip_envs.sh
```

`envs/chgnet/requirements.txt` is intentionally just:

```text
chgnet==0.4.2
```

That exact path was verified in a temporary `uv` Python 3.13 environment with:
- `uv pip install chgnet`
- `from chgnet.model.model import CHGNet; CHGNet.load()`

2. Switch the model to the legacy backend in `config.toml` and enable `chgnet`.

3. Set `mlip.rootstock.models.chgnet.checkpoint` to CHGNet's upstream model
selector. Today that means `"0.3.0"` for the default pretrained model or
`"r2scan"` for the R2SCAN transfer-learned model.

## Entrypoints

Single entrypoint:

```bash
moira --config config.toml
moira --config config.toml --run-tag dev data/raw_data/example_adsorption.json
```

`moira` and `python -m moira` both run the config-driven MLIP workflow:
- Enabled models come from `config.toml`
- Optional dataset arguments override `mlip.dataset` or `mlip.datasets`
- Optional `mlip.shard_size` or `mlip.num_shards` split large adsorption JSONs
  into one task per `model x shard`
- Optional `mlip.results_dir` redirects CatBench output from `./result` to a
  path you control, for example `data/results/MamunHighT2019`; when
  `mlip.dev_run = true`, Moira automatically uses a `_dev` suffixed sibling
- Moira patches CatBench's adsorption loader so the resolved dataset JSON path is
  used directly; you do not need a `raw_data/` directory in the execution cwd
- Sharded tasks materialize shard-local dataset JSONs at runtime and write
  shard-isolated result directories to avoid collisions between array jobs
- `--run-tag` controls the Slurm task/result grouping

Module entrypoints:

```bash
python -m moira --config config.toml
python -m moira.mlip --config config.toml
```

Sharding helpers:

```bash
python -m moira.mlip make-tasks --config config.toml --run-tag screen --out slurm_output/mlip_tasks_screen.jsonl
python -m moira.mlip collect-shards data/results/trimetallic_n_dev
```

`make-tasks` writes JSONL task records for Slurm array execution. When sharding is
enabled, each task record contains the full dataset path plus shard bounds, and the
runner materializes the shard locally before invoking CatBench.

## Ingest

Use the ingest pipeline directly with:

```bash
python -m moira.ingest --config config.toml
```

This reads `[ingest]` from `config.toml`, loads the source dataset, builds the
CatBench coefficient map from `[ingest.stoich]`, and writes the emitted
adsorption JSON.

- If `ingest.catbench_folder` is set, Moira keeps the full emitted CatBench-style dataset tree there.
- If `ingest.catbench_folder` is unset, Moira stages the intermediate dataset in a temporary directory for the current run.
- `moira.ingest` is a Python module entrypoint, not a shell command. `moira.ingest` by itself will not work in `zsh`.

## Sharded Runs

Use one sharding mode at a time:

- `mlip.shard_size = 1000`: create as many shards as needed with at most 1000
  reactions per shard
- `mlip.num_shards = 32`: partition the dataset into 32 shards
- `mlip.shard_index = 0`: optional manual selection for one shard, mainly useful
  for local debugging

Typical Slurm flow:

```bash
python -m moira.mlip make-tasks --config config.toml --run-tag screen --out slurm_output/mlip_tasks_screen.jsonl
sbatch --array=0-$(($(wc -l < slurm_output/mlip_tasks_screen.jsonl)-1)) slurm/mlip_one.sbatch slurm_output/mlip_tasks_screen.jsonl config.toml
python -m moira.mlip collect-shards data/results/trimetallic_n_dev
```

Each shard writes to a shard-specific result directory under `mlip.results_dir`.
This keeps CatBench restart logic local to that shard and prevents concurrent jobs
from rewriting the same `*_result.json`.

When you point `collect-shards` at one shard-root directory and omit `--mlip`,
Moira autodetects every MLIP present across the shards and writes merged
`*_result.json` files to that directory. If shard efficiency JSONs are present,
it also writes per-MLIP `*_efficiency.csv` and `*_efficiency.json` summaries.

## Slab Cache

Moira can reuse relaxed clean-slab calculations across adsorption reactions that
share the same underlying slab.

- Cache identity: the key is derived from source metadata `parent_slab_id` plus
  model identity (`model_name`, `mlip_name`) and relaxation settings such as
  optimizer, force threshold, step limit, damping, rate, and chemical bond cutoff.
- Cache location: non-sharded runs default to a local `slab_cache/` directory
  under the CatBench result directory for that run. Sharded runs use a shared
  location under `mlip.results_dir/<base_dataset_name>/_shared/slab_cache` so all
  shards for the same dataset can see the same entries.
- Invalidation: changing any field that participates in cache identity produces a
  new cache key automatically. In practice, changing `parent_slab_id`, model
  selection, or slab relaxation settings invalidates reuse. Manual invalidation is
  just deleting the corresponding cache files or the shared `slab_cache/` folder.
- Shard interaction: shard-local result JSONs remain isolated, but slab-cache
  entries are intentionally shared for one dataset run. Concurrent writers use
  atomic replace with unique temporary files so a cache collision does not leave a
  partial JSON artifact.
- Diagnostics: result payloads and `*_efficiency.json` summaries record
  `slab_cache_hit`, `slab_cache_hit_count`, and
  `saved_slab_time_estimate_seconds`.

## Scope

This repo contains:

- `moira.mlip` runtime, artifact loading, and result parsing
- `moira.adapters` integration with Rootstock/CatBench
- `moira.ingest` dataset ingestion helpers
- MLIP-facing config models in `moira.mlip_config`

## Testing

```bash
PYTHONPATH=src python -m unittest tests.test_mlip_cli tests.test_mlip_artifacts
```
