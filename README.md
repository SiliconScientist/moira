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

## Entrypoints

Single entrypoint:

```bash
moira --config mlip.toml
moira --config mlip.toml --run-tag dev data/raw_data/example_adsorption.json
```

`moira` and `python -m moira` both run the config-driven MLIP workflow:
- Enabled models come from `mlip.toml`
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
python -m moira --config mlip.toml
python -m moira.mlip --config mlip.toml
```

Sharding helpers:

```bash
python -m moira.mlip make-tasks --config mlip.toml --run-tag screen --out slurm_output/mlip_tasks_screen.jsonl
python -m moira.mlip merge-shards --out data/results/mace_result.json shard_a.json shard_b.json
```

`make-tasks` writes JSONL task records for Slurm array execution. When sharding is
enabled, each task record contains the full dataset path plus shard bounds, and the
runner materializes the shard locally before invoking CatBench.

## Sharded Runs

Use one sharding mode at a time:

- `mlip.shard_size = 1000`: create as many shards as needed with at most 1000
  reactions per shard
- `mlip.num_shards = 32`: partition the dataset into 32 shards
- `mlip.shard_index = 0`: optional manual selection for one shard, mainly useful
  for local debugging

Typical Slurm flow:

```bash
python -m moira.mlip make-tasks --config mlip.toml --run-tag screen --out slurm_output/mlip_tasks_screen.jsonl
sbatch --array=0-$(($(wc -l < slurm_output/mlip_tasks_screen.jsonl)-1)) slurm/mlip_one.sbatch slurm_output/mlip_tasks_screen.jsonl mlip.toml
python -m moira.mlip merge-shards --out data/results/mace_result.json data/results/run/example_shard_00_of_32/mace-mh-1/mace-mh-1_result.json data/results/run/example_shard_01_of_32/mace-mh-1/mace-mh-1_result.json
```

Each shard writes to a shard-specific result directory under `mlip.results_dir`.
This keeps CatBench restart logic local to that shard and prevents concurrent jobs
from rewriting the same `*_result.json`.

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
