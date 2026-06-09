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
- Optional `mlip.results_dir` redirects CatBench output from `./result` to a
  path you control, for example `data/results/MamunHighT2019`; when
  `mlip.dev_run = true`, Moira automatically uses a `_dev` suffixed sibling
- Moira patches CatBench's adsorption loader so the resolved dataset JSON path is
  used directly; you do not need a `raw_data/` directory in the execution cwd
- `--run-tag` controls the Slurm task/result grouping

Module entrypoints:

```bash
python -m moira --config mlip.toml
python -m moira.mlip --config mlip.toml
```

Lower-level task creation and single-task execution are internal implementation details behind this workflow, not the public CLI surface.

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
