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

Console script:

```bash
moira submit --config mlip.toml
moira make-tasks --config mlip.toml --run-tag dev --out slurm_output/tasks.txt
moira run-one --config mlip.toml --line "mace example data/raw_data/example.json data/results/mlips/dev/example/mace.json"
```

Module entrypoints:

```bash
python -m moira submit --config mlip.toml
python -m moira.mlip submit --config mlip.toml
```

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
