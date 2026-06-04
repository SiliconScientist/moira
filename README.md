# Moirai
MLIP ingestion and prediction workflows extracted from Oasis.

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
moirai submit --config mlip.toml
moirai make-tasks --config mlip.toml --run-tag dev --out slurm_output/tasks.txt
moirai run-one --config mlip.toml --line "mace example data/raw_data/example.json data/results/mlips/dev/example/mace.json"
```

Module entrypoints:

```bash
python -m oasis submit --config mlip.toml
python -m oasis.mlip submit --config mlip.toml
```

## Scope

This repo contains:

- `oasis.mlip` runtime, artifact loading, and result parsing
- `oasis.adapters` integration with Rootstock/CatBench
- `oasis.ingest` dataset ingestion helpers
- MLIP-facing config models in `oasis.mlip_config`

## Testing

```bash
PYTHONPATH=src python -m unittest tests.test_mlip_cli tests.test_mlip_artifacts
```
