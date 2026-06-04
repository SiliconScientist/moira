# oasis-mlip
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
oasis-mlip submit --config mlip.toml
oasis-mlip make-tasks --config mlip.toml --run-tag dev --out slurm_output/tasks.txt
oasis-mlip run-one --config mlip.toml --line "mace example data/raw_data/example.json data/results/mlips/dev/example/mace.json"
```

Module entrypoint:

```bash
python -m oasis.mlip submit --config mlip.toml
python -m oasis.mlip make-tasks --config mlip.toml --run-tag dev --out slurm_output/tasks.txt
python -m oasis.mlip run-one --config mlip.toml --line "mace example data/raw_data/example.json data/results/mlips/dev/example/mace.json"
```

`python -m oasis` also forwards directly to the MLIP CLI in this extracted repo.

## Scope

This repo contains:

- `oasis.mlip` runtime, artifact loading, and result parsing
- `oasis.adapters` integration with Rootstock/CatBench
- `oasis.ingest` dataset ingestion helpers
- MLIP-facing config models in `oasis.mlip_config`

Targeted graph/config test commands:

```bash
PYTHONPATH=src python -m unittest tests.test_mlip_cli tests.test_mlip_artifacts
```
  No emitted batch may mix train, val, or test examples.
- Train batching and eval batching are configured separately through
  `TrainEvalLoaderPolicy`.
- Train loaders may shuffle. Validation and test loaders default to no shuffle.
- `eval_batch_size` may differ from `batch_size`, so validation/test throughput
  can be tuned independently of training behavior.
- The default helper path, `SweepDatasetBatchLoaderAdapter`, emits deterministic
  split-safe batches from `SweepDataset` subsets.
- If train shuffling is enabled, batch order may change inside the train split,
  but train batches still contain only train examples.
- Validation/test loaders remain stable unless a caller explicitly opts into
  different behavior.
- Held-out outer-test data must not be consumed during candidate ranking. It is
  reserved for one final post-selection evaluation pass.

## Split Feasibility Policy

Learning-curve sweep sizes are outer training budgets.

- For train/test-only families, `sweep_size` is the full training set size.
- For validation-aware families, `sweep_size` must cover both inner training
  and inner validation.
- `test_idx` is always an outer holdout and is never part of `sweep_size`.

Validation-aware runs size validation as:

```text
max(
  floor(validation_fraction * sweep_size),
  min_val_size,
  min_tuning_val_size,
)
```

A validation-aware sweep point is emitted only if all of these can be satisfied
together:

- caller-requested `min_train` / `max_train`
- family-level `min_train_size`
- `min_tuning_val_size` and `min_val_size`
- `min_inner_train_size`
- `min_test_size`

That means some requested sweep sizes may be skipped, and some whole sweep
regions may collapse to an empty split collection, when the budget is too small
to leave:

- enough validation samples for meaningful scoring
- enough remaining inner-train samples after validation
- enough outer-test samples for final evaluation

This behavior is intentional. Oasis now prefers dropping infeasible
validation-aware points over producing train/val/test splits that are too small
to support sensible model selection.
