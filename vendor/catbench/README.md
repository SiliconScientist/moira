Place a patched CatBench source checkout in this directory.

Expected layout:

- `vendor/catbench/catbench/__init__.py`
- `vendor/catbench/catbench/...`

`oasis` will prepend `vendor/catbench` to `PYTHONPATH` for adapter subprocesses.

If you store the checkout somewhere else, set `mlip.catbench_source` in `config.toml`.

Moira also monkey-patches CatBench's adsorption dataset lookup at runtime so
`mlip.dataset` / `mlip.datasets` input paths are used directly instead of being
rewritten to `./raw_data/<benchmark>_adsorption.json`.

Moira can also monkey-patch CatBench's result directory lookup so
`mlip.results_dir` is used instead of writing to `./result`.

This vendored checkout also includes a direct source patch in
`catbench/utils/calculation_utils.py`: `energy_cal()` no longer reapplies
`fixatom(atoms, z_target)`, because that was overwriting existing ASE
constraints from the dataset.
