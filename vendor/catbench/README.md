Place a patched CatBench source checkout in this directory.

Expected layout:

- `vendor/catbench/catbench/__init__.py`
- `vendor/catbench/catbench/...`

`oasis` will prepend `vendor/catbench` to `PYTHONPATH` for adapter subprocesses.

If you store the checkout somewhere else, set `mlip.catbench_source` in `config.toml`.

Moira also monkey-patches CatBench's adsorption dataset lookup at runtime so
`mlip.dataset` / `mlip.datasets` input paths are used directly instead of being
rewritten to `./raw_data/<benchmark>_adsorption.json`.
