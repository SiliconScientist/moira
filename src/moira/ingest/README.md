# Ingest Pipeline Contracts

The ingest pipeline is split into three phases:

1. `sources/*`: load source-specific files into a `DatasetBundle`
2. `transforms/*`: derive source-agnostic intermediate products from the bundle
3. `writers/*`: emit a concrete output format from the bundle plus transform output

## Loader Output

Loaders return `DatasetBundle` from [models.py](/Users/averyhill/github/moira/src/moira/ingest/models.py:1).

- `DatasetBundle.name`: dataset identifier
- `DatasetBundle.source`: optional source path string
- `DatasetBundle.structures`: all discovered `StructureRecord`s
- `DatasetBundle.references`: reference groupings used by downstream transforms
- `DatasetBundle.reactions`: optional reaction-level records for sources that have them
- `DatasetBundle.metadata`: source-specific extras that are not part of the stable core schema

For the current VASP mapping loader in [sources/vasp_mapping.py](/Users/averyhill/github/moira/src/moira/ingest/sources/vasp_mapping.py:1):

- `StructureRecord.kind` is typically `gas`, `slab`, or `adslab`
- `StructureRecord.source_path` points at the original folder on disk
- `StructureRecord.metadata["catbench_relpath"]` is the relative path the CatBench writer materializes
- `ReferenceSet.slab`, `ReferenceSet.adslab`, and `ReferenceSet.gas` are populated enough for CatBench coefficient generation

For the current ASE DB loader in [sources/ase_db.py](/Users/averyhill/github/moira/src/moira/ingest/sources/ase_db.py:1):

- `StructureRecord.metadata` keeps `row.key_value_pairs`
- if `row.data["structure_metadata"]` exists, those typed values override the stringified key/value pairs
- source fields such as `adslab_id`, `parent_slab_id`, `surface_type`, `swap_indices`, and `initial_site_coordinate` are expected to survive downstream unchanged unless a transform adds derived keys

## Optional Fields

Most model fields are intentionally optional so partial datasets can still load.

- `StructureRecord.formula`, `symbols`, `positions`, `cell`, `pbc`, and `energy_ev` may be missing
- `StructureRecord.source_id` and `source_path` may be missing for synthetic or in-memory records
- `ReferenceSet.slab`, `adslab`, `adsorbate`, and `gas` may be incomplete
- `ReactionRecord.adsorbate`, `stoichiometry`, `energy_ev`, and `references` may be missing
- `DatasetBundle.reactions` may be empty for sources that do not expose reaction data

Transforms and writers should only require the fields they actually consume.

## Reference Completeness

There are two distinct completeness modes for emitted references:

- geometry-complete: emitted `ReferenceSet` entries must point to `slab`, `adslab`, and `gas` structures with usable geometry
- energy-optional: those same structures may still have `energy_ev=None`

In other words:

- emitted references must have geometries
- energies may be absent

This is the supported partial-energy ingest mode for structural datasets.

## CatBench Writer Requirements

The CatBench writer lives in [writers/catbench.py](/Users/averyhill/github/moira/src/moira/ingest/writers/catbench.py:1).

It requires:

- a `DatasetBundle` whose emitted references are geometry-complete
- `StructureRecord.metadata["catbench_relpath"]` for each structure that should be copied
- a CatBench coefficient map such as the output of [transforms/catbench.py](/Users/averyhill/github/moira/src/moira/ingest/transforms/catbench.py:1)
- a destination dataset folder, output directory, and output dataset name

It supports two output modes:

- full-energy mode: structures have geometry plus energies, so the writer can materialize VASP-style inputs and run CatBench preprocessing
- partial-energy mode: structures have geometry but may omit energies, so the writer still materializes the dataset layout without inventing missing energy labels

The emitted adsorption JSON now also carries a top-level per-reaction `metadata` block:

- `metadata["reference"]`: `ReferenceSet.metadata`
- `metadata["structures"]["slab"]`: slab structure metadata
- `metadata["structures"]["adslab"]`: adslab structure metadata
- `metadata["structures"]["gas"]`: gas structure metadata keyed by gas record id

This block is additive and does not replace CatBench's existing `raw`, `ref_ads_eng`, or `adsorbate_indices` fields.

## MLIP Metadata Handoff

MLIP runs consume the emitted adsorption JSON and write `*_result.json` files under the configured result directory.

- Moira immediately enriches each reaction entry after CatBench finishes
- The enriched result entry keeps dataset `metadata` and a persisted `moira_analysis` block with labels, anomaly details, and derived energies
- [moira.mlip.result_parsing](/Users/averyhill/github/moira/src/moira/mlip/result_parsing.py:1) defines the persisted analysis payload
- [moira.mlip.artifacts](/Users/averyhill/github/moira/src/moira/mlip/artifacts.py:1) reads that stored analysis into the wide predictions table and only falls back to recomputing it for older unenriched result files

That is the current contract for preserving source metadata and anomaly analysis from ASE DB ingestion through MLIP result extraction.

Moira no longer depends on running:

```python
from catbench.adsorption import AdsorptionAnalysis

AdsorptionAnalysis().analysis()
```

as a separate post-processing step. For Moira-managed workflows, the analyzed `*_result.json` file is the source of truth.

When loading results downstream, prefer explicit result JSON paths over directory auto-discovery.

It does not require reaction records or a fully populated bundle.

## Expected Plug Points

Branch 2 should plug in at the loader boundary:

- implement another source adapter that returns `DatasetBundle`
- keep source-specific naming and filesystem assumptions inside `sources/*`

Branch 3 should plug in at the transform and/or writer boundary:

- reuse the same loader contract if the source can produce `DatasetBundle`
- add a new transform if reference or reaction derivation differs
- add a new writer if the target format is not CatBench

The stable interface is therefore:

- loader: `source-specific input -> DatasetBundle`
- transform: `DatasetBundle -> derived records or coefficient maps`
- writer: `DatasetBundle + transform output -> emitted dataset`
