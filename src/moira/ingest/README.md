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
