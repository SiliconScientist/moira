# Moira

Moira runs machine-learned interatomic potentials (MLIPs) on adsorption datasets
and keeps each sample's predictions linked to its source metadata and anomaly
flags. MLIPs predict atomic energies and forces; comparing their predictions
helps assess how strongly an atom or molecule binds to a catalyst surface.

The project grew out of a need to run ten MLIPs with different implementations
and software environments through a common workflow. Extracted from Oasis,
Moira handles dataset preparation, model execution, and result extraction so
that predictions remain traceable across models and calculations.

That traceability supports prediction abstention: a downstream study can exclude
an anomalous prediction from one model while retaining other models' predictions
for the same sample. [Oasis](https://github.com/SiliconScientist/oasis) evaluates
anomaly-aware filtering in its ensemble studies. Moira supplies the predictions
and diagnostics needed to test such policies on further datasets.

## Capabilities

- Prepare CatBench adsorption datasets from mapped VASP calculations or ASE
  databases, preserving source metadata and supported structural constraints.
- Run models through Rootstock or dedicated adapters with separate model
  environments. The ten registered adapters cover Allegro, AlphaNet, AQCat25,
  CHGNet, GRACE, MACE, MatterSim, ORB v3, SevenNet, and UMA.
- Retain adsorption energies, sample metadata, and CatBench-derived anomaly
  classifications and details in each model's result files.
- Submit Slurm arrays, split large datasets into isolated shards, collect their
  results, and reuse relaxed clean slabs shared by multiple adsorption samples.
- Load results into a table with model-specific energy, label, and diagnostic
  columns aligned by reaction identifier.

**Scope:** Moira produces predictions and diagnostics for downstream analysis.
Anomaly flags depend on the configured calculations and detection thresholds;
an unflagged prediction is not a guarantee of accuracy. Filtering policies and
comparisons of their benefits belong to the downstream study.

## Sample tracking and CatBench changes

Moira uses a [vendored CatBench source tree](vendor/catbench) with a targeted
constraint patch. CatBench's `energy_cal()` previously reapplied its own
height-based fixed-atom selection, overwriting existing ASE constraints. The
patched function preserves the supplied constraints during slab and
adsorbate–slab relaxation, including fixed surface layers.

CatBench already computes per-reaction anomaly summaries as well as aggregate
reports. Moira makes those associations persistent: after a model finishes,
each reaction in its `*_result.json` receives source `metadata` and a
`moira_analysis` block containing energies, a classification, and detailed
anomaly indicators. This keeps the sample, model prediction, and diagnostics
together without requiring a separate plotting or reporting step.

Diagnostics cover convergence, structural displacement, variation across
repeated calculations, adsorbate migration, and disagreement with reference
energy. The reference-energy check requires a known reference and cannot serve
as an inference-only abstention rule for unlabeled samples. The detailed flags
remain available alongside the priority-based classification; see the
[result parser](src/moira/mlip/result_parsing.py) for thresholds and precedence.

## Quickstart

From a checkout, install the project and development dependencies:

```bash
pip install -e ".[dev]"
cp config.example.toml config.toml
```

The package declares Python 3.10 or later; model environments have their own
version requirements. The environment setup script defaults to Python 3.13,
with Python 3.12 for ORB v3. See [environment setup](envs/README.md).

Edit [config.toml](config.example.toml) for your dataset, enabled models,
checkpoints, backend, and output directory. The example contains placeholder
paths. Rootstock uses its own model environments; the `legacy` backend uses
Moira's per-model environments, created with `./envs/setup_mlip_envs.sh`.
Follow the [model setup instructions](docs/workflows.md) before running.

If starting from VASP calculations or an ASE database, prepare the dataset:

```bash
python -m moira.ingest --config config.toml
```

Set `mlip.dataset` to the emitted adsorption JSON, or supply an existing
CatBench adsorption JSON. For cluster execution, copy and customize the Slurm
script for your account and resources:

```bash
cp slurm/mlip_one.example.sbatch slurm/mlip_one.sbatch
# Edit slurm/mlip_one.sbatch before submitting.
moira --config config.toml --run-tag screen
```

This command runs environment preflight checks and submits a Slurm array; it
requires `sbatch`. Results go under `mlip.results_dir`, or `./result` by default.
See [entrypoints](docs/workflows.md#entrypoints) and
[sharded runs](docs/workflows.md#sharded-runs) for task generation and collection.

## Documentation

<!-- Preserve anchors from the original README. -->
<a id="naming"></a>
<a id="environment"></a>
<a id="aqcat25-setup"></a>
<a id="chgnet-setup"></a>
<a id="allegro-setup"></a>
<a id="entrypoints"></a>
<a id="ingest"></a>
<a id="sharded-runs"></a>
<a id="slab-cache"></a>
<a id="scope"></a>
<a id="testing"></a>

- [Setup and workflows](docs/workflows.md): model installation, CLI usage,
  sharding, slab caching, and test commands.
- [Configuration example](config.example.toml) and [model environments](envs/README.md).
- [Ingestion contracts](src/moira/ingest/README.md): geometry, metadata, and result handoff.
- [Result loading](src/moira/mlip/artifacts.py) and
  [persisted analysis](src/moira/mlip/result_metadata.py).

## Related projects

| Project | Role |
| --- | --- |
| [**Intergen**](https://github.com/SiliconScientist/intergen) | Generates candidate atomic structures and adsorption sites. |
| **Moira** | Runs MLIP predictions and preserves sample-level anomaly diagnostics. |
| [**Oasis**](https://github.com/SiliconScientist/oasis) | Evaluates and adapts predictions using limited reference data, including anomaly-aware filtering. |

These projects have complementary roles but do not form an automatically
integrated pipeline; connecting their workflows requires separate setup.
