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

```mermaid
flowchart TD
    V["VASP calculations"] --> P["Prepare adsorption dataset; preserve source metadata"]
    A["ASE databases"] --> P
    P --> D["CatBench adsorption dataset"]
    C["Existing CatBench dataset"] -->|Skip preparation| D
    D --> M["Run MLIPs through supported model adapters"]
    M --> R["Collect predictions and anomaly diagnostics per sample and model"]
    R -.->|Optional table loading| T["Align results by sample for downstream analysis"]
```

Existing datasets retain the metadata they supply. Result files preserve
sample–model associations; optional table loading aligns compatible results
across models. See [alignment limits](docs/results-and-diagnostics.md#aligned-results).

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

<a id="sample-tracking-and-catbench-changes"></a>

## Sample traceability

Moira keeps each sample's source metadata, model predictions, and anomaly
diagnostics together. These associations let researchers trace a result back to
its input and examine why a prediction was flagged, supporting reproducible
comparisons and explicit abstention decisions for individual model predictions.

Anomaly detection identifies warning signs; it does not establish prediction
accuracy. In particular, disagreement with reference energy requires a known
reference and cannot support abstention on unlabeled samples. See
[sample tracking and diagnostics](docs/results-and-diagnostics.md) for the
CatBench constraint patch, stored results, diagnostic rules, and limitations.

## Quickstart

The workflow below submits MLIP calculations to a Slurm cluster. You need an
input dataset, supported model environments, model checkpoints or pretrained
selectors, and access to the compute resources required by your models. Some
checkpoints require separate downloads or access approval.

**1. Install Moira.** Run these commands from the repository root in an active
Python environment. Use Python 3.11 or newer for the current runtime; see the
[Python compatibility note](docs/workflows.md#environment).

```bash
python -m pip install -e ".[dev]"
```

**2. Configure a run.** Copy the [configuration example](config.example.toml),
then replace its placeholder paths:

```bash
cp config.example.toml config.toml
```

Set `mlip.models.enabled`, `mlip.adapter_backend`, the model specifications under
`mlip.rootstock.models`, and `mlip.results_dir`. Supply an existing CatBench
adsorption JSON through `mlip.dataset`, or configure `[ingest]` and
`[ingest.stoich]` to prepare one in step 4. Start with one configured model.

**3. Set up model environments and checkpoints.** Installation of Moira alone
does not configure the MLIPs. Choose the setup matching your backend:

- **Rootstock:** configure `mlip.rootstock.root` and `mlip.rootstock.python`,
  and provision the model environments used by Rootstock. Submission preflight
  also checks model imports in the configured Python interpreter.
- **Legacy adapters:** with `uv` installed, run `./envs/setup_mlip_envs.sh` to
  create the repository's per-model environments. This script processes all
  model requirements, defaulting to Python 3.13 and Python 3.12 for ORB v3.

Obtain the required checkpoints or compiled model artifacts and set their paths
or supported selectors in `config.toml`. Follow [environment setup](envs/README.md)
and the model-specific guides for [AQCat25](docs/workflows.md#aqcat25-setup),
[CHGNet](docs/workflows.md#chgnet-setup), and [Allegro](docs/workflows.md#allegro-setup).

**4. Prepare input, if needed.** Skip this step for an existing CatBench dataset.
For configured VASP or ASE database inputs, run:

```bash
python -m moira.ingest --config config.toml
```

This writes `data/raw_data/<ingest.dataset_name>_adsorption.json`. Set
`mlip.dataset` to that path; ingestion does not update the configuration file.
See [ingestion contracts](src/moira/ingest/README.md) for supported input layouts.

**5. Submit and inspect results.** On a system with `sbatch`, copy the Slurm
script and edit its account, partition, resources, and email settings:

```bash
cp slurm/mlip_one.example.sbatch slurm/mlip_one.sbatch
```

After editing, submit from the repository root:

```bash
moira --config config.toml --run-tag screen
```

Moira checks model environments, freezes the configuration, and submits an
array. The run directory under `slurm_output/runs/` holds the configuration
snapshot, task list, and job logs. Completed tasks write `*_result.json` files
with predictions and sample diagnostics under `mlip.results_dir`, or `./result`
for unsharded runs without an output override. For optional sharding and its
output layout, see [sharded runs](docs/workflows.md#sharded-runs).

There is no self-contained local inference example bundled with this checkout:
the test fixtures do not include model checkpoints or ready-to-run environments.
Direct task execution is available, but still needs those prerequisites; see
[local execution and verification limits](docs/workflows.md#local-execution-and-verification-limits).

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
- [Sample tracking and diagnostics](docs/results-and-diagnostics.md): CatBench
  integration, anomaly thresholds, and result alignment limits.
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
