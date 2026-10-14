# Sample tracking and diagnostics

[Project introduction](../README.md) · [Setup and workflows](workflows.md)

## Preserving sample associations

Moira's adapters run CatBench and then enrich each reaction in the model's
`*_result.json`. Dataset `metadata` is copied into the reaction when available,
and `moira_analysis` stores reference and predicted energies, metadata, a
classification (`label`), its contributing indicators (`labels`), and diagnostic
counts or flags (`details`). Existing datasets can only contribute the metadata
they contain; Moira cannot recover absent source provenance.

CatBench already computes per-reaction anomaly summaries as well as aggregate
reports. Moira persists sample-level analysis alongside the predictions, so a
separate plotting or reporting step is unnecessary. This supports auditing a
prediction and excluding it independently of other models' predictions for the
same sample. It does not establish that filtering improves a particular study.

See [result enrichment](../src/moira/mlip/result_metadata.py),
[result parsing](../src/moira/mlip/result_parsing.py), and the
[ingestion metadata contract](../src/moira/ingest/README.md#mlip-metadata-handoff).

## CatBench constraint patch

Moira uses a [vendored CatBench source tree](../vendor/catbench).
In [`energy_cal()`](../vendor/catbench/catbench/utils/calculation_utils.py),
CatBench previously reapplied `atoms.set_constraint(fixatom(atoms, z_target))`.
That height-based fixed-atom selection replaced the existing ASE constraints.
The vendored source disables this reassignment, preserving supplied constraints
during slab and adsorbate–slab relaxation, including fixed surface layers.

This is a targeted patch to `energy_cal()`, not a guarantee that every constraint
type survives every input format or that gas relaxation behaves identically.
The ingestion layer must also preserve the constraints supported by its format.
Moira separately patches CatBench's dataset and result path lookup at runtime;
see [path handling](../src/moira/adapters/catbench_paths.py).

## Diagnostic rules

The [result parser](../src/moira/mlip/result_parsing.py) derives CatBench-style
anomalies from saved calculation quantities. Automatic result enrichment uses
these defaults:

| Diagnostic | Default condition |
| --- | --- |
| Slab or adsorbate–slab convergence | Recorded steps equal `n_crit_relax` from calculation settings (fallback: 999). |
| Slab displacement | `slab_max_disp` exceeds 1.0 Å in a calculation. |
| Substrate displacement in the adsorbed structure | `substrate_displacement` exceeds 1.0 Å for the median-energy calculation. |
| Variation across repeated calculations | A recorded slab, adsorbate, or adsorption-energy seed range exceeds 0.2 eV. |
| Adsorbate migration | The median calculation's `max_bond_change` exceeds 20%, provided no convergence or displacement anomaly was detected. |
| Reference-energy disagreement | Absolute difference between median predicted and reference adsorption energy exceeds 2.0 eV. |

The detector requires `chemical_bond_cutoff` in the saved calculation settings.
Slab-specific indicators are disabled when the result is identified as lacking
slab calculations. Several checks rely on optional quantities: an absent flag
can mean the requisite data were unavailable, not that a check established
physical validity. Repeated-calculation checks require repeated results.

The detector's Python arguments allow threshold overrides; these are not exposed
as equivalent Quickstart configuration fields. Record any custom thresholds
with the downstream analysis. Stored analysis is reused by the result loader;
changing detector arguments does not retroactively update persisted results.

Classification uses this precedence:

1. `reproduction_failure`
2. `unphysical_relaxation`
3. `adsorbate_migration`
4. `energy_anomaly`
5. `normal`

The `labels` list describes the selected classification; `details` retains all
computed indicators, including those outside that classification. Some details
are counts across calculations rather than Boolean values. Migration detection
is conditional, so these are not independent tests of every possible anomaly.

Anomaly detection is distinct from prediction accuracy: an unflagged prediction
is not necessarily accurate, and a flagged prediction is not by itself proof of
large energy error. Reference-energy disagreement requires a known reference;
it cannot support inference-only abstention on unlabeled samples. Policies for
such samples must use diagnostics available without reference labels.

## Aligned results

[`load_result_analysis()`](../src/moira/mlip/artifacts.py) reads persisted
`moira_analysis` blocks and falls back to detection for older files without
stored analysis. Per-model result files retain the reaction identifiers and
available source metadata.

`load_wide_predictions()` builds a table with model-specific energy, label, and
inference-diagnostic columns. It checks overlapping reactions for conflicting
reference energies, adsorbates, and metadata, then joins by reaction identifier.
It does not automatically perform abstention.

**Current limitation:** this loader uses inner joins and drops rows missing
reference energy or required model values. The table therefore retains only
samples with usable predictions from every supplied model and a non-null
reference energy. It is not a complete export of unlabeled or partially
predicted samples. Use per-model result files or per-reaction analysis when
those samples must be retained, and apply an explicit downstream alignment and
filtering policy. The wide table exposes inference diagnostics separately from
the reference-dependent energy-anomaly check.
