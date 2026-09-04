# PFArena Protein-Model Baselines

This module contains the preprocessing, inference, and result-finalization code
for the six protein-model baselines evaluated on PFArena T1--T4. It also
retains the complete reported predictions and metrics. Benchmark data and the
shared evaluator are maintained by the parent PFArena project and are supplied
to this module through explicit paths.

## Models

| Directory | Baseline | Required inputs | Candidate score |
|---|---|---|---|
| `models/esm2` | ESM-2 650M | sequence | masked-marginal log-odds |
| `models/progen2_base` | ProGen2-base | sequence | mutant-minus-WT bidirectional likelihood |
| `models/prosst_2048` | ProSST-2048 | sequence + chain structure | wild-type marginal log-odds |
| `models/s3f` | S3F | sequence + chain structure/surface | structure-aware masked-marginal score |
| `models/s3f_msa` | S3F-MSA | S3F score + chain MSA | mean of query-standardized S3F and EVE scores |
| `models/venusrem` | VenusREM | sequence + structure tokens + chain MSA | aligned sequence/structure marginal score |

ProGen2-base and the EVE component of S3F-MSA score a multi-substitution
sequence jointly within each chain. Sitewise models sum substitution-level
log-odds within a chain. All baselines sum contributions across mutated chains.

## Contents

- `models/`: model-specific preprocessing and inference implementations.
- `model_adapters/`: the minimal scoring adapters used by the runners.
- `common_io.py`, `dataset_io.py`, and `model_dataset_io.py`: shared validated I/O.
- `finalize_results.py`: shard aggregation, coverage validation, and generation
  of evaluator-ready prediction tables.
- `export_evaluator_predictions.py`: reconstruction of evaluator-ready task
  tables from the released final predictions.
- `validate_results.py`: self-contained integrity check for released predictions,
  evaluation summaries, and the leaderboard.
- `results/`: complete predictions and unified-evaluator outputs for all six
  reported baselines.
- `docs/`: input contract, external dependencies, and result organization.
- `environments/`: one reproducible Conda specification per protein model and
  the runtime versions used for the reported experiments.

The module intentionally excludes benchmark copies, ground truth, the shared
evaluation package, scheduler-specific submission scripts, model checkpoints,
predicted structures, and rebuildable model caches. The required WT chain PDBs
are released separately as the accompanying PFArena PLM Structures dataset.

## Running a baseline

The parent project supplies a canonical dataset directory matching
`docs/DATA_FORMAT.md`. Structure-aware models use the manifest distributed with
the accompanying structure dataset, and MSA-aware models resolve the alignment
paths recorded in `msa_contexts.csv`.

ESM-2, ProGen2-base, ProSST-2048, S3F, and VenusREM use model-specific
`run_candidates.py` entry points with explicit data, output, cache, model
resource, and sharding arguments. S3F-MSA uses a staged EVE workflow instead.
Model-specific preparation is required before S3F, S3F-MSA, and VenusREM
inference:

- S3F: `prepare_inputs.py`, then `prepare_candidate_surfaces.py`.
- S3F-MSA: `prepare_inputs.py`, followed by the `weights`, `train`, and
  `score` stages in `run_stage.py`, then `combine.py`.
- VenusREM: run ProSST first, then use `prepare_inputs.py`.

The complete platform-independent command sequence is provided in
`docs/RUNNING.md`.

After all inference shards finish, finalize one model run with:

```bash
python finalize_results.py \
  --dataset-dir /path/to/canonical_dataset \
  --run-dir /path/to/model_run \
  --output-schema models/<model>/output_schema_candidates.json
```

Evaluator-ready task tables can be generated from released results with
`export_evaluator_predictions.py`. Evaluate these tables with the official
evaluator distributed by the parent PFArena project. Required upstream
repositories and checkpoints are listed in `docs/DEPENDENCIES.md`; no model
parameters are included here.

Validate all released protein-model results from the module root with:

```bash
python validate_results.py
```
