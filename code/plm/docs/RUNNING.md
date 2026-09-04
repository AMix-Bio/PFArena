# Running the protein-model baselines

Run commands from the repository root. The examples below are independent of
any scheduler. Use the model-specific environments and official resources
listed in `DEPENDENCIES.md`.

Set the following paths for the local installation:

```bash
DATASET=/path/to/frozen_pfarena_dataset
STRUCTURES=/path/to/PFArena_PLM_Structures/manifest.csv
WORK=/path/to/new_work_directory
N=1
I=0
```

`N` is the total number of inference shards and `I` is one shard index in
`0,...,N-1`. Run every shard with the same `N` and a distinct `I`. Output and
cache directories must be new or contain only validated artifacts from the
identical configuration.

## Direct sequence and structure models

```bash
python models/esm2/run_candidates.py \
  --dataset-dir "$DATASET" --output-dir "$WORK/esm2" \
  --model-dir /path/to/esm2_t33_650M_UR50D \
  --cache-dir "$WORK/cache/esm2" --num-shards "$N" --shard-id "$I"

python models/progen2_base/run_candidates.py \
  --dataset-dir "$DATASET" --output-dir "$WORK/progen2_base" \
  --model-dir /path/to/progen2-base \
  --cache-dir "$WORK/cache/progen2_base" --num-shards "$N" --shard-id "$I"

python models/prosst_2048/run_candidates.py \
  --dataset-dir "$DATASET" --structure-manifest "$STRUCTURES" \
  --output-dir "$WORK/prosst_2048" --model-dir /path/to/ProSST-2048 \
  --prosst-repo-dir /path/to/ProSST --cache-dir "$WORK/cache/prosst_2048" \
  --num-shards "$N" --shard-id "$I"
```

## S3F

Prepare the chain PDB inputs once, then generate every surface shard before
starting inference:

```bash
python models/s3f/prepare_inputs.py \
  --dataset-dir "$DATASET" --structure-manifest "$STRUCTURES" \
  --output-dir "$WORK/s3f_inputs"

python models/s3f/prepare_candidate_surfaces.py \
  --input-dir "$WORK/s3f_inputs" --surface-dir "$WORK/s3f_surfaces" \
  --s3f-script /path/to/S3F/score_s3f_fitness.py \
  --num-shards "$N" --shard-id "$I"

python models/s3f/run_candidates.py \
  --dataset-dir "$DATASET" --input-dir "$WORK/s3f_inputs" \
  --surface-dir "$WORK/s3f_surfaces" --output-dir "$WORK/s3f" \
  --cache-dir "$WORK/cache/s3f" \
  --s3f-script /path/to/S3F/score_s3f_fitness.py \
  --checkpoint /path/to/s3f.pth --esm-model-dir /path/to/S3F/esm2 \
  --num-shards "$N" --shard-id "$I"
```

## VenusREM

VenusREM uses the validated structure-token cache produced by the completed
ProSST run. Finalize ProSST first, then prepare synchronized model inputs:

```bash
python finalize_results.py \
  --dataset-dir "$DATASET" --run-dir "$WORK/prosst_2048" \
  --output-schema models/prosst_2048/output_schema_candidates.json

python models/venusrem/prepare_inputs.py \
  --dataset-dir "$DATASET" --prosst-run "$WORK/prosst_2048/run.json" \
  --output-dir "$WORK/venusrem_inputs"

python models/venusrem/run_candidates.py \
  --dataset-dir "$DATASET" --input-dir "$WORK/venusrem_inputs" \
  --output-dir "$WORK/venusrem" --model-dir /path/to/ProSST-2048 \
  --cache-dir "$WORK/cache/venusrem" --num-shards "$N" --shard-id "$I"
```

## S3F-MSA

S3F-MSA requires a finalized S3F run and one EVE workflow per row of
`mapping.csv`. For each context index `J`, compute sequence weights, train all
five seeds, and then score the context:

```bash
python models/s3f_msa/prepare_inputs.py \
  --dataset-dir "$DATASET" --output-dir "$WORK/s3f_msa_inputs"

python models/s3f_msa/run_stage.py weights \
  --input-dir "$WORK/s3f_msa_inputs" --cache-dir "$WORK/cache/s3f_msa" \
  --eve-root /path/to/EVE --index "$J"

python models/s3f_msa/run_stage.py train \
  --input-dir "$WORK/s3f_msa_inputs" --cache-dir "$WORK/cache/s3f_msa" \
  --eve-root /path/to/EVE --index "$J" --seed "$SEED"

python models/s3f_msa/run_stage.py score \
  --input-dir "$WORK/s3f_msa_inputs" --cache-dir "$WORK/cache/s3f_msa" \
  --eve-root /path/to/EVE --index "$J"

python models/s3f_msa/combine.py \
  --dataset-dir "$DATASET" --input-dir "$WORK/s3f_msa_inputs" \
  --cache-dir "$WORK/cache/s3f_msa" --s3f-run-dir "$WORK/s3f" \
  --output-dir "$WORK/s3f_msa"
```

Run `train` once for each `SEED` in `0,...,4`. The stages validate weights,
checkpoints, logs, score direction, five-seed aggregation, and source hashes
before reuse.

## Finalization and evaluation

After every shard has completed, finalize each run exactly once:

```bash
python finalize_results.py \
  --dataset-dir "$DATASET" --run-dir "$WORK/MODEL" \
  --output-schema models/MODEL/output_schema_candidates.json
```

Finalization verifies shard configuration, numerical validity, sample identity,
complete candidate coverage, and multi-chain aggregation. Evaluate the emitted
`tasks/<task>/evaluator_predictions.csv` files with the evaluator distributed
by the parent PFArena project.

For released results, reconstruct the same evaluator-ready tables without
retaining duplicate copies:

```bash
python export_evaluator_predictions.py \
  --result-dir results/MODEL \
  --output-dir /path/to/evaluator_inputs/MODEL
```
