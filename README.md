# PFArena

PFArena is a protein-mutation benchmark with four evaluation tasks and three
inference backends: API-based LLMs, Biomni agents, and protein language models
(PLMs). The runnable code is under [`code/`](code/).

## 1. Create the shared Python environment

Create one Python 3.12 environment for the API LLM, Agent, and evaluator code.
The pinned dependencies are in [`code/requirements.txt`](code/requirements.txt).

```bash
cd /path/to/PFArena
export PF_REPO="$PWD"
conda create --name pfarena python=3.12 -y
conda activate pfarena
python -m pip install --upgrade pip
python -m pip install --requirement code/requirements.txt
export PYTHONPATH="$PF_REPO/code${PYTHONPATH:+:$PYTHONPATH}"
```

Do not install PLM dependencies into this environment. Each PLM has its own
Conda specification.

## 2. Download the benchmark data

Download the `PFArena` dataset from
[`AMix-Bio/PFArena`](https://huggingface.co/datasets/AMix-Bio/PFArena) before
running inference or evaluation. The examples below assume that the dataset is
downloaded to `PFArena/PFArena` relative to this repository:

```bash
export PF_DATA="$PF_REPO/PFArena"

# Install the Hugging Face CLI once if it is not already available.
python -m pip install --upgrade huggingface_hub
# Run `hf auth login` first if the dataset requires authentication.
hf download AMix-Bio/PFArena \
  --repo-type dataset \
  --local-dir "$PF_DATA"

# The download should contain these files/directories.
test -f "$PF_DATA/T1_single_mutant_generation/assay.csv"
test -d "$PF_DATA/T1_single_mutant_generation/norm_data"
```

If the downloaded files are placed elsewhere, set `PF_DATA` to that directory
for the commands below. The four task directories are:

| Task | Directory | Task purpose |
| --- | --- | --- |
| T1 | `T1_single_mutant_generation` | Generate the top 40 single-mutant substitutions |
| T2 | `T2_measurement_free_multi_mutant_ranking` | Rank multi-mutant candidates without measurements provided |
| T3 | `T3_anchor_informed_multi_mutant_ranking` | Rank multi-mutant candidates given one measured anchor mutant |
| T4 | `T4_single_mutant_informed_multi_mutant_ranking` | Rank multi-mutant candidates given all relevant single-mutant measurements |

The `norm_data` tables contain the candidate mutations; `DMS_score` is the
ground-truth label used by evaluation. Keep the downloaded benchmark data
unchanged and DO NOT expose ground-truth columns to an inference backend.

## 3. Inference

### 3.1 LLM inference

The API runner uses an OpenAI-compatible endpoint. Setup the endpoint `API_BASE_URL` and key `API_KEY` in the environment, or directly in `code/llm/settings.py` where it is read.

```bash
export API_BASE_URL="https://your-openai-compatible-endpoint/v1"
export API_KEY="<your-api-key>"
```

The supplied wrapper is [`code/llm/run_infer_api_mutation.sh`](code/llm/run_infer_api_mutation.sh).
Edit its `model` and `task_id` variables, then run it from the repository root:

```bash
cd "$PF_REPO"
bash code/llm/run_infer_api_mutation.sh
```

For a smoke test or an explicit dataset location, call the Python entry point
directly. T1 takes the task-level `assay.csv`; T2--T4 take the task-level
`norm_data` directory.

```bash
# T1 smoke test: infer one assay with one concurrent request.
python code/llm/infer_api_mutation.py \
  --model gpt-5.6-sol \
  --task-id 1 \
  --input-dir "$PF_DATA/T1_single_mutant_generation/assay.csv" \
  --output-root "$PF_REPO/code/llm/results" \
  --max-files 1 \
  --num-workers 1 \
  --retry-runs 3

# Example full T2 run. Use --task-id 3 or 4 and the corresponding norm_data
# directory for T3 or T4.
python code/llm/infer_api_mutation.py \
  --model gpt-5.6-sol \
  --task-id 2 \
  --input-dir "$PF_DATA/T2_measurement_free_multi_mutant_ranking/norm_data" \
  --output-root "$PF_REPO/code/llm/results" \
  --num-workers 32 \
  --retry-runs 10
```

Outputs are written as one CSV per assay under
`code/llm/results/<model>/T*_*/norm_data/`. Existing non-empty output files are
skipped; add `--force` to recompute them. Use `--max-files` for a smoke test,
and lower `--num-workers` when the endpoint has a strict rate limit.

### 3.2 Agent inference

The Agent runner wraps Biomni and supports DeepSeek, GPT, and Claude through an
OpenAI-compatible interface. The provider presets in `code/agent/run.py` map
to these variables:

| Provider | Model preset | API key variable | Base URL variable |
| --- | --- | --- | --- |
| `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` |
| `gpt` | `gpt-4o` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| `claude` | `claude-3-5-sonnet-latest` | `CLAUDE_API_KEY` | `CLAUDE_BASE_URL` |

Set the variables for the provider you will use. `BIOMNI_PROJECT_ROOT` is the
project root used by Biomni tools; `BIOMNI_DATA_ROOT` is a convenient default
for the benchmark location.

```bash
cd "$PF_REPO/code/agent"
# Point this to the Biomni project/tool root used by your installation.
export BIOMNI_PROJECT_ROOT="/path/to/your/biomni/project"
export BIOMNI_DATA_ROOT="$PF_DATA"
export BIOMNI_RESULT_ROOT="$PF_REPO/code/agent/results"
export DEEPSEEK_API_KEY="<your-api-key>"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

Run one task with `run.py`. The `--data-dir` and output paths are explicit here
so that the result can be evaluated without renaming files later.

```bash
# T1 smoke test: one query, no optional tool retriever calls.
python run.py \
  --task T1_single_mutant_generation \
  --provider deepseek \
  --data-dir "$PF_DATA/T1_single_mutant_generation" \
  --max-items 1 \
  --no-use-tool-retriever \
  --output-csv "$PF_REPO/code/agent/results/T1_single_mutant_generation/T1_single_mutant_generation_agent_native_deepseek_v4_pro_predictions.csv"

# Full runs use the same form; replace TASK with one of the four task names.
TASK=T2_measurement_free_multi_mutant_ranking
python run.py \
  --task "$TASK" \
  --provider deepseek \
  --data-dir "$PF_DATA/$TASK" \
  --output-csv "$PF_REPO/code/agent/results/$TASK/${TASK}_agent_native_deepseek_v4_pro_predictions.csv"
```

Use `--provider gpt` or `--provider claude` after setting the corresponding
API variables. For a custom OpenAI-compatible model, pass
`--agent-llm`, `--agent-base-url`, and `--agent-api-key-env` instead of a
provider preset. Each task also has a standalone script, for example
`run_T1_single_mutant_generation_agent_native.py`.

The Agent writes a prediction CSV plus an audit JSONL and an execution log.
Keep the audit and log files beside the predictions; they are useful for
checking tool calls, failures, and resumable runs. `--resume-existing` can
continue a partially completed run. GPU dispatch is optional and disabled by
default; enable it only after configuring a compatible dispatcher and shared
job directory (see [`code/agent/README.md`](code/agent/README.md)).

### 3.3 PLM inference

PLMs are deliberately isolated from the shared `pfarena` environment. The
environment files under
[`code/plm/environments/`](code/plm/environments/) pin the Python, PyTorch,
CUDA, and model-specific dependencies required by each baseline.

Create only the environments you need; the name is stored in each YAML file:

```bash
cd "$PF_REPO"
PLM_ENV_FILE="code/plm/environments/esm2.yml"
conda env create --file "$PLM_ENV_FILE"
conda activate pfarena-esm2
```

The same command works for `progen2_base.yml`, `prosst_2048.yml`, `s3f.yml`,
`s3f_msa.yml`, and `venusrem.yml`. Do not mix packages between these
environments. Model weights and upstream repositories are not bundled; obtain
them from the official sources listed in
[`code/plm/docs/DEPENDENCIES.md`](code/plm/docs/DEPENDENCIES.md).

The PLM runners consume the frozen, canonical model-input release: a directory
containing `dataset.json`, `proteins.csv`, `samples.csv`, `substitutions.csv`,
`queries.csv`, the context tables, and `msa_contexts.csv`. Set that directory
as `PLM_DATA`. The Hugging Face benchmark download above is the default input
for the LLM/Agent paths; use it for PLMs only when it contains this canonical
`dataset.json` layout.

For a sequence-only baseline, run all shards with the same `--num-shards` and
distinct `--shard-id` values. The model directory, cache directory, and output
directory must contain the resources expected by that model.

```bash
conda activate pfarena-esm2
export PLM_DATA=/path/to/frozen_pfarena_model_dataset
export PLM_WORK="$PF_REPO/work/plm"
mkdir -p "$PLM_WORK"

python code/plm/models/esm2/run_candidates.py \
  --dataset-dir "$PLM_DATA" \
  --output-dir "$PLM_WORK/esm2" \
  --model-dir /path/to/esm2_t33_650M_UR50D \
  --cache-dir "$PLM_WORK/cache/esm2" \
  --num-shards 1 \
  --shard-id 0
```

The other direct runner entry points are:

```text
code/plm/models/progen2_base/run_candidates.py
code/plm/models/prosst_2048/run_candidates.py
code/plm/models/s3f/run_candidates.py
code/plm/models/venusrem/run_candidates.py
```

ProSST-2048, S3F, and VenusREM additionally require structure/model resources;
S3F-MSA requires the staged EVE workflow. Follow the complete command
sequences in [`code/plm/docs/RUNNING.md`](code/plm/docs/RUNNING.md), including
input preparation, structure/MSA manifests, sharding, and finalization.

After all shards for a model finish, finalize the run into evaluator-ready
task tables:

```bash
python code/plm/finalize_results.py \
  --dataset-dir "$PLM_DATA" \
  --run-dir "$PLM_WORK/esm2" \
  --output-schema code/plm/models/esm2/output_schema_candidates.json
```

Finalization validates shard coverage, sample identity, score direction, and
the model output schema. Do not evaluate an incomplete run as if missing rows
were zero.

For model-specific inputs, dependency versions, output schemas, and resource
provenance, see the documentation under [`code/plm/docs/`](code/plm/docs/).

## 4. Evaluation

The shared evaluator is wrapped by
[`code/evaluation/run_evaluate.sh`](code/evaluation/run_evaluate.sh). It uses
the downloaded `PFArena/PFArena` directory as ground truth by default. Set
`GROUND_TRUTH_DIR` when the benchmark is elsewhere. The wrapper currently
passes `--lenient`, which scores the available assays when coverage is
incomplete; remove that flag in the wrapper or call `evaluation.evaluate`
directly for strict coverage checking.

### 4.1 Evaluate LLM predictions

```bash
cd "$PF_REPO"
MODEL=gpt-5.6-sol \
SETTING=T1_single_mutant_generation \
RESULTS_SOURCE=llm \
RESULTS_ROOT="$PF_REPO/code/llm/results" \
GROUND_TRUTH_DIR="$PF_DATA" \
bash code/evaluation/run_evaluate.sh
```

For T2, T3, or T4, change `SETTING` to the corresponding full task directory
name. The evaluator expects the LLM files at
`<RESULTS_ROOT>/<MODEL>/<SETTING>/norm_data/`.

### 4.2 Evaluate Agent predictions

Set `AGENT_PREDICTIONS_NAME` to the filename passed to `--output-csv`:

```bash
MODEL=deepseek-v4-pro \
SETTING=T1_single_mutant_generation \
RESULTS_SOURCE=agent \
RESULTS_ROOT="$PF_REPO/code/agent/results" \
AGENT_PREDICTIONS_NAME=T1_single_mutant_generation_agent_native_deepseek_v4_pro_predictions.csv \
GROUND_TRUTH_DIR="$PF_DATA" \
bash "$PF_REPO/code/evaluation/run_evaluate.sh"
```

The Agent result directory is expected to be
`<RESULTS_ROOT>/<SETTING>/`.

### 4.3 Evaluate PLM predictions

After `finalize_results.py` creates
`<RESULTS_ROOT>/<MODEL>/tasks/<SETTING>/evaluator_predictions.csv`, run:

```bash
MODEL=esm2 \
SETTING=T1_single_mutant_generation \
RESULTS_SOURCE=plm \
RESULTS_ROOT="$PF_REPO/work/plm" \
GROUND_TRUTH_DIR="$PF_DATA" \
bash "$PF_REPO/code/evaluation/run_evaluate.sh"
```

Evaluation writes `summary_metrics.json`, `summary_metrics.csv`, and
`per_assay_metrics.csv` under the selected output directory. T1 reports
`NMS@40` and `Recall@40`; T2--T4 report global ranking agreement
(`spearman`, `NDCG`) and top-five quality/recovery (`NMS@5`, `Recall@5`).
