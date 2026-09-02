#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"

# Edit these values for the run you want to launch.
model="gpt-5.6-sol"             # OpenAI-compatible endpoint model name
task_id="1"                     # Task id in ("1", "2", "3", "4")
max_completion_tokens="65536"   # API max completion tokens, needs to be large enough for LLM thinking
num_workers="32"                # API client concurrency
max_files=""                    # Limit files for smoke tests, leave blank for regular evaluation
retry_runs="10"                 # Rerun failed responses, default 10 times until no new successes produced

args=(
    --model "${model}"
    --task-id "${task_id}"
    --max-completion-tokens "${max_completion_tokens}"
    --num-workers "${num_workers}"
    --retry-runs "${retry_runs}"
)
if [[ -n "${max_files}" ]]; then
    args+=(--max-files "${max_files}")
fi

exec "${python_bin}" "${SCRIPT_DIR}/infer_api_mutation.py" "${args[@]}"
