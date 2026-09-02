#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ==============================
# Edit these values, or override them through environment variables.
MODEL="${MODEL:-gpt-5.6-sol}"                           # For LLM or PLM source
SETTING="${SETTING:-T1_single_mutant_generation}"       # Tasks: ("T1_single_mutant_generation", "T2_measurement_free_multi_mutant_ranking", "T3_anchor_informed_multi_mutant_ranking", "T4_mutation_informed_multi_mutant_ranking")
RESULTS_SOURCE="${RESULTS_SOURCE:-llm}"                 # Source: ("llm", "plm", "agent")
# ==============================

RESULTS_ROOT="${RESULTS_ROOT:-${CODE_DIR}/${RESULTS_SOURCE}/results}"
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-}"

case "${RESULTS_SOURCE}" in
    llm|plm|agent)
        ;;
    *)
        echo "RESULTS_SOURCE must be one of: llm, plm, agent (got ${RESULTS_SOURCE})" >&2
        exit 2
        ;;
esac

case "${RESULTS_SOURCE}" in
    llm)
        PREDICTIONS="${RESULTS_ROOT}/${MODEL}/${SETTING}/norm_data"
        OUTPUT_DIR="${RESULTS_ROOT}/${MODEL}/${SETTING}/evaluation"
        ;;
    plm)
        PREDICTIONS="${RESULTS_ROOT}/${MODEL}/tasks/${SETTING}/evaluator_predictions.csv"
        OUTPUT_DIR="${RESULTS_ROOT}/${MODEL}/evaluation/${SETTING}"
        ;;
    agent)
        AGENT_PREDICTIONS_NAME="${AGENT_PREDICTIONS_NAME:-${SETTING}_agent_native_deepseek_v4_pro_predictions.csv}"
        PREDICTIONS="${RESULTS_ROOT}/${SETTING}/${AGENT_PREDICTIONS_NAME}"
        OUTPUT_DIR="${RESULTS_ROOT}/${SETTING}/evaluation"
        ;;
esac

export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

args=(
    --setting "${SETTING}"
    --predictions "${PREDICTIONS}"
    --output-dir "${OUTPUT_DIR}"
    --lenient                       # Enable for random substitution with missing assays
)
if [[ -n "${GROUND_TRUTH_DIR}" ]]; then
    args+=(--ground-truth-dir "${GROUND_TRUTH_DIR}")
fi

exec "${PYTHON_BIN}" -m evaluation.evaluate "${args[@]}"
