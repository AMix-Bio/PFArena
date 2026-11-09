#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ==============================
# Edit these values, or override them through environment variables.
MODEL="${MODEL:-gpt-6-astra}"                           # For LLM or PLM source
SETTING="${SETTING:-}"                                  # Empty: available tasks, then average if all four succeed; Single task: set to one of ("T1_single_mutant_generation", "T2_measurement_free_multi_mutant_ranking", "T3_anchor_informed_multi_mutant_ranking", "T4_single_mutant_informed_multi_mutant_ranking")
RESULTS_SOURCE="${RESULTS_SOURCE:-llm}"                 # Source: ("llm", "plm", "agent")
# ==============================

RESULTS_ROOT="${RESULTS_ROOT:-${CODE_DIR}/${RESULTS_SOURCE}/results}"
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-}"
LENIENT="${LENIENT:-1}"                                  # 0: fail; 1: random replacement; 2: skip and report

case "${RESULTS_SOURCE}" in
    llm|plm|agent)
        ;;
    *)
        echo "RESULTS_SOURCE must be one of: llm, plm, agent (got ${RESULTS_SOURCE})" >&2
        exit 2
        ;;
esac

configure_task_paths() {
    local setting="$1"
    case "${RESULTS_SOURCE}" in
        llm)
            PREDICTIONS="${RESULTS_ROOT}/${MODEL}/${setting}/norm_data"
            OUTPUT_DIR="${RESULTS_ROOT}/${MODEL}/${setting}/evaluation"
            ;;
        plm)
            PREDICTIONS="${RESULTS_ROOT}/${MODEL}/tasks/${setting}/evaluator_predictions.csv"
            OUTPUT_DIR="${RESULTS_ROOT}/${MODEL}/evaluation/${setting}"
            ;;
        agent)
            local predictions_name="${AGENT_PREDICTIONS_NAME:-${setting}_agent_native_deepseek_v4_pro_predictions.csv}"
            PREDICTIONS="${RESULTS_ROOT}/${setting}/${predictions_name}"
            OUTPUT_DIR="${RESULTS_ROOT}/${setting}/evaluation"
            ;;
    esac
}

has_task_results() (
    if [[ -d "${PREDICTIONS}" ]]; then
        shopt -s globstar nullglob
        for path in "${PREDICTIONS}"/**/*.csv; do
            if [[ -f "${path}" && -s "${path}" ]]; then
                return 0
            fi
        done
        return 1
    fi
    [[ -f "${PREDICTIONS}" && -s "${PREDICTIONS}" ]]
)

export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${SETTING}" ]]; then
    settings=(
        "T1_single_mutant_generation"
        "T2_measurement_free_multi_mutant_ranking"
        "T3_anchor_informed_multi_mutant_ranking"
        "T4_single_mutant_informed_multi_mutant_ranking"
    )
else
    settings=("${SETTING}")
fi

completed=0
exit_status=0
for setting in "${settings[@]}"; do
    configure_task_paths "${setting}"
    if ! has_task_results; then
        printf '(%s results missing)\n\n' "${setting}"
        continue
    fi

    args=(
        --setting "${setting}"
        --predictions "${PREDICTIONS}"
        --output-dir "${OUTPUT_DIR}"
        --lenient "${LENIENT}"
    )
    if [[ -n "${GROUND_TRUTH_DIR}" ]]; then
        args+=(--ground-truth-dir "${GROUND_TRUTH_DIR}")
    fi
    if "${PYTHON_BIN}" -m evaluation.evaluate "${args[@]}"; then
        completed=$((completed + 1))
    else
        exit_status=$?
    fi
done

if [[ -z "${SETTING}" && "${completed}" -eq 4 ]]; then
    if [[ "${RESULTS_SOURCE}" == "agent" ]]; then
        average_root="${RESULTS_ROOT}"
    else
        average_root="${RESULTS_ROOT}/${MODEL}"
    fi
    bash "${SCRIPT_DIR}/average_nms.sh" "${average_root}"
fi
exit "${exit_status}"
