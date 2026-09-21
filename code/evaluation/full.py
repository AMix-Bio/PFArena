"""Evaluation for mutation benchmark settings."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from router import DATASET_INPUTS

DEFAULT_T_BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "PFArena"
TASK_NAMES = {task_id: task_dir for task_id, (task_dir, _) in DATASET_INPUTS.items()}
DEFAULT_T1_GROUND_TRUTH_DIR = DEFAULT_T_BENCHMARK_ROOT / TASK_NAMES["1"]
DEFAULT_T2_GROUND_TRUTH_DIR = DEFAULT_T_BENCHMARK_ROOT / TASK_NAMES["2"]
DEFAULT_T3_GROUND_TRUTH_DIR = DEFAULT_T_BENCHMARK_ROOT / TASK_NAMES["3"]
DEFAULT_T4_GROUND_TRUTH_DIR = DEFAULT_T_BENCHMARK_ROOT / TASK_NAMES["4"]
DEFAULT_GROUND_TRUTH_DIRS = {
    TASK_NAMES[task_id]: DEFAULT_T_BENCHMARK_ROOT / task_dir
    for task_id, (task_dir, _) in DATASET_INPUTS.items()
}
SUPPORTED_SETTINGS = tuple(DEFAULT_GROUND_TRUTH_DIRS)
TOP_K = 40
T_RANKING_K = 5
NMS_METRIC = "NMS@40"
RECALL_METRIC = "Recall@40"
T_RANKING_NMS_METRIC = "NMS@5"
T_RANKING_RECALL_METRIC = "Recall@5"
RANKING_NDCG_METRIC = "NDCG"
RANKING_SPEARMAN_METRIC = "spearman"
T_RANKING_SETTINGS = frozenset(TASK_NAMES[task_id] for task_id in ("2", "3", "4"))
NESTED_GROUP_SETTINGS = frozenset(TASK_NAMES[task_id] for task_id in ("3", "4"))
FULL_METRICS = (NMS_METRIC, RECALL_METRIC)
T_RANKING_METRICS = (
    RANKING_SPEARMAN_METRIC,
    RANKING_NDCG_METRIC,
    T_RANKING_NMS_METRIC,
    T_RANKING_RECALL_METRIC,
)


@dataclass(frozen=True)
class PredictionRow:
    assay: str
    mutant: str
    order_value: float
    input_index: int
    source: str


@dataclass(frozen=True)
class AssayTruth:
    assay: str
    rows: tuple[tuple[str, float], ...]
    by_mutant: dict[str, float]
    source_path: Path
    raw_rows: int
    invalid_score_rows: int
    invalid_mutant_rows: int
    duplicate_mutants_collapsed: int

    @property
    def score_min(self) -> float:
        return min(score for _, score in self.rows)

    @property
    def score_max(self) -> float:
        return max(score for _, score in self.rows)


@dataclass(frozen=True)
class FullEvaluationResult:
    per_assay: list[dict[str, Any]]
    summary: dict[str, Any]


def normalize_assay_id(value: str) -> str:
    assay = value.strip()
    if assay.endswith(".csv"):
        assay = assay[:-4]
    return assay


def default_ground_truth_dir(setting: str) -> Path:
    setting = normalize_setting(setting)
    try:
        return DEFAULT_GROUND_TRUTH_DIRS[setting]
    except KeyError as exc:
        raise ValueError(f"Unsupported setting: {setting}") from exc


def resolve_cli_ground_truth_dir(setting: str, ground_truth_dir: Path | None) -> Path:
    if ground_truth_dir is None:
        return default_ground_truth_dir(setting)

    setting_dir = default_ground_truth_dir(setting).name
    nested_setting_dir = ground_truth_dir / setting_dir
    if nested_setting_dir.is_dir() and (nested_setting_dir / "norm_data").is_dir():
        return nested_setting_dir
    return ground_truth_dir


def normalize_setting(setting: str) -> str:
    return setting


def setting_top_k(setting: str) -> int:
    return T_RANKING_K if normalize_setting(setting) in T_RANKING_SETTINGS else TOP_K


def setting_metrics(setting: str) -> tuple[str, ...]:
    return T_RANKING_METRICS if normalize_setting(setting) in T_RANKING_SETTINGS else FULL_METRICS


def setting_requires_full_ranking(setting: str) -> bool:
    return normalize_setting(setting) in T_RANKING_SETTINGS


def setting_uses_nested_groups(setting: str) -> bool:
    return normalize_setting(setting) in NESTED_GROUP_SETTINGS


def setting_item_column(setting: str) -> str:
    return "mutant"


def setting_prediction_item_columns(setting: str) -> tuple[str, ...]:
    return ("mutant", "annots", "mutation")


def resolve_ground_truth_dir(path: Path) -> Path:
    if path.exists() and path.is_dir():
        nested = path / "norm_data"
        if nested.exists() and nested.is_dir() and (any(nested.glob("*.csv")) or any(nested.glob("*/*.csv"))):
            return nested
        if any(path.glob("*.csv")):
            return path
    return path


def ground_truth_csv_paths(ground_truth_dir: Path, *, nested_groups: bool, setting: str) -> list[Path]:
    setting = normalize_setting(setting)
    if setting == TASK_NAMES["3"]:
        return sorted(ground_truth_dir.glob("*/anchor_*.csv"))
    if setting == TASK_NAMES["4"]:
        return sorted(ground_truth_dir.glob("*/single_context_combo*.csv"))
    if nested_groups:
        return sorted(ground_truth_dir.glob("*/*.csv"))
    return sorted(ground_truth_dir.glob("*.csv"))


def fallback_assay_id(path: Path, *, nested_groups: bool, setting: str) -> str:
    if setting == TASK_NAMES["3"]:
        return f"{path.parent.name}_T3"
    if setting == TASK_NAMES["4"]:
        return f"{path.parent.name}_T4"
    if nested_groups:
        return f"{path.parent.name}|{path.stem}"
    return path.stem


def _parse_finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def load_full_ground_truth(
    ground_truth_dir: Path,
    *,
    nested_groups: bool = False,
    item_column: str = "mutant",
    setting: str = "full",
) -> dict[str, AssayTruth]:
    ground_truth_dir = resolve_ground_truth_dir(ground_truth_dir)
    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {ground_truth_dir}")
    if not ground_truth_dir.is_dir():
        raise NotADirectoryError(f"Ground-truth path is not a directory: {ground_truth_dir}")

    assays: dict[str, AssayTruth] = {}
    csv_paths = ground_truth_csv_paths(ground_truth_dir, nested_groups=nested_groups, setting=setting)
    for path in csv_paths:
        fallback_assay = fallback_assay_id(path, nested_groups=nested_groups, setting=setting)
        best_by_mutant: dict[str, float] = {}
        group_ids: set[str] = set()
        raw_rows = 0
        invalid_score_rows = 0
        invalid_mutant_rows = 0
        duplicate_rows = 0

        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = {item_column, "DMS_score"} - fieldnames
            if missing:
                raise ValueError(f"{path} missing required columns: {sorted(missing)}")

            for row in reader:
                raw_rows += 1
                group_id = (row.get("candidate_group_id") or "").strip()
                if group_id:
                    group_ids.add(group_id)
                mutant = (row.get(item_column) or "").strip()
                score = _parse_finite_float(row.get("DMS_score", ""))
                if not mutant:
                    invalid_mutant_rows += 1
                if score is None:
                    invalid_score_rows += 1
                if not mutant or score is None:
                    continue
                if mutant in best_by_mutant:
                    duplicate_rows += 1
                    if score > best_by_mutant[mutant]:
                        best_by_mutant[mutant] = score
                else:
                    best_by_mutant[mutant] = score

        if not best_by_mutant:
            raise ValueError(f"{path} has no valid ground-truth rows")

        if nested_groups:
            if len(group_ids) > 1:
                raise ValueError(f"{path} has multiple candidate_group_id values: {sorted(group_ids)[:5]}")
            assay = next(iter(group_ids)) if group_ids else fallback_assay
        else:
            assay = fallback_assay
        if assay in assays:
            raise ValueError(f"Duplicate ground-truth query id: {assay}")

        rows = tuple(sorted(best_by_mutant.items(), key=lambda item: (-item[1], item[0])))
        assays[assay] = AssayTruth(
            assay=assay,
            rows=rows,
            by_mutant=dict(best_by_mutant),
            source_path=path,
            raw_rows=raw_rows,
            invalid_score_rows=invalid_score_rows,
            invalid_mutant_rows=invalid_mutant_rows,
            duplicate_mutants_collapsed=duplicate_rows,
        )

    if not assays:
        raise ValueError(f"No assay CSV files found in {ground_truth_dir}")
    return assays


def _choose_column(fieldnames: Iterable[str], requested: str | None, aliases: tuple[str, ...]) -> str | None:
    available = set(fieldnames)
    if requested:
        return requested if requested in available else None
    for alias in aliases:
        if alias in available:
            return alias
    return None


def load_predictions(
    predictions_path: Path,
    *,
    assay_column: str | None = None,
    mutant_column: str | None = None,
    item_column_aliases: tuple[str, ...] = ("mutant", "annots", "mutation"),
    rank_column: str = "rank",
    score_column: str = "score",
    setting: str = "full",
) -> dict[str, list[PredictionRow]]:
    if predictions_path.is_dir():
        return _load_prediction_dir(
            predictions_path,
            assay_column=assay_column,
            mutant_column=mutant_column,
            item_column_aliases=item_column_aliases,
            rank_column=rank_column,
            score_column=score_column,
            setting=setting,
        )
    return _load_prediction_file(
        predictions_path,
        default_assay=None,
        assay_column=assay_column,
        mutant_column=mutant_column,
        item_column_aliases=item_column_aliases,
        rank_column=rank_column,
        score_column=score_column,
    )


def _derive_nested_default_assay(path: Path, setting: str) -> str:
    if "__" in path.stem:
        return path.stem.replace("__", "|")
    if setting == TASK_NAMES["3"]:
        return f"{path.parent.name}_T3"
    if setting == TASK_NAMES["4"]:
        return f"{path.parent.name}_T4"
    return f"{path.parent.name}|{path.stem}"


def _load_prediction_dir(
    predictions_dir: Path,
    *,
    assay_column: str | None,
    mutant_column: str | None,
    item_column_aliases: tuple[str, ...],
    rank_column: str,
    score_column: str,
    setting: str,
) -> dict[str, list[PredictionRow]]:
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {predictions_dir}")
    per_assay: dict[str, list[PredictionRow]] = defaultdict(list)
    csv_paths = sorted(predictions_dir.rglob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No prediction CSV files found in {predictions_dir}")
    for path in csv_paths:
        is_nested = path.parent != predictions_dir
        default_assay = _derive_nested_default_assay(path, setting) if is_nested else path.stem
        loaded = _load_prediction_file(
            path,
            default_assay=default_assay,
            assay_column=assay_column,
            mutant_column=mutant_column,
            item_column_aliases=item_column_aliases,
            rank_column=rank_column,
            score_column=score_column,
            force_default_assay=is_nested,
        )
        for assay, rows in loaded.items():
            per_assay[assay].extend(rows)
    return _sort_predictions(per_assay)


def _load_prediction_file(
    predictions_file: Path,
    *,
    default_assay: str | None,
    assay_column: str | None,
    mutant_column: str | None,
    item_column_aliases: tuple[str, ...],
    rank_column: str,
    score_column: str,
    force_default_assay: bool = False,
) -> dict[str, list[PredictionRow]]:
    if not predictions_file.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {predictions_file}")
    if not predictions_file.is_file():
        raise ValueError(f"Prediction path is not a file: {predictions_file}")

    per_assay: dict[str, list[PredictionRow]] = defaultdict(list)
    with predictions_file.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"Prediction file has no header: {predictions_file}")

        chosen_assay_column = (
            None
            if force_default_assay
            else _choose_column(fieldnames, assay_column, ("assay", "assay_id", "dataset"))
        )
        chosen_mutant_column = _choose_column(fieldnames, mutant_column, item_column_aliases)
        if chosen_mutant_column is None:
            raise ValueError(
                f"{predictions_file} missing prediction item column; "
                f"expected one of {', '.join(item_column_aliases)}"
            )
        if chosen_assay_column is None and default_assay is None:
            raise ValueError(f"{predictions_file} missing assay column")

        has_score = score_column in fieldnames
        has_rank = rank_column in fieldnames
        ordering_mode = "score" if has_score else "rank" if has_rank else "input_order"

        for input_index, row in enumerate(reader):
            assay = default_assay if chosen_assay_column is None else row.get(chosen_assay_column, "")
            assay = normalize_assay_id(assay or "")
            mutant = (row.get(chosen_mutant_column) or "").strip()
            if not assay or not mutant:
                continue

            if ordering_mode == "score":
                parsed = _parse_finite_float(row.get(score_column, ""))
                if parsed is None:
                    raise ValueError(f"Invalid score at {predictions_file}:{input_index + 2}")
                order_value = -parsed
            elif ordering_mode == "rank":
                parsed = _parse_finite_float(row.get(rank_column, ""))
                if parsed is None:
                    raise ValueError(f"Invalid rank at {predictions_file}:{input_index + 2}")
                order_value = parsed
            else:
                order_value = float(input_index + 1)

            per_assay[assay].append(
                PredictionRow(
                    assay=assay,
                    mutant=mutant,
                    order_value=order_value,
                    input_index=input_index,
                    source=f"{predictions_file}:{input_index + 2}",
                )
            )

    if not per_assay:
        raise ValueError(f"No predictions found in {predictions_file}")
    return _sort_predictions(per_assay)


def _sort_predictions(per_assay: dict[str, list[PredictionRow]]) -> dict[str, list[PredictionRow]]:
    return {
        assay: sorted(rows, key=lambda row: (row.order_value, row.input_index, row.mutant))
        for assay, rows in per_assay.items()
    }


def evaluate_full_setting(
    *,
    setting: str = TASK_NAMES["1"],
    ground_truth_dir: Path | None = None,
    predictions_path: Path,
    top_k: int = TOP_K,
    strict: bool = True,
    assay_column: str | None = None,
    mutant_column: str | None = None,
    rank_column: str = "rank",
    score_column: str = "score",
) -> FullEvaluationResult:
    setting = normalize_setting(setting)
    top_k = (
        setting_top_k(setting)
        if top_k == TOP_K and setting in T_RANKING_SETTINGS
        else top_k
    )
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    ground_truth_dir = resolve_ground_truth_dir(resolve_cli_ground_truth_dir(setting, ground_truth_dir))
    truth = load_full_ground_truth(
        ground_truth_dir,
        nested_groups=setting_uses_nested_groups(setting),
        item_column=setting_item_column(setting),
        setting=setting,
    )
    predictions = load_predictions(
        predictions_path,
        assay_column=assay_column,
        mutant_column=mutant_column,
        item_column_aliases=setting_prediction_item_columns(setting),
        rank_column=rank_column,
        score_column=score_column,
        setting=setting,
    )
    _validate_predictions(truth, predictions, top_k=top_k, strict=strict, setting=setting)

    per_assay = [
        evaluate_one_assay(
            truth=assay_truth,
            pred_rows=predictions.get(assay)
            or missing_query_random_predictions(assay_truth, enabled=not strict),
            top_k=top_k,
            setting=setting,
        )
        for assay, assay_truth in sorted(truth.items())
    ]
    summary = summarize_full_results(
        per_assay,
        truth=truth,
        predictions=predictions,
        setting=setting,
        ground_truth_dir=ground_truth_dir,
        predictions_path=predictions_path,
        top_k=top_k,
        strict=strict,
    )
    return FullEvaluationResult(per_assay=per_assay, summary=summary)


def _validate_predictions(
    truth: dict[str, AssayTruth],
    predictions: dict[str, list[PredictionRow]],
    *,
    top_k: int,
    strict: bool,
    setting: str,
) -> None:
    truth_assays = set(truth)
    prediction_assays = set(predictions)
    unknown = sorted(prediction_assays - truth_assays)
    missing = sorted(truth_assays - prediction_assays)
    short = sorted(
        assay
        for assay, rows in predictions.items()
        if assay in truth_assays and len(rows) < required_prediction_count(truth[assay], top_k)
    )
    if strict and (unknown or missing or short):
        messages = []
        if unknown:
            messages.append(f"unknown assay ids: {unknown[:10]}{'...' if len(unknown) > 10 else ''}")
        if missing:
            messages.append(f"missing assays: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if short:
            preview = [
                (assay, len(predictions[assay]), required_prediction_count(truth[assay], top_k))
                for assay in short[:10]
            ]
            messages.append(
                f"assays below required prediction count (assay, provided, required): "
                f"{preview}{'...' if len(short) > 10 else ''}"
            )
        raise ValueError("; ".join(messages))


def legal_prediction_mutants(rows: list[PredictionRow], truth: AssayTruth) -> set[str]:
    return {row.mutant for row in rows if row.mutant in truth.by_mutant}


def required_prediction_count(truth: AssayTruth, top_k: int) -> int:
    return min(top_k, len(truth.rows))


def missing_query_random_predictions(truth: AssayTruth, *, enabled: bool) -> list[PredictionRow]:
    if not enabled:
        return []
    ordered_items = sorted(
        (mutant for mutant, _ in truth.rows),
        key=lambda item: missing_query_random_key(truth.assay, item),
    )
    return [
        PredictionRow(
            assay=truth.assay,
            mutant=mutant,
            order_value=float(index),
            input_index=index - 1,
            source="synthetic_missing_query_random",
        )
        for index, mutant in enumerate(ordered_items, start=1)
    ]


def missing_query_random_key(assay: str, item: str) -> str:
    return hashlib.sha256(f"missing-query-random\0{assay}\0{item}".encode("utf-8")).hexdigest()


def evaluate_one_assay(
    *,
    truth: AssayTruth,
    pred_rows: list[PredictionRow],
    top_k: int,
    setting: str,
) -> dict[str, Any]:
    selected = pred_rows[:top_k]
    span = truth.score_max - truth.score_min
    selected_annots = [row.mutant for row in selected]

    seen_legal: set[str] = set()
    unique_legal_selected: list[str] = []
    duplicate_in_budget = 0
    illegal_in_budget = 0
    for annot in selected_annots:
        if annot not in truth.by_mutant:
            illegal_in_budget += 1
            continue
        if annot in seen_legal:
            duplicate_in_budget += 1
            continue
        seen_legal.add(annot)
        unique_legal_selected.append(annot)

    selected_scores = [truth.by_mutant[annot] for annot in unique_legal_selected]
    best_selected = max(selected_scores) if selected_scores else truth.score_min
    normalized_max_score = 1.0 if span == 0 else (best_selected - truth.score_min) / span

    recall_denominator = required_prediction_count(truth, top_k)
    if recall_denominator > 0:
        # tie-inclusive top-k: include every legal mutant whose score is at
        # least the score at the k-th position, so score ties do not depend on
        # mutant string ordering. truth.rows is sorted by (-score, mutant).
        threshold_score = truth.rows[recall_denominator - 1][1]
        top_recall = {annot for annot, score in truth.rows if score >= threshold_score}
        recalled = sum(annot in top_recall for annot in unique_legal_selected)
        recall = recalled / recall_denominator
    else:
        recall = 0.0

    all_pred_annots = [row.mutant for row in pred_rows]
    num_illegal_predictions = sum(annot not in truth.by_mutant for annot in all_pred_annots)
    num_duplicate_predictions = len(all_pred_annots) - len(set(all_pred_annots))

    row = {
        "assay": truth.assay,
        "num_ground_truth_mutants": len(truth.rows),
        "raw_ground_truth_rows": truth.raw_rows,
        "invalid_ground_truth_score_rows": truth.invalid_score_rows,
        "invalid_ground_truth_mutant_rows": truth.invalid_mutant_rows,
        "duplicate_ground_truth_mutants_collapsed": truth.duplicate_mutants_collapsed,
        "num_predictions": len(pred_rows),
        "num_selected": len(selected),
        "num_unique_legal_selected": len(unique_legal_selected),
        "num_illegal_predictions": num_illegal_predictions,
        "num_duplicate_predictions": num_duplicate_predictions,
        "illegal_predictions_within_budget": illegal_in_budget,
        "duplicate_predictions_within_budget": duplicate_in_budget,
    }
    if normalize_setting(setting) in T_RANKING_SETTINGS:
        row.update(
            {
                RANKING_SPEARMAN_METRIC: spearman_score(truth=truth, pred_rows=pred_rows),
                RANKING_NDCG_METRIC: ndcg_score(
                    truth_rows=truth.rows,
                    truth_by_mutant=truth.by_mutant,
                    pred_annots=all_pred_annots,
                    budget=len(truth.rows),
                ),
                T_RANKING_NMS_METRIC: float(max(0.0, min(1.0, normalized_max_score))),
                T_RANKING_RECALL_METRIC: float(max(0.0, min(1.0, recall))),
            }
        )
    else:
        row.update(
            {
                NMS_METRIC: float(max(0.0, min(1.0, normalized_max_score))),
                RECALL_METRIC: float(max(0.0, min(1.0, recall))),
            }
        )
    return row


def ndcg_score(
    *,
    truth_rows: Iterable[tuple[str, float]],
    truth_by_mutant: dict[str, float],
    pred_annots: list[str],
    budget: int,
) -> float:
    truth_rows = tuple(truth_rows)
    if budget <= 0 or not truth_rows:
        return 0.0

    score_min = min(score for _, score in truth_rows)
    score_max = max(score for _, score in truth_rows)
    span = score_max - score_min

    def relevance(annot: str) -> float:
        if annot not in truth_by_mutant:
            return 0.0
        if span <= 0:
            return 1.0
        value = (truth_by_mutant[annot] - score_min) / span
        return max(0.0, min(1.0, float(value)))

    def discount(rank_index: int) -> float:
        return 1.0 / math.log2(rank_index + 2.0)

    seen: set[str] = set()
    dcg = 0.0
    for rank_index in range(budget):
        annot = pred_annots[rank_index] if rank_index < len(pred_annots) else ""
        rel = 0.0
        if annot in truth_by_mutant and annot not in seen:
            seen.add(annot)
            rel = relevance(annot)
        dcg += rel * discount(rank_index)

    ideal_relevances = sorted((relevance(annot) for annot, _ in truth_rows), reverse=True)[:budget]
    idcg = sum(rel * discount(rank_index) for rank_index, rel in enumerate(ideal_relevances))
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def spearman_score(*, truth: AssayTruth, pred_rows: list[PredictionRow]) -> float:
    truth_ranks = average_ranks(
        [(mutant, -score) for mutant, score in truth.by_mutant.items()]
    )
    predicted_ranks = prediction_ranks(pred_rows=pred_rows, truth=truth)
    ordered_mutants = sorted(truth.by_mutant)
    xs = [truth_ranks[mutant] for mutant in ordered_mutants]
    ys = [predicted_ranks[mutant] for mutant in ordered_mutants]
    return pearson(xs, ys)


def prediction_ranks(*, pred_rows: list[PredictionRow], truth: AssayTruth) -> dict[str, float]:
    first_legal_rows: dict[str, PredictionRow] = {}
    for row in pred_rows:
        if row.mutant not in truth.by_mutant or row.mutant in first_legal_rows:
            continue
        first_legal_rows[row.mutant] = row
    ranks = average_ranks(
        [(mutant, row.order_value) for mutant, row in first_legal_rows.items()]
    )
    next_rank = len(first_legal_rows) + 1
    missing_mutants = sorted(
        set(truth.by_mutant) - set(first_legal_rows),
        key=lambda item: stable_tail_key(truth.assay, item),
    )
    for mutant in missing_mutants:
        ranks[mutant] = float(next_rank)
        next_rank += 1
    return ranks


def stable_tail_key(assay: str, item: str) -> str:
    return hashlib.sha256(f"{assay}\0{item}".encode("utf-8")).hexdigest()


def average_ranks(items: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for item_index in range(index, end):
            ranks[ordered[item_index][0]] = average_rank
        index = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    value = sum(x * y for x, y in zip(centered_x, centered_y)) / (denom_x * denom_y)
    return float(max(-1.0, min(1.0, value)))


def summarize_full_results(
    per_assay: list[dict[str, Any]],
    *,
    truth: dict[str, AssayTruth],
    predictions: dict[str, list[PredictionRow]],
    setting: str,
    ground_truth_dir: Path,
    predictions_path: Path,
    top_k: int,
    strict: bool,
) -> dict[str, Any]:
    if not per_assay:
        raise ValueError("No per-assay results to summarize")

    def mean(metric: str) -> float:
        return float(sum(float(row[metric]) for row in per_assay) / len(per_assay))

    missing_assays = sorted(set(truth) - set(predictions))
    unknown_assays = sorted(set(predictions) - set(truth))
    short_assays = sorted(
        assay
        for assay, rows in predictions.items()
        if assay in truth and len(rows) < required_prediction_count(truth[assay], top_k)
    )
    incomplete_ranking_assays = sorted(
        assay
        for assay, rows in predictions.items()
        if assay in truth
        and setting_requires_full_ranking(setting)
        and legal_prediction_mutants(rows, truth[assay]) != set(truth[assay].by_mutant)
    )
    metrics = setting_metrics(setting)
    num_queries = len(per_assay)
    num_assays = (
        len({assay.source_path.parent.name for assay in truth.values()})
        if setting_uses_nested_groups(setting)
        else num_queries
    )

    return {
        "setting": setting,
        "strict": strict,
        "num_assays": num_assays,
        "num_queries": num_queries,
        "ground_truth_dir": str(ground_truth_dir),
        "predictions_path": str(predictions_path),
        "metrics": {metric: mean(metric) for metric in metrics},
        "quality": {
            "missing_assays": len(missing_assays),
            "unknown_assays": len(unknown_assays),
            "assays_below_budget": len(short_assays),
            "assays_with_incomplete_candidate_ranking": len(incomplete_ranking_assays),
            "mean_illegal_predictions_within_budget": mean("illegal_predictions_within_budget"),
            "mean_duplicate_predictions_within_budget": mean("duplicate_predictions_within_budget"),
            "total_invalid_ground_truth_score_rows": sum(
                assay.invalid_score_rows for assay in truth.values()
            ),
            "total_invalid_ground_truth_mutant_rows": sum(
                assay.invalid_mutant_rows for assay in truth.values()
            ),
            "total_duplicate_ground_truth_mutants_collapsed": sum(
                assay.duplicate_mutants_collapsed for assay in truth.values()
            ),
        },
        "problem_assays": {
            "missing_assays": missing_assays,
            "unknown_assays": unknown_assays,
            "assays_below_budget": {
                assay: {
                    "provided": len(predictions[assay]),
                    "required": required_prediction_count(truth[assay], top_k),
                }
                for assay in short_assays
            },
            "assays_with_incomplete_candidate_ranking": {
                assay: {
                    "legal_provided": len(legal_prediction_mutants(predictions[assay], truth[assay])),
                    "required": len(truth[assay].by_mutant),
                }
                for assay in incomplete_ranking_assays
            },
        },
    }


def write_outputs(result: FullEvaluationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_per_assay_csv(output_dir / "per_assay_metrics.csv", result.per_assay)
    write_summary_json(output_dir / "summary_metrics.json", result.summary)
    write_summary_csv(output_dir / "summary_metrics.csv", result.summary)


def write_per_assay_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty per-assay metrics")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    quality = summary["quality"]
    row = {
        "setting": summary["setting"],
        "num_assays": summary["num_assays"],
        "num_queries": summary["num_queries"],
        **metrics,
        "missing_assays": quality["missing_assays"],
        "unknown_assays": quality["unknown_assays"],
        "assays_below_budget": quality["assays_below_budget"],
        "assays_with_incomplete_candidate_ranking": quality["assays_with_incomplete_candidate_ranking"],
        "mean_illegal_predictions_within_budget": quality["mean_illegal_predictions_within_budget"],
        "mean_duplicate_predictions_within_budget": quality["mean_duplicate_predictions_within_budget"],
        "total_invalid_ground_truth_score_rows": quality["total_invalid_ground_truth_score_rows"],
        "total_invalid_ground_truth_mutant_rows": quality["total_invalid_ground_truth_mutant_rows"],
        "total_duplicate_ground_truth_mutants_collapsed": quality[
            "total_duplicate_ground_truth_mutants_collapsed"
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
