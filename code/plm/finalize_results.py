#!/usr/bin/env python3
"""Merge model shards, validate coverage, and export evaluator-ready scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common_io import atomic_write_csv, atomic_write_json, sha256_file, sha256_json, utc_now
from v7_io import load_v7_dataset


IDENTITY = ["sample_id", "subset", "candidate_group_id", "source_assay", "mutant", "status"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-schema", type=Path, required=True)
    parser.add_argument("--primary-score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    published = [args.run_dir / name for name in ("predictions.csv", "context_predictions.csv", "summary.json", "run.json", "tasks")]
    if any(path.exists() for path in published):
        raise FileExistsError("run directory already contains finalized outputs")
    dataset, _, samples, _, queries, contexts, _, _ = load_v7_dataset(args.dataset_dir)
    schema = json.loads(args.output_schema.read_text())
    if schema.get("identity_columns") != IDENTITY:
        raise ValueError("output schema identity columns differ")
    outputs = schema.get("sample_outputs", {})
    primary = args.primary_score or schema.get("primary_score")
    if not outputs or primary not in outputs or not outputs[primary].get("higher_is_better", False):
        raise ValueError("invalid primary model score")
    expected_columns = IDENTITY + list(outputs)

    metadata_paths = sorted((args.run_dir / "shards").glob("shard_*.json"))
    if not metadata_paths:
        raise ValueError("no completed shards")
    shard_metadata = [json.loads(path.read_text()) for path in metadata_paths]
    num_shards = {int(item["num_shards"]) for item in shard_metadata}
    if len(num_shards) != 1:
        raise ValueError("inconsistent shard count")
    num_shards = num_shards.pop()
    if sorted(int(item["shard_id"]) for item in shard_metadata) != list(range(num_shards)):
        raise ValueError("incomplete shard IDs")
    schema_hash = sha256_file(args.output_schema)
    config_hashes, frames = set(), []
    for item in shard_metadata:
        if item.get("partial_run", False):
            raise ValueError("smoke-test shard cannot be finalized")
        if item.get("dataset_id") != dataset["dataset_id"] or item.get("dataset_hash") != dataset["dataset_hash"]:
            raise ValueError("shard targets a different dataset")
        config = item["run_config"]
        config_hash = sha256_json(config)
        if config_hash != item.get("run_config_sha256") or config.get("output_schema_sha256") != schema_hash:
            raise ValueError("shard configuration provenance differs")
        config_hashes.add(config_hash)
        path = args.run_dir / "shards" / f"shard_{int(item['shard_id']):04d}.csv"
        frame = pd.read_csv(path)
        if frame.columns.tolist() != expected_columns or len(frame) != int(item["samples"]):
            raise ValueError(f"{path}: output schema or row count differs")
        frames.append(frame)
    if len(config_hashes) != 1:
        raise ValueError("shards used different configurations")

    raw = pd.concat(frames, ignore_index=True)
    if not raw["status"].eq("ok").all():
        raise ValueError("formal run contains failed rows")
    numeric = list(outputs)
    if any(not pd.api.types.is_numeric_dtype(raw[column]) for column in numeric):
        raise ValueError("model output contains non-numeric values")
    if not np.isfinite(raw[numeric].to_numpy()).all():
        raise ValueError("model output contains non-finite values")
    identity = raw[IDENTITY[:-1]].drop_duplicates("sample_id")
    if raw.groupby("sample_id")[IDENTITY[1:-1]].nunique().gt(1).any().any():
        raise ValueError("chain contributions disagree on sample identity")
    aggregated = raw.groupby("sample_id", sort=False)[numeric].sum().reset_index()
    predictions = identity.merge(aggregated, on="sample_id", validate="one_to_one")
    predictions["status"] = "ok"
    predictions = predictions[expected_columns]

    expected_all = pd.concat([samples, contexts], ignore_index=True)
    checked = expected_all[["sample_id", "task", "query_id", "source_assay", "sample_role", "mutant"]].merge(
        predictions[["sample_id", "subset", "candidate_group_id", "source_assay", "mutant"]],
        on="sample_id", how="outer", suffixes=("_expected", "_result"), indicator=True, validate="one_to_one",
    )
    if not checked["_merge"].eq("both").all():
        raise ValueError("model sample coverage differs from the dataset")
    for expected_column, result_column in (("task", "subset"), ("query_id", "candidate_group_id"), ("source_assay_expected", "source_assay_result"), ("mutant_expected", "mutant_result")):
        if not checked[expected_column].eq(checked[result_column]).all():
            raise ValueError(f"model identity differs: {expected_column}")
    roles = expected_all[["sample_id", "sample_role", "task", "query_id"]]
    predictions = predictions.merge(roles, on="sample_id", validate="one_to_one")
    evaluation = predictions[predictions["sample_role"].eq("evaluation_candidate")].copy()
    context_predictions = predictions[~predictions["sample_role"].eq("evaluation_candidate")].copy()
    if len(evaluation) != len(samples) or len(context_predictions) != len(contexts):
        raise ValueError("evaluation/context split differs")

    task_counts = {}
    for task, task_queries in queries.groupby("task", sort=True):
        task_predictions = evaluation[evaluation["task"].eq(task)].copy()
        observed = task_predictions.groupby("query_id").size()
        expected = task_queries.set_index("query_id")["expected_samples"].astype(int)
        if not observed.sort_index().equals(expected.sort_index()):
            raise ValueError(f"{task}: incomplete candidate ranking")
        evaluator = task_predictions[["query_id", "mutant", primary]].rename(columns={"query_id": "assay", primary: "score"})
        task_dir = args.run_dir / "tasks" / task
        atomic_write_csv(task_predictions.sort_values(["query_id", "mutant"]), task_dir / "predictions.csv")
        atomic_write_csv(evaluator.sort_values(["assay", "score"], ascending=[True, False]), task_dir / "evaluator_predictions.csv")
        task_counts[task] = {"queries": len(task_queries), "samples": len(task_predictions)}

    atomic_write_csv(evaluation.sort_values("sample_id"), args.run_dir / "predictions.csv")
    atomic_write_csv(context_predictions.sort_values("sample_id"), args.run_dir / "context_predictions.csv")
    config = shard_metadata[0]["run_config"]
    summary = {
        "dataset_id": dataset["dataset_id"], "dataset_hash": dataset["dataset_hash"],
        "model_id": config["model_id"], "run_config_sha256": next(iter(config_hashes)),
        "completed_at_utc": utc_now(), "evaluation_samples": len(evaluation),
        "context_samples": len(context_predictions), "primary_score": primary,
        "numeric_columns": numeric, "chain_contribution_rows": len(raw),
        "multichain_samples": int(raw.groupby("sample_id").size().gt(1).sum()),
        "tasks": task_counts, "run_config": config,
    }
    atomic_write_json(summary, args.run_dir / "summary.json")
    atomic_write_json({"dataset_id": dataset["dataset_id"], "dataset_hash": dataset["dataset_hash"], "model_id": config["model_id"], "run_config_sha256": next(iter(config_hashes)), "shards": shard_metadata}, args.run_dir / "run.json")
    print(f"evaluation_samples={len(evaluation)} context_samples={len(context_predictions)} chain_rows={len(raw)}")


if __name__ == "__main__":
    main()
