#!/usr/bin/env python3
"""Combine joint EVE scores with a compatible finalized S3F run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common_io import atomic_write_csv, atomic_write_json, sha256_file, sha256_files, sha256_json, utc_now
from model_dataset_io import load_model_dataset
from run_stage import EVE_SOURCE_SHA256, validate_score_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--s3f-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"S3F-MSA output already exists: {args.output_dir}")
    dataset, _, samples, _, _ = load_model_dataset(args.dataset_dir)
    inputs = json.loads((args.input_dir / "inputs.json").read_text())
    if inputs["dataset_hash"] != dataset["dataset_hash"] or inputs["partial_input"]:
        raise ValueError("S3F-MSA inputs target another dataset")
    mapping = pd.read_csv(args.input_dir / "mapping.csv")
    sample_mapping = pd.read_csv(args.input_dir / "sample_mapping.csv")
    score_frames, artifacts = [], []
    for row in mapping.itertuples(index=False):
        path = args.cache_dir / "eve_scores" / f"{row.context_id}.csv"
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"missing EVE score: {row.context_id}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("release_dataset_hash", metadata.get("dataset_hash")) != dataset["dataset_hash"] or metadata.get("release_input_hash", metadata.get("input_hash")) != inputs["input_hash"] or metadata.get("stage") != "score" or metadata.get("eve_source_sha256") != EVE_SOURCE_SHA256 or metadata.get("score_sha256") != sha256_file(path) or metadata.get("random_seeds") != list(range(5)) or metadata.get("num_samples_compute_evol_indices") != 20_000 or metadata.get("batch_size") != 1024 or metadata.get("aggregation_method") != "full":
            raise ValueError(f"{row.context_id}: EVE score provenance differs")
        frame = pd.read_csv(path)
        validate_score_frame(frame, pd.read_csv(args.input_dir / "mutations" / row.DMS_filename)["mutant"], row.context_id)
        score_source = json.loads(metadata_path.read_text()).get(
            "eve_score_source", "trained_local"
        )
        if score_source not in {"trained_local", "validated_reuse"}:
            raise ValueError(f"{row.context_id}: invalid EVE score provenance")
        frame.insert(0, "context_id", row.context_id)
        frame.insert(1, "eve_score_source", score_source)
        score_frames.append(frame); artifacts.extend([path, metadata_path])
    eve = pd.concat(score_frames, ignore_index=True)
    chain_scores = sample_mapping.merge(eve, left_on=["context_id", "eve_mutant"], right_on=["context_id", "mutant"], how="left", validate="many_to_one", suffixes=("", "_eve"))
    score_columns = [*[f"eve_score_seed_{seed}" for seed in range(5)], "eve_ensemble"]
    if chain_scores[score_columns].isna().any().any():
        raise ValueError("S3F-MSA sample-chain EVE scores are incomplete")
    aggregated = chain_scores.groupby("sample_id", sort=False)[score_columns].sum().reset_index()
    source_groups = chain_scores.groupby("sample_id")["eve_score_source"]
    if source_groups.nunique().gt(1).any():
        raise ValueError("a sample mixes EVE artifact provenance across chains")
    source = source_groups.first().map(
        {"trained_local": 0, "validated_reuse": 1}
    ).rename("eve_source_code")
    identity = samples[["sample_id", "subset", "candidate_group_id", "source_assay", "sample_role", "mutant", "mutated_sequence"]]
    frame = identity.merge(aggregated, on="sample_id", validate="one_to_one").merge(source, on="sample_id", validate="one_to_one")
    s3f_files = [
        next(path for path in (args.s3f_run_dir / "predictions.csv", args.s3f_run_dir / "predictions.csv.gz") if path.is_file()),
        next(path for path in (args.s3f_run_dir / "context_predictions.csv", args.s3f_run_dir / "context_predictions.csv.gz") if path.is_file()),
    ]
    s3f = pd.concat([pd.read_csv(path) for path in s3f_files], ignore_index=True)
    summary_path = args.s3f_run_dir / "summary.json"
    summary_path = summary_path if summary_path.is_file() else args.s3f_run_dir / "run.json"
    s3f_summary = json.loads(summary_path.read_text())
    if s3f_summary["dataset_hash"] != dataset["dataset_hash"] or len(s3f) != len(samples):
        raise ValueError("S3F run is incomplete or targets another dataset")
    frame = frame.merge(s3f[["sample_id", "s3f_score"]], on="sample_id", validate="one_to_one")
    frame["eve_single"] = frame["eve_score_seed_0"]
    candidates = frame[frame.sample_role.eq("evaluation_candidate")]
    statistics = candidates.groupby(["subset", "candidate_group_id"]).agg(
        s3f_mean=("s3f_score", "mean"),
        s3f_std=("s3f_score", "std"),
        eve_mean=("eve_ensemble", "mean"),
        eve_std=("eve_ensemble", "std"),
    )
    if (statistics[["s3f_std", "eve_std"]] <= 0).any().any():
        raise ValueError("query component variance is zero")
    frame = frame.merge(
        statistics.reset_index(),
        on=["subset", "candidate_group_id"],
        validate="many_to_one",
    )
    frame["s3f_z"] = (frame.s3f_score - frame.s3f_mean) / frame.s3f_std
    frame["eve_z"] = (frame.eve_ensemble - frame.eve_mean) / frame.eve_std
    frame["s3f_msa_score"] = (frame.s3f_z + frame.eve_z) / 2
    frame["status"] = "ok"
    output_columns = ["sample_id", "subset", "candidate_group_id", "source_assay", "mutant", "status", "s3f_score", "eve_single", "eve_ensemble", "eve_source_code", "s3f_z", "eve_z", "s3f_msa_score"]
    output = frame[output_columns].sort_values("sample_id")
    if not np.isfinite(output.drop(columns=[*output_columns[:6]]).to_numpy()).all():
        raise ValueError("S3F-MSA output contains non-finite values")
    atomic_write_csv(output, args.output_dir / "shards/shard_0000.csv")
    source_descriptions = {
        0: "locally trained five-seed model",
        1: "validated compatible five-seed model",
    }
    run_config = {
        "model_id": "S3F-MSA", "dataset_hash": dataset["dataset_hash"], "input_hash": inputs["input_hash"],
        "s3f_run_config_sha256": s3f_summary.get("run_config_sha256", sha256_json(s3f_summary["run_config"])), "eve_score_artifacts_sha256": sha256_files(artifacts),
        "locally_trained_eve_seeds": list(range(5)), "eve_score_direction": "higher is better",
        "eve_sources": {
            str(code): source_descriptions[code]
            for code in sorted(output.eve_source_code.unique())
        },
        "standardization_scope": "evaluation candidates within each query", "standard_deviation_correction": 1,
        "combination": "(z(S3F) + z(EVE ensemble)) / 2", "multi_mutant_eve_scoring": "direct joint EVE score within each chain; independent sum across chains",
        "output_schema_sha256": sha256_file(MODEL_ROOT / "output_schema_candidates.json"), "combine_script_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_json({"model_id": "S3F-MSA", "dataset_id": dataset["dataset_id"], "dataset_hash": dataset["dataset_hash"], "created_at_utc": utc_now(), "shard_id": 0, "num_shards": 1, "partial_run": False, "samples": len(output), "run_config": run_config, "run_config_sha256": sha256_json(run_config)}, args.output_dir / "shards/shard_0000.json")
    print(f"samples={len(output)} contexts={len(mapping)}")


if __name__ == "__main__":
    main()
