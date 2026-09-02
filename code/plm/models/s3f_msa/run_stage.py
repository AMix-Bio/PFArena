#!/usr/bin/env python3
"""Run and validate one WT task in the EVE weights, train, or score stage."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common_io import atomic_write_csv, atomic_write_json, sha256_file, sha256_json, utc_now


SEEDS = (0, 1, 2, 3, 4)
EVE_SOURCE_SHA256 = {
    "calc_weights.py": "37ad9c040b7dc62cb7c41a4df6c156893e47473abce8d62d2718cd7a188ccaed",
    "train_VAE.py": "fb7ded93c37d873b14ebe364b6f1cfcb83537bbf0cd679b393fbb92e2bff92f5",
    "compute_evol_indices_DMS.py": "3295508423b96cb8fe672c5dd2804e9b0b80d4ac8c2e845a91a83535f4d2a45a",
    "EVE/default_model_params.json": "86d6003c6f6af948cc2d696dbfae2f6ba85af4fc10ffda20c7db57b73b338018",
    "EVE/VAE_model.py": "84457ced16db4521acff98946aeb128c6f9ba9416a24cd6a15d76ed18a1b1419",
    "utils/data_utils.py": "3366775d1683739e193ddcce5fb4308d8843adeafb9822d666df62cc2d0b72f7",
    "utils/weights.py": "caecec03ad3cacbf9e2326aa191fbce76d4b33766c42355b1fbf0f858ac27c92",
}


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["weights", "train", "score"])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--artifact-cache-dir", type=Path)
    parser.add_argument("--artifact-input-dir", type=Path)
    parser.add_argument("--eve-root", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--num-cpus", type=int, default=40)
    parser.add_argument("--num-samples", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--reuse-valid", action="store_true")
    return parser.parse_args()


def command_base(args: argparse.Namespace, row: pd.Series) -> list[str]:
    return [
        sys.executable,
        "--",
        "--MSA_data_folder", str(args.input_dir / "alignments"),
        "--DMS_reference_file_path", str(args.input_dir / "mapping.csv"),
    ]


def validate_eve_source(eve_root: Path) -> None:
    for relative, expected in EVE_SOURCE_SHA256.items():
        path = eve_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"EVE source differs from the registered reference version: {path}")


def run(command: list[str], script: Path, eve_root: Path) -> None:
    if not script.is_file():
        raise FileNotFoundError(f"EVE entry point not found: {script}")
    command[1] = str(script)
    subprocess.run(command, cwd=eve_root, check=True)


def base_metadata(args: argparse.Namespace, row: pd.Series) -> dict:
    inputs = json.loads((args.input_dir / "inputs.json").read_text())
    return {
        "schema_version": 1,
        "dataset_id": inputs["dataset_id"],
        "dataset_hash": inputs["dataset_hash"],
        "input_hash": inputs["input_hash"],
        "wt_id": row.wt_id,
        "protein_index": int(row.protein_index),
        "theta": float(row.MSA_theta),
        "focus_column_gap_threshold": 1.0,
        "sequence_gap_threshold": 0.5,
        "eve_git_commit": "144fe22b07dfaeec2b366f2346203a9838a55b4c",
        "eve_source_sha256": EVE_SOURCE_SHA256,
        "python": sys.version.split()[0],
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "torch", "scikit-learn", "numba", "numba-progress"]
        },
        "created_at_utc": utc_now(),
    }


def validate_score_frame(frame: pd.DataFrame, expected_mutants: pd.Series, wt_id: str) -> None:
    seed_raw = [f"evol_indices_seed_{seed}" for seed in SEEDS]
    seed_scores = [f"eve_score_seed_{seed}" for seed in SEEDS]
    expected_columns = ["mutant", *seed_raw, *seed_scores, "evol_indices_ensemble", "eve_ensemble"]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(f"{wt_id}: unexpected EVE score columns")
    if frame["mutant"].duplicated().any() or set(frame["mutant"]) != set(expected_mutants):
        raise ValueError(f"{wt_id}: EVE score mutant set is incomplete")
    if not np.isfinite(frame.drop(columns="mutant").to_numpy()).all():
        raise ValueError(f"{wt_id}: non-finite EVE scores")
    if not np.allclose(frame[seed_scores], -frame[seed_raw], rtol=0, atol=1e-7):
        raise ValueError(f"{wt_id}: EVE score direction is inconsistent")
    if not np.allclose(frame["evol_indices_ensemble"], frame[seed_raw].mean(axis=1), rtol=0, atol=1e-7):
        raise ValueError(f"{wt_id}: EVE ensemble is inconsistent")
    if not np.allclose(frame["eve_ensemble"], -frame["evol_indices_ensemble"], rtol=0, atol=1e-7):
        raise ValueError(f"{wt_id}: EVE ensemble direction is inconsistent")


def validate_training_log(path: Path, params: Path, wt_id: str) -> None:
    lines = path.read_text().splitlines()
    updates = [int(match.group(1)) for line in lines if (match := re.search(r"Update (\d+)\.", line))]
    training = json.loads(params.read_text())["training_parameters"]
    expected = list(range(0, training["num_training_steps"], training["log_training_freq"]))
    if updates != expected or any(re.search(r"\b(?:nan|inf)\b", line, re.IGNORECASE) for line in lines):
        raise ValueError(f"{wt_id}: incomplete or non-finite EVE training log")


def run_weights(args: argparse.Namespace, row: pd.Series) -> None:
    output = args.cache_dir / "weights" / row.weight_file_name
    metadata_path = output.with_suffix(".json")
    if output.is_file() and metadata_path.is_file() and args.reuse_valid:
        record = json.loads(metadata_path.read_text())
        weights = np.load(output)
        inputs = json.loads((args.input_dir / "inputs.json").read_text())
        if (
            record.get("dataset_id") == inputs["dataset_id"]
            and record.get("dataset_hash") == inputs["dataset_hash"]
            and record.get("input_hash") == inputs["input_hash"]
            and record.get("eve_source_sha256") == EVE_SOURCE_SHA256
            and record.get("wt_id") == row.wt_id
            and int(record.get("protein_index", -1)) == int(row.protein_index)
            and float(record.get("theta")) == float(row.MSA_theta)
            and record.get("stage") == "weights"
            and record.get("num_sequences") == len(weights)
            and record.get("weight_sha256") == sha256_file(output)
        ):
            if (
                len(weights) == int(row.msa_depth_after_eve_filter)
                and np.isfinite(weights).all()
                and (weights > 0).all()
                and np.isclose(record.get("neff", np.nan), weights.sum(), rtol=0, atol=1e-9)
            ):
                print(f"Reused validated weights: {row.wt_id}")
                return
        raise ValueError(f"existing weight artifact is not valid for {row.wt_id}")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"weight artifact already exists: {output}")
    temporary = args.cache_dir / f".weights_{row.wt_id}_{os.getpid()}"
    temporary.mkdir(parents=True)
    try:
        command = command_base(args, row) + [
            "--DMS_index", str(int(row.protein_index)),
            "--MSA_weights_location", str(temporary),
            "--num_cpus", str(args.num_cpus),
            "--calc_method", "evcouplings",
            "--threshold_focus_cols_frac_gaps", "1",
        ]
        run(command, args.eve_root / "calc_weights.py", args.eve_root)
        generated = temporary / row.weight_file_name
        weights = np.load(generated)
        if len(weights) != int(row.msa_depth_after_eve_filter):
            raise ValueError(f"{row.wt_id}: weight count differs from EVE-filtered MSA depth")
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError(f"{row.wt_id}: invalid sequence weights")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(generated, output)
        metadata = {
            **base_metadata(args, row),
            "stage": "weights",
            "num_cpus": args.num_cpus,
            "num_sequences": len(weights),
            "neff": float(weights.sum()),
            "weight_sha256": sha256_file(output),
        }
        atomic_write_json(metadata, metadata_path)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def run_train_locked(args: argparse.Namespace, row: pd.Series) -> None:
    weight = args.cache_dir / "weights" / row.weight_file_name
    weight_metadata = weight.with_suffix(".json")
    if not weight.is_file() or not weight_metadata.is_file():
        raise FileNotFoundError(f"validated weights are missing for {row.wt_id}")
    weight_record = json.loads(weight_metadata.read_text())
    weights = np.load(weight)
    inputs = json.loads((args.input_dir / "inputs.json").read_text())
    valid_weight = (
        weight_record.get("dataset_hash") == inputs["dataset_hash"]
        and weight_record.get("input_hash") == inputs["input_hash"]
        and weight_record.get("eve_source_sha256") == EVE_SOURCE_SHA256
        and weight_record.get("wt_id") == row.wt_id
        and int(weight_record.get("protein_index", -1)) == int(row.protein_index)
        and float(weight_record.get("theta")) == float(row.MSA_theta)
        and weight_record.get("stage") == "weights"
        and weight_record.get("num_sequences") == len(weights)
        and weight_record.get("weight_sha256") == sha256_file(weight)
        and len(weights) == int(row.msa_depth_after_eve_filter)
        and np.isfinite(weights).all()
        and (weights > 0).all()
        and np.isclose(weight_record.get("neff", np.nan), weights.sum(), rtol=0, atol=1e-9)
    )
    if not valid_weight:
        raise ValueError(f"{row.wt_id}: weight artifact is invalid")
    output = args.cache_dir / "checkpoints" / f"{row.wt_id}_seed_{args.seed}"
    metadata_path = output.with_suffix(".json")
    params = args.eve_root / "EVE/default_model_params.json"
    if output.is_file() and metadata_path.is_file() and args.reuse_valid:
        record = json.loads(metadata_path.read_text())
        inputs = json.loads((args.input_dir / "inputs.json").read_text())
        log_path = resolve_recorded_path(record.get("training_log", ""))
        if (
            record.get("stage") == "train"
            and record.get("dataset_hash") == inputs["dataset_hash"]
            and record.get("input_hash") == inputs["input_hash"]
            and record.get("eve_source_sha256") == EVE_SOURCE_SHA256
            and record.get("wt_id") == row.wt_id
            and int(record.get("protein_index", -1)) == int(row.protein_index)
            and record.get("seed") == args.seed
            and record.get("model_parameters_sha256") == sha256_file(params)
            and record.get("weight_sha256") == sha256_file(weight)
        ):
            if (
                record["checkpoint_sha256"] == sha256_file(output)
                and log_path.is_file()
                and record["training_log_sha256"] == sha256_file(log_path)
            ):
                validate_training_log(log_path, params, row.wt_id)
                print(f"Reused validated checkpoint: {row.wt_id} seed={args.seed}")
                return
        raise ValueError(f"existing checkpoint artifact is not valid for {row.wt_id} seed={args.seed}")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"checkpoint artifact already exists: {output}")

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".train_{row.wt_id}_{args.seed}_", dir=args.cache_dir
        )
    )
    temporary.chmod(0o777)
    checkpoint_dir = temporary / "checkpoints"
    log_dir = temporary / "logs"
    checkpoint_dir.mkdir(parents=True)
    log_dir.mkdir()
    try:
        command = command_base(args, row) + [
            "--protein_index", str(int(row.protein_index)),
            "--MSA_weights_location", str(args.cache_dir / "weights"),
            "--VAE_checkpoint_location", str(checkpoint_dir),
            "--model_parameters_location", str(params),
            "--training_logs_location", str(log_dir),
            "--threshold_focus_cols_frac_gaps", "1",
            "--seed", str(args.seed),
            "--experimental_stream_data",
            "--force_load_weights",
        ]
        run(command, args.eve_root / "train_VAE.py", args.eve_root)
        generated = checkpoint_dir / f"{row.wt_id}_seed_{args.seed}"
        if not generated.is_file() or generated.stat().st_size == 0:
            raise ValueError(f"{row.wt_id}: EVE did not produce a checkpoint")
        generated_log = log_dir / f"{row.wt_id}_seed_{args.seed}_losses.csv"
        log_output = args.cache_dir / "training_logs" / generated_log.name
        if not generated_log.is_file():
            raise ValueError(f"{row.wt_id}: EVE did not produce a training log")
        validate_training_log(generated_log, params, row.wt_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        log_output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(generated, output)
        os.replace(generated_log, log_output)
        metadata = {
            **base_metadata(args, row),
            "stage": "train",
            "seed": args.seed,
            "model_parameters_sha256": sha256_file(params),
            "weight_sha256": sha256_file(weight),
            "checkpoint_sha256": sha256_file(output),
            "training_log": str(log_output.resolve()),
            "training_log_sha256": sha256_file(log_output),
        }
        atomic_write_json(metadata, metadata_path)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def run_train(args: argparse.Namespace, row: pd.Series) -> None:
    if args.seed is None:
        raise ValueError("train stage requires --seed")
    lock_path = args.cache_dir / ".locks" / f"train_{row.wt_id}_{args.seed}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"training is already running for {row.wt_id} seed={args.seed}"
            ) from error
        run_train_locked(args, row)


def run_locked(args: argparse.Namespace, row: pd.Series, stage: str) -> None:
    lock_path = args.cache_dir / ".locks" / f"{stage}_{row.wt_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"{stage} is already running for {row.wt_id}") from error
        if stage == "weights":
            run_weights(args, row)
        else:
            run_score(args, row)


def run_score(args: argparse.Namespace, row: pd.Series) -> None:
    artifact_cache = args.artifact_cache_dir or args.cache_dir
    artifact_inputs = args.artifact_input_dir or args.input_dir
    artifact_input_hash = json.loads((artifact_inputs / "inputs.json").read_text())["input_hash"]
    output = args.cache_dir / "eve_scores" / f"{row.wt_id}.csv"
    metadata_path = output.with_suffix(".json")
    expected_mutants = pd.read_csv(args.input_dir / "mutations" / row.DMS_filename)["mutant"]
    if output.is_file() and metadata_path.is_file() and args.reuse_valid:
        record = json.loads(metadata_path.read_text())
        inputs = json.loads((args.input_dir / "inputs.json").read_text())
        recorded_cache = resolve_recorded_path(
            record.get("artifact_cache_dir", str(args.cache_dir.resolve()))
        ).resolve()
        valid_config = (
            record.get("release_input_hash", record["input_hash"])
            == inputs["input_hash"]
            and record.get("eve_source_sha256") == EVE_SOURCE_SHA256
            and record["random_seeds"] == list(SEEDS)
            and record["num_samples_compute_evol_indices"] == args.num_samples
            and record["batch_size"] == args.batch_size
            and record["aggregation_method"] == "full"
            and record.get(
                "release_input_hash",
                record.get("artifact_input_hash", record["input_hash"]),
            ) == artifact_input_hash
            and recorded_cache == artifact_cache.resolve()
        )
        checkpoints_valid = all(
            (artifact_cache / "checkpoints" / f"{row.wt_id}_seed_{seed}").is_file()
            and record["checkpoint_sha256"][str(seed)]
            == sha256_file(artifact_cache / "checkpoints" / f"{row.wt_id}_seed_{seed}")
            for seed in SEEDS
        )
        if valid_config and checkpoints_valid and record["score_sha256"] == sha256_file(output):
            validate_score_frame(pd.read_csv(output), expected_mutants, row.wt_id)
            print(f"Reused validated EVE scores: {row.wt_id}")
            return
        raise ValueError(f"existing score artifact is not valid for {row.wt_id}")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"score artifact already exists: {output}")
    params = args.eve_root / "EVE/default_model_params.json"
    weight = artifact_cache / "weights" / row.weight_file_name
    if not weight.is_file():
        raise FileNotFoundError(f"validated weights are missing: {weight}")
    checkpoint_hashes: dict[str, str] = {}
    for seed in SEEDS:
        checkpoint = artifact_cache / "checkpoints" / f"{row.wt_id}_seed_{seed}"
        metadata = checkpoint.with_suffix(".json")
        if not checkpoint.is_file() or not metadata.is_file():
            raise FileNotFoundError(f"validated checkpoint is missing: {checkpoint}")
        checkpoint_record = json.loads(metadata.read_text())
        checkpoint_sha256 = sha256_file(checkpoint)
        if (
            checkpoint_record.get("stage") != "train"
            or checkpoint_record.get(
                "release_input_hash", checkpoint_record.get("input_hash")
            ) != artifact_input_hash
            or checkpoint_record.get("wt_id") != row.wt_id
            or checkpoint_record.get("seed") != seed
        ):
            raise ValueError(f"{row.wt_id}: checkpoint input identity mismatch")
        if checkpoint_record.get("eve_source_sha256") != EVE_SOURCE_SHA256:
            raise ValueError(f"{row.wt_id}: checkpoint provenance mismatch")
        if checkpoint_record.get("model_parameters_sha256") != sha256_file(params):
            raise ValueError(f"{row.wt_id}: checkpoint model parameters mismatch")
        if checkpoint_record.get("weight_sha256") != sha256_file(weight):
            raise ValueError(f"{row.wt_id}: checkpoint weight identity mismatch")
        if checkpoint_record["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError(f"{row.wt_id}: checkpoint SHA256 mismatch")
        training_log = resolve_recorded_path(checkpoint_record.get("training_log", ""))
        if (
            not training_log.is_file()
            or checkpoint_record.get("training_log_sha256") != sha256_file(training_log)
        ):
            raise ValueError(f"{row.wt_id}: checkpoint training log mismatch")
        validate_training_log(training_log, params, row.wt_id)
        checkpoint_hashes[str(seed)] = checkpoint_sha256

    temporary = args.cache_dir / f".score_{row.wt_id}_{os.getpid()}"
    temporary.mkdir(parents=True)
    try:
        command = command_base(args, row) + [
            "--protein_index", str(int(row.protein_index)),
            "--MSA_weights_location", str(artifact_cache / "weights"),
            "--random_seeds", *map(str, SEEDS),
            "--VAE_checkpoint_location", str(artifact_cache / "checkpoints"),
            "--model_parameters_location", str(params),
            "--DMS_data_folder", str(args.input_dir / "mutations"),
            "--output_scores_folder", str(temporary),
            "--num_samples_compute_evol_indices", str(args.num_samples),
            "--batch_size", str(args.batch_size),
            "--aggregation_method", "full",
            "--threshold_focus_cols_frac_gaps", "1",
        ]
        run(command, args.eve_root / "compute_evol_indices_DMS.py", args.eve_root)
        raw = pd.read_csv(temporary / f"{row.wt_id}.csv")
        seed_raw = [f"evol_indices_seed_{seed}" for seed in SEEDS]
        if raw.columns.tolist() != ["mutant", *seed_raw]:
            raise ValueError(f"{row.wt_id}: unexpected EVE score columns")
        raw = raw.loc[~raw["mutant"].eq("wt")].copy()
        for seed in SEEDS:
            raw[f"eve_score_seed_{seed}"] = -raw[f"evol_indices_seed_{seed}"]
        raw["evol_indices_ensemble"] = raw[seed_raw].mean(axis=1)
        raw["eve_ensemble"] = -raw["evol_indices_ensemble"]
        raw = raw.sort_values("mutant").reset_index(drop=True)
        validate_score_frame(raw, expected_mutants, row.wt_id)
        atomic_write_csv(raw, output)
        metadata = {
            **base_metadata(args, row),
            "stage": "score",
            "random_seeds": list(SEEDS),
            "num_samples_compute_evol_indices": args.num_samples,
            "batch_size": args.batch_size,
            "aggregation_method": "full",
            "checkpoint_sha256": checkpoint_hashes,
            "artifact_input_hash": artifact_input_hash,
            "artifact_cache_dir": str(artifact_cache.resolve()),
            "scores": len(raw),
            "score_sha256": sha256_file(output),
        }
        metadata["score_config_sha256"] = sha256_json({
            key: metadata[key] for key in ["random_seeds", "num_samples_compute_evol_indices", "batch_size", "aggregation_method"]
        })
        atomic_write_json(metadata, metadata_path)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    if args.artifact_cache_dir:
        args.artifact_cache_dir = args.artifact_cache_dir.resolve()
    if args.artifact_input_dir:
        args.artifact_input_dir = args.artifact_input_dir.resolve()
    args.eve_root = args.eve_root.resolve()
    validate_eve_source(args.eve_root)
    inputs = json.loads((args.input_dir / "inputs.json").read_text())
    mapping_path = args.input_dir / "mapping.csv"
    if inputs.get("mapping_sha256") != sha256_file(mapping_path):
        raise ValueError("mapping.csv differs from frozen input metadata")
    mapping = pd.read_csv(mapping_path)
    if args.index < 0 or args.index >= len(mapping):
        raise ValueError(f"index must be in [0, {len(mapping) - 1}]")
    row = mapping.iloc[args.index]
    if int(row.protein_index) != args.index:
        raise ValueError("mapping protein_index is not contiguous")
    alignment = args.input_dir / "alignments" / row.MSA_filename
    if not alignment.is_file() or sha256_file(alignment) != row.a2m_sha256:
        raise ValueError(f"{row.wt_id}: alignment differs from frozen mapping")
    if "mutation_sha256" in mapping.columns:
        mutation = args.input_dir / "mutations" / row.DMS_filename
        if not mutation.is_file() or sha256_file(mutation) != row.mutation_sha256:
            raise ValueError(f"{row.wt_id}: mutation input differs from frozen mapping")
    if args.stage == "weights":
        run_locked(args, row, "weights")
    elif args.stage == "train":
        run_train(args, row)
    else:
        run_locked(args, row, "score")
    print(f"Completed {args.stage}: index={args.index} wt_id={row.wt_id}")


if __name__ == "__main__":
    main()
