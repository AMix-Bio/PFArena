#!/usr/bin/env python3
"""Precompute ESM2 masked-position probabilities and single-mutant scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common_io import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    sha256_files,
    sha256_json,
    utc_now,
)


DEFAULT_PROENV_ROOT = PROJECT_ROOT / "ProEnv"
MODEL_METADATA = json.loads((SCRIPT_DIR / "model.json").read_text())
LONG_SEQUENCE_PROTOCOLS = ("centered-special", "proteingym")
CACHE_FORMAT_VERSION = MODEL_METADATA["cache_format_version"]


def validate_model_dir(model_dir: Path) -> dict[str, object]:
    config_sha256 = sha256_file(model_dir / "config.json")
    if config_sha256 != MODEL_METADATA["config_sha256"]:
        raise ValueError("model config does not match the registered checkpoint")

    tokenizer_files = [
        model_dir / filename for filename in MODEL_METADATA["tokenizer_files"]
    ]
    tokenizer_sha256 = sha256_files(tokenizer_files)
    if tokenizer_sha256 != MODEL_METADATA["tokenizer_sha256"]:
        raise ValueError("model tokenizer does not match the registered checkpoint")

    weights = model_dir / MODEL_METADATA["weights_file"]
    if weights.stat().st_size != MODEL_METADATA["weights_size_bytes"]:
        raise ValueError("model weights size does not match the registered checkpoint")
    weights_sha256 = sha256_file(weights)
    if weights_sha256 != MODEL_METADATA["weights_sha256"]:
        raise ValueError("model weights do not match the registered checkpoint")
    return {
        "config_sha256": config_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "weights_sha256": weights_sha256,
        "weights_size_bytes": weights.stat().st_size,
    }


def validate_dataset(
    dataset_dir: Path, dataset: dict[str, object], proteins: pd.DataFrame
) -> None:
    required_columns = {
        "assay",
        "wt_id",
        "sequence_sha256",
        "wildtype_sequence",
        "sequence_length",
    }
    if dataset.get("schema_version") != 2:
        raise ValueError("dataset schema_version must be 2")
    missing_columns = required_columns - set(proteins.columns)
    if missing_columns:
        raise ValueError(f"proteins.csv missing columns: {sorted(missing_columns)}")
    if proteins["assay"].duplicated().any():
        raise ValueError("proteins.csv contains duplicate assays")

    paths = [
        dataset_dir / "proteins.csv",
        dataset_dir / "deferred_samples.csv",
        dataset_dir / "deferred_assays.csv",
    ]
    paths.extend(
        dataset_dir / "mutations" / f"{assay}.csv" for assay in proteins["assay"]
    )
    actual_hash = sha256_files(paths)
    if actual_hash != dataset["dataset_hash"]:
        raise ValueError("dataset files do not match dataset.json")
    if len(proteins) != dataset["ready_assays"]:
        raise ValueError("protein assay count does not match dataset.json")
    if int(proteins["expected_samples"].sum()) != dataset["ready_samples"]:
        raise ValueError("protein sample count does not match dataset.json")
    if proteins["wt_id"].nunique() != dataset["unique_wildtype_sequences"]:
        raise ValueError("unique WT count does not match dataset.json")

    sequence_hashes: dict[str, str] = {}
    for protein in proteins.itertuples(index=False):
        sequence = protein.wildtype_sequence
        sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
        if protein.wt_id != sequence_sha256[:16]:
            raise ValueError(f"{protein.assay}: wt_id does not match the WT sequence")
        if int(protein.sequence_length) != len(sequence):
            raise ValueError(f"{protein.assay}: sequence length differs from metadata")
        if protein.sequence_sha256 != sequence_sha256:
            raise ValueError(f"{protein.assay}: sequence_sha256 does not match the WT")
        previous = sequence_hashes.setdefault(protein.wt_id, sequence_sha256)
        if previous != sequence_sha256:
            raise ValueError(f"{protein.wt_id}: truncated WT hash collision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proenv-root", type=Path, default=DEFAULT_PROENV_ROOT)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=SCRIPT_DIR / "cache")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-window", type=int, default=1022)
    parser.add_argument(
        "--long-sequence-protocol",
        choices=LONG_SEQUENCE_PROTOCOLS,
        default=MODEL_METADATA["default_long_sequence_protocol"],
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def atomic_save_cache(
    path: Path,
    cache_config_sha256: str,
    sequence_sha256: str,
    sequence_length: int,
    positions: np.ndarray,
    tokens: np.ndarray,
    token_ids: np.ndarray,
    log_probs: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cache_format_version=np.int32(CACHE_FORMAT_VERSION),
            cache_config_sha256=np.asarray(cache_config_sha256),
            sequence_sha256=np.asarray(sequence_sha256),
            sequence_length=np.int32(sequence_length),
            positions=positions,
            tokens=tokens,
            token_ids=token_ids,
            log_probs=log_probs,
        )
    os.replace(temporary, path)


def centered_residue_window(
    sequence: str, position: int, max_window: int
) -> tuple[str, int]:
    if len(sequence) <= max_window:
        return sequence, position
    start = max(1, position - max_window // 2)
    end = min(len(sequence), start + max_window - 1)
    if end - start + 1 < max_window:
        start = end - max_window + 1
    return sequence[start - 1 : end], position - start + 1


def masked_inputs(
    scorer,
    sequence: str,
    positions: list[int],
    max_window: int,
    long_sequence_protocol: str,
) -> list[tuple[int, object, object, int]]:
    tokenizer = scorer.tokenizer
    max_tokens = max_window + 2
    full = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    full_ids = full["input_ids"][0]
    full_mask = full["attention_mask"][0]
    if len(full_ids) != len(sequence) + 2:
        raise ValueError("tokenizer did not produce one token per residue plus BOS/EOS")
    prepared = []

    for position in positions:
        if not 1 <= position <= len(sequence):
            raise ValueError(f"position {position} is outside the wild-type sequence")
        if len(sequence) <= max_window:
            input_ids = full_ids.clone()
            attention_mask = full_mask.clone()
            token_index = position
        elif long_sequence_protocol == "centered-special":
            window, relative_position = centered_residue_window(
                sequence, position, max_window
            )
            encoded = tokenizer(window, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"][0]
            attention_mask = encoded["attention_mask"][0]
            token_index = relative_position
        else:
            target_index = position
            half_window = max_tokens // 2
            if target_index < half_window:
                start = 0
            elif target_index >= len(full_ids) - half_window:
                start = len(full_ids) - max_tokens
            else:
                start = target_index - half_window
            input_ids = full_ids[start : start + max_tokens].clone()
            attention_mask = full_mask[start : start + max_tokens].clone()
            token_index = target_index - start

        expected_token = tokenizer.convert_tokens_to_ids(sequence[position - 1])
        if int(input_ids[token_index]) != expected_token:
            raise ValueError(f"token mismatch at position {position}")
        input_ids[token_index] = scorer.mask_token_id
        prepared.append((position, input_ids, attention_mask, token_index))
    return prepared


def precompute_full_log_probs(
    scorer,
    sequence: str,
    positions: list[int],
    max_window: int,
    long_sequence_protocol: str,
) -> np.ndarray:
    import torch

    prepared = masked_inputs(
        scorer, sequence, positions, max_window, long_sequence_protocol
    )
    rows = []
    for offset in range(0, len(prepared), scorer.batch_size):
        batch = prepared[offset : offset + scorer.batch_size]
        input_ids = torch.stack([item[1] for item in batch]).to(scorer.device)
        attention_mask = torch.stack([item[2] for item in batch]).to(scorer.device)
        with torch.inference_mode():
            logits = scorer.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        for row, (_, _, _, token_index) in enumerate(batch):
            rows.append(
                torch.log_softmax(logits[row, token_index], dim=-1)
                .float()
                .cpu()
                .numpy()
            )
    return np.asarray(rows, dtype=np.float32)


def position_log_probs(
    scorer,
    wt_sequence: str,
    sequence_sha256: str,
    required_positions: list[int],
    cache_path: Path,
    cache_config_sha256: str,
    max_window: int,
    long_sequence_protocol: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    tokens = [
        scorer.tokenizer.convert_ids_to_tokens(token_id)
        for token_id in range(len(scorer.tokenizer))
    ]
    if len(tokens) != scorer.model.config.vocab_size:
        raise ValueError("tokenizer and model vocabulary sizes differ")
    token_ids = np.arange(len(tokens), dtype=np.int32)
    positions = np.empty(0, dtype=np.int32)
    log_probs = np.empty((0, len(tokens)), dtype=np.float32)
    if cache_path.exists():
        with np.load(cache_path) as cache:
            if int(cache["cache_format_version"].item()) != CACHE_FORMAT_VERSION:
                raise ValueError(f"{cache_path}: unsupported cache format")
            if str(cache["cache_config_sha256"].item()) != cache_config_sha256:
                raise ValueError(f"{cache_path}: inference configuration differs")
            if str(cache["sequence_sha256"].item()) != sequence_sha256:
                raise ValueError(f"{cache_path}: WT sequence hash differs")
            if int(cache["sequence_length"].item()) != len(wt_sequence):
                raise ValueError(f"{cache_path}: WT sequence length differs")
            cached_tokens = cache["tokens"].astype(str).tolist()
            cached_token_ids = cache["token_ids"].astype(np.int32)
            if cached_tokens != tokens or not np.array_equal(
                cached_token_ids, token_ids
            ):
                raise ValueError(f"{cache_path}: tokenizer vocabulary differs")
            positions = cache["positions"].astype(np.int32)
            log_probs = cache["log_probs"].astype(np.float32)

        if positions.ndim != 1 or log_probs.shape != (len(positions), len(tokens)):
            raise ValueError(f"{cache_path}: invalid cache shape")
        if len(np.unique(positions)) != len(positions):
            raise ValueError(f"{cache_path}: duplicate cached positions")
        if len(positions) and not (
            positions.min() >= 1 and positions.max() <= len(wt_sequence)
        ):
            raise ValueError(f"{cache_path}: cached position is outside the WT")
        if not np.isfinite(log_probs).all():
            raise ValueError(f"{cache_path}: non-finite cached log probabilities")

    missing = sorted(set(required_positions) - set(positions.tolist()))
    cache_stats = {
        "required": len(set(required_positions)),
        "reused": len(set(required_positions) & set(positions.tolist())),
        "computed": len(missing),
    }
    if not missing:
        return positions, log_probs, tokens, cache_stats

    new_log_probs = precompute_full_log_probs(
        scorer,
        wt_sequence,
        missing,
        max_window,
        long_sequence_protocol,
    )
    if new_log_probs.shape != (len(missing), len(tokens)):
        raise ValueError("model returned an unexpected log-probability shape")
    if not np.isfinite(new_log_probs).all():
        raise ValueError("model returned non-finite log probabilities")
    positions = np.concatenate([positions, np.asarray(missing, dtype=np.int32)])
    log_probs = np.concatenate([log_probs, new_log_probs])
    order = np.argsort(positions)
    positions = positions[order]
    log_probs = log_probs[order]
    atomic_save_cache(
        cache_path,
        cache_config_sha256,
        sequence_sha256,
        len(wt_sequence),
        positions,
        np.asarray(tokens),
        token_ids,
        log_probs,
    )
    return positions, log_probs, tokens, cache_stats


def score_mutations(
    mutations: pd.DataFrame,
    positions: np.ndarray,
    log_probs: np.ndarray,
    tokens: list[str],
) -> pd.DataFrame:
    position_index = {int(position): i for i, position in enumerate(positions)}
    token_index = {token: i for i, token in enumerate(tokens)}
    rows = mutations["position"].map(position_index)
    wt_columns = mutations["wt_aa"].map(token_index)
    mutant_columns = mutations["mut_aa"].map(token_index)
    if rows.isna().any() or wt_columns.isna().any() or mutant_columns.isna().any():
        raise ValueError("mutation references a missing position or tokenizer residue")
    rows = rows.to_numpy(dtype=np.int64)
    wt_columns = wt_columns.to_numpy(dtype=np.int64)
    mutant_columns = mutant_columns.to_numpy(dtype=np.int64)
    wt_logp = log_probs[rows, wt_columns]
    mutant_logp = log_probs[rows, mutant_columns]

    output = mutations[["sample_id", "assay", "mutant"]].copy()
    output["status"] = "ok"
    output["masked_marginal"] = mutant_logp - wt_logp
    output["mutant_logp"] = mutant_logp
    output["wt_logp"] = wt_logp
    return output


def balanced_wt_shards(
    proteins: pd.DataFrame, dataset_dir: Path, max_window: int, num_shards: int
) -> tuple[list[list[str]], list[int]]:
    work_items = []
    for wt_id, group in proteins.groupby("wt_id", sort=True):
        positions: set[int] = set()
        for assay in group["assay"]:
            assay_positions = pd.read_csv(
                dataset_dir / "mutations" / f"{assay}.csv", usecols=["position"]
            )["position"]
            positions.update(assay_positions.astype(int))
        sequence_length = int(group["sequence_length"].iloc[0])
        cost = len(positions) * min(sequence_length, max_window)
        work_items.append((cost, wt_id))

    shards: list[list[str]] = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for cost, wt_id in sorted(work_items, key=lambda item: (-item[0], item[1])):
        shard_id = min(range(num_shards), key=lambda i: (loads[i], i))
        shards[shard_id].append(wt_id)
        loads[shard_id] += cost
    return shards, loads


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not 1 <= args.max_window <= MODEL_METADATA["default_max_residues"]:
        raise ValueError(
            f"max-window must be in 1..{MODEL_METADATA['default_max_residues']}"
        )

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.proenv_root))

    import safetensors
    import torch
    import transformers
    from proenv.metrics.sequence.esm2_de_fitness import ESM2FitnessScorer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dataset = json.loads((args.dataset_dir / "dataset.json").read_text())
    model_validation = validate_model_dir(args.model_dir)
    proteins = pd.read_csv(args.dataset_dir / "proteins.csv")
    validate_dataset(args.dataset_dir, dataset, proteins)
    if args.num_shards > proteins["wt_id"].nunique():
        raise ValueError("num-shards exceeds the number of unique WT sequences")
    shards, estimated_loads = balanced_wt_shards(
        proteins, args.dataset_dir, args.max_window, args.num_shards
    )
    selected_wt_ids = shards[args.shard_id]
    started_at = utc_now()

    scorer = ESM2FitnessScorer(
        str(args.model_dir),
        device=args.device,
        batch_size=args.batch_size,
        max_window=args.max_window,
    )
    model_name_or_path = scorer.model.config.name_or_path
    model_dtype = str(next(scorer.model.parameters()).dtype)
    if model_dtype != "torch.float32":
        raise ValueError(f"expected FP32 model weights, found {model_dtype}")

    scorer_path = (
        args.proenv_root / "proenv" / "metrics" / "sequence" / "esm2_de_fitness.py"
    )
    output_schema_path = SCRIPT_DIR / "output_schema.json"
    gpu = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if scorer.device == "cuda"
        else None
    )
    cache_config = {
        "model_id": MODEL_METADATA["model_id"],
        "checkpoint": MODEL_METADATA["checkpoint"],
        "checkpoint_revision": MODEL_METADATA["checkpoint_revision"],
        "model_config_sha256": model_validation["config_sha256"],
        "model_weights_sha256": model_validation["weights_sha256"],
        "model_weights_size_bytes": model_validation["weights_size_bytes"],
        "tokenizer_sha256": model_validation["tokenizer_sha256"],
        "proenv_scorer_sha256": sha256_file(scorer_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "device": scorer.device,
        "model_dtype": model_dtype,
        "gpu": gpu,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "batch_size": args.batch_size,
        "max_window": args.max_window,
        "scoring_strategy": MODEL_METADATA["scoring_strategy"],
        "long_sequence_protocol": args.long_sequence_protocol,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "cache_namespace": MODEL_METADATA["cache_namespace"],
    }
    cache_config_sha256 = sha256_json(cache_config)
    cache_root = (
        args.cache_dir
        / MODEL_METADATA["cache_namespace"]
        / cache_config_sha256[:16]
        / args.long_sequence_protocol
        / f"max_tokens_{args.max_window + 2}"
    )
    completed_assays = 0
    completed_samples = 0
    required_positions_total = 0
    computed_positions_total = 0
    reused_positions_total = 0
    try:
        for wt_id in selected_wt_ids:
            protein_rows = proteins[proteins["wt_id"] == wt_id]
            sequences = protein_rows["wildtype_sequence"].drop_duplicates()
            if len(sequences) != 1:
                raise ValueError(f"{wt_id}: inconsistent model sequences")
            sequence = sequences.iloc[0]
            sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()

            mutation_frames = [
                pd.read_csv(args.dataset_dir / "mutations" / f"{assay}.csv")
                for assay in protein_rows["assay"]
            ]
            mutations = pd.concat(mutation_frames, ignore_index=True)
            required_positions = sorted(mutations["position"].astype(int).unique())
            cache_path = cache_root / f"{wt_id}.npz"
            positions, log_probs, tokens, cache_stats = position_log_probs(
                scorer,
                sequence,
                sequence_sha256,
                required_positions,
                cache_path,
                cache_config_sha256,
                args.max_window,
                args.long_sequence_protocol,
            )
            required_positions_total += cache_stats["required"]
            computed_positions_total += cache_stats["computed"]
            reused_positions_total += cache_stats["reused"]
            print(
                f"{wt_id}: positions={cache_stats['required']} "
                f"computed={cache_stats['computed']} reused={cache_stats['reused']}",
                flush=True,
            )
            scored = score_mutations(mutations, positions, log_probs, tokens)

            for assay, assay_scores in scored.groupby("assay", sort=True):
                atomic_write_csv(
                    assay_scores.sort_values("sample_id"),
                    args.output_dir / "assays" / f"{assay}.csv",
                )
                completed_assays += 1
                completed_samples += len(assay_scores)
                print(f"{assay}: {len(assay_scores)}", flush=True)
    finally:
        scorer.teardown()

    run_config = {
        **cache_config,
        "cache_config_sha256": cache_config_sha256,
        "output_schema_sha256": sha256_file(output_schema_path),
    }
    shard_metadata = {
        "model_id": MODEL_METADATA["model_id"],
        "run_config": run_config,
        "run_config_sha256": sha256_json(run_config),
        "dataset_id": dataset["dataset_id"],
        "dataset_hash": dataset["dataset_hash"],
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "wt_sequences": len(selected_wt_ids),
        "estimated_position_token_work": estimated_loads[args.shard_id],
        "assays": completed_assays,
        "samples": completed_samples,
        "model_dir": str(args.model_dir.resolve()),
        "model_name_or_path": model_name_or_path,
        "proenv_root": str(args.proenv_root.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "cache_root": str(cache_root.resolve()),
        "required_positions": required_positions_total,
        "computed_positions": computed_positions_total,
        "reused_positions": reused_positions_total,
        "partial_run": False,
    }
    atomic_write_json(
        shard_metadata,
        args.output_dir / "shards" / f"shard_{args.shard_id:04d}.json",
    )
    print(f"Shard {args.shard_id}: {completed_samples} samples")


if __name__ == "__main__":
    main()
