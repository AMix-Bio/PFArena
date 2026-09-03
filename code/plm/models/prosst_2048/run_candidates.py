#!/usr/bin/env python3
"""Precompute ProSST-2048 outputs for a frozen canonical model dataset."""

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

from model_dataset_io import load_model_dataset
from common_io import atomic_write_csv, atomic_write_json, sha256_file, sha256_json, utc_now
from core import (
    DEFAULT_ADAPTER_ROOT,
    MODEL_METADATA,
    validate_model_dir,
    validate_prosst_repo,
)


OUTPUT_SCHEMA = SCRIPT_DIR / "output_schema_candidates.json"
CACHE_FORMAT_VERSION = 1


def resolve_structure_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    manifest_relative = (manifest_path.resolve().parent / path).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prosst-repo-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--structure-batch-size", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--context-sha256", action="append")
    return parser.parse_args()


def build_components(
    proteins: pd.DataFrame,
    samples: pd.DataFrame,
    substitutions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    chain_lookup = {}
    chain_sequences = {}
    for protein in proteins.itertuples(index=False):
        for chain_id, sequence in enumerate(json.loads(protein.chain_sequences), start=1):
            context_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
            chain_lookup[(protein.sequence_sha256, chain_id)] = context_sha256
            if chain_sequences.setdefault(context_sha256, sequence) != sequence:
                raise ValueError("chain sequence SHA256 collision")

    identity = samples[
        ["sample_id", "subset", "candidate_group_id", "source_assay", "mutant", "sequence_sha256"]
    ]
    components = substitutions.merge(identity, on="sample_id", validate="many_to_one")
    components["context_sha256"] = [
        chain_lookup[(sequence_sha256, int(chain_id))]
        for sequence_sha256, chain_id in zip(
            components["sequence_sha256"], components["chain_id"]
        )
    ]
    return components, chain_sequences


def validate_structures(
    manifest: pd.DataFrame,
    dataset: dict,
    chain_sequences: dict[str, str],
) -> pd.DataFrame:
    required = {
        "dataset_hash", "context_sha256", "sequence_length", "structure_status",
        "pdb_path", "pdb_sha256",
    }
    if required - set(manifest.columns):
        raise ValueError("chain structure manifest columns are incomplete")
    if manifest["context_sha256"].duplicated().any():
        raise ValueError("chain structure manifest contains duplicate contexts")
    if set(manifest["dataset_hash"]) != {dataset["dataset_hash"]}:
        raise ValueError("chain structure manifest targets a different dataset")
    if not manifest["structure_status"].eq("ready").all():
        raise ValueError("chain structure manifest contains unavailable structures")
    if set(manifest["context_sha256"]) != set(chain_sequences):
        raise ValueError("chain structure manifest context set differs from the dataset")
    lengths = manifest.set_index("context_sha256")["sequence_length"].astype(int)
    if any(lengths[context] != len(sequence) for context, sequence in chain_sequences.items()):
        raise ValueError("chain structure manifest sequence length differs")
    return manifest.set_index("context_sha256")


def inference_window(length: int, positions: list[int], max_residues: int) -> tuple[int, int]:
    if length <= max_residues:
        return 1, length
    if positions[-1] - positions[0] + 1 > max_residues:
        raise ValueError("required positions do not fit in one ProSST input window")
    midpoint = (positions[0] + positions[-1]) // 2
    start = max(1, midpoint - (max_residues - 1) // 2)
    end = start + max_residues - 1
    if end > length:
        end = length
        start = end - max_residues + 1
    if start > positions[0] or end < positions[-1]:
        raise ValueError("ProSST window does not cover every required position")
    return start, end


def save_cache(
    path: Path,
    cache_config_sha256: str,
    context_sha256: str,
    pdb_sha256: str,
    positions: np.ndarray,
    structure_tokens: np.ndarray,
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
            context_sha256=np.asarray(context_sha256),
            pdb_sha256=np.asarray(pdb_sha256),
            sequence_length=np.int32(len(structure_tokens)),
            positions=positions,
            structure_tokens=structure_tokens,
            tokens=tokens,
            token_ids=token_ids,
            log_probs=log_probs,
        )
    os.replace(temporary, path)


def context_outputs(
    scorer,
    sequence: str,
    context_sha256: str,
    required_positions: list[int],
    pdb_path: Path,
    pdb_sha256: str,
    cache_path: Path,
    cache_config_sha256: str,
    max_residues: int,
) -> tuple[np.ndarray, np.ndarray, list[str], bool]:
    start, end = inference_window(len(sequence), required_positions, max_residues)
    expected_positions = np.arange(start, end + 1, dtype=np.int32)
    tokens = [
        scorer.tokenizer.convert_ids_to_tokens(token_id)
        for token_id in range(len(scorer.tokenizer))
    ]
    token_ids = np.arange(len(tokens), dtype=np.int32)

    if cache_path.exists():
        with np.load(cache_path) as cache:
            if int(cache["cache_format_version"].item()) != CACHE_FORMAT_VERSION:
                raise ValueError(f"{cache_path}: unsupported cache format")
            if str(cache["cache_config_sha256"].item()) != cache_config_sha256:
                raise ValueError(f"{cache_path}: inference configuration differs")
            if str(cache["context_sha256"].item()) != context_sha256:
                raise ValueError(f"{cache_path}: chain context differs")
            if str(cache["pdb_sha256"].item()) != pdb_sha256:
                raise ValueError(f"{cache_path}: structure differs")
            positions = cache["positions"].astype(np.int32)
            structure_tokens = cache["structure_tokens"].astype(np.int32)
            sequence_length = int(cache["sequence_length"].item())
            cached_tokens = cache["tokens"].astype(str).tolist()
            cached_token_ids = cache["token_ids"].astype(np.int32)
            log_probs = cache["log_probs"].astype(np.float32)
        if sequence_length != len(sequence):
            raise ValueError(f"{cache_path}: sequence length differs")
        if not np.array_equal(positions, expected_positions):
            raise ValueError(f"{cache_path}: inference window differs")
        if structure_tokens.shape != (len(sequence),):
            raise ValueError(f"{cache_path}: structure-token shape differs")
        if cached_tokens != tokens or not np.array_equal(cached_token_ids, token_ids):
            raise ValueError(f"{cache_path}: tokenizer vocabulary differs")
        reused = True
    else:
        pdb_sequence, structure_token_list = scorer._structure_tokens_from_pdb(pdb_path)
        if pdb_sequence != sequence:
            raise ValueError(f"{pdb_path}: PDB-derived sequence differs from the chain")
        structure_tokens = np.asarray(structure_token_list, dtype=np.int32)
        if structure_tokens.shape != (len(sequence),):
            raise ValueError(f"{pdb_path}: structure-token shape differs")
        window_sequence = sequence[start - 1 : end]
        window_structure = structure_tokens[start - 1 : end]
        encoded = scorer.tokenizer(window_sequence, add_special_tokens=True)["input_ids"]
        if len(encoded) != len(window_sequence) + 2:
            raise ValueError("tokenizer did not produce one token per residue plus BOS/EOS")
        if scorer.tokenizer.convert_ids_to_tokens(encoded[1:-1]) != list(window_sequence):
            raise ValueError("tokenizer residue mapping differs from the chain sequence")
        log_probs = scorer._log_probs(window_sequence, window_structure)[0].numpy().astype(np.float32)
        positions = expected_positions
        save_cache(
            cache_path, cache_config_sha256, context_sha256, pdb_sha256,
            positions, structure_tokens, np.asarray(tokens), token_ids, log_probs,
        )
        reused = False

    if (structure_tokens < 0).any() or (structure_tokens >= MODEL_METADATA["structure_vocab_size"]).any():
        raise ValueError("invalid ProSST structure token")
    if log_probs.shape != (len(positions), len(tokens)) or not np.isfinite(log_probs).all():
        raise ValueError("invalid ProSST log-probability matrix")
    if not np.allclose(np.logaddexp.reduce(log_probs, axis=1), 0.0, atol=1e-5):
        raise ValueError("ProSST log probabilities are not normalized")
    return positions, log_probs, tokens, reused


def score_context(
    context: pd.DataFrame,
    positions: np.ndarray,
    log_probs: np.ndarray,
    tokens: list[str],
) -> pd.DataFrame:
    position_index = {int(position): index for index, position in enumerate(positions)}
    token_index = {token: index for index, token in enumerate(tokens)}
    rows = context["chain_position"].map(position_index)
    wt_columns = context["wt_aa"].map(token_index)
    mutant_columns = context["mut_aa"].map(token_index)
    if rows.isna().any() or wt_columns.isna().any() or mutant_columns.isna().any():
        raise ValueError("candidate references a missing position or amino-acid token")
    row_ids = rows.to_numpy(dtype=np.int64)
    wt_logp = log_probs[row_ids, wt_columns.to_numpy(dtype=np.int64)].astype(np.float64)
    mutant_logp = log_probs[row_ids, mutant_columns.to_numpy(dtype=np.int64)].astype(np.float64)
    components = pd.DataFrame(
        {"sample_id": context["sample_id"].to_numpy(), "prosst_score": mutant_logp - wt_logp,
         "mutant_logp": mutant_logp, "wt_logp": wt_logp}
    )
    scores = components.groupby("sample_id", sort=False)[
        ["prosst_score", "mutant_logp", "wt_logp"]
    ].sum()
    identity = context[
        ["sample_id", "subset", "candidate_group_id", "source_assay", "mutant"]
    ].drop_duplicates("sample_id")
    output = identity.merge(scores, on="sample_id", validate="one_to_one")
    output.insert(5, "status", "ok")
    return output


def balanced_shards(
    components: pd.DataFrame,
    chain_sequences: dict[str, str],
    max_residues: int,
    num_shards: int,
) -> tuple[list[list[str]], list[int]]:
    work = []
    for context_sha256, context in components.groupby("context_sha256"):
        length = len(chain_sequences[context_sha256])
        positions = sorted(context["chain_position"].astype(int).unique())
        start, end = inference_window(length, positions, max_residues)
        work.append((length * length + (end - start + 1) ** 2, context_sha256))
    shards = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for cost, context_sha256 in sorted(work, key=lambda item: (-item[0], item[1])):
        shard_id = min(range(num_shards), key=lambda index: (loads[index], index))
        shards[shard_id].append(context_sha256)
        loads[shard_id] += cost
    return shards, loads


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    if args.structure_batch_size < 1:
        raise ValueError("structure-batch-size must be positive")
    if args.context_sha256 and (args.num_shards != 1 or args.shard_id != 0):
        raise ValueError("explicit context selection require one shard")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.adapter_root))
    import joblib
    import torch
    import transformers
    from pfarena_models.metrics.sequence.prosst2048 import ProSST2048FitnessScorer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dataset, proteins, samples, substitutions, _ = load_model_dataset(args.dataset_dir)
    components, chain_sequences = build_components(proteins, samples, substitutions)
    structure_frame = pd.read_csv(args.structure_manifest)
    structure_frame["pdb_path"] = structure_frame["pdb_path"].map(
        lambda value: str(resolve_structure_path(value, args.structure_manifest))
    )
    structures = validate_structures(structure_frame, dataset, chain_sequences)
    structure_manifest_sha256 = sha256_file(args.structure_manifest)
    model_validation = validate_model_dir(args.model_dir)
    prosst_validation = validate_prosst_repo(args.prosst_repo_dir)
    scorer = ProSST2048FitnessScorer(
        model_dir=args.model_dir,
        prosst_repo_dir=args.prosst_repo_dir,
        device=args.device,
        batch_size=args.structure_batch_size,
        structure_vocab_size=MODEL_METADATA["structure_vocab_size"],
    )
    model_dtype = str(next(scorer.model.parameters()).dtype)
    if model_dtype != "torch.float32":
        raise ValueError(f"expected FP32 model weights, found {model_dtype}")
    tokenizer_max_length = int(scorer.tokenizer.model_max_length)
    max_residues = tokenizer_max_length - 2
    all_contexts = set(components["context_sha256"])
    if args.context_sha256:
        selected_contexts = sorted(set(args.context_sha256))
        unknown = set(selected_contexts) - all_contexts
        if unknown:
            raise ValueError(f"unknown selected contexts: {sorted(unknown)}")
        selected = components[components["context_sha256"].isin(selected_contexts)]
        _, loads = balanced_shards(selected, chain_sequences, max_residues, 1)
        estimated_load = loads[0]
        partial_run = True
    else:
        if args.num_shards > len(all_contexts):
            raise ValueError("num-shards exceeds the number of chain contexts")
        shards, loads = balanced_shards(components, chain_sequences, max_residues, args.num_shards)
        selected_contexts = shards[args.shard_id]
        estimated_load = loads[args.shard_id]
        partial_run = False

    scorer_path = args.adapter_root / "pfarena_models/metrics/sequence/prosst2048.py"
    cache_config = {
        "model_id": MODEL_METADATA["model_id"],
        "checkpoint": MODEL_METADATA["checkpoint"],
        "checkpoint_revision": MODEL_METADATA["checkpoint_revision"],
        "model_config_sha256": model_validation["config_sha256"],
        "model_weights_sha256": model_validation["weights_sha256"],
        "model_weights_size_bytes": model_validation["weights_size_bytes"],
        "tokenizer_sha256": model_validation["tokenizer_sha256"],
        "custom_model_code_sha256": model_validation["custom_model_code_sha256"],
        "adapter_sha256": sha256_file(scorer_path),
        "core_sha256": sha256_file(SCRIPT_DIR / "core.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        **prosst_validation,
        "python": platform.python_version(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__, "joblib": joblib.__version__,
        "device": str(scorer.device), "model_dtype": model_dtype,
        "gpu": torch.cuda.get_device_name(0) if scorer.device.type == "cuda" else None,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "structure_batch_size": args.structure_batch_size,
        "structure_vocab_size": MODEL_METADATA["structure_vocab_size"],
        "tokenizer_model_max_length": tokenizer_max_length,
        "model_max_relative_positions": int(scorer.model.config.max_relative_positions),
        "scoring_strategy": "wildtype_marginal_sum_over_substitutions",
        "multichain_protocol": "sum_independent_mutated_chain_log_odds",
        "long_sequence_protocol": "full_pdb_tokens_required_interval_centered_window",
        "max_residues": max_residues,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "cache_namespace": "candidate_chain_prosst2048_v1",
    }
    cache_config_sha256 = sha256_json(cache_config)
    cache_root = args.cache_dir / cache_config_sha256[:16]
    started_at = utc_now()
    frames = []
    computed_contexts = reused_contexts = 0
    try:
        for context_sha256 in selected_contexts:
            context = components[components["context_sha256"].eq(context_sha256)]
            sequence = chain_sequences[context_sha256]
            structure = structures.loc[context_sha256]
            pdb_path = Path(structure["pdb_path"])
            pdb_sha256 = sha256_file(pdb_path)
            if pdb_sha256 != structure["pdb_sha256"]:
                raise ValueError(f"{context_sha256}: PDB checksum differs from manifest")
            required_positions = sorted(context["chain_position"].astype(int).unique())
            positions, log_probs, tokens, reused = context_outputs(
                scorer, sequence, context_sha256, required_positions, pdb_path,
                pdb_sha256, cache_root / f"{context_sha256}.npz",
                cache_config_sha256, max_residues,
            )
            frames.append(score_context(context, positions, log_probs, tokens))
            computed_contexts += int(not reused)
            reused_contexts += int(reused)
            print(
                f"{context_sha256[:16]}: length={len(sequence)} window={positions[0]}-{positions[-1]} "
                f"context={'reused' if reused else 'computed'}",
                flush=True,
            )
    finally:
        scorer.teardown()

    results = pd.concat(frames, ignore_index=True).sort_values("sample_id")
    run_config = {
        **cache_config,
        "cache_config_sha256": cache_config_sha256,
        "structure_manifest": str(args.structure_manifest.resolve()),
        "structure_manifest_sha256": structure_manifest_sha256,
        "output_schema_sha256": sha256_file(OUTPUT_SCHEMA),
    }
    atomic_write_csv(results, args.output_dir / "shards" / f"shard_{args.shard_id:04d}.csv")
    atomic_write_json(
        {
            "model_id": MODEL_METADATA["model_id"],
            "run_config": run_config,
            "run_config_sha256": sha256_json(run_config),
            "dataset_id": dataset["dataset_id"], "dataset_hash": dataset["dataset_hash"],
            "started_at_utc": started_at, "completed_at_utc": utc_now(),
            "shard_id": args.shard_id, "num_shards": args.num_shards,
            "chain_contexts": len(selected_contexts),
            "estimated_quadratic_work": int(estimated_load),
            "samples": len(results), "computed_contexts": computed_contexts,
            "reused_contexts": reused_contexts,
            "model_dir": str(args.model_dir.resolve()),
            "prosst_repo_dir": str(args.prosst_repo_dir.resolve()),
            "adapter_root": str(args.adapter_root.resolve()),
            "cache_root": str(cache_root.resolve()), "partial_run": partial_run,
        },
        args.output_dir / "shards" / f"shard_{args.shard_id:04d}.json",
    )
    print(f"Shard {args.shard_id}: {len(results)} samples")


if __name__ == "__main__":
    main()
