#!/usr/bin/env python3
"""Precompute ESM-2 masked-marginal outputs for T1--T4."""

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
from common_io import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    sha256_json,
    utc_now,
)
from run import (
    CACHE_FORMAT_VERSION,
    DEFAULT_PROENV_ROOT,
    LONG_SEQUENCE_PROTOCOLS,
    MODEL_METADATA,
    position_log_probs,
    validate_model_dir,
)


OUTPUT_SCHEMA = SCRIPT_DIR / "output_schema_candidates.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proenv-root", type=Path, default=DEFAULT_PROENV_ROOT)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
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
    parser.add_argument(
        "--context-sha256",
        action="append",
        help="score only an explicitly selected chain context for a smoke test",
    )
    return parser.parse_args()


def build_component_contexts(
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
            previous = chain_sequences.setdefault(context_sha256, sequence)
            if previous != sequence:
                raise ValueError("chain sequence SHA256 collision")

    identity = samples[
        [
            "sample_id",
            "subset",
            "candidate_group_id",
            "source_assay",
            "mutant",
            "sequence_sha256",
        ]
    ]
    components = substitutions.merge(
        identity,
        on="sample_id",
        validate="many_to_one",
    )
    components["context_sha256"] = [
        chain_lookup[(sequence_sha256, int(chain_id))]
        for sequence_sha256, chain_id in zip(
            components["sequence_sha256"], components["chain_id"]
        )
    ]
    return components, chain_sequences


def balanced_context_shards(
    components: pd.DataFrame,
    chain_sequences: dict[str, str],
    max_window: int,
    num_shards: int,
) -> tuple[list[list[str]], list[int]]:
    work_items = []
    for context_sha256, context in components.groupby("context_sha256"):
        positions = context["chain_position"].nunique()
        cost = int(positions) * min(len(chain_sequences[context_sha256]), max_window)
        work_items.append((cost, context_sha256))

    shards = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for cost, context_sha256 in sorted(work_items, key=lambda item: (-item[0], item[1])):
        shard_id = min(range(num_shards), key=lambda index: (loads[index], index))
        shards[shard_id].append(context_sha256)
        loads[shard_id] += cost
    return shards, loads


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

    rows = rows.to_numpy(dtype=np.int64)
    wt_logp = log_probs[rows, wt_columns.to_numpy(dtype=np.int64)].astype(np.float64)
    mutant_logp = log_probs[
        rows, mutant_columns.to_numpy(dtype=np.int64)
    ].astype(np.float64)
    components = pd.DataFrame(
        {
            "sample_id": context["sample_id"].to_numpy(),
            "masked_marginal": mutant_logp - wt_logp,
            "mutant_logp": mutant_logp,
            "wt_logp": wt_logp,
        }
    )
    scores = components.groupby("sample_id", sort=False)[
        ["masked_marginal", "mutant_logp", "wt_logp"]
    ].sum()
    identity = context[
        [
            "sample_id",
            "subset",
            "candidate_group_id",
            "source_assay",
            "mutant",
        ]
    ].drop_duplicates("sample_id")
    output = identity.merge(scores, on="sample_id", validate="one_to_one")
    output.insert(5, "status", "ok")
    return output


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not 1 <= args.max_window <= MODEL_METADATA["default_max_residues"]:
        raise ValueError("max-window is outside the registered model context")
    if args.context_sha256 and (args.num_shards != 1 or args.shard_id != 0):
        raise ValueError("explicit smoke contexts require one shard")

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

    dataset, proteins, samples, substitutions, _ = load_model_dataset(
        args.dataset_dir
    )
    components, chain_sequences = build_component_contexts(
        proteins, samples, substitutions
    )
    all_contexts = set(components["context_sha256"])
    if args.context_sha256:
        selected_contexts = sorted(set(args.context_sha256))
        unknown = set(selected_contexts) - all_contexts
        if unknown:
            raise ValueError(f"unknown smoke contexts: {sorted(unknown)}")
        estimated_load = sum(
            components.loc[
                components["context_sha256"].eq(context_sha256), "chain_position"
            ].nunique()
            * min(len(chain_sequences[context_sha256]), args.max_window)
            for context_sha256 in selected_contexts
        )
        partial_run = True
    else:
        if args.num_shards > len(all_contexts):
            raise ValueError("num-shards exceeds the number of chain contexts")
        shards, loads = balanced_context_shards(
            components, chain_sequences, args.max_window, args.num_shards
        )
        selected_contexts = shards[args.shard_id]
        estimated_load = loads[args.shard_id]
        partial_run = False

    model_validation = validate_model_dir(args.model_dir)
    scorer = ESM2FitnessScorer(
        str(args.model_dir),
        device=args.device,
        batch_size=args.batch_size,
        max_window=args.max_window,
    )
    model_dtype = str(next(scorer.model.parameters()).dtype)
    if model_dtype != "torch.float32":
        raise ValueError(f"expected FP32 model weights, found {model_dtype}")
    model_name_or_path = scorer.model.config.name_or_path

    scorer_path = (
        args.proenv_root / "proenv" / "metrics" / "sequence" / "esm2_de_fitness.py"
    )
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
        "base_runner_sha256": sha256_file(SCRIPT_DIR / "run.py"),
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
        "scoring_strategy": "masked_marginal_sum_over_substitutions",
        "multichain_protocol": "sum_independent_mutated_chain_log_odds",
        "long_sequence_protocol": args.long_sequence_protocol,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "cache_namespace": "candidate_chain_masked_marginal_v1",
    }
    cache_config_sha256 = sha256_json(cache_config)
    cache_root = (
        args.cache_dir
        / cache_config_sha256[:16]
        / args.long_sequence_protocol
        / f"max_tokens_{args.max_window + 2}"
    )

    started_at = utc_now()
    result_frames = []
    required_positions_total = 0
    computed_positions_total = 0
    reused_positions_total = 0
    try:
        for context_sha256 in selected_contexts:
            context = components[components["context_sha256"].eq(context_sha256)]
            sequence = chain_sequences[context_sha256]
            required_positions = sorted(context["chain_position"].astype(int).unique())
            positions, log_probs, tokens, cache_stats = position_log_probs(
                scorer,
                sequence,
                context_sha256,
                required_positions,
                cache_root / f"{context_sha256}.npz",
                cache_config_sha256,
                args.max_window,
                args.long_sequence_protocol,
            )
            result_frames.append(score_context(context, positions, log_probs, tokens))
            required_positions_total += cache_stats["required"]
            computed_positions_total += cache_stats["computed"]
            reused_positions_total += cache_stats["reused"]
            print(
                f"{context_sha256[:16]}: positions={cache_stats['required']} "
                f"computed={cache_stats['computed']} reused={cache_stats['reused']}",
                flush=True,
            )
    finally:
        scorer.teardown()

    results = pd.concat(result_frames, ignore_index=True).sort_values("sample_id")
    run_config = {
        **cache_config,
        "cache_config_sha256": cache_config_sha256,
        "output_schema_sha256": sha256_file(OUTPUT_SCHEMA),
    }
    atomic_write_csv(
        results,
        args.output_dir / "shards" / f"shard_{args.shard_id:04d}.csv",
    )
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
        "chain_contexts": len(selected_contexts),
        "estimated_position_token_work": int(estimated_load),
        "samples": len(results),
        "required_positions": required_positions_total,
        "computed_positions": computed_positions_total,
        "reused_positions": reused_positions_total,
        "model_dir": str(args.model_dir.resolve()),
        "model_name_or_path": model_name_or_path,
        "proenv_root": str(args.proenv_root.resolve()),
        "cache_root": str(cache_root.resolve()),
        "partial_run": partial_run,
    }
    atomic_write_json(
        shard_metadata,
        args.output_dir / "shards" / f"shard_{args.shard_id:04d}.json",
    )
    print(f"Shard {args.shard_id}: {len(results)} samples")


if __name__ == "__main__":
    main()
