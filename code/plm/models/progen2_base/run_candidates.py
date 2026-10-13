#!/usr/bin/env python3
"""Precompute ProGen2-base likelihoods for T1--T4."""

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
    CACHE_FORMAT_VERSION,
    DEFAULT_ADAPTER_ROOT,
    METRIC_NAMES,
    MODEL_METADATA,
    atomic_save_cache,
    load_cache,
    score_sequences,
    validate_model,
)


OUTPUT_SCHEMA = SCRIPT_DIR / "output_schema_candidates.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-block-size", type=int, default=2048)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--context-sha256", action="append")
    return parser.parse_args()


def build_variants(
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

    ordered = substitutions.sort_values(["sample_id", "component_index"]).copy()
    ordered["annotation"] = (
        ordered["wt_aa"]
        + ordered["chain_position"].astype(str)
        + ordered["mut_aa"]
    )
    variant_keys = ordered.groupby(["sample_id", "chain_id"], sort=False)["annotation"].agg(":".join)
    sample_context = ordered[["sample_id", "chain_id"]].drop_duplicates()
    sample_context = sample_context.merge(
        samples[["sample_id", "sequence_sha256"]],
        on="sample_id",
        validate="many_to_one",
    )
    sample_context["context_sha256"] = [
        chain_lookup[(sequence_sha256, int(chain_id))]
        for sequence_sha256, chain_id in zip(
            sample_context["sequence_sha256"], sample_context["chain_id"]
        )
    ]

    variants = samples[
        [
            "sample_id",
            "subset",
            "candidate_group_id",
            "source_assay",
            "mutant",
            "mutated_sequence",
        ]
    ].merge(
        sample_context[["sample_id", "chain_id", "context_sha256"]],
        on="sample_id",
        validate="one_to_many",
    )
    variants["variant_key"] = [
        variant_keys.loc[(sample_id, chain_id)]
        for sample_id, chain_id in zip(variants["sample_id"], variants["chain_id"])
    ]
    variants["variant_sequence"] = [
        sequence.split(":")[int(chain_id) - 1]
        for sequence, chain_id in zip(
            variants["mutated_sequence"], variants["chain_id"]
        )
    ]
    consistency = variants.groupby(["context_sha256", "variant_key"])[
        "variant_sequence"
    ].nunique()
    if consistency.gt(1).any():
        raise ValueError("one chain-local variant key maps to multiple sequences")
    return variants.drop(columns=["mutated_sequence"]), chain_sequences


def attention_cost(sequence_length: int, context_length: int) -> int:
    terminalized_length = sequence_length + 2
    return sum(
        min(context_length, terminalized_length - start) ** 2
        for start in range(0, terminalized_length, context_length)
    )


def balanced_context_shards(
    variants: pd.DataFrame,
    chain_sequences: dict[str, str],
    context_length: int,
    num_shards: int,
) -> tuple[list[list[str]], list[int]]:
    work = []
    for context_sha256, context in variants.groupby("context_sha256"):
        unique_sequences = context["variant_key"].nunique() + 1
        cost = unique_sequences * attention_cost(
            len(chain_sequences[context_sha256]), context_length
        )
        work.append((cost, context_sha256))
    shards = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for cost, context_sha256 in sorted(work, key=lambda item: (-item[0], item[1])):
        shard_id = min(range(num_shards), key=lambda index: (loads[index], index))
        shards[shard_id].append(context_sha256)
        loads[shard_id] += cost
    return shards, loads


def score_context(
    sequence: str,
    variants: pd.DataFrame,
    model,
    tokenizer,
    context_length: int,
    batch_size: int,
    cache_block_size: int,
    cache_path: Path,
    cache_config_sha256: str,
    context_sha256: str,
) -> tuple[pd.DataFrame, dict[str, float], int, int]:
    unique = variants.drop_duplicates("variant_key").set_index("variant_key")
    required_keys = ["__WT__"] + sorted(unique.index.astype(str))
    keys, values = load_cache(cache_path, cache_config_sha256, context_sha256)
    cached_keys = set(keys)
    missing = [key for key in required_keys if key not in cached_keys]

    for start in range(0, len(missing), cache_block_size):
        block_keys = missing[start : start + cache_block_size]
        block_sequences = [
            sequence if key == "__WT__" else unique.at[key, "variant_sequence"]
            for key in block_keys
        ]
        block_values = score_sequences(
            model, tokenizer, block_sequences, context_length, batch_size, model.device
        )
        keys.extend(block_keys)
        for name in METRIC_NAMES:
            values[name] = np.concatenate([values[name], block_values[name]])
        atomic_save_cache(
            cache_path, cache_config_sha256, context_sha256, keys, values
        )
        print(
            f"{context_sha256[:16]}: cached "
            f"{min(start + len(block_keys), len(missing))}/{len(missing)} new sequences",
            flush=True,
        )

    index = {key: position for position, key in enumerate(keys)}
    scores = {
        key: {name: values[name][index[key]].item() for name in METRIC_NAMES}
        for key in required_keys
    }
    mapped = variants["variant_key"].map(scores)
    if mapped.isna().any():
        raise ValueError("missing candidate sequence likelihood")
    output = variants[
        ["sample_id", "subset", "candidate_group_id", "source_assay", "mutant"]
    ].copy()
    output["status"] = "ok"
    for name in METRIC_NAMES:
        output[name] = mapped.map(lambda item, metric=name: item[metric])
    output["progen2_delta"] = output["progen2_score"] - scores["__WT__"]["progen2_score"]
    return output, scores["__WT__"], len(required_keys) - len(missing), len(missing)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    if args.batch_size < 1 or args.cache_block_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.context_sha256 and (args.num_shards != 1 or args.shard_id != 0):
        raise ValueError("explicit context selection require one shard")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.adapter_root))

    import tokenizers
    import torch
    import transformers
    from pfarena_models.models.progen import ProGenForCausalLM
    from tokenizers import Tokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dataset, proteins, samples, substitutions, _ = load_model_dataset(
        args.dataset_dir
    )
    variants, chain_sequences = build_variants(proteins, samples, substitutions)
    all_contexts = set(variants["context_sha256"])
    validation = validate_model(args.model_dir, args.adapter_root)
    tokenizer = Tokenizer.from_file(str(validation["tokenizer_path"]))
    if tokenizer.get_vocab_size() != 30:
        raise ValueError("deployed ProGen tokenizer vocabulary is not 30 tokens")
    model = ProGenForCausalLM.from_pretrained(
        str(args.model_dir), dtype=torch.float32
    ).to(args.device).eval()
    if str(next(model.parameters()).dtype) != "torch.float32":
        raise ValueError("ProGen2 model did not load in FP32")
    if int(model.config.vocab_size) <= MODEL_METADATA["scored_token_id_last"]:
        raise ValueError("checkpoint vocabulary is incompatible with the tokenizer")
    context_length = int(model.config.n_positions)
    if context_length < 2:
        raise ValueError("invalid checkpoint context length")

    if args.context_sha256:
        selected_contexts = sorted(set(args.context_sha256))
        unknown = set(selected_contexts) - all_contexts
        if unknown:
            raise ValueError(f"unknown selected contexts: {sorted(unknown)}")
        estimated_load = sum(
            (variants.loc[variants["context_sha256"].eq(context), "variant_key"].nunique() + 1)
            * attention_cost(len(chain_sequences[context]), context_length)
            for context in selected_contexts
        )
        partial_run = True
    else:
        if args.num_shards > len(all_contexts):
            raise ValueError("num-shards exceeds the number of chain contexts")
        shards, loads = balanced_context_shards(
            variants, chain_sequences, context_length, args.num_shards
        )
        selected_contexts = shards[args.shard_id]
        estimated_load = loads[args.shard_id]
        partial_run = False

    gpu = torch.cuda.get_device_name(0) if args.device == "cuda" else None
    cache_config = {
        "model_id": MODEL_METADATA["model_id"],
        "checkpoint": MODEL_METADATA["checkpoint"],
        "checkpoint_config_sha256": validation["config_sha256"],
        "checkpoint_weights_sha256": validation["weights_sha256"],
        "checkpoint_weights_size_bytes": validation["weights_size_bytes"],
        "checkpoint_weight_files": validation["weight_files"],
        "checkpoint_declared_dtype": validation["config"].get("torch_dtype"),
        "checkpoint_architecture": {
            key: getattr(model.config, key)
            for key in ("vocab_size", "n_positions", "n_embd", "n_layer", "n_head", "rotary_dim")
        },
        "tokenizer_sha256": MODEL_METADATA["tokenizer_sha256"],
        "configuration_code_sha256": MODEL_METADATA["configuration_code_sha256"],
        "modeling_code_sha256": MODEL_METADATA["modeling_code_sha256"],
        "core_sha256": sha256_file(SCRIPT_DIR / "core.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "device": args.device,
        "model_dtype": str(next(model.parameters()).dtype),
        "gpu": gpu,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "batch_size": args.batch_size,
        "context_length": context_length,
        "terminal_protocol": "prefix_1_suffix_2",
        "chunk_protocol": "non_overlapping_no_empty_tail",
        "scored_token_ids": [MODEL_METADATA["scored_token_id_first"], MODEL_METADATA["scored_token_id_last"]],
        "scoring_strategy": MODEL_METADATA["scoring_strategy"],
        "multichain_protocol": "sum_independent_mutated_chain_score_deltas",
        "cache_format_version": CACHE_FORMAT_VERSION,
        "cache_namespace": "candidate_chain_bidirectional_v1",
    }
    cache_config_sha256 = sha256_json(cache_config)
    cache_root = args.cache_dir / cache_config_sha256[:16]
    started_at = utc_now()
    frames = []
    required_sequences = computed_sequences = reused_sequences = 0
    for context_sha256 in selected_contexts:
        context = variants[variants["context_sha256"].eq(context_sha256)]
        output, wt_scores, reused, computed = score_context(
            chain_sequences[context_sha256],
            context,
            model,
            tokenizer,
            context_length,
            args.batch_size,
            args.cache_block_size,
            cache_root / f"{context_sha256}.npz",
            cache_config_sha256,
            context_sha256,
        )
        atomic_write_json(
            {
                "context_sha256": context_sha256,
                "sequence_length": len(chain_sequences[context_sha256]),
                **wt_scores,
            },
            args.output_dir / "wildtypes" / f"{context_sha256}.json",
        )
        frames.append(output)
        required_sequences += context["variant_key"].nunique() + 1
        computed_sequences += computed
        reused_sequences += reused
        print(
            f"{context_sha256[:16]}: sequences={context['variant_key'].nunique() + 1} "
            f"computed={computed} reused={reused}",
            flush=True,
        )

    results = pd.concat(frames, ignore_index=True).sort_values("sample_id")
    run_config = {
        **cache_config,
        "cache_config_sha256": cache_config_sha256,
        "output_schema_sha256": sha256_file(OUTPUT_SCHEMA),
    }
    atomic_write_csv(results, args.output_dir / "shards" / f"shard_{args.shard_id:04d}.csv")
    atomic_write_json(
        {
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
            "estimated_attention_work": int(estimated_load),
            "samples": len(results),
            "required_sequences": required_sequences,
            "computed_sequences": computed_sequences,
            "reused_sequences": reused_sequences,
            "model_dir": str(args.model_dir.resolve()),
            "adapter_root": str(args.adapter_root.resolve()),
            "cache_root": str(cache_root.resolve()),
            "partial_run": partial_run,
        },
        args.output_dir / "shards" / f"shard_{args.shard_id:04d}.json",
    )
    print(f"Shard {args.shard_id}: {len(results)} samples", flush=True)


if __name__ == "__main__":
    main()
