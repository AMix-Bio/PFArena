#!/usr/bin/env python3
"""Precompute VenusREM matrices and scores for a frozen candidate dataset."""

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
from run import DEFAULT_PROENV_ROOT, MODEL_METADATA, validate_model_dir


OUTPUT_SCHEMA = SCRIPT_DIR / "output_schema_candidates.json"
CACHE_FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proenv-root", type=Path, default=DEFAULT_PROENV_ROOT)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--chain-sha256", action="append")
    return parser.parse_args()


def read_fasta(path: Path) -> str:
    return "".join(
        line.strip() for line in path.read_text().splitlines() if not line.startswith(">")
    )


def validate_inputs(
    input_dir: Path,
    inputs: dict,
    contexts: pd.DataFrame,
    dataset: dict,
    proteins: pd.DataFrame,
    samples: pd.DataFrame,
    substitutions: pd.DataFrame,
) -> pd.DataFrame:
    if inputs.get("schema_version") != 1:
        raise ValueError("candidate VenusREM input schema_version must be 1")
    if inputs.get("dataset_hash") != dataset["dataset_hash"]:
        raise ValueError("candidate VenusREM inputs target a different dataset")
    if sha256_file(input_dir / "contexts.csv") != inputs["contexts_sha256"]:
        raise ValueError("candidate VenusREM contexts.csv checksum differs")
    required = {
        "context_id", "context_sha256", "parent_sequence_sha256", "chain_id",
        "chain_sha256", "chain_length", "window_start", "window_end",
        "window_length", "expected_samples", "residue_fasta", "residue_sha256",
        "structure_fasta", "structure_sha256", "aa_alignment_file",
        "aa_alignment_sha256", "aa_alignment_size_bytes",
    }
    if required - set(contexts.columns):
        raise ValueError("candidate VenusREM context columns are incomplete")
    if contexts.duplicated(["parent_sequence_sha256", "chain_id"]).any():
        raise ValueError("duplicate parent-chain VenusREM context")
    if contexts["context_sha256"].duplicated().any():
        raise ValueError("duplicate VenusREM context identity")
    if len(contexts) != int(inputs["chain_contexts"]):
        raise ValueError("VenusREM context count differs from inputs.json")
    chain_lookup = {}
    for protein in proteins.itertuples(index=False):
        for chain_id, sequence in enumerate(json.loads(protein.chain_sequences), start=1):
            chain_lookup[(protein.sequence_sha256, chain_id)] = hashlib.sha256(sequence.encode()).hexdigest()
    identity = samples[
        ["sample_id", "subset", "candidate_group_id", "source_assay", "mutant", "sequence_sha256"]
    ]
    components = substitutions.merge(identity, on="sample_id", validate="many_to_one")
    components["chain_sha256"] = [
        chain_lookup[(sequence_sha256, int(chain_id))]
        for sequence_sha256, chain_id in zip(components["sequence_sha256"], components["chain_id"])
    ]
    components = components.merge(
        contexts,
        on="chain_sha256",
        validate="many_to_one",
    )
    if len(components) != len(substitutions):
        raise ValueError("VenusREM contexts do not cover every substitution component")
    assignments = components.drop_duplicates(["sample_id", "context_sha256"])
    if int(contexts["expected_samples"].sum()) != len(assignments):
        raise ValueError("VenusREM context sample count differs from the dataset")
    observed = assignments.groupby("context_sha256").size()
    expected = contexts.set_index("context_sha256")["expected_samples"].astype(int)
    if not observed.sort_index().equals(expected.sort_index()):
        raise ValueError("VenusREM context sample counts differ")
    return components


def balanced_shards(
    contexts: pd.DataFrame, num_shards: int
) -> tuple[list[list[str]], list[int]]:
    work = []
    for row in contexts.itertuples(index=False):
        cost = int(row.aa_alignment_size_bytes) + int(row.window_length) ** 2
        work.append((cost, row.context_sha256))
    shards = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for cost, context_sha256 in sorted(work, key=lambda item: (-item[0], item[1])):
        shard_id = min(range(num_shards), key=lambda index: (loads[index], index))
        shards[shard_id].append(context_sha256)
        loads[shard_id] += cost
    return shards, loads


def save_cache(
    path: Path,
    cache_config_sha256: str,
    context,
    positions: np.ndarray,
    tokens: np.ndarray,
    token_ids: np.ndarray,
    values: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cache_format_version=np.int32(CACHE_FORMAT_VERSION),
            cache_config_sha256=np.asarray(cache_config_sha256),
            context_sha256=np.asarray(context.context_sha256),
            chain_sha256=np.asarray(context.chain_sha256),
            chain_length=np.int32(context.chain_length),
            window_start=np.int32(context.window_start),
            window_end=np.int32(context.window_end),
            positions=positions,
            tokens=tokens,
            token_ids=token_ids,
            values=values,
        )
    os.replace(temporary, path)


def context_values(
    scorer,
    context,
    cache_path: Path,
    cache_config_sha256: str,
    tokens: list[str],
    token_ids: np.ndarray,
    protein_context_class,
) -> tuple[np.ndarray, np.ndarray, bool]:
    resources = (
        (Path(context.residue_fasta), context.residue_sha256),
        (Path(context.structure_fasta), context.structure_sha256),
        (Path(context.aa_alignment_file), context.aa_alignment_sha256),
    )
    for path, expected_sha256 in resources:
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"VenusREM resource checksum differs: {path}")
    sequence = read_fasta(Path(context.residue_fasta)).upper()
    if len(sequence) != int(context.window_length):
        raise ValueError("VenusREM residue FASTA length differs from the context")
    positions = np.arange(
        int(context.window_start), int(context.window_end) + 1, dtype=np.int32
    )

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if int(cache["cache_format_version"].item()) != CACHE_FORMAT_VERSION:
                raise ValueError(f"{cache_path}: unsupported cache format")
            if str(cache["cache_config_sha256"].item()) != cache_config_sha256:
                raise ValueError(f"{cache_path}: inference configuration differs")
            if str(cache["context_sha256"].item()) != context.context_sha256:
                raise ValueError(f"{cache_path}: input context differs")
            if str(cache["chain_sha256"].item()) != context.chain_sha256:
                raise ValueError(f"{cache_path}: natural chain differs")
            if int(cache["chain_length"].item()) != int(context.chain_length):
                raise ValueError(f"{cache_path}: chain length differs")
            cached_positions = cache["positions"].astype(np.int32)
            cached_tokens = cache["tokens"].astype(str).tolist()
            cached_token_ids = cache["token_ids"].astype(np.int32)
            values = cache["values"].astype(np.float32)
        if not np.array_equal(cached_positions, positions):
            raise ValueError(f"{cache_path}: inference window differs")
        if cached_tokens != tokens or not np.array_equal(cached_token_ids, token_ids):
            raise ValueError(f"{cache_path}: tokenizer vocabulary differs")
        reused = True
    else:
        protein_context = protein_context_class(
            protein_name=context.context_id,
            residue_fasta=context.residue_fasta,
            structure_fasta=context.structure_fasta,
            aa_seq_aln_file=context.aa_alignment_file,
        )
        values = (
            scorer._precompute_logits(sequence, protein_context)
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        if values.shape != (len(positions), len(tokens)) or not np.isfinite(values).all():
            raise ValueError("VenusREM returned an invalid retrieval-enhanced value matrix")
        save_cache(
            cache_path, cache_config_sha256, context, positions,
            np.asarray(tokens), token_ids, values,
        )
        reused = False
    if values.shape != (len(positions), len(tokens)) or not np.isfinite(values).all():
        raise ValueError("VenusREM returned an invalid retrieval-enhanced value matrix")
    return positions, values, reused


def score_context(
    components: pd.DataFrame,
    positions: np.ndarray,
    values: np.ndarray,
    tokens: list[str],
) -> pd.DataFrame:
    position_index = {int(position): index for index, position in enumerate(positions)}
    token_index = {token: index for index, token in enumerate(tokens)}
    rows = components["chain_position"].map(position_index)
    wt_columns = components["wt_aa"].map(token_index)
    mutant_columns = components["mut_aa"].map(token_index)
    if rows.isna().any() or wt_columns.isna().any() or mutant_columns.isna().any():
        raise ValueError("candidate references a missing VenusREM position or residue token")
    row_ids = rows.to_numpy(dtype=np.int64)
    wt_values = values[row_ids, wt_columns.to_numpy(dtype=np.int64)].astype(np.float64)
    mutant_values = values[
        row_ids, mutant_columns.to_numpy(dtype=np.int64)
    ].astype(np.float64)
    component_scores = pd.DataFrame(
        {
            "sample_id": components["sample_id"].to_numpy(),
            "venusrem_score": mutant_values - wt_values,
            "mutant_value": mutant_values,
            "wt_value": wt_values,
        }
    )
    scores = component_scores.groupby("sample_id", sort=False)[
        ["venusrem_score", "mutant_value", "wt_value"]
    ].sum()
    identity = components[
        ["sample_id", "subset", "candidate_group_id", "source_assay", "mutant"]
    ].drop_duplicates("sample_id")
    output = identity.merge(scores, on="sample_id", validate="one_to_one")
    output.insert(5, "status", "ok")
    return output


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    if args.chain_sha256 and (args.num_shards != 1 or args.shard_id != 0):
        raise ValueError("explicit smoke contexts require one shard")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.proenv_root))
    import torch
    import transformers
    from proenv.metrics.sequence.venusrem_de_fitness import (
        VenusREMFitnessScorer,
        VenusREMProteinContext,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dataset, proteins, samples, substitutions, _ = load_model_dataset(args.dataset_dir)
    inputs = json.loads((args.input_dir / "inputs.json").read_text())
    contexts = pd.read_csv(args.input_dir / "contexts.csv", keep_default_na=False)
    components = validate_inputs(
        args.input_dir, inputs, contexts, dataset, proteins, samples, substitutions
    )
    model_validation = validate_model_dir(args.model_dir)
    all_chain_hashes = set(contexts["chain_sha256"])
    if args.chain_sha256:
        selected_hashes = sorted(set(args.chain_sha256))
        unknown = set(selected_hashes) - all_chain_hashes
        if unknown:
            raise ValueError(f"unknown smoke chain contexts: {sorted(unknown)}")
        selected_rows = contexts[contexts["chain_sha256"].isin(selected_hashes)]
        selected_contexts = selected_rows["context_sha256"].tolist()
        _, loads = balanced_shards(selected_rows, 1)
        estimated_load = loads[0]
        partial_run = True
    else:
        if args.num_shards > len(contexts):
            raise ValueError("num-shards exceeds the number of VenusREM contexts")
        shards, loads = balanced_shards(contexts, args.num_shards)
        selected_contexts = shards[args.shard_id]
        estimated_load = loads[args.shard_id]
        partial_run = False

    scorer_path = args.proenv_root / "proenv/metrics/sequence/venusrem_de_fitness.py"
    scorer_sha256 = sha256_file(scorer_path)
    scorer = VenusREMFitnessScorer(
        str(args.model_dir),
        device=args.device,
        logit_mode=MODEL_METADATA["logit_mode"],
        alpha=MODEL_METADATA["alpha"],
        structure_vocab_size=MODEL_METADATA["structure_vocab_size"],
        sample_ratio=MODEL_METADATA["sample_ratio"],
        sample_times=MODEL_METADATA["sample_times"],
        local_files_only=True,
        trust_remote_code=True,
    )
    model_dtype = str(next(scorer.model.parameters()).dtype)
    if model_dtype != "torch.float32":
        raise ValueError(f"expected FP32 model weights, found {model_dtype}")
    tokenizer_max_length = int(scorer.tokenizer.model_max_length)
    if tokenizer_max_length != inputs["max_residues"] + 2:
        raise ValueError("prepared VenusREM window differs from checkpoint token limit")
    tokens = [
        scorer.tokenizer.convert_ids_to_tokens(token_id)
        for token_id in range(len(scorer.tokenizer))
    ]
    token_ids = np.arange(len(tokens), dtype=np.int32)
    cache_config = {
        "model_id": MODEL_METADATA["model_id"],
        "checkpoint": MODEL_METADATA["checkpoint"],
        "checkpoint_revision": MODEL_METADATA["checkpoint_revision"],
        "model_config_sha256": model_validation["config_sha256"],
        "model_weights_sha256": model_validation["weights_sha256"],
        "model_weights_size_bytes": model_validation["weights_size_bytes"],
        "tokenizer_sha256": model_validation["tokenizer_sha256"],
        "custom_model_code_sha256": model_validation["custom_model_code_sha256"],
        "proenv_scorer_sha256": scorer_sha256,
        "base_runner_sha256": sha256_file(SCRIPT_DIR / "run.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "device": str(scorer.device),
        "gpu": torch.cuda.get_device_name(0) if scorer.device.type == "cuda" else None,
        "model_dtype": model_dtype,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "tokenizer_model_max_length": tokenizer_max_length,
        "structure_vocab_size": MODEL_METADATA["structure_vocab_size"],
        "logit_mode": MODEL_METADATA["logit_mode"],
        "alpha": MODEL_METADATA["alpha"],
        "sample_ratio": MODEL_METADATA["sample_ratio"],
        "sample_times": MODEL_METADATA["sample_times"],
        "scoring_strategy": "wildtype_marginal_sum_over_substitutions",
        "multichain_protocol": inputs["multichain_protocol"],
        "long_sequence_protocol": inputs["long_sequence_protocol"],
        "max_residues": inputs["max_residues"],
        "input_contexts_sha256": inputs["contexts_sha256"],
        "cache_format_version": CACHE_FORMAT_VERSION,
        "cache_namespace": "candidate_chain_venusrem_v1",
    }
    cache_config_sha256 = sha256_json(cache_config)
    cache_root = args.cache_dir / cache_config_sha256[:16]
    started_at = utc_now()
    frames = []
    computed_contexts = reused_contexts = 0
    try:
        for context_sha256 in selected_contexts:
            context = next(
                contexts[contexts["context_sha256"].eq(context_sha256)].itertuples(index=False)
            )
            context_components = components[
                components["context_sha256"].eq(context_sha256)
            ]
            positions, values, reused = context_values(
                scorer, context, cache_root / f"{context.chain_sha256}.npz",
                cache_config_sha256, tokens, token_ids, VenusREMProteinContext,
            )
            frames.append(score_context(context_components, positions, values, tokens))
            computed_contexts += int(not reused)
            reused_contexts += int(reused)
            print(
                f"{context.context_id}: chain={context.chain_length} "
                f"window={context.window_start}-{context.window_end} "
                f"msa={context.msa_records} context={'reused' if reused else 'computed'}",
                flush=True,
            )
    finally:
        scorer.teardown()

    results = pd.concat(frames, ignore_index=True).sort_values("sample_id")
    run_config = {
        **cache_config,
        "cache_config_sha256": cache_config_sha256,
        "inputs_sha256": sha256_file(args.input_dir / "inputs.json"),
        "prosst_run_config_sha256": inputs["prosst_run_config_sha256"],
        "output_schema_sha256": sha256_file(OUTPUT_SCHEMA),
    }
    if "resource_manifest_sha256" in inputs:
        run_config["resource_manifest_sha256"] = inputs["resource_manifest_sha256"]
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
            "estimated_work": int(estimated_load),
            "samples": len(results),
            "computed_contexts": computed_contexts,
            "reused_contexts": reused_contexts,
            "model_dir": str(args.model_dir.resolve()),
            "proenv_root": str(args.proenv_root.resolve()),
            "input_dir": str(args.input_dir.resolve()),
            "cache_root": str(cache_root.resolve()),
            "partial_run": partial_run,
        },
        args.output_dir / "shards" / f"shard_{args.shard_id:04d}.json",
    )
    print(f"Shard {args.shard_id}: {len(results)} samples")


if __name__ == "__main__":
    main()
