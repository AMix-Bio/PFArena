#!/usr/bin/env python3
"""Precompute S3F outputs for the frozen T1--T4 candidates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
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
    sha256_files,
    sha256_json,
    utc_now,
)
from core import (
    AMINO_ACIDS,
    DEFAULT_ADAPTER_ROOT,
    MODEL,
    MODEL_WINDOW,
    atomic_save_cache,
    compute_score_matrix,
    read_cache,
    surface_artifacts,
    validate_context_metadata,
    validate_precomputed_surface,
)
from prepare_surfaces import validate_surface


OUTPUT_SCHEMA = SCRIPT_DIR / "output_schema_candidates.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--surface-dir", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--s3f-script", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--esm-model-dir", type=Path, required=True)
    parser.add_argument("--position-batch-size", type=int, default=2)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--context-id", action="append")
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
        [
            "sample_id",
            "subset",
            "candidate_group_id",
            "source_assay",
            "mutant",
            "sequence_sha256",
        ]
    ]
    components = substitutions.merge(identity, on="sample_id", validate="many_to_one")
    components["context_sha256"] = [
        chain_lookup[(sequence_sha256, int(chain_id))]
        for sequence_sha256, chain_id in zip(
            components["sequence_sha256"], components["chain_id"]
        )
    ]
    return components, chain_sequences


def validate_inputs(
    input_dir: Path,
    dataset: dict,
    components: pd.DataFrame,
    chain_sequences: dict[str, str],
) -> tuple[dict, pd.DataFrame]:
    inputs = json.loads((input_dir / "inputs.json").read_text())
    contexts = pd.read_csv(input_dir / "contexts.csv")
    if (
        inputs.get("schema_version") != 1
        or inputs.get("model_id") != "S3F"
        or inputs.get("dataset_id") != dataset["dataset_id"]
        or inputs.get("dataset_hash") != dataset["dataset_hash"]
    ):
        raise ValueError("invalid S3F candidate input manifest")
    if sha256_file(input_dir / "contexts.csv") != inputs["context_manifest_sha256"]:
        raise ValueError("S3F candidate context manifest changed")
    if sha256_files([input_dir / "contexts.csv"]) != inputs["input_hash"]:
        raise ValueError("S3F candidate input hash differs")
    if contexts["context_id"].duplicated().any() or contexts["context_sha256"].duplicated().any():
        raise ValueError("S3F candidate contexts are duplicated")
    if set(contexts["context_sha256"]) != set(components["context_sha256"]):
        raise ValueError("S3F candidate context set differs from substitutions")

    sample_counts = components.groupby("context_sha256")["sample_id"].nunique()
    component_counts = components.groupby("context_sha256").size()
    position_counts = components.groupby("context_sha256")["chain_position"].nunique()
    for context in contexts.itertuples(index=False):
        sequence = chain_sequences[context.context_sha256]
        if (
            context.context_id != context.context_sha256[:16]
            or context.sequence != sequence
            or int(context.sequence_length) != len(sequence)
            or int(context.candidate_sample_count) != int(sample_counts[context.context_sha256])
            or int(context.substitution_component_count) != int(component_counts[context.context_sha256])
            or int(context.mutation_position_count) != int(position_counts[context.context_sha256])
            or sha256_file(resolve_project_path(context.pdb_path)) != context.pdb_sha256
        ):
            raise ValueError(f"{context.context_id}: invalid S3F candidate context")
    if (
        len(contexts) != inputs["contexts"]
        or int(contexts["candidate_sample_count"].sum()) != inputs["samples"]
        or int(contexts["substitution_component_count"].sum()) != inputs["substitution_components"]
        or int(contexts["mutation_position_count"].sum()) != inputs["mutation_positions"]
    ):
        raise ValueError("S3F candidate input counts differ")
    return inputs, contexts


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def candidate_surface(context: pd.Series, surface_root: Path) -> tuple[Path, Path]:
    context_id = str(context.name)
    surface_dir = surface_root / context_id
    surface_path = surface_dir / f"{context_id}.pkl"
    metadata_path = surface_dir / f"{context_id}.json"
    if not surface_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"{context_id}: candidate S3F surface is missing")
    metadata = json.loads(metadata_path.read_text())
    surface_summary = validate_surface(surface_path, int(context["sequence_length"]))
    expected = {
        "context_id": context_id,
        "context_sha256": str(context["context_sha256"]),
        "sequence_length": int(context["sequence_length"]),
        "pdb_sha256": str(context["pdb_sha256"]),
        "surface_sha256": sha256_file(surface_path),
        "s3f_script_sha256": MODEL["s3f_script_sha256"],
        "curvature_implementation": "chunked_pytorch_equivalent_v1",
        "curvature_chunk_size": 512,
        **surface_summary,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"{surface_dir}: surface {field} differs")
    if metadata.get("preprocessor_sha256") != sha256_file(
        SCRIPT_DIR / "prepare_candidate_surfaces.py"
    ):
        raise ValueError(f"{surface_dir}: surface preprocessor differs")
    return surface_path, surface_dir


def score_context(
    context: pd.DataFrame,
    positions: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    position_index = {int(position): index for index, position in enumerate(positions)}
    amino_acid_index = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
    rows = context["chain_position"].map(position_index)
    columns = context["mut_aa"].map(amino_acid_index)
    if rows.isna().any() or columns.isna().any():
        raise ValueError("candidate references a missing S3F position or amino acid")
    component_scores = scores[
        rows.to_numpy(dtype=np.int64), columns.to_numpy(dtype=np.int64)
    ].astype(np.float64)
    totals = pd.DataFrame(
        {"sample_id": context["sample_id"].to_numpy(), "s3f_score": component_scores}
    ).groupby("sample_id", sort=False)["s3f_score"].sum()
    identity = context[
        ["sample_id", "subset", "candidate_group_id", "source_assay", "mutant"]
    ].drop_duplicates("sample_id")
    output = identity.merge(totals, on="sample_id", validate="one_to_one")
    output.insert(5, "status", "ok")
    return output


def balanced_shards(contexts: pd.DataFrame, num_shards: int) -> tuple[list[list[str]], list[int]]:
    work = [
        (
            min(int(context.sequence_length), MODEL_WINDOW)
            * int(context.mutation_position_count),
            context.context_id,
        )
        for context in contexts.itertuples(index=False)
    ]
    shards = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for cost, context_id in sorted(work, key=lambda item: (-item[0], item[1])):
        shard_id = min(range(num_shards), key=lambda index: (loads[index], index))
        shards[shard_id].append(context_id)
        loads[shard_id] += cost
    return shards, loads


def main() -> None:
    args = parse_args()
    if args.position_batch_size < 1:
        raise ValueError("position-batch-size must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    if args.context_id and (args.num_shards != 1 or args.shard_id != 0):
        raise ValueError("explicit candidate contexts require one shard")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.adapter_root))
    import torch
    from pfarena_models.metrics.sequence.s3f import S3FFitnessScorer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset, proteins, samples, substitutions, _ = load_model_dataset(args.dataset_dir)
    components, chain_sequences = build_components(proteins, samples, substitutions)
    inputs, contexts = validate_inputs(args.input_dir, dataset, components, chain_sequences)
    if args.num_shards > len(contexts):
        raise ValueError("num-shards exceeds the number of S3F chain contexts")

    scorer_path = args.adapter_root / "pfarena_models/metrics/sequence/s3f.py"
    s3f_code_files = sorted(args.s3f_script.parent.rglob("*.py"))
    esm_files = sorted(path for path in args.esm_model_dir.rglob("*") if path.is_file())
    if not s3f_code_files or not esm_files:
        raise FileNotFoundError("S3F code or ESM model files are missing")
    model_files = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "s3f_script_sha256": sha256_file(args.s3f_script),
        "s3f_code_sha256": sha256_files(s3f_code_files),
        "esm_model_sha256": sha256_files(esm_files),
        "adapter_sha256": sha256_file(scorer_path),
    }
    for field, observed in model_files.items():
        if observed != MODEL[field]:
            raise ValueError(f"deployed S3F {field} differs from model.json")

    cache_config = {
        "model_id": MODEL["model_id"],
        "method": MODEL["method"],
        **model_files,
        "core_sha256": sha256_file(SCRIPT_DIR / "core.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "input_hash": inputs["input_hash"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "device": "cuda",
        "structure_start": 1,
        "plddt_threshold": MODEL["plddt_threshold"],
        "esm_max_input_length": MODEL_WINDOW,
        "position_batch_size": args.position_batch_size,
        "position_batch_policy": "min(requested, floor(1022^2 / min(sequence_length, 1022)^2))",
        "seed_policy": "global seed 0; per-chain seed from context_sha256",
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "scoring_strategy": "S3F masked-marginal sum over substitution components",
        "multichain_protocol": "sum independent mutated-chain component scores",
        "long_sequence_protocol": "official mutation-centered 1022-residue window",
        "amino_acids": MODEL["amino_acids"],
        "cache_format_version": MODEL["cache_format_version"],
        "cache_namespace": "candidate_chain_s3f_v1",
    }
    cache_config_sha256 = sha256_json(cache_config)
    cache_root = args.cache_dir / cache_config_sha256[:16]
    run_config = {
        **cache_config,
        "cache_config_sha256": cache_config_sha256,
        "output_schema_sha256": sha256_file(OUTPUT_SCHEMA),
    }

    if args.context_id:
        selected_contexts = sorted(set(args.context_id))
        unknown = set(selected_contexts) - set(contexts["context_id"])
        if unknown:
            raise ValueError(f"unknown S3F candidate contexts: {sorted(unknown)}")
        selected = contexts[contexts["context_id"].isin(selected_contexts)]
        estimated_load = sum(
            min(int(row.sequence_length), MODEL_WINDOW) * int(row.mutation_position_count)
            for row in selected.itertuples(index=False)
        )
        partial_run = True
    else:
        shards, loads = balanced_shards(contexts, args.num_shards)
        selected_contexts = shards[args.shard_id]
        estimated_load = loads[args.shard_id]
        partial_run = False

    context_by_id = contexts.set_index("context_id")
    surface_paths = {}
    surface_dirs = {}
    for context_id in selected_contexts:
        surface_paths[context_id], surface_dirs[context_id] = candidate_surface(
            context_by_id.loc[context_id], args.surface_dir
        )

    scorer = S3FFitnessScorer(
        script_path=args.s3f_script,
        checkpoint_path=args.checkpoint,
        esm_model_dir=args.esm_model_dir,
        structure_pdb_dir=None,
        surface_pkl_dir=None,
        surface_cache_dir=None,
        device="cuda",
        structure_start=1,
        plddt_threshold=MODEL["plddt_threshold"],
    )
    started_at = utc_now()
    frames = []
    computed_contexts = reused_contexts = 0
    try:
        for context_id in selected_contexts:
            manifest = context_by_id.loc[context_id]
            context = components[components["context_sha256"].eq(manifest["context_sha256"])]
            sequence = str(manifest["sequence"])
            positions = np.asarray(
                sorted(context["chain_position"].astype(int).unique()), dtype=np.int32
            )
            pdb_path = resolve_project_path(manifest["pdb_path"])
            pdb_sha256 = sha256_file(pdb_path)
            cache_path = cache_root / f"{context_id}.npz"
            metadata_path = cache_root / f"{context_id}.json"
            scorer.surface_pkl_path = surface_paths[context_id]
            if cache_path.exists():
                if not metadata_path.is_file():
                    raise FileNotFoundError(f"{context_id}: score-cache metadata is missing")
                scores = read_cache(
                    cache_path,
                    cache_config_sha256,
                    str(manifest["context_sha256"]),
                    pdb_sha256,
                    positions,
                    sequence,
                )
                validate_context_metadata(metadata_path, cache_path, surface_dirs[context_id])
                reused_contexts += 1
                source = "reused"
            else:
                seed = int(str(manifest["context_sha256"])[:8], 16)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                scores = compute_score_matrix(
                    scorer, sequence, positions, pdb_path, args.position_batch_size
                )
                artifacts = surface_artifacts(surface_dirs[context_id])
                atomic_save_cache(
                    cache_path,
                    cache_config_sha256,
                    str(manifest["context_sha256"]),
                    pdb_sha256,
                    positions,
                    sequence,
                    scores,
                )
                atomic_write_json(
                    {
                        "context_id": context_id,
                        "context_sha256": str(manifest["context_sha256"]),
                        "pdb_path": str(pdb_path.resolve()),
                        "pdb_sha256": pdb_sha256,
                        "score_cache": str(cache_path.resolve()),
                        "score_cache_sha256": sha256_file(cache_path),
                        "surface_artifacts": artifacts,
                    },
                    metadata_path,
                )
                computed_contexts += 1
                source = "computed"
            frames.append(score_context(context, positions, scores))
            print(
                f"{context_id}: length={len(sequence)} positions={len(positions)} cache={source}",
                flush=True,
            )
            scorer.teardown()
            gc.collect()
    finally:
        scorer.teardown()

    results = pd.concat(frames, ignore_index=True).sort_values("sample_id")
    atomic_write_csv(results, args.output_dir / "shards" / f"shard_{args.shard_id:04d}.csv")
    atomic_write_json(
        {
            "model_id": MODEL["model_id"],
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
            "computed_contexts": computed_contexts,
            "reused_contexts": reused_contexts,
            "input_dir": str(args.input_dir.resolve()),
            "cache_root": str(cache_root.resolve()),
            "partial_run": partial_run,
        },
        args.output_dir / "shards" / f"shard_{args.shard_id:04d}.json",
    )
    print(f"Shard {args.shard_id}: {len(results)} samples")


if __name__ == "__main__":
    main()
