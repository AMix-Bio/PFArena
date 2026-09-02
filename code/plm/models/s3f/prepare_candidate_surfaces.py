#!/usr/bin/env python3
"""Precompute and validate S3F surface graphs for chain-level inputs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CURVATURE_CHUNK_SIZE = 512
LEGACY_PREPROCESSOR_SHA256 = "cc546fa1d13558143b23c86cdbf9e133e16664cc8f9b4e4a0ff98f9dfad7c5f8"
sys.path.insert(0, str(PROJECT_ROOT))

from common_io import atomic_write_json, sha256_file, sha256_files
from prepare_surfaces import install_chunked_curvature, load_s3f_module, validate_surface
from run import MODEL


CONTEXT_COLUMNS = [
    "context_id", "context_sha256", "sequence", "sequence_length",
    "candidate_sample_count", "substitution_component_count",
    "mutation_position_count", "structure_qc", "structure_source", "pdb_path",
    "source_pdb_path", "pdb_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--surface-dir", type=Path, required=True)
    parser.add_argument("--s3f-script", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--context-id")
    parser.add_argument(
        "--all-contexts",
        action="store_true",
        help="process every input context",
    )
    return parser.parse_args()


def load_inputs(input_dir: Path) -> tuple[dict, pd.DataFrame]:
    inputs = json.loads((input_dir / "inputs.json").read_text())
    contexts = pd.read_csv(input_dir / "contexts.csv")
    if (
        inputs.get("schema_version") != 1
        or inputs.get("model_id") != "S3F"
        or not inputs.get("dataset_id")
        or not inputs.get("dataset_hash")
    ):
        raise ValueError("invalid S3F candidate input manifest")
    if contexts.columns.tolist() != CONTEXT_COLUMNS:
        raise ValueError("S3F candidate context columns differ")
    if sha256_file(input_dir / "contexts.csv") != inputs["context_manifest_sha256"]:
        raise ValueError("S3F candidate context manifest changed")
    if sha256_files([input_dir / "contexts.csv"]) != inputs["input_hash"]:
        raise ValueError("S3F candidate input hash differs")
    if (
        len(contexts) != inputs["contexts"]
        or contexts["context_id"].duplicated().any()
        or contexts["context_sha256"].duplicated().any()
        or int(contexts["candidate_sample_count"].sum()) != inputs["samples"]
        or int(contexts["substitution_component_count"].sum())
        != inputs["substitution_components"]
        or int(contexts["mutation_position_count"].sum()) != inputs["mutation_positions"]
    ):
        raise ValueError("S3F candidate input counts differ")
    for context in contexts.itertuples(index=False):
        if (
            context.context_id != context.context_sha256[:16]
            or len(context.sequence) != int(context.sequence_length)
            or sha256_file(resolve_project_path(context.pdb_path)) != context.pdb_sha256
        ):
            raise ValueError(f"{context.context_id}: invalid S3F candidate context")
    return inputs, contexts


def balanced_shards(contexts: pd.DataFrame, num_shards: int) -> list[list[str]]:
    items = [
        (int(row.sequence_length) ** 2, row.context_id)
        for row in contexts.itertuples(index=False)
    ]
    shards = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for cost, context_id in sorted(items, key=lambda item: (-item[0], item[1])):
        shard = min(range(num_shards), key=lambda index: (loads[index], index))
        shards[shard].append(context_id)
        loads[shard] += cost
    return shards


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_existing(
    surface_path: Path,
    metadata_path: Path,
    context: pd.Series,
    s3f_script: Path,
) -> dict[str, int]:
    if not surface_path.is_file() or not metadata_path.is_file():
        raise FileExistsError(f"incomplete candidate surface cache: {surface_path.parent}")
    summary = validate_surface(surface_path, int(context["sequence_length"]))
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "context_id": str(context.name),
        "context_sha256": str(context["context_sha256"]),
        "sequence_length": int(context["sequence_length"]),
        "pdb_sha256": str(context["pdb_sha256"]),
        "surface_sha256": sha256_file(surface_path),
        "s3f_script_sha256": sha256_file(s3f_script),
        "curvature_implementation": "chunked_pytorch_equivalent_v1",
        "curvature_chunk_size": CURVATURE_CHUNK_SIZE,
        **summary,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"{surface_path.parent}: cached surface {field} differs")
    if metadata.get("preprocessor_sha256") not in {
        LEGACY_PREPROCESSOR_SHA256,
        sha256_file(Path(__file__)),
    }:
        raise ValueError(f"{surface_path.parent}: cached surface preprocessor differs")
    return summary


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    _, contexts = load_inputs(args.input_dir)
    if sha256_file(args.s3f_script) != MODEL["s3f_script_sha256"]:
        raise ValueError("deployed S3F script differs from model.json")
    selected_contexts = contexts
    if args.context_id:
        if args.context_id not in set(selected_contexts["context_id"]):
            raise ValueError(f"unknown S3F context: {args.context_id}")
        selected = [args.context_id]
    else:
        if args.num_shards > len(selected_contexts):
            raise ValueError("num-shards exceeds the number of S3F contexts")
        selected = balanced_shards(selected_contexts, args.num_shards)[args.shard_id]

    import torch
    from torch.nn import functional as F

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    module = load_s3f_module(args.s3f_script)
    install_chunked_curvature()
    context_by_id = selected_contexts.set_index("context_id")
    for context_id in selected:
        context = context_by_id.loc[context_id]
        pdb_path = resolve_project_path(context["pdb_path"])
        output_dir = args.surface_dir / context_id
        surface_path = output_dir / f"{context_id}.pkl"
        metadata_path = output_dir / f"{context_id}.json"
        if surface_path.exists() or metadata_path.exists():
            summary = validate_existing(surface_path, metadata_path, context, args.s3f_script)
            print(f"{context_id}: {summary['surface_points']} surface points (reused)", flush=True)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)

        seed = int(str(context["context_sha256"])[:8], 16)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        temporary = output_dir / f".{context_id}.{os.getpid()}.tmp.pkl"
        try:
            module._write_surface_from_pdb(str(pdb_path), str(temporary), torch, F, device)
            summary = validate_surface(temporary, int(context["sequence_length"]))
            os.replace(temporary, surface_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        atomic_write_json(
            {
                "context_id": context_id,
                "context_sha256": str(context["context_sha256"]),
                "sequence_length": int(context["sequence_length"]),
                "pdb_path": str(pdb_path.resolve()),
                "pdb_sha256": str(context["pdb_sha256"]),
                "surface_path": str(surface_path.resolve()),
                "surface_sha256": sha256_file(surface_path),
                "s3f_script_sha256": sha256_file(args.s3f_script),
                "preprocessor_sha256": sha256_file(Path(__file__)),
                "curvature_implementation": "chunked_pytorch_equivalent_v1",
                "curvature_chunk_size": CURVATURE_CHUNK_SIZE,
                "seed": seed,
                "device": str(device),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                **summary,
            },
            metadata_path,
        )
        print(f"{context_id}: {summary['surface_points']} surface points", flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
