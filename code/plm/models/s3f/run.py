#!/usr/bin/env python3
"""Precompute S3F score matrices and single-mutant scores."""

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

from common_io import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    sha256_files,
    sha256_json,
    utc_now,
)


MODEL = json.loads((SCRIPT_DIR / "model.json").read_text())
AMINO_ACIDS = list(MODEL["amino_acids"])
MODEL_WINDOW = 1022
DEFAULT_PROENV_ROOT = PROJECT_ROOT / "ProEnv"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--surface-dir", type=Path, required=True)
    parser.add_argument("--proenv-root", type=Path, default=DEFAULT_PROENV_ROOT)
    parser.add_argument("--s3f-script", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--esm-model-dir", type=Path, required=True)
    parser.add_argument("--position-batch-size", type=int, default=16)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--wt-id")
    parser.add_argument("--partial-run", action="store_true")
    return parser.parse_args()


def validate_dataset(
    dataset_dir: Path, dataset: dict[str, object], proteins: pd.DataFrame
) -> None:
    if dataset.get("schema_version") != 2:
        raise ValueError("dataset schema_version must be 2")
    files = [
        dataset_dir / "proteins.csv",
        dataset_dir / "deferred_samples.csv",
        dataset_dir / "deferred_assays.csv",
    ]
    files.extend(
        dataset_dir / "mutations" / f"{assay}.csv" for assay in proteins["assay"]
    )
    if sha256_files(files) != dataset["dataset_hash"]:
        raise ValueError("dataset files do not match dataset.json")
    if len(proteins) != dataset["ready_assays"]:
        raise ValueError("protein assay count does not match dataset.json")
    if int(proteins["expected_samples"].sum()) != dataset["ready_samples"]:
        raise ValueError("protein sample count does not match dataset.json")
    if proteins["wt_id"].nunique() != dataset["unique_wildtype_sequences"]:
        raise ValueError("unique WT count does not match dataset.json")


def validate_inputs(
    input_dir: Path,
    dataset: dict[str, object],
    proteins: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    inputs = json.loads((input_dir / "inputs.json").read_text())
    contexts = pd.read_csv(input_dir / "contexts.csv")
    if inputs.get("schema_version") != 1 or inputs.get("model_id") != "S3F":
        raise ValueError("invalid S3F input manifest")
    if inputs["dataset_hash"] != dataset["dataset_hash"]:
        raise ValueError("S3F inputs target a different dataset")
    if inputs["dataset_id"] != dataset["dataset_id"]:
        raise ValueError("S3F input dataset ID differs")
    if sha256_file(input_dir / "contexts.csv") != inputs["context_manifest_sha256"]:
        raise ValueError("S3F context manifest changed")
    if contexts["wt_id"].duplicated().any():
        raise ValueError("S3F contexts contain duplicate WT entries")
    if set(contexts["wt_id"]) != set(proteins["wt_id"]):
        raise ValueError("S3F context WT set differs from the dataset")
    if len(contexts) != inputs["contexts"]:
        raise ValueError("S3F context count does not match inputs.json")
    if int(contexts["assay_count"].sum()) != inputs["assays"]:
        raise ValueError("S3F assay count does not match inputs.json")
    if int(contexts["sample_count"].sum()) != inputs["samples"]:
        raise ValueError("S3F sample count does not match inputs.json")
    if int(contexts["mutation_position_count"].sum()) != inputs["mutation_positions"]:
        raise ValueError("S3F position count does not match inputs.json")
    pdb_stems = contexts["pdb_path"].map(lambda value: Path(value).stem)
    if not pdb_stems.eq(contexts["wt_id"]).all():
        raise ValueError("S3F PDB names do not match their WT IDs")
    expected = proteins[
        ["wt_id", "sequence_sha256", "sequence_length"]
    ].drop_duplicates()
    checked = expected.merge(
        contexts[["wt_id", "sequence_sha256", "sequence_length"]],
        on="wt_id",
        suffixes=("_dataset", "_context"),
        validate="one_to_one",
    )
    for field in ("sequence_sha256", "sequence_length"):
        if not checked[f"{field}_dataset"].eq(checked[f"{field}_context"]).all():
            raise ValueError(f"S3F context {field} differs from the dataset")
    for row in contexts.itertuples(index=False):
        pdb_path = Path(row.pdb_path)
        if sha256_file(pdb_path) != row.pdb_sha256:
            raise ValueError(f"{row.wt_id}: S3F PDB differs from its manifest")
    return inputs, contexts


def required_positions(dataset_dir: Path, protein_rows: pd.DataFrame) -> np.ndarray:
    positions: set[int] = set()
    for assay in protein_rows["assay"]:
        mutations = pd.read_csv(
            dataset_dir / "mutations" / f"{assay}.csv", usecols=["position"]
        )
        positions.update(mutations["position"].astype(int))
    return np.asarray(sorted(positions), dtype=np.int32)


def effective_position_batch_size(sequence_length: int, requested: int) -> int:
    inference_length = min(sequence_length, MODEL_WINDOW)
    memory_limited = MODEL_WINDOW**2 // inference_length**2
    return min(requested, max(1, memory_limited))


def atomic_save_cache(
    path: Path,
    cache_config_sha256: str,
    sequence_sha256: str,
    pdb_sha256: str,
    positions: np.ndarray,
    sequence: str,
    scores: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cache_format_version=np.int32(MODEL["cache_format_version"]),
            cache_config_sha256=np.asarray(cache_config_sha256),
            sequence_sha256=np.asarray(sequence_sha256),
            pdb_sha256=np.asarray(pdb_sha256),
            sequence_length=np.int32(len(sequence)),
            positions=positions,
            amino_acids=np.asarray(AMINO_ACIDS),
            score_matrix=scores.astype(np.float32),
        )
    os.replace(temporary, path)


def read_cache(
    path: Path,
    cache_config_sha256: str,
    sequence_sha256: str,
    pdb_sha256: str,
    positions: np.ndarray,
    sequence: str,
) -> np.ndarray:
    with np.load(path) as cache:
        if int(cache["cache_format_version"].item()) != MODEL["cache_format_version"]:
            raise ValueError(f"{path}: unsupported cache format")
        if str(cache["cache_config_sha256"].item()) != cache_config_sha256:
            raise ValueError(f"{path}: inference configuration differs")
        if str(cache["sequence_sha256"].item()) != sequence_sha256:
            raise ValueError(f"{path}: WT sequence differs")
        if str(cache["pdb_sha256"].item()) != pdb_sha256:
            raise ValueError(f"{path}: WT structure differs")
        if int(cache["sequence_length"].item()) != len(sequence):
            raise ValueError(f"{path}: WT length differs")
        if not np.array_equal(cache["positions"].astype(np.int32), positions):
            raise ValueError(f"{path}: required positions differ")
        if cache["amino_acids"].astype(str).tolist() != AMINO_ACIDS:
            raise ValueError(f"{path}: amino-acid order differs")
        scores = cache["score_matrix"].astype(np.float32)
    if scores.shape != (len(positions), len(AMINO_ACIDS)):
        raise ValueError(f"{path}: score matrix shape differs")
    if not np.isfinite(scores).all():
        raise ValueError(f"{path}: non-finite S3F scores")
    wt_columns = np.asarray([AMINO_ACIDS.index(sequence[position - 1]) for position in positions])
    if not np.array_equal(scores[np.arange(len(positions)), wt_columns], np.zeros(len(positions), dtype=np.float32)):
        raise ValueError(f"{path}: WT scores are not zero")
    return scores


def surface_artifacts(surface_dir: Path) -> list[dict[str, object]]:
    files = sorted(path for path in surface_dir.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"S3F did not persist a surface graph in {surface_dir}")
    return [
        {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def validate_precomputed_surface(
    surface_dir: Path,
    wt_id: str,
    sequence_sha256: str,
    sequence_length: int,
    pdb_sha256: str,
) -> Path:
    surface_path = surface_dir / f"{wt_id}.pkl"
    metadata_path = surface_dir / f"{wt_id}.json"
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "wt_id": wt_id,
        "sequence_sha256": sequence_sha256,
        "sequence_length": sequence_length,
        "pdb_sha256": pdb_sha256,
        "surface_sha256": sha256_file(surface_path),
        "s3f_script_sha256": MODEL["s3f_script_sha256"],
        "preprocessor_sha256": sha256_file(SCRIPT_DIR / "prepare_surfaces.py"),
        "curvature_implementation": "chunked_pytorch_equivalent_v1",
        "curvature_chunk_size": 512,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"{surface_dir}: surface {field} differs")
    return surface_path


def validate_context_metadata(path: Path, cache_path: Path, surface_dir: Path) -> None:
    metadata = json.loads(path.read_text())
    if metadata["score_cache_sha256"] != sha256_file(cache_path):
        raise ValueError(f"{path}: S3F score cache changed")
    expected = metadata["surface_artifacts"]
    observed = surface_artifacts(surface_dir)
    if observed != expected:
        raise ValueError(f"{path}: S3F surface artifacts changed")


def compute_score_matrix(
    scorer,
    sequence: str,
    positions: np.ndarray,
    pdb_path: Path,
    position_batch_size: int,
) -> np.ndarray:
    batch_size = effective_position_batch_size(len(sequence), position_batch_size)
    print(
        f"sequence_length={len(sequence)} position_batch_size={batch_size}",
        flush=True,
    )
    scores = np.zeros((len(positions), len(AMINO_ACIDS)), dtype=np.float32)
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        variants: list[str] = []
        matrix_indices: list[tuple[int, int]] = []
        for row, position in enumerate(batch_positions, start=start):
            wt_aa = sequence[position - 1]
            for column, mut_aa in enumerate(AMINO_ACIDS):
                if mut_aa == wt_aa:
                    continue
                variants.append(f"{wt_aa}{position}{mut_aa}")
                matrix_indices.append((row, column))
        outputs = scorer.score_batch(sequence, variants, pdb_file=str(pdb_path))
        if len(outputs) != len(variants):
            raise ValueError("S3F returned an unexpected number of scores")
        for variant, (row, column), output in zip(variants, matrix_indices, outputs):
            if not output.get("valid") or output.get("fitness_score") is None:
                raise RuntimeError(f"S3F failed for {variant}: {output.get('error', '')}")
            value = float(output["fitness_score"])
            if not np.isfinite(value):
                raise ValueError(f"S3F returned a non-finite score for {variant}")
            scores[row, column] = value
        print(
            f"positions {start + 1}-{start + len(batch_positions)} / {len(positions)}",
            flush=True,
        )
    return scores


def score_assay(
    mutations: pd.DataFrame,
    sequence: str,
    positions: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    position_index = {int(position): index for index, position in enumerate(positions)}
    amino_acid_index = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
    rows = mutations["position"].map(position_index)
    columns = mutations["mut_aa"].map(amino_acid_index)
    if rows.isna().any() or columns.isna().any():
        raise ValueError("assay mutation is absent from the S3F cache")
    sequence_rows = mutations["position"].to_numpy(dtype=np.int64) - 1
    if (sequence_rows < 0).any() or (sequence_rows >= len(sequence)).any():
        raise ValueError("mutation position is outside the WT sequence")
    if not np.array_equal(
        mutations["wt_aa"].to_numpy(dtype=str), np.asarray(list(sequence))[sequence_rows]
    ):
        raise ValueError("mutation WT residue differs from the WT sequence")
    output = mutations[["sample_id", "assay", "mutant"]].copy()
    output["status"] = "ok"
    output["s3f_score"] = scores[
        rows.to_numpy(dtype=np.int64), columns.to_numpy(dtype=np.int64)
    ]
    return output


def balanced_shards(contexts: pd.DataFrame, num_shards: int) -> list[list[str]]:
    items = [
        (int(row.sequence_length) * int(row.mutation_position_count), row.wt_id)
        for row in contexts.itertuples(index=False)
    ]
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for cost, wt_id in sorted(items, key=lambda item: (-item[0], item[1])):
        shard = min(range(num_shards), key=lambda index: (loads[index], index))
        shards[shard].append(wt_id)
        loads[shard] += cost
    return shards


def main() -> None:
    args = parse_args()
    if args.position_batch_size < 1:
        raise ValueError("position-batch-size must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.proenv_root))
    import torch
    from proenv.metrics.sequence.s3f_de_fitness import S3FFitnessScorer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset = json.loads((args.dataset_dir / "dataset.json").read_text())
    proteins = pd.read_csv(args.dataset_dir / "proteins.csv")
    validate_dataset(args.dataset_dir, dataset, proteins)
    inputs, contexts = validate_inputs(args.input_dir, dataset, proteins)
    if args.num_shards > proteins["wt_id"].nunique():
        raise ValueError("num-shards exceeds the number of unique WT sequences")

    scorer_path = args.proenv_root / "proenv/metrics/sequence/s3f_de_fitness.py"
    s3f_code_files = sorted(args.s3f_script.parent.rglob("*.py"))
    esm_files = sorted(path for path in args.esm_model_dir.rglob("*") if path.is_file())
    if not s3f_code_files or not esm_files:
        raise FileNotFoundError("S3F code or ESM model files are missing")
    model_files = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "s3f_script_sha256": sha256_file(args.s3f_script),
        "s3f_code_sha256": sha256_files(s3f_code_files),
        "esm_model_sha256": sha256_files(esm_files),
        "proenv_scorer_sha256": sha256_file(scorer_path),
    }
    for field, observed in model_files.items():
        if observed != MODEL[field]:
            raise ValueError(f"deployed S3F {field} differs from model.json")
    cache_config = {
        "model_id": MODEL["model_id"],
        "method": MODEL["method"],
        "checkpoint": str(args.checkpoint.resolve()),
        "esm_model_dir": str(args.esm_model_dir.resolve()),
        **model_files,
        "runner_sha256": sha256_file(Path(__file__)),
        "surface_preprocessor_sha256": sha256_file(SCRIPT_DIR / "prepare_surfaces.py"),
        "input_hash": inputs["input_hash"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "gpu": torch.cuda.get_device_name(0),
        "device": "cuda",
        "structure_start": 1,
        "plddt_threshold": MODEL["plddt_threshold"],
        "esm_max_input_length": MODEL_WINDOW,
        "position_batch_size": args.position_batch_size,
        "position_batch_policy": "min(requested, floor(1022^2 / min(sequence_length, 1022)^2))",
        "seed_policy": "global seed 0; per-WT seed from the first 8 hex digits of sequence_sha256",
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "scoring_strategy": MODEL["scoring_strategy"],
        "amino_acids": MODEL["amino_acids"],
        "cache_format_version": MODEL["cache_format_version"],
        "surface_source": "precomputed from the benchmark AF3 WT PDB",
    }
    cache_config_sha256 = sha256_json(cache_config)
    output_schema = SCRIPT_DIR / "output_schema.json"
    run_config = {
        **cache_config,
        "cache_config_sha256": cache_config_sha256,
        "output_schema_sha256": sha256_file(output_schema),
    }

    if args.wt_id:
        if args.wt_id not in set(proteins["wt_id"]):
            raise ValueError(f"unknown wt-id: {args.wt_id}")
        selected_wt_ids = [args.wt_id]
    else:
        selected_wt_ids = balanced_shards(contexts, args.num_shards)[args.shard_id]
    context_by_wt = contexts.set_index("wt_id")
    surface_paths = {}
    for wt_id in selected_wt_ids:
        context = context_by_wt.loc[wt_id]
        surface_paths[wt_id] = validate_precomputed_surface(
            args.surface_dir / wt_id,
            wt_id,
            str(context["sequence_sha256"]),
            int(context["sequence_length"]),
            sha256_file(Path(context["pdb_path"])),
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
    assay_count = 0
    sample_count = 0
    computed_contexts = 0
    reused_contexts = 0
    for wt_id in selected_wt_ids:
        protein_rows = proteins[proteins["wt_id"] == wt_id]
        sequences = protein_rows["wildtype_sequence"].drop_duplicates()
        if len(sequences) != 1:
            raise ValueError(f"{wt_id}: inconsistent WT sequences")
        sequence = sequences.iloc[0]
        sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
        context = context_by_wt.loc[wt_id]
        if sequence_sha256 != context["sequence_sha256"]:
            raise ValueError(f"{wt_id}: context sequence differs from the dataset")
        pdb_path = Path(context["pdb_path"])
        pdb_sha256 = sha256_file(pdb_path)
        positions = required_positions(args.dataset_dir, protein_rows)
        if len(positions) != int(context["mutation_position_count"]):
            raise ValueError(f"{wt_id}: required position count differs")
        if int(protein_rows["expected_samples"].sum()) != int(context["sample_count"]):
            raise ValueError(f"{wt_id}: sample count differs")
        cache_path = args.cache_dir / f"{wt_id}.npz"
        context_metadata_path = args.cache_dir / f"{wt_id}.json"
        surface_context_dir = args.surface_dir / wt_id
        scorer.surface_pkl_path = surface_paths[wt_id]
        if cache_path.exists():
            if not context_metadata_path.is_file():
                raise FileNotFoundError(f"missing S3F context metadata: {context_metadata_path}")
            scores = read_cache(
                cache_path,
                cache_config_sha256,
                sequence_sha256,
                pdb_sha256,
                positions,
                sequence,
            )
            validate_context_metadata(
                context_metadata_path, cache_path, surface_context_dir
            )
            reused_contexts += 1
            source = "reused"
        else:
            context_seed = int(sequence_sha256[:8], 16)
            random.seed(context_seed)
            np.random.seed(context_seed)
            torch.manual_seed(context_seed)
            torch.cuda.manual_seed_all(context_seed)
            scores = compute_score_matrix(
                scorer,
                sequence,
                positions,
                pdb_path,
                args.position_batch_size,
            )
            artifacts = surface_artifacts(surface_context_dir)
            atomic_save_cache(
                cache_path,
                cache_config_sha256,
                sequence_sha256,
                pdb_sha256,
                positions,
                sequence,
                scores,
            )
            atomic_write_json(
                {
                    "wt_id": wt_id,
                    "sequence_sha256": sequence_sha256,
                    "pdb_path": str(pdb_path.resolve()),
                    "pdb_sha256": pdb_sha256,
                    "score_cache": str(cache_path.resolve()),
                    "score_cache_sha256": sha256_file(cache_path),
                    "surface_artifacts": artifacts,
                },
                context_metadata_path,
            )
            computed_contexts += 1
            source = "computed"
        print(
            f"{wt_id}: length={len(sequence)} positions={len(positions)} cache={source}",
            flush=True,
        )
        for protein in protein_rows.itertuples(index=False):
            mutations = pd.read_csv(
                args.dataset_dir / "mutations" / f"{protein.assay}.csv"
            )
            output = score_assay(mutations, sequence, positions, scores)
            atomic_write_csv(
                output.sort_values("sample_id"),
                args.output_dir / "assays" / f"{protein.assay}.csv",
            )
            assay_count += 1
            sample_count += len(output)
        scorer.teardown()
        gc.collect()

    shard_metadata = {
        "model_id": MODEL["model_id"],
        "run_config": run_config,
        "run_config_sha256": sha256_json(run_config),
        "dataset_id": dataset["dataset_id"],
        "dataset_hash": dataset["dataset_hash"],
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "wt_sequences": len(selected_wt_ids),
        "assays": assay_count,
        "samples": sample_count,
        "computed_contexts": computed_contexts,
        "reused_contexts": reused_contexts,
        "input_dir": str(args.input_dir.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "surface_dir": str(args.surface_dir.resolve()),
        "partial_run": args.partial_run or bool(args.wt_id),
    }
    atomic_write_json(
        shard_metadata,
        args.output_dir / "shards" / f"shard_{args.shard_id:04d}.json",
    )
    print(f"Shard {args.shard_id}: {sample_count} samples")


if __name__ == "__main__":
    main()
