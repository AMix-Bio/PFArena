#!/usr/bin/env python3
"""Precompute ProSST-2048 WT-context probabilities and single-mutant scores."""

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


MODEL_METADATA = json.loads((SCRIPT_DIR / "model.json").read_text())
DEFAULT_PROENV_ROOT = PROJECT_ROOT / "ProEnv"
CACHE_FORMAT_VERSION = MODEL_METADATA["cache_format_version"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proenv-root", type=Path, default=DEFAULT_PROENV_ROOT)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prosst-repo-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=SCRIPT_DIR / "cache")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--structure-batch-size", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--partial-run", action="store_true")
    return parser.parse_args()


def validate_model_dir(model_dir: Path) -> dict[str, object]:
    config_sha256 = sha256_file(model_dir / "config.json")
    if config_sha256 != MODEL_METADATA["config_sha256"]:
        raise ValueError("model config does not match the registered checkpoint")

    tokenizer_sha256 = sha256_files(
        [model_dir / name for name in MODEL_METADATA["tokenizer_files"]]
    )
    if tokenizer_sha256 != MODEL_METADATA["tokenizer_sha256"]:
        raise ValueError("model tokenizer does not match the registered checkpoint")

    custom_code_sha256 = sha256_files(
        [model_dir / name for name in MODEL_METADATA["custom_model_files"]]
    )
    if custom_code_sha256 != MODEL_METADATA["custom_model_code_sha256"]:
        raise ValueError("custom model code does not match the registered checkpoint")

    weights = model_dir / MODEL_METADATA["weights_file"]
    if weights.stat().st_size != MODEL_METADATA["weights_size_bytes"]:
        raise ValueError("model weights size does not match the registered checkpoint")
    weights_sha256 = sha256_file(weights)
    if weights_sha256 != MODEL_METADATA["weights_sha256"]:
        raise ValueError("model weights do not match the registered checkpoint")
    return {
        "config_sha256": config_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "custom_model_code_sha256": custom_code_sha256,
        "weights_sha256": weights_sha256,
        "weights_size_bytes": weights.stat().st_size,
    }


def validate_dataset(
    dataset_dir: Path, dataset: dict[str, object], proteins: pd.DataFrame
) -> None:
    if dataset.get("schema_version") != 2:
        raise ValueError("dataset schema_version must be 2")
    if proteins["assay"].duplicated().any():
        raise ValueError("proteins.csv contains duplicate assays")
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


def validate_structure_manifest(
    manifest: pd.DataFrame, dataset: dict[str, object], proteins: pd.DataFrame
) -> None:
    required = {
        "dataset_hash",
        "wt_id",
        "sequence_sha256",
        "sequence_length",
        "structure_status",
        "pdb_path",
        "pdb_sha256",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"structure manifest is missing columns: {sorted(missing)}")
    if manifest["wt_id"].duplicated().any():
        raise ValueError("structure manifest contains duplicate WT entries")
    if set(manifest["dataset_hash"]) != {dataset["dataset_hash"]}:
        raise ValueError("structure manifest targets a different dataset")
    if not manifest["structure_status"].eq("ready").all():
        raise ValueError("structure manifest contains unavailable structures")

    expected = proteins[
        ["wt_id", "sequence_sha256", "sequence_length"]
    ].drop_duplicates()
    checked = expected.merge(
        manifest[["wt_id", "sequence_sha256", "sequence_length"]],
        on="wt_id",
        how="outer",
        suffixes=("_dataset", "_structure"),
        indicator=True,
        validate="one_to_one",
    )
    if not checked["_merge"].eq("both").all():
        raise ValueError("structure manifest WT set differs from the dataset")
    for field in ("sequence_sha256", "sequence_length"):
        if not checked[f"{field}_dataset"].eq(checked[f"{field}_structure"]).all():
            raise ValueError(f"structure manifest {field} differs from the dataset")


def validate_prosst_repo(prosst_repo_dir: Path) -> dict[str, object]:
    structure_root = prosst_repo_dir / "prosst/structure"
    encoder_weights = structure_root / "static/AE.pt"
    cluster_file = structure_root / "static/2048.joblib"
    code_files = sorted(structure_root.rglob("*.py"))
    if not code_files:
        raise FileNotFoundError(f"ProSST structure code is missing: {structure_root}")
    return {
        "structure_encoder_weights_sha256": sha256_file(encoder_weights),
        "structure_cluster_sha256": sha256_file(cluster_file),
        "structure_code_sha256": sha256_files(code_files),
    }


def atomic_save_cache(
    path: Path,
    cache_config_sha256: str,
    sequence_sha256: str,
    pdb_sha256: str,
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
            sequence_sha256=np.asarray(sequence_sha256),
            pdb_sha256=np.asarray(pdb_sha256),
            sequence_length=np.int32(len(structure_tokens)),
            positions=np.arange(1, len(structure_tokens) + 1, dtype=np.int32),
            structure_tokens=structure_tokens,
            tokens=tokens,
            token_ids=token_ids,
            log_probs=log_probs,
        )
    os.replace(temporary, path)


def context_log_probs(
    scorer,
    sequence: str,
    sequence_sha256: str,
    pdb_path: Path,
    pdb_sha256: str,
    cache_path: Path,
    cache_config_sha256: str,
) -> tuple[np.ndarray, list[str], bool]:
    tokens = [
        scorer.tokenizer.convert_ids_to_tokens(token_id)
        for token_id in range(len(scorer.tokenizer))
    ]
    token_ids = np.arange(len(tokens), dtype=np.int32)
    encoded = scorer.tokenizer(sequence, add_special_tokens=True)
    input_ids = encoded["input_ids"]
    if len(input_ids) != len(sequence) + 2:
        raise ValueError("tokenizer did not produce one token per residue plus BOS/EOS")
    residue_tokens = scorer.tokenizer.convert_ids_to_tokens(input_ids[1:-1])
    if residue_tokens != list(sequence):
        raise ValueError("tokenizer residue mapping differs from the WT sequence")

    if cache_path.exists():
        with np.load(cache_path) as cache:
            if int(cache["cache_format_version"].item()) != CACHE_FORMAT_VERSION:
                raise ValueError(f"{cache_path}: unsupported cache format")
            if str(cache["cache_config_sha256"].item()) != cache_config_sha256:
                raise ValueError(f"{cache_path}: inference configuration differs")
            if str(cache["sequence_sha256"].item()) != sequence_sha256:
                raise ValueError(f"{cache_path}: WT sequence hash differs")
            if str(cache["pdb_sha256"].item()) != pdb_sha256:
                raise ValueError(f"{cache_path}: WT structure hash differs")
            if int(cache["sequence_length"].item()) != len(sequence):
                raise ValueError(f"{cache_path}: WT sequence length differs")
            if cache["tokens"].astype(str).tolist() != tokens:
                raise ValueError(f"{cache_path}: tokenizer vocabulary differs")
            if not np.array_equal(cache["token_ids"].astype(np.int32), token_ids):
                raise ValueError(f"{cache_path}: tokenizer IDs differ")
            positions = cache["positions"].astype(np.int32)
            structure_tokens = cache["structure_tokens"].astype(np.int32)
            log_probs = cache["log_probs"].astype(np.float32)
        if not np.array_equal(positions, np.arange(1, len(sequence) + 1)):
            raise ValueError(f"{cache_path}: residue positions differ")
        if structure_tokens.shape != (len(sequence),):
            raise ValueError(f"{cache_path}: structure token shape differs")
        if (structure_tokens < 0).any() or (
            structure_tokens >= MODEL_METADATA["structure_vocab_size"]
        ).any():
            raise ValueError(f"{cache_path}: invalid structure tokens")
        if log_probs.shape != (len(sequence), len(tokens)):
            raise ValueError(f"{cache_path}: log-probability shape differs")
        if not np.isfinite(log_probs).all():
            raise ValueError(f"{cache_path}: non-finite log probabilities")
        if not np.allclose(np.logaddexp.reduce(log_probs, axis=1), 0.0, atol=1e-5):
            raise ValueError(f"{cache_path}: log probabilities are not normalized")
        return log_probs, tokens, True

    pdb_sequence, structure_tokens = scorer._structure_tokens_from_pdb(pdb_path)
    if pdb_sequence != sequence:
        raise ValueError(f"{pdb_path}: PDB-derived sequence differs from the WT")
    if len(structure_tokens) != len(sequence) or not all(
        0 <= token < MODEL_METADATA["structure_vocab_size"]
        for token in structure_tokens
    ):
        raise ValueError(f"{pdb_path}: invalid ProSST structure tokens")
    log_probs = (
        scorer._log_probs(sequence, structure_tokens)[0].float().numpy().astype(np.float32)
    )
    if log_probs.shape != (len(sequence), len(tokens)):
        raise ValueError("model returned an unexpected log-probability shape")
    if not np.isfinite(log_probs).all():
        raise ValueError("model returned non-finite log probabilities")
    if not np.allclose(np.logaddexp.reduce(log_probs, axis=1), 0.0, atol=1e-5):
        raise ValueError("model returned unnormalized log probabilities")
    atomic_save_cache(
        cache_path,
        cache_config_sha256,
        sequence_sha256,
        pdb_sha256,
        np.asarray(structure_tokens, dtype=np.int32),
        np.asarray(tokens),
        token_ids,
        log_probs,
    )
    return log_probs, tokens, False


def score_mutations(
    mutations: pd.DataFrame,
    sequence: str,
    log_probs: np.ndarray,
    tokens: list[str],
) -> pd.DataFrame:
    rows = mutations["position"].to_numpy(dtype=np.int64) - 1
    if (rows < 0).any() or (rows >= len(sequence)).any():
        raise ValueError("mutation position is outside the WT sequence")
    if not np.array_equal(
        mutations["wt_aa"].to_numpy(dtype=str), np.asarray(list(sequence))[rows]
    ):
        raise ValueError("mutation WT residue differs from the WT sequence")
    token_index = {token: index for index, token in enumerate(tokens)}
    wt_columns = mutations["wt_aa"].map(token_index)
    mutant_columns = mutations["mut_aa"].map(token_index)
    if wt_columns.isna().any() or mutant_columns.isna().any():
        raise ValueError("mutation contains a residue absent from the tokenizer")
    wt_logp = log_probs[rows, wt_columns.to_numpy(dtype=np.int64)]
    mutant_logp = log_probs[rows, mutant_columns.to_numpy(dtype=np.int64)]

    output = mutations[["sample_id", "assay", "mutant"]].copy()
    output["status"] = "ok"
    output["prosst_score"] = mutant_logp - wt_logp
    output["mutant_logp"] = mutant_logp
    output["wt_logp"] = wt_logp
    return output


def balanced_wt_shards(
    proteins: pd.DataFrame, num_shards: int
) -> tuple[list[list[str]], list[int]]:
    unique_wt = proteins[["wt_id", "sequence_length"]].drop_duplicates()
    work_items = [
        (int(row.sequence_length) ** 2, row.wt_id)
        for row in unique_wt.itertuples(index=False)
    ]
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for cost, wt_id in sorted(work_items, key=lambda item: (-item[0], item[1])):
        shard_id = min(range(num_shards), key=lambda index: (loads[index], index))
        shards[shard_id].append(wt_id)
        loads[shard_id] += cost
    return shards, loads


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    if args.structure_batch_size < 1:
        raise ValueError("structure-batch-size must be positive")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.proenv_root))

    import joblib
    import torch
    import transformers
    from proenv.metrics.sequence.prosst2048_de_fitness import ProSST2048FitnessScorer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dataset = json.loads((args.dataset_dir / "dataset.json").read_text())
    proteins = pd.read_csv(args.dataset_dir / "proteins.csv")
    validate_dataset(args.dataset_dir, dataset, proteins)
    structure_manifest = pd.read_csv(args.structure_manifest)
    validate_structure_manifest(structure_manifest, dataset, proteins)
    structure_manifest_sha256 = sha256_file(args.structure_manifest)
    if args.num_shards > proteins["wt_id"].nunique():
        raise ValueError("num-shards exceeds the number of unique WT sequences")

    model_validation = validate_model_dir(args.model_dir)
    prosst_validation = validate_prosst_repo(args.prosst_repo_dir)
    scorer_path = (
        args.proenv_root
        / "proenv/metrics/sequence/prosst2048_de_fitness.py"
    )
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
    max_sequence_length = int(proteins["sequence_length"].max())
    if max_sequence_length + 2 > tokenizer_max_length:
        raise ValueError("dataset contains a sequence longer than the tokenizer limit")

    output_schema_path = SCRIPT_DIR / "output_schema.json"
    gpu = torch.cuda.get_device_name(0) if scorer.device.type == "cuda" else None
    cache_config = {
        "model_id": MODEL_METADATA["model_id"],
        "checkpoint": MODEL_METADATA["checkpoint"],
        "checkpoint_revision": MODEL_METADATA["checkpoint_revision"],
        "model_config_sha256": model_validation["config_sha256"],
        "model_weights_sha256": model_validation["weights_sha256"],
        "model_weights_size_bytes": model_validation["weights_size_bytes"],
        "tokenizer_sha256": model_validation["tokenizer_sha256"],
        "custom_model_code_sha256": model_validation["custom_model_code_sha256"],
        "proenv_scorer_sha256": sha256_file(scorer_path),
        "runner_sha256": sha256_file(Path(__file__)),
        **prosst_validation,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "joblib": joblib.__version__,
        "device": str(scorer.device),
        "model_dtype": model_dtype,
        "gpu": gpu,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "structure_batch_size": args.structure_batch_size,
        "structure_vocab_size": MODEL_METADATA["structure_vocab_size"],
        "tokenizer_model_max_length": tokenizer_max_length,
        "model_max_relative_positions": int(
            scorer.model.config.max_relative_positions
        ),
        "scoring_strategy": MODEL_METADATA["scoring_strategy"],
        "cache_format_version": CACHE_FORMAT_VERSION,
        "cache_namespace": MODEL_METADATA["cache_namespace"],
    }
    cache_config_sha256 = sha256_json(cache_config)
    cache_root = (
        args.cache_dir
        / MODEL_METADATA["cache_namespace"]
        / cache_config_sha256[:16]
    )
    shards, estimated_loads = balanced_wt_shards(proteins, args.num_shards)
    selected_wt_ids = shards[args.shard_id]
    structures = structure_manifest.set_index("wt_id")
    started_at = utc_now()
    completed_assays = 0
    completed_samples = 0
    computed_contexts = 0
    reused_contexts = 0

    try:
        for wt_id in selected_wt_ids:
            protein_rows = proteins[proteins["wt_id"] == wt_id]
            sequences = protein_rows["wildtype_sequence"].drop_duplicates()
            if len(sequences) != 1:
                raise ValueError(f"{wt_id}: inconsistent WT sequences")
            sequence = sequences.iloc[0]
            sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
            structure = structures.loc[wt_id]
            pdb_path = Path(structure["pdb_path"])
            pdb_sha256 = sha256_file(pdb_path)
            if pdb_sha256 != structure["pdb_sha256"]:
                raise ValueError(f"{wt_id}: WT structure hash differs from the manifest")

            cache_path = cache_root / f"{wt_id}.npz"
            log_probs, tokens, reused = context_log_probs(
                scorer,
                sequence,
                sequence_sha256,
                pdb_path,
                pdb_sha256,
                cache_path,
                cache_config_sha256,
            )
            reused_contexts += int(reused)
            computed_contexts += int(not reused)
            print(
                f"{wt_id}: length={len(sequence)} "
                f"context={'reused' if reused else 'computed'}",
                flush=True,
            )

            for protein in protein_rows.itertuples(index=False):
                mutations = pd.read_csv(
                    args.dataset_dir / "mutations" / f"{protein.assay}.csv"
                )
                scored = score_mutations(mutations, sequence, log_probs, tokens)
                atomic_write_csv(
                    scored.sort_values("sample_id"),
                    args.output_dir / "assays" / f"{protein.assay}.csv",
                )
                completed_assays += 1
                completed_samples += len(scored)
                print(f"{protein.assay}: {len(scored)}", flush=True)
    finally:
        scorer.teardown()

    run_config = {
        **cache_config,
        "cache_config_sha256": cache_config_sha256,
        "structure_manifest": str(args.structure_manifest.resolve()),
        "structure_manifest_sha256": structure_manifest_sha256,
        "dataset_max_sequence_length": max_sequence_length,
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
        "estimated_quadratic_work": estimated_loads[args.shard_id],
        "assays": completed_assays,
        "samples": completed_samples,
        "computed_contexts": computed_contexts,
        "reused_contexts": reused_contexts,
        "model_dir": str(args.model_dir.resolve()),
        "prosst_repo_dir": str(args.prosst_repo_dir.resolve()),
        "proenv_root": str(args.proenv_root.resolve()),
        "cache_root": str(cache_root.resolve()),
        "partial_run": args.partial_run,
    }
    atomic_write_json(
        shard_metadata,
        args.output_dir / "shards" / f"shard_{args.shard_id:04d}.json",
    )
    print(f"Shard {args.shard_id}: {completed_samples} samples")


if __name__ == "__main__":
    main()
