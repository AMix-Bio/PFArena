#!/usr/bin/env python3
"""Precompute VenusREM matrices and single-mutant scores."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proenv-root", type=Path, default=DEFAULT_PROENV_ROOT)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=SCRIPT_DIR / "cache")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
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
    dataset_dir: Path, dataset: dict, proteins: pd.DataFrame
) -> None:
    if dataset.get("schema_version") != 2:
        raise ValueError("dataset schema_version must be 2")
    required_columns = {
        "assay",
        "wt_id",
        "sequence_sha256",
        "wildtype_sequence",
        "sequence_length",
        "expected_samples",
    }
    missing = required_columns - set(proteins.columns)
    if missing:
        raise ValueError(f"proteins.csv missing columns: {sorted(missing)}")
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
    if sha256_files(paths) != dataset["dataset_hash"]:
        raise ValueError("dataset files do not match dataset.json")
    if len(proteins) != int(dataset["ready_assays"]):
        raise ValueError("protein assay count does not match dataset.json")
    if int(proteins["expected_samples"].sum()) != int(dataset["ready_samples"]):
        raise ValueError("protein sample count does not match dataset.json")
    if proteins["wt_id"].nunique() != int(dataset["unique_wildtype_sequences"]):
        raise ValueError("unique WT count does not match dataset.json")


def validate_inputs(
    input_dir: Path,
    inputs: dict,
    contexts: pd.DataFrame,
    dataset: dict,
    proteins: pd.DataFrame,
) -> None:
    if inputs.get("schema_version") != 2:
        raise ValueError("VenusREM input schema_version must be 2")
    if sha256_file(input_dir / "contexts.csv") != inputs["contexts_sha256"]:
        raise ValueError("contexts.csv does not match inputs.json")
    if inputs["dataset_hash"] != dataset["dataset_hash"]:
        raise ValueError("VenusREM inputs target a different dataset")
    if contexts["assay"].duplicated().any():
        raise ValueError("VenusREM contexts contain duplicate assays")
    if set(contexts["assay"]) != set(proteins["assay"]):
        raise ValueError("VenusREM context assays differ from the dataset")
    if not contexts["status"].eq("ready").all():
        raise ValueError("all VenusREM contexts must be ready")
    if len(contexts) != int(inputs["assays"]):
        raise ValueError("VenusREM context count does not match inputs.json")
    if int(contexts["expected_samples"].sum()) != int(inputs["samples"]):
        raise ValueError("VenusREM sample count does not match inputs.json")
    if contexts["sequence_sha256"].nunique() != int(
        inputs["unique_wildtype_sequences"]
    ):
        raise ValueError("VenusREM WT count does not match inputs.json")
    if contexts["context_sha256"].nunique() != int(inputs["ready_contexts"]):
        raise ValueError("VenusREM unique context count does not match inputs.json")

    checked = proteins[
        ["assay", "wt_id", "sequence_sha256", "sequence_length", "expected_samples"]
    ].merge(
        contexts[
            ["assay", "wt_id", "sequence_sha256", "sequence_length", "expected_samples"]
        ],
        on="assay",
        suffixes=("_dataset", "_context"),
        validate="one_to_one",
    )
    for field in ("wt_id", "sequence_sha256", "sequence_length", "expected_samples"):
        if not checked[f"{field}_dataset"].eq(checked[f"{field}_context"]).all():
            raise ValueError(f"VenusREM context {field} differs from the dataset")

    required = {
        "context_id",
        "context_sha256",
        "residue_fasta",
        "residue_sha256",
        "structure_fasta",
        "structure_sha256",
        "aa_alignment_file",
        "aa_alignment_sha256",
    }
    missing = required - set(contexts.columns)
    if missing:
        raise ValueError(f"VenusREM contexts missing columns: {sorted(missing)}")
    if any(contexts[field].eq("").any() for field in required):
        raise ValueError("VenusREM context is missing a resource field")


def balanced_context_shards(
    contexts: pd.DataFrame, num_shards: int
) -> tuple[list[list[str]], list[int]]:
    work_items = []
    for context_sha256, group in contexts.groupby("context_sha256", sort=True):
        alignment_bytes = int(group["aa_alignment_size_bytes"].iloc[0])
        sequence_length = int(group["sequence_length"].iloc[0])
        cost = alignment_bytes + sequence_length * sequence_length
        work_items.append((cost, context_sha256))

    shards: list[list[str]] = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for cost, context_sha256 in sorted(
        work_items, key=lambda item: (-item[0], item[1])
    ):
        shard_id = min(range(num_shards), key=lambda index: (loads[index], index))
        shards[shard_id].append(context_sha256)
        loads[shard_id] += cost
    return shards, loads


def atomic_save_cache(
    path: Path,
    cache_config_sha256: str,
    context_sha256: str,
    sequence_sha256: str,
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
            cache_format_version=np.int32(MODEL_METADATA["cache_format_version"]),
            cache_config_sha256=np.asarray(cache_config_sha256),
            context_sha256=np.asarray(context_sha256),
            sequence_sha256=np.asarray(sequence_sha256),
            sequence_length=np.int32(len(positions)),
            positions=positions,
            tokens=tokens,
            token_ids=token_ids,
            values=values,
        )
    os.replace(temporary, path)


def context_values(
    scorer,
    context_row,
    cache_path: Path,
    cache_config_sha256: str,
    tokens: list[str],
    token_ids: np.ndarray,
    protein_context_class,
) -> tuple[np.ndarray, bool]:
    resources = (
        (Path(context_row.residue_fasta), context_row.residue_sha256),
        (Path(context_row.structure_fasta), context_row.structure_sha256),
        (Path(context_row.aa_alignment_file), context_row.aa_alignment_sha256),
    )
    for path, expected_sha256 in resources:
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"VenusREM resource changed: {path}")

    sequence_length = int(context_row.sequence_length)
    positions = np.arange(1, sequence_length + 1, dtype=np.int32)
    if cache_path.exists():
        with np.load(cache_path) as cache:
            if int(cache["cache_format_version"].item()) != MODEL_METADATA[
                "cache_format_version"
            ]:
                raise ValueError(f"{cache_path}: unsupported cache format")
            if str(cache["cache_config_sha256"].item()) != cache_config_sha256:
                raise ValueError(f"{cache_path}: cache configuration differs")
            if str(cache["context_sha256"].item()) != context_row.context_sha256:
                raise ValueError(f"{cache_path}: context identity differs")
            if str(cache["sequence_sha256"].item()) != context_row.sequence_sha256:
                raise ValueError(f"{cache_path}: WT sequence differs")
            if int(cache["sequence_length"].item()) != sequence_length:
                raise ValueError(f"{cache_path}: WT length differs")
            if not np.array_equal(cache["positions"], positions):
                raise ValueError(f"{cache_path}: cached positions differ")
            if cache["tokens"].astype(str).tolist() != tokens:
                raise ValueError(f"{cache_path}: tokenizer vocabulary differs")
            if not np.array_equal(cache["token_ids"], token_ids):
                raise ValueError(f"{cache_path}: token IDs differ")
            values = cache["values"].astype(np.float32)
        if values.shape != (sequence_length, len(tokens)) or not np.isfinite(
            values
        ).all():
            raise ValueError(f"{cache_path}: invalid cached values")
        return values, True

    protein_context = protein_context_class(
        protein_name=context_row.context_id,
        residue_fasta=context_row.residue_fasta,
        structure_fasta=context_row.structure_fasta,
        aa_seq_aln_file=context_row.aa_alignment_file,
    )
    values = (
        scorer._precompute_logits(context_row.wildtype_sequence, protein_context)
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    if values.shape != (sequence_length, len(tokens)) or not np.isfinite(
        values
    ).all():
        raise ValueError("VenusREM returned invalid context values")
    atomic_save_cache(
        cache_path,
        cache_config_sha256,
        context_row.context_sha256,
        context_row.sequence_sha256,
        positions,
        np.asarray(tokens),
        token_ids,
        values,
    )
    return values, False


def score_mutations(
    mutations: pd.DataFrame,
    sequence: str,
    values: np.ndarray,
    vocab: dict[str, int],
) -> pd.DataFrame:
    rows = mutations["position"].astype(int).to_numpy() - 1
    if (rows < 0).any() or (rows >= len(sequence)).any():
        raise ValueError("mutation position is outside the WT sequence")
    if not np.array_equal(
        mutations["wt_aa"].to_numpy(dtype=str), np.asarray(list(sequence))[rows]
    ):
        raise ValueError("mutation WT residue differs from the WT sequence")
    wt_columns = mutations["wt_aa"].map(vocab)
    mutant_columns = mutations["mut_aa"].map(vocab)
    if wt_columns.isna().any() or mutant_columns.isna().any():
        raise ValueError("mutation contains a residue absent from the tokenizer")
    wt_columns = wt_columns.to_numpy(dtype=np.int64)
    mutant_columns = mutant_columns.to_numpy(dtype=np.int64)
    wt_values = values[rows, wt_columns]
    mutant_values = values[rows, mutant_columns]

    output = mutations[["sample_id", "assay", "mutant"]].copy()
    output["status"] = "ok"
    output["venusrem_score"] = mutant_values - wt_values
    output["mutant_value"] = mutant_values
    output["wt_value"] = wt_values
    return output


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")

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

    dataset = json.loads((args.dataset_dir / "dataset.json").read_text())
    inputs = json.loads((args.input_dir / "inputs.json").read_text())
    proteins = pd.read_csv(args.dataset_dir / "proteins.csv")
    contexts = pd.read_csv(args.input_dir / "contexts.csv", keep_default_na=False)
    validate_dataset(args.dataset_dir, dataset, proteins)
    validate_inputs(args.input_dir, inputs, contexts, dataset, proteins)
    model_validation = validate_model_dir(args.model_dir)

    ready = contexts.merge(
        proteins[["assay", "wildtype_sequence"]], on="assay", validate="one_to_one"
    )
    context_count = ready["context_sha256"].nunique()
    if context_count == 0:
        raise ValueError("VenusREM input manifest contains no ready contexts")
    if args.num_shards > context_count:
        raise ValueError("num-shards exceeds the number of ready VenusREM contexts")
    shards, estimated_loads = balanced_context_shards(ready, args.num_shards)
    selected_contexts = shards[args.shard_id]
    started_at = utc_now()

    scorer_path = (
        args.proenv_root / "proenv/metrics/sequence/venusrem_de_fitness.py"
    )
    output_schema_path = SCRIPT_DIR / "output_schema.json"
    runner_sha256 = sha256_file(Path(__file__))
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
    device_type = scorer.device.type
    gpu = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if device_type == "cuda"
        else None
    )
    tokens = [
        scorer.tokenizer.convert_ids_to_tokens(token_id)
        for token_id in range(len(scorer.tokenizer))
    ]
    token_ids = np.arange(len(tokens), dtype=np.int32)
    vocab = scorer.tokenizer.get_vocab()

    cache_config = {
        "model_id": MODEL_METADATA["model_id"],
        "checkpoint": MODEL_METADATA["checkpoint"],
        "checkpoint_revision": MODEL_METADATA["checkpoint_revision"],
        "model_config_sha256": model_validation["config_sha256"],
        "model_weights_sha256": model_validation["weights_sha256"],
        "tokenizer_sha256": model_validation["tokenizer_sha256"],
        "custom_model_code_sha256": model_validation["custom_model_code_sha256"],
        "proenv_scorer_sha256": scorer_sha256,
        "runner_sha256": runner_sha256,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "device": device_type,
        "gpu": gpu,
        "model_dtype": model_dtype,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "structure_vocab_size": MODEL_METADATA["structure_vocab_size"],
        "logit_mode": MODEL_METADATA["logit_mode"],
        "alpha": MODEL_METADATA["alpha"],
        "sample_ratio": MODEL_METADATA["sample_ratio"],
        "sample_times": MODEL_METADATA["sample_times"],
        "cache_format_version": MODEL_METADATA["cache_format_version"],
        "cache_namespace": MODEL_METADATA["cache_namespace"],
    }
    cache_config_sha256 = sha256_json(cache_config)
    cache_root = (
        args.cache_dir
        / MODEL_METADATA["cache_namespace"]
        / cache_config_sha256[:16]
    )

    completed_contexts = 0
    computed_contexts = 0
    reused_contexts = 0
    completed_assays = 0
    completed_samples = 0
    try:
        for context_sha256 in selected_contexts:
            context_rows = ready[ready["context_sha256"] == context_sha256]
            context_row = next(context_rows.itertuples(index=False))
            cache_path = cache_root / f"{context_sha256}.npz"
            values, reused = context_values(
                scorer,
                context_row,
                cache_path,
                cache_config_sha256,
                tokens,
                token_ids,
                VenusREMProteinContext,
            )
            completed_contexts += 1
            reused_contexts += int(reused)
            computed_contexts += int(not reused)
            print(
                f"{context_row.context_id}: length={context_row.sequence_length} "
                f"cache={'reused' if reused else 'computed'}",
                flush=True,
            )

            for assay in sorted(context_rows["assay"]):
                mutations = pd.read_csv(
                    args.dataset_dir / "mutations" / f"{assay}.csv"
                )
                scores = score_mutations(
                    mutations,
                    context_row.wildtype_sequence,
                    values,
                    vocab,
                )
                atomic_write_csv(
                    scores.sort_values("sample_id"),
                    args.output_dir / "assays" / f"{assay}.csv",
                )
                completed_assays += 1
                completed_samples += len(scores)
                print(f"{assay}: {len(scores)}", flush=True)
    finally:
        scorer.teardown()

    run_config = {
        **cache_config,
        "model_weights_size_bytes": model_validation["weights_size_bytes"],
        "output_schema_sha256": sha256_file(output_schema_path),
        "inputs_sha256": sha256_file(args.input_dir / "inputs.json"),
        "contexts_sha256": inputs["contexts_sha256"],
        "dataset_hash": dataset["dataset_hash"],
        "cache_config_sha256": cache_config_sha256,
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
        "contexts": completed_contexts,
        "computed_contexts": computed_contexts,
        "reused_contexts": reused_contexts,
        "estimated_work": estimated_loads[args.shard_id],
        "assays": completed_assays,
        "samples": completed_samples,
        "dataset_assays_total": len(proteins),
        "model_dir": str(args.model_dir.resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "partial_run": args.partial_run,
    }
    atomic_write_json(
        shard_metadata,
        args.output_dir / "shards" / f"shard_{args.shard_id:04d}.json",
    )
    print(f"Shard {args.shard_id}: {completed_samples} samples")


if __name__ == "__main__":
    main()
