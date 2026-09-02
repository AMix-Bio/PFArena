#!/usr/bin/env python3
"""Precompute ProGen2-base bidirectional likelihoods for single mutants."""

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
CACHE_FORMAT_VERSION = MODEL_METADATA["cache_format_version"]
METRIC_NAMES = [
    "progen2_score",
    "bidirectional_mean_nll",
    "forward_mean_nll",
    "reverse_mean_nll",
    "forward_nll_sum",
    "reverse_nll_sum",
    "forward_chunk_mean_nll_sum",
    "reverse_chunk_mean_nll_sum",
    "forward_token_count",
    "reverse_token_count",
    "chunk_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--proenv-root", type=Path, default=DEFAULT_PROENV_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=SCRIPT_DIR / "cache")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-block-size", type=int, default=2048)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def checkpoint_files(model_dir: Path) -> list[Path]:
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = model_dir / index_name
        if index_path.exists():
            filenames = sorted(
                set(json.loads(index_path.read_text())["weight_map"].values())
            )
            return [index_path, *(model_dir / name for name in filenames)]
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        files = sorted(model_dir.glob("pytorch_model*.bin"))
    if not files:
        raise FileNotFoundError(f"no model weights found in {model_dir}")
    return files


def validate_model(model_dir: Path, proenv_root: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing checkpoint config: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "progen":
        raise ValueError("checkpoint model_type is not progen")

    tokenizer_path = proenv_root / MODEL_METADATA["tokenizer_file"]
    configuration_path = (
        proenv_root / "proenv/models/progen/configuration_progen.py"
    )
    modeling_path = proenv_root / "proenv/models/progen/modeling_progen.py"
    expected_hashes = {
        tokenizer_path: MODEL_METADATA["tokenizer_sha256"],
        configuration_path: MODEL_METADATA["configuration_code_sha256"],
        modeling_path: MODEL_METADATA["modeling_code_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"deployed ProGen file differs from model.json: {path}")

    weights = checkpoint_files(model_dir)
    actual = {
        "config": config,
        "config_sha256": sha256_file(config_path),
        "weights_sha256": sha256_files(weights),
        "weights_size_bytes": sum(path.stat().st_size for path in weights),
        "weight_files": [path.name for path in weights],
        "tokenizer_path": tokenizer_path,
        "configuration_path": configuration_path,
        "modeling_path": modeling_path,
    }
    for key in ("config_sha256", "weights_sha256", "weights_size_bytes"):
        if actual[key] != MODEL_METADATA[key]:
            raise ValueError(f"checkpoint {key} does not match model.json")
    if actual["weight_files"] != MODEL_METADATA["weight_files"]:
        raise ValueError("checkpoint weight files do not match model.json")
    return actual


def validate_dataset(
    dataset_dir: Path, dataset: dict[str, object], proteins: pd.DataFrame
) -> None:
    required = {
        "assay",
        "wt_id",
        "sequence_sha256",
        "wildtype_sequence",
        "sequence_length",
    }
    if dataset.get("schema_version") != 2:
        raise ValueError("dataset schema_version must be 2")
    if required - set(proteins.columns):
        raise ValueError("proteins.csv does not match the canonical schema")
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

    seen: dict[str, str] = {}
    for row in proteins.itertuples(index=False):
        sequence_hash = hashlib.sha256(row.wildtype_sequence.encode()).hexdigest()
        if row.sequence_sha256 != sequence_hash or row.wt_id != sequence_hash[:16]:
            raise ValueError(f"{row.assay}: WT identity mismatch")
        if int(row.sequence_length) != len(row.wildtype_sequence):
            raise ValueError(f"{row.assay}: WT length mismatch")
        if seen.setdefault(row.wt_id, sequence_hash) != sequence_hash:
            raise ValueError(f"{row.wt_id}: truncated WT hash collision")


def atomic_save_cache(
    path: Path,
    cache_config_sha256: str,
    sequence_sha256: str,
    keys: list[str],
    values: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cache_format_version=np.int32(CACHE_FORMAT_VERSION),
            cache_config_sha256=np.asarray(cache_config_sha256),
            sequence_sha256=np.asarray(sequence_sha256),
            keys=np.asarray(keys),
            **values,
        )
    os.replace(temporary, path)


def load_cache(
    path: Path, cache_config_sha256: str, sequence_sha256: str
) -> tuple[list[str], dict[str, np.ndarray]]:
    if not path.exists():
        return [], {
            name: np.empty(
                0,
                dtype=np.int32
                if name.endswith("token_count") or name == "chunk_count"
                else np.float64,
            )
            for name in METRIC_NAMES
        }
    with np.load(path) as cache:
        if int(cache["cache_format_version"].item()) != CACHE_FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported cache format")
        if str(cache["cache_config_sha256"].item()) != cache_config_sha256:
            raise ValueError(f"{path}: inference configuration differs")
        if str(cache["sequence_sha256"].item()) != sequence_sha256:
            raise ValueError(f"{path}: WT sequence differs")
        keys = cache["keys"].astype(str).tolist()
        values = {name: cache[name].copy() for name in METRIC_NAMES}
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate cache keys")
    if any(value.shape != (len(keys),) for value in values.values()):
        raise ValueError(f"{path}: invalid cache shape")
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError(f"{path}: non-finite cached value")
    return keys, values


def mutate_sequence(sequence: str, row: object) -> str:
    position = int(row.position)
    if sequence[position - 1] != row.wt_aa:
        raise ValueError(f"{row.model_mutant}: wild-type residue mismatch")
    return sequence[: position - 1] + row.mut_aa + sequence[position:]


def score_sequences(
    model,
    tokenizer,
    sequences: list[str],
    context_length: int,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F

    terminalized = [
        MODEL_METADATA["terminal_prefix"] + sequence + MODEL_METADATA["terminal_suffix"]
        for sequence in sequences
    ]
    lengths = {len(sequence) for sequence in terminalized}
    if len(lengths) != 1:
        raise ValueError("a scoring batch must contain equal-length sequences")
    total_length = lengths.pop()
    chunks = [
        [sequence[start : start + context_length] for sequence in terminalized]
        for start in range(0, total_length, context_length)
    ]
    if any(len(chunk[0]) < 2 for chunk in chunks):
        raise ValueError("context chunk is too short to score")

    size = len(sequences)
    directional_sum = {
        "forward": np.zeros(size, dtype=np.float64),
        "reverse": np.zeros(size, dtype=np.float64),
    }
    directional_count = {
        "forward": np.zeros(size, dtype=np.int32),
        "reverse": np.zeros(size, dtype=np.int32),
    }
    directional_chunk_mean = {
        "forward": np.zeros(size, dtype=np.float64),
        "reverse": np.zeros(size, dtype=np.float64),
    }

    first_token = MODEL_METADATA["scored_token_id_first"]
    last_token = MODEL_METADATA["scored_token_id_last"]
    for chunk in chunks:
        for direction in ("forward", "reverse"):
            texts = chunk if direction == "forward" else [text[::-1] for text in chunk]
            for offset in range(0, size, batch_size):
                batch = texts[offset : offset + batch_size]
                encoded = [tokenizer.encode(text).ids for text in batch]
                if any(len(ids) != len(text) for ids, text in zip(encoded, batch)):
                    raise ValueError("tokenizer did not produce one token per character")
                ids = torch.tensor(encoded, dtype=torch.long, device=device)
                targets = ids[:, 1:]
                with torch.inference_mode():
                    logits = model(ids[:, :-1], use_cache=False).logits

                terminal = (targets[:, -1] == 3) | (targets[:, -1] == 4)
                if terminal.any():
                    if not terminal.all():
                        raise ValueError("inconsistent terminal tokens within a batch")
                    targets = targets[:, :-1]
                    logits = logits[:, :-1]
                if ((targets == 3) | (targets == 4)).any():
                    raise ValueError("terminal token remained in likelihood targets")

                logits = logits[:, :, first_token : last_token + 1]
                targets = targets - first_token
                if targets.min() < 0 or targets.max() > last_token - first_token:
                    raise ValueError("sequence contains an unsupported ProGen token")
                token_losses = F.cross_entropy(
                    logits.transpose(1, 2), targets, reduction="none"
                )
                token_sums = token_losses.sum(dim=1).double().cpu().numpy()
                chunk_means = token_losses.mean(dim=1).double().cpu().numpy()
                token_count = token_losses.shape[1]
                stop = offset + len(batch)
                directional_sum[direction][offset:stop] += token_sums
                directional_count[direction][offset:stop] += token_count
                directional_chunk_mean[direction][offset:stop] += (
                    chunk_means
                )

    forward_mean = directional_sum["forward"] / directional_count["forward"]
    reverse_mean = directional_sum["reverse"] / directional_count["reverse"]
    return {
        "progen2_score": -0.5
        * (
            directional_chunk_mean["forward"]
            + directional_chunk_mean["reverse"]
        )
        / total_length,
        "bidirectional_mean_nll": 0.5 * (forward_mean + reverse_mean),
        "forward_mean_nll": forward_mean,
        "reverse_mean_nll": reverse_mean,
        "forward_nll_sum": directional_sum["forward"],
        "reverse_nll_sum": directional_sum["reverse"],
        "forward_chunk_mean_nll_sum": directional_chunk_mean["forward"],
        "reverse_chunk_mean_nll_sum": directional_chunk_mean["reverse"],
        "forward_token_count": directional_count["forward"],
        "reverse_token_count": directional_count["reverse"],
        "chunk_count": np.full(size, len(chunks), dtype=np.int32),
    }


def score_wild_type(
    sequence: str,
    mutations: pd.DataFrame,
    model,
    tokenizer,
    context_length: int,
    batch_size: int,
    cache_block_size: int,
    cache_path: Path,
    cache_config_sha256: str,
    sequence_sha256: str,
) -> tuple[dict[str, dict[str, float]], int, int]:
    unique = mutations.drop_duplicates("model_mutant").set_index("model_mutant")
    required_keys = ["__WT__"] + sorted(unique.index.astype(str))
    keys, values = load_cache(cache_path, cache_config_sha256, sequence_sha256)
    cached_keys = set(keys)
    missing = [key for key in required_keys if key not in cached_keys]

    for start in range(0, len(missing), cache_block_size):
        block_keys = missing[start : start + cache_block_size]
        block_sequences = [
            sequence
            if key == "__WT__"
            else mutate_sequence(sequence, unique.loc[key])
            for key in block_keys
        ]
        block_values = score_sequences(
            model, tokenizer, block_sequences, context_length, batch_size, model.device
        )
        keys.extend(block_keys)
        for name in METRIC_NAMES:
            values[name] = np.concatenate([values[name], block_values[name]])
        atomic_save_cache(
            cache_path, cache_config_sha256, sequence_sha256, keys, values
        )
        print(
            f"{cache_path.stem}: cached {min(start + len(block_keys), len(missing))}/"
            f"{len(missing)} new sequences",
            flush=True,
        )

    index = {key: i for i, key in enumerate(keys)}
    scores = {
        key: {name: values[name][index[key]].item() for name in METRIC_NAMES}
        for key in required_keys
    }
    return scores, len(required_keys) - len(missing), len(missing)


def balanced_wt_shards(
    proteins: pd.DataFrame,
    dataset_dir: Path,
    context_length: int,
    num_shards: int,
) -> tuple[list[list[str]], list[int]]:
    work = []
    for wt_id, group in proteins.groupby("wt_id", sort=True):
        mutations = pd.concat(
            [
                pd.read_csv(
                    dataset_dir / "mutations" / f"{assay}.csv",
                    usecols=["model_mutant"],
                )
                for assay in group["assay"]
            ],
            ignore_index=True,
        )
        length = int(group["sequence_length"].iloc[0]) + 2
        chunk_lengths = [
            min(context_length, length - start)
            for start in range(0, length, context_length)
        ]
        cost = (mutations["model_mutant"].nunique() + 1) * sum(
            chunk_length**2 for chunk_length in chunk_lengths
        )
        work.append((cost, wt_id))

    shards = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for cost, wt_id in sorted(work, key=lambda item: (-item[0], item[1])):
        shard_id = min(range(num_shards), key=lambda i: (loads[i], i))
        shards[shard_id].append(wt_id)
        loads[shard_id] += cost
    return shards, loads


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    if args.batch_size < 1 or args.cache_block_size < 1:
        raise ValueError("batch sizes must be positive")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.proenv_root))

    import tokenizers
    import torch
    import transformers
    from proenv.models.progen import ProGenForCausalLM
    from tokenizers import Tokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dataset = json.loads((args.dataset_dir / "dataset.json").read_text())
    proteins = pd.read_csv(args.dataset_dir / "proteins.csv")
    validate_dataset(args.dataset_dir, dataset, proteins)
    if args.num_shards > proteins["wt_id"].nunique():
        raise ValueError("num-shards exceeds the number of unique WT sequences")

    validation = validate_model(args.model_dir, args.proenv_root)
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

    shards, estimated_loads = balanced_wt_shards(
        proteins, args.dataset_dir, context_length, args.num_shards
    )
    selected_wt_ids = shards[args.shard_id]
    output_schema_path = SCRIPT_DIR / "output_schema.json"
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
            for key in (
                "vocab_size",
                "n_positions",
                "n_embd",
                "n_layer",
                "n_head",
                "rotary_dim",
            )
        },
        "tokenizer_sha256": MODEL_METADATA["tokenizer_sha256"],
        "configuration_code_sha256": MODEL_METADATA["configuration_code_sha256"],
        "modeling_code_sha256": MODEL_METADATA["modeling_code_sha256"],
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
        "scored_token_ids": [
            MODEL_METADATA["scored_token_id_first"],
            MODEL_METADATA["scored_token_id_last"],
        ],
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
    started_at = utc_now()
    completed_assays = 0
    completed_samples = 0
    required_sequences = 0
    computed_sequences = 0
    reused_sequences = 0

    for wt_id in selected_wt_ids:
        protein_rows = proteins[proteins["wt_id"] == wt_id]
        sequences = protein_rows["wildtype_sequence"].drop_duplicates()
        if len(sequences) != 1:
            raise ValueError(f"{wt_id}: inconsistent WT sequences")
        sequence = sequences.iloc[0]
        sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
        mutations = pd.concat(
            [
                pd.read_csv(args.dataset_dir / "mutations" / f"{assay}.csv")
                for assay in protein_rows["assay"]
            ],
            ignore_index=True,
        )
        scores, reused, computed = score_wild_type(
            sequence,
            mutations,
            model,
            tokenizer,
            context_length,
            args.batch_size,
            args.cache_block_size,
            cache_root / f"{wt_id}.npz",
            cache_config_sha256,
            sequence_sha256,
        )
        required_sequences += len(scores)
        reused_sequences += reused
        computed_sequences += computed
        print(
            f"{wt_id}: sequences={len(scores)} computed={computed} reused={reused}",
            flush=True,
        )

        atomic_write_json(
            {
                "wt_id": wt_id,
                "sequence_sha256": sequence_sha256,
                "sequence_length": len(sequence),
                **scores["__WT__"],
            },
            args.output_dir / "wildtypes" / f"{wt_id}.json",
        )
        key_index = mutations["model_mutant"].map(scores)
        if key_index.isna().any():
            raise ValueError(f"{wt_id}: missing mutant likelihood")
        output = mutations[["sample_id", "assay", "mutant"]].copy()
        output["status"] = "ok"
        for name in METRIC_NAMES:
            output[name] = key_index.map(lambda item, metric=name: item[metric])

        for assay, assay_scores in output.groupby("assay", sort=True):
            atomic_write_csv(
                assay_scores.sort_values("sample_id"),
                args.output_dir / "assays" / f"{assay}.csv",
            )
            completed_assays += 1
            completed_samples += len(assay_scores)
            print(f"{assay}: {len(assay_scores)}", flush=True)

    run_config = {
        **cache_config,
        "cache_config_sha256": cache_config_sha256,
        "output_schema_sha256": sha256_file(output_schema_path),
    }
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
            "wt_sequences": len(selected_wt_ids),
            "estimated_attention_work": estimated_loads[args.shard_id],
            "assays": completed_assays,
            "samples": completed_samples,
            "model_dir": str(args.model_dir.resolve()),
            "proenv_root": str(args.proenv_root.resolve()),
            "cache_dir": str(args.cache_dir.resolve()),
            "cache_root": str(cache_root.resolve()),
            "required_sequences": required_sequences,
            "computed_sequences": computed_sequences,
            "reused_sequences": reused_sequences,
            "partial_run": False,
        },
        args.output_dir / "shards" / f"shard_{args.shard_id:04d}.json",
    )
    print(f"Shard {args.shard_id}: {completed_samples} samples", flush=True)


if __name__ == "__main__":
    main()
