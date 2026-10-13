#!/usr/bin/env python3
"""Shared ProGen2-base loading, caching, and likelihood utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

from common_io import sha256_file, sha256_files


DEFAULT_ADAPTER_ROOT = PROJECT_ROOT / "model_adapters"
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


def validate_model(model_dir: Path, adapter_root: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing checkpoint config: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "progen":
        raise ValueError("checkpoint model_type is not progen")

    tokenizer_path = adapter_root / MODEL_METADATA["tokenizer_file"]
    configuration_path = (
        adapter_root / "pfarena_models/models/progen/configuration_progen.py"
    )
    modeling_path = adapter_root / "pfarena_models/models/progen/modeling_progen.py"
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
