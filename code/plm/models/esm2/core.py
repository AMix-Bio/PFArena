#!/usr/bin/env python3
"""Shared ESM-2 loading and masked-marginal utilities."""

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
LONG_SEQUENCE_PROTOCOLS = ("centered-special", "proteingym")
CACHE_FORMAT_VERSION = MODEL_METADATA["cache_format_version"]


def atomic_save_cache(
    path: Path,
    cache_config_sha256: str,
    sequence_sha256: str,
    sequence_length: int,
    positions: np.ndarray,
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
            sequence_length=np.int32(sequence_length),
            positions=positions,
            tokens=tokens,
            token_ids=token_ids,
            log_probs=log_probs,
        )
    os.replace(temporary, path)


def validate_model_dir(model_dir: Path) -> dict[str, object]:
    config_sha256 = sha256_file(model_dir / "config.json")
    if config_sha256 != MODEL_METADATA["config_sha256"]:
        raise ValueError("model config does not match the registered checkpoint")

    tokenizer_files = [
        model_dir / filename for filename in MODEL_METADATA["tokenizer_files"]
    ]
    tokenizer_sha256 = sha256_files(tokenizer_files)
    if tokenizer_sha256 != MODEL_METADATA["tokenizer_sha256"]:
        raise ValueError("model tokenizer does not match the registered checkpoint")

    weights = model_dir / MODEL_METADATA["weights_file"]
    if weights.stat().st_size != MODEL_METADATA["weights_size_bytes"]:
        raise ValueError("model weights size does not match the registered checkpoint")
    weights_sha256 = sha256_file(weights)
    if weights_sha256 != MODEL_METADATA["weights_sha256"]:
        raise ValueError("model weights do not match the registered checkpoint")
    return {
        "config_sha256": config_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "weights_sha256": weights_sha256,
        "weights_size_bytes": weights.stat().st_size,
    }


def centered_residue_window(
    sequence: str, position: int, max_window: int
) -> tuple[str, int]:
    if len(sequence) <= max_window:
        return sequence, position
    start = max(1, position - max_window // 2)
    end = min(len(sequence), start + max_window - 1)
    if end - start + 1 < max_window:
        start = end - max_window + 1
    return sequence[start - 1 : end], position - start + 1


def masked_inputs(
    scorer,
    sequence: str,
    positions: list[int],
    max_window: int,
    long_sequence_protocol: str,
) -> list[tuple[int, object, object, int]]:
    tokenizer = scorer.tokenizer
    max_tokens = max_window + 2
    full = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    full_ids = full["input_ids"][0]
    full_mask = full["attention_mask"][0]
    if len(full_ids) != len(sequence) + 2:
        raise ValueError("tokenizer did not produce one token per residue plus BOS/EOS")
    prepared = []

    for position in positions:
        if not 1 <= position <= len(sequence):
            raise ValueError(f"position {position} is outside the wild-type sequence")
        if len(sequence) <= max_window:
            input_ids = full_ids.clone()
            attention_mask = full_mask.clone()
            token_index = position
        elif long_sequence_protocol == "centered-special":
            window, relative_position = centered_residue_window(
                sequence, position, max_window
            )
            encoded = tokenizer(window, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"][0]
            attention_mask = encoded["attention_mask"][0]
            token_index = relative_position
        else:
            target_index = position
            half_window = max_tokens // 2
            if target_index < half_window:
                start = 0
            elif target_index >= len(full_ids) - half_window:
                start = len(full_ids) - max_tokens
            else:
                start = target_index - half_window
            input_ids = full_ids[start : start + max_tokens].clone()
            attention_mask = full_mask[start : start + max_tokens].clone()
            token_index = target_index - start

        expected_token = tokenizer.convert_tokens_to_ids(sequence[position - 1])
        if int(input_ids[token_index]) != expected_token:
            raise ValueError(f"token mismatch at position {position}")
        input_ids[token_index] = scorer.mask_token_id
        prepared.append((position, input_ids, attention_mask, token_index))
    return prepared


def precompute_full_log_probs(
    scorer,
    sequence: str,
    positions: list[int],
    max_window: int,
    long_sequence_protocol: str,
) -> np.ndarray:
    import torch

    prepared = masked_inputs(
        scorer, sequence, positions, max_window, long_sequence_protocol
    )
    rows = []
    for offset in range(0, len(prepared), scorer.batch_size):
        batch = prepared[offset : offset + scorer.batch_size]
        input_ids = torch.stack([item[1] for item in batch]).to(scorer.device)
        attention_mask = torch.stack([item[2] for item in batch]).to(scorer.device)
        with torch.inference_mode():
            logits = scorer.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        for row, (_, _, _, token_index) in enumerate(batch):
            rows.append(
                torch.log_softmax(logits[row, token_index], dim=-1)
                .float()
                .cpu()
                .numpy()
            )
    return np.asarray(rows, dtype=np.float32)


def position_log_probs(
    scorer,
    wt_sequence: str,
    sequence_sha256: str,
    required_positions: list[int],
    cache_path: Path,
    cache_config_sha256: str,
    max_window: int,
    long_sequence_protocol: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    tokens = [
        scorer.tokenizer.convert_ids_to_tokens(token_id)
        for token_id in range(len(scorer.tokenizer))
    ]
    if len(tokens) != scorer.model.config.vocab_size:
        raise ValueError("tokenizer and model vocabulary sizes differ")
    token_ids = np.arange(len(tokens), dtype=np.int32)
    positions = np.empty(0, dtype=np.int32)
    log_probs = np.empty((0, len(tokens)), dtype=np.float32)
    if cache_path.exists():
        with np.load(cache_path) as cache:
            if int(cache["cache_format_version"].item()) != CACHE_FORMAT_VERSION:
                raise ValueError(f"{cache_path}: unsupported cache format")
            if str(cache["cache_config_sha256"].item()) != cache_config_sha256:
                raise ValueError(f"{cache_path}: inference configuration differs")
            if str(cache["sequence_sha256"].item()) != sequence_sha256:
                raise ValueError(f"{cache_path}: WT sequence hash differs")
            if int(cache["sequence_length"].item()) != len(wt_sequence):
                raise ValueError(f"{cache_path}: WT sequence length differs")
            cached_tokens = cache["tokens"].astype(str).tolist()
            cached_token_ids = cache["token_ids"].astype(np.int32)
            if cached_tokens != tokens or not np.array_equal(
                cached_token_ids, token_ids
            ):
                raise ValueError(f"{cache_path}: tokenizer vocabulary differs")
            positions = cache["positions"].astype(np.int32)
            log_probs = cache["log_probs"].astype(np.float32)

        if positions.ndim != 1 or log_probs.shape != (len(positions), len(tokens)):
            raise ValueError(f"{cache_path}: invalid cache shape")
        if len(np.unique(positions)) != len(positions):
            raise ValueError(f"{cache_path}: duplicate cached positions")
        if len(positions) and not (
            positions.min() >= 1 and positions.max() <= len(wt_sequence)
        ):
            raise ValueError(f"{cache_path}: cached position is outside the WT")
        if not np.isfinite(log_probs).all():
            raise ValueError(f"{cache_path}: non-finite cached log probabilities")

    missing = sorted(set(required_positions) - set(positions.tolist()))
    cache_stats = {
        "required": len(set(required_positions)),
        "reused": len(set(required_positions) & set(positions.tolist())),
        "computed": len(missing),
    }
    if not missing:
        return positions, log_probs, tokens, cache_stats

    new_log_probs = precompute_full_log_probs(
        scorer,
        wt_sequence,
        missing,
        max_window,
        long_sequence_protocol,
    )
    if new_log_probs.shape != (len(missing), len(tokens)):
        raise ValueError("model returned an unexpected log-probability shape")
    if not np.isfinite(new_log_probs).all():
        raise ValueError("model returned non-finite log probabilities")
    positions = np.concatenate([positions, np.asarray(missing, dtype=np.int32)])
    log_probs = np.concatenate([log_probs, new_log_probs])
    order = np.argsort(positions)
    positions = positions[order]
    log_probs = log_probs[order]
    atomic_save_cache(
        cache_path,
        cache_config_sha256,
        sequence_sha256,
        len(wt_sequence),
        positions,
        np.asarray(tokens),
        token_ids,
        log_probs,
    )
    return positions, log_probs, tokens, cache_stats
