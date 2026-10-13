#!/usr/bin/env python3
"""Shared VenusREM model-validation utilities."""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

from common_io import sha256_file, sha256_files


MODEL_METADATA = json.loads((SCRIPT_DIR / "model.json").read_text())
DEFAULT_ADAPTER_ROOT = PROJECT_ROOT / "model_adapters"


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
