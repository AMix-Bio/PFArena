#!/usr/bin/env python3
"""Shared S3F caching and structure-aware scoring utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

from common_io import sha256_file


MODEL = json.loads((SCRIPT_DIR / "model.json").read_text())
AMINO_ACIDS = list(MODEL["amino_acids"])
MODEL_WINDOW = 1022
DEFAULT_ADAPTER_ROOT = PROJECT_ROOT / "model_adapters"


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
