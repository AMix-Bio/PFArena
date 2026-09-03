"""Small helpers for constructing VenusREM sequence, structure, and MSA inputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from common_io import sha256_json


MAX_RESIDUES = 2046


def inference_window(length: int, positions: list[int]) -> tuple[int, int]:
    if length <= MAX_RESIDUES:
        return 1, length
    if positions[-1] - positions[0] + 1 > MAX_RESIDUES:
        raise ValueError("required positions do not fit in one VenusREM input window")
    midpoint = (positions[0] + positions[-1]) // 2
    start = max(1, midpoint - (MAX_RESIDUES - 1) // 2)
    end = start + MAX_RESIDUES - 1
    if end > length:
        end = length
        start = end - MAX_RESIDUES + 1
    if start > positions[0] or end < positions[-1]:
        raise ValueError("VenusREM window does not cover every required position")
    return start, end


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def normalize_a3m(source: Path, output: Path, query: str, start: int, end: int) -> tuple[int, int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    records = lowercase_insertions = unknown_residues = 0
    sequence_parts: list[str] = []
    canonical = set("ACDEFGHIKLMNPQRSTVWY-.")
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ-.")

    def emit(handle) -> None:
        nonlocal records, lowercase_insertions, unknown_residues, sequence_parts
        raw = "".join(sequence_parts)
        lowercase_insertions += sum(character.islower() for character in raw)
        aligned = "".join(character for character in raw if not character.islower()).upper()
        if len(aligned) != len(query) or (records == 0 and aligned != query):
            raise ValueError(f"{source}: A3M query or columns differ from the chain")
        if set(aligned) - allowed:
            raise ValueError(f"{source}: unsupported A3M character")
        window = aligned[start - 1 : end]
        unknown_residues += sum(character not in canonical for character in window)
        header = f">query/1-{len(window)}" if records == 0 else f">sequence_{records}"
        handle.write(f"{header}\n{window}\n")
        records += 1
        sequence_parts = []

    try:
        with source.open() as source_handle, temporary.open("w") as output_handle:
            for line in source_handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if sequence_parts:
                        emit(output_handle)
                else:
                    sequence_parts.append(line)
            if sequence_parts:
                emit(output_handle)
        if records == 0:
            raise ValueError(f"{source}: empty A3M")
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return records, lowercase_insertions, unknown_residues


def load_prosst_cache_root(run_path: Path, dataset_hash: str) -> tuple[Path, dict]:
    run = json.loads(run_path.read_text())
    if run.get("dataset_hash") != dataset_hash:
        raise ValueError("ProSST run targets a different dataset")
    shards = run.get("shards", [])
    if not shards or sorted(int(shard["shard_id"]) for shard in shards) != list(range(int(shards[0]["num_shards"]))):
        raise ValueError("ProSST formal run has incomplete shards")
    roots = {Path(shard["cache_root"]) for shard in shards}
    config_hash = run.get("run_config_sha256")
    shard_config = shards[0].get("run_config")
    if (
        len(roots) != 1
        or any(shard.get("partial_run", False) for shard in shards)
        or not config_hash
        or {shard.get("run_config_sha256") for shard in shards} != {config_hash}
        or not shard_config
        or sha256_json(shard_config) != config_hash
        or any(shard.get("run_config") != shard_config for shard in shards[1:])
    ):
        raise ValueError("ProSST run provenance is inconsistent")
    run["run_config"] = shard_config
    return roots.pop(), run
