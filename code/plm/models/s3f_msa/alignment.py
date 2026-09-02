"""Alignment conversion used by the EVE component of S3F-MSA."""

from __future__ import annotations

import os
import re
from pathlib import Path


ALPHABET = set("ACDEFGHIKLMNPQRSTVWY-")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = ""
    sequence: list[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(sequence)))
                header, sequence = line[1:], []
            else:
                if not header:
                    raise ValueError(f"{path}: sequence before first FASTA header")
                sequence.append(line)
    if header:
        records.append((header, "".join(sequence)))
    if not records:
        raise ValueError(f"{path}: empty alignment")
    return records


def convert_a3m(
    source: Path,
    target: Path,
    context_id: str,
    wt_sequence: str,
    focus_start: int = 1,
    source_wt_sequence: str | None = None,
) -> dict:
    converted: list[str] = []
    for index, (_, sequence) in enumerate(read_fasta(source)):
        aligned = re.sub(r"[a-z.]", "", sequence).upper()
        expected_source = source_wt_sequence or wt_sequence
        if len(aligned) != len(expected_source):
            raise ValueError(f"{context_id}: A3M row {index} has incompatible length")
        if source_wt_sequence is not None:
            aligned = aligned[focus_start - 1 : focus_start - 1 + len(wt_sequence)]
        converted.append(aligned)
    if converted[0] != wt_sequence:
        raise ValueError(f"{context_id}: first A3M record does not equal the WT")

    gap_fractions = [sequence.count("-") / len(wt_sequence) for sequence in converted]
    fragment_rows = [sequence for sequence, gap in zip(converted, gap_fractions) if gap <= 0.5]
    valid_rows = [sequence for sequence in fragment_rows if set(sequence) <= ALPHABET]
    if not valid_rows or valid_rows[0] != wt_sequence:
        raise ValueError(f"{context_id}: EVE preprocessing would remove the WT")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        focus_end = focus_start + len(wt_sequence) - 1
        for index, sequence in enumerate(converted):
            header = f"{context_id}/{focus_start}-{focus_end}" if index == 0 else f"seq_{index}"
            handle.write(f">{header}\n{sequence}\n")
    os.replace(temporary, target)
    return {
        "msa_depth_raw": len(converted),
        "msa_depth_after_fragment_filter": len(fragment_rows),
        "msa_depth_after_eve_filter": len(valid_rows),
    }
