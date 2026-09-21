#!/usr/bin/env python3
"""Load and validate a canonical frozen protein-model dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from common_io import sha256_files


DATA_FILES = (
    "proteins.csv",
    "samples.csv",
    "substitutions.csv",
    "queries.csv",
    "context_samples.csv",
    "context_substitutions.csv",
    "provided_context.csv",
    "msa_contexts.csv",
)


def resolve_dataset_path(dataset_dir: Path, value: str) -> Path:
    """Resolve an absolute or dataset-relative resource path."""
    path = Path(value)
    return path if path.is_absolute() else (dataset_dir / path).resolve()


def _validate_components(
    samples: pd.DataFrame,
    substitutions: pd.DataFrame,
    proteins: pd.DataFrame,
) -> None:
    protein_contexts = {}
    for protein in proteins.itertuples(index=False):
        chains = json.loads(protein.chain_sequences)
        boundaries = json.loads(protein.chain_boundaries)
        sequence = str(protein.wildtype_sequence)
        if hashlib.sha256(sequence.encode()).hexdigest() != protein.sequence_sha256:
            raise ValueError(f"{protein.wt_id}: WT SHA256 mismatch")
        if chains != sequence.split(":") or int(protein.n_chains) != len(chains):
            raise ValueError(f"{protein.wt_id}: invalid chain representation")
        expected_boundaries = []
        start = 1
        for chain in chains:
            expected_boundaries.append([start, start + len(chain) - 1])
            start += len(chain)
        if boundaries != expected_boundaries or int(protein.sequence_length) != start - 1:
            raise ValueError(f"{protein.wt_id}: invalid chain boundaries")
        protein_contexts[protein.sequence_sha256] = (chains, boundaries, "".join(chains))

    if samples["sample_id"].duplicated().any():
        raise ValueError("duplicate sample_id")
    if substitutions.duplicated(["sample_id", "component_index"]).any():
        raise ValueError("duplicate substitution component")
    counts = substitutions.groupby("sample_id").size()
    expected = samples.set_index("sample_id")["mutation_count"].astype(int)
    if set(counts.index) != set(expected.index) or not counts.reindex(expected.index).eq(expected).all():
        raise ValueError("mutation_count differs from substitution components")

    linked = substitutions.merge(
        samples[["sample_id", "sequence_sha256"]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    protein_lookup = proteins.set_index("sequence_sha256")
    boundaries = {
        key: json.loads(value)
        for key, value in protein_lookup["chain_boundaries"].items()
    }
    flat_sequences = {
        key: str(value).replace(":", "")
        for key, value in protein_lookup["wildtype_sequence"].items()
    }
    component_indices = substitutions["component_index"].astype(int)
    expected_indices = substitutions.groupby("sample_id", sort=False).cumcount() + 1
    if not component_indices.eq(expected_indices).all():
        raise ValueError("non-contiguous or unordered component indices")
    if substitutions.duplicated(["sample_id", "global_position"]).any():
        raise ValueError("duplicate substitution position")
    previous_positions = substitutions.groupby("sample_id", sort=False)["global_position"].shift()
    if substitutions["global_position"].astype(int).lt(previous_positions.fillna(0).astype(int)).any():
        raise ValueError("substitution positions are not ordered")
    for row in linked.itertuples(index=False):
        chain_id = int(row.chain_id)
        sequence_boundaries = boundaries[row.sequence_sha256]
        if not 1 <= chain_id <= len(sequence_boundaries):
            raise ValueError(f"{row.sample_id}: invalid chain ID")
        start, end = sequence_boundaries[chain_id - 1]
        position = int(row.global_position)
        flat_wt = flat_sequences[row.sequence_sha256]
        if (
            not start <= position <= end
            or int(row.chain_position) != position - start + 1
            or flat_wt[position - 1] != row.wt_aa
            or row.wt_aa == row.mut_aa
        ):
            raise ValueError(f"{row.sample_id}: invalid substitution mapping")



def load_dataset(
    dataset_dir: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = json.loads((dataset_dir / "dataset.json").read_text())
    if metadata.get("schema_version") != 4 or metadata.get("status") != "frozen":
        raise ValueError("expected frozen schema_version 4 dataset")
    paths = [dataset_dir / name for name in DATA_FILES]
    if sha256_files(paths) != metadata.get("dataset_hash"):
        raise ValueError("normalized files do not match dataset.json")
    proteins, samples, substitutions, queries, context_samples, context_substitutions, provided_context, msa = (
        pd.read_csv(path, keep_default_na=False) for path in paths
    )

    expected_counts = {
        "proteins": len(proteins),
        "samples": len(samples),
        "substitution_components": len(substitutions),
        "queries": len(queries),
        "context_samples": len(context_samples),
        "context_substitution_components": len(context_substitutions),
        "provided_context_rows": len(provided_context),
        "msa_contexts": len(msa),
    }
    for key, value in expected_counts.items():
        if metadata.get(key) != value:
            raise ValueError(f"{key} count differs from dataset.json")
    if queries.duplicated(["task", "query_id"]).any():
        raise ValueError("duplicate query identity")
    actual = samples.groupby(["task", "query_id"]).size().rename("actual_samples").reset_index()
    checked = queries.merge(actual, on=["task", "query_id"], how="outer", validate="one_to_one")
    if checked.isna().any().any() or not checked["expected_samples"].astype(int).eq(checked["actual_samples"]).all():
        raise ValueError("query membership or size differs")
    _validate_components(samples, substitutions, proteins)
    _validate_components(context_samples, context_substitutions, proteins)

    all_samples = pd.concat([samples, context_samples], ignore_index=True)
    if all_samples["sample_id"].duplicated().any():
        raise ValueError("evaluation and context sample IDs overlap")
    if not all_samples["sample_id"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("sample_id must be a lowercase SHA256 identifier")

    if msa["context_sha256"].duplicated().any():
        raise ValueError("duplicate MSA chain context")
    chain_hashes = set()
    for protein in proteins.itertuples(index=False):
        chain_hashes.update(
            hashlib.sha256(chain.encode()).hexdigest()
            for chain in json.loads(protein.chain_sequences)
        )
    if set(msa["context_sha256"]) != chain_hashes:
        raise ValueError("MSA context coverage differs from protein chains")
    return (
        metadata,
        proteins,
        samples,
        substitutions,
        queries,
        context_samples,
        context_substitutions,
        msa,
    )
