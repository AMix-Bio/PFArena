#!/usr/bin/env python3
"""Prepare the chain-level structures required by S3F on T1--T4."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model_dataset_io import load_model_dataset
from common_io import atomic_write_csv, atomic_write_json, sha256_file, sha256_files, utc_now


def resolve_structure_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    manifest_relative = (manifest_path.resolve().parent / path).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"S3F candidate input already exists: {args.output_dir}")

    dataset, proteins, samples, substitutions, _ = load_model_dataset(args.dataset_dir)
    structures = pd.read_csv(args.structure_manifest)
    required_structure_columns = {
        "dataset_hash",
        "context_sha256",
        "sequence_length",
        "structure_status",
        "structure_qc",
        "structure_source",
        "pdb_path",
        "pdb_sha256",
    }
    missing_columns = required_structure_columns - set(structures.columns)
    if missing_columns:
        raise ValueError(
            "structure manifest is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    chain_sequences: dict[str, str] = {}
    chain_lookup: dict[tuple[str, int], str] = {}
    for protein in proteins.itertuples(index=False):
        for chain_id, sequence in enumerate(json.loads(protein.chain_sequences), start=1):
            context_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
            chain_sequences[context_sha256] = sequence
            chain_lookup[(protein.sequence_sha256, chain_id)] = context_sha256

    components = substitutions.merge(
        samples[["sample_id", "sequence_sha256"]], on="sample_id", validate="many_to_one"
    )
    components["context_sha256"] = [
        chain_lookup[(sequence_sha256, int(chain_id))]
        for sequence_sha256, chain_id in zip(
            components["sequence_sha256"], components["chain_id"]
        )
    ]
    required_contexts = set(components["context_sha256"])
    chain_contribution_rows = int(
        components[["sample_id", "context_sha256"]].drop_duplicates().shape[0]
    )
    if structures["context_sha256"].duplicated().any():
        raise ValueError("structure manifest contains duplicate contexts")
    if set(structures["dataset_hash"]) != {dataset["dataset_hash"]}:
        raise ValueError("structure manifest targets a different dataset")
    if not structures["structure_status"].eq("ready").all():
        raise ValueError("structure manifest contains unavailable structures")
    if set(structures["context_sha256"]) != required_contexts:
        raise ValueError("structure manifest context set differs from mutated chains")

    sample_counts = components.groupby("context_sha256")["sample_id"].nunique()
    component_counts = components.groupby("context_sha256").size()
    position_counts = components.groupby("context_sha256")["chain_position"].nunique()
    structure_dir = args.output_dir / "structures"
    structure_dir.mkdir(parents=True)
    rows = []
    for structure in structures.sort_values("context_sha256").itertuples(index=False):
        sequence = chain_sequences[structure.context_sha256]
        if len(sequence) != int(structure.sequence_length):
            raise ValueError(f"{structure.context_sha256}: sequence length differs")
        context_id = structure.context_sha256[:16]
        source = resolve_structure_path(structure.pdb_path, args.structure_manifest)
        if sha256_file(source) != structure.pdb_sha256:
            raise ValueError(f"{context_id}: PDB differs from the structure manifest")
        target = structure_dir / f"{context_id}.pdb"
        shutil.copyfile(source, target)
        if sha256_file(target) != structure.pdb_sha256:
            raise ValueError(f"{context_id}: copied PDB checksum differs")
        rows.append(
            {
                "context_id": context_id,
                "context_sha256": structure.context_sha256,
                "sequence": sequence,
                "sequence_length": len(sequence),
                "candidate_sample_count": int(sample_counts[structure.context_sha256]),
                "substitution_component_count": int(component_counts[structure.context_sha256]),
                "mutation_position_count": int(position_counts[structure.context_sha256]),
                "structure_qc": structure.structure_qc,
                "structure_source": structure.structure_source,
                "pdb_path": str(target.resolve()),
                "source_pdb_path": str(source),
                "pdb_sha256": structure.pdb_sha256,
            }
        )

    contexts = pd.DataFrame(rows)
    if (
        contexts["context_id"].duplicated().any()
        or contexts["context_sha256"].duplicated().any()
        or contexts["candidate_sample_count"].sum() != components.drop_duplicates(["sample_id", "context_sha256"]).shape[0]
        or contexts["substitution_component_count"].sum() != len(substitutions)
    ):
        raise ValueError("S3F candidate context coverage differs from the dataset")
    context_path = args.output_dir / "contexts.csv"
    atomic_write_csv(contexts, context_path)
    atomic_write_json(
        {
            "schema_version": 1,
            "model_id": "S3F",
            "dataset_id": dataset["dataset_id"],
            "dataset_hash": dataset["dataset_hash"],
            "created_at_utc": utc_now(),
            "structure_manifest": str(args.structure_manifest.resolve()),
            "structure_manifest_sha256": sha256_file(args.structure_manifest),
            "contexts": len(contexts),
            "samples": chain_contribution_rows,
            "unique_samples": len(samples),
            "chain_contribution_rows": chain_contribution_rows,
            "substitution_components": int(contexts["substitution_component_count"].sum()),
            "mutation_positions": int(contexts["mutation_position_count"].sum()),
            "context_manifest_sha256": sha256_file(context_path),
            "input_hash": sha256_files([context_path]),
        },
        args.output_dir / "inputs.json",
    )
    print(
        f"Prepared {len(contexts)} S3F chain contexts for "
        f"{len(samples)} samples ({chain_contribution_rows} chain contributions)"
    )


if __name__ == "__main__":
    main()
