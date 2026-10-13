#!/usr/bin/env python3
"""Validate released chain PDBs and bind their manifest to local model inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from common_io import atomic_write_csv, sha256_file
from dataset_io import load_dataset


AA3 = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    "ACDEFGHIKLMNPQRSTVWY",
))


def pdb_sequence(path: Path) -> str:
    residues = []
    chains = set()
    models = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith("MODEL "):
                models += 1
            if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                chains.add(line[21])
                if line[16] not in (" ", "A") or line[26] != " ":
                    raise ValueError(f"{path}: unsupported alternate location or insertion code")
                residues.append((int(line[22:26]), AA3[line[17:20]]))
    if models > 1 or len(chains) != 1 or [n for n, _ in residues] != list(range(1, len(residues) + 1)):
        raise ValueError(f"{path}: expected one chain with consecutive residue numbering")
    return "".join(aa for _, aa in residues)


def prepare_structures(dataset_dir: Path, source_manifest: Path) -> Path:
    output = dataset_dir / "structures.csv"
    if output.exists():
        raise FileExistsError(output)
    metadata, proteins, samples, substitutions, *_ = load_dataset(dataset_dir)
    sequences = {}
    chains = {}
    for protein in proteins.itertuples(index=False):
        for index, sequence in enumerate(json.loads(protein.chain_sequences), 1):
            context = hashlib.sha256(sequence.encode()).hexdigest()
            chains[(protein.sequence_sha256, index)] = context
            sequences[context] = sequence
    components = substitutions.merge(samples[["sample_id", "sequence_sha256"]],
                                     on="sample_id", validate="many_to_one")
    required = {chains[(r.sequence_sha256, int(r.chain_id))] for r in components.itertuples(index=False)}
    manifest = pd.read_csv(source_manifest, keep_default_na=False)
    columns = {"context_sha256", "sequence_length", "structure_status", "structure_qc",
               "structure_source", "pdb_path", "pdb_sha256"}
    if columns - set(manifest.columns) or manifest["context_sha256"].duplicated().any():
        raise ValueError("structure manifest has missing columns or duplicate contexts")
    if required - set(manifest["context_sha256"]):
        raise ValueError("structure manifest does not cover all mutated chains")
    selected = manifest[manifest["context_sha256"].isin(required)].copy()
    for index, row in selected.iterrows():
        path = Path(row.pdb_path)
        if not path.is_absolute():
            path = source_manifest.resolve().parent / path
        sequence = sequences[row.context_sha256]
        if (row.structure_status != "ready" or int(row.sequence_length) != len(sequence)
                or sha256_file(path) != row.pdb_sha256 or pdb_sequence(path) != sequence):
            raise ValueError(f"{path}: structure sequence, status or checksum mismatch")
        selected.at[index, "pdb_path"] = os.path.relpath(path.resolve(), dataset_dir.resolve())
    # A new benchmark identity does not imply a different structure. Rebind only
    # after validating the actual PDB bytes, chain sequence and residue mapping.
    selected["dataset_hash"] = metadata["dataset_hash"]
    atomic_write_csv(selected.sort_values("context_sha256"), output)
    print(f"structures={len(selected)} manifest={output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    args = parser.parse_args()
    prepare_structures(args.dataset_dir, args.structure_manifest)
