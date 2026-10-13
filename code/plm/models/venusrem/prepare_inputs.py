#!/usr/bin/env python3
"""Prepare VenusREM sequence, structure-token, and MSA inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common_io import atomic_write_csv, atomic_write_json, sha256_file, sha256_json, utc_now
from model_dataset_io import load_model_dataset
from input_utils import MAX_RESIDUES, inference_window, load_prosst_cache_root, normalize_a3m, write_text
from dataset_io import load_dataset, resolve_dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--prosst-run", type=Path, required=True)
    parser.add_argument("--prosst-cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"VenusREM inputs already exist: {args.output_dir}")
    dataset, proteins, samples, substitutions, _ = load_model_dataset(args.dataset_dir)
    *_, msa = load_dataset(args.dataset_dir)
    msa_index = msa.set_index("context_sha256")
    chain_lookup, chain_sequences, representatives = {}, {}, {}
    for protein in proteins.itertuples(index=False):
        for chain_id, sequence in enumerate(json.loads(protein.chain_sequences), start=1):
            digest = hashlib.sha256(sequence.encode()).hexdigest()
            chain_lookup[(protein.sequence_sha256, chain_id)] = digest
            chain_sequences[digest] = sequence
            representatives.setdefault(digest, (protein.sequence_sha256, chain_id))
    components = substitutions.merge(
        samples[["sample_id", "sequence_sha256", "source_assay"]],
        on="sample_id",
        validate="many_to_one",
    )
    components["chain_sha256"] = [
        chain_lookup[(sequence_hash, int(chain_id))]
        for sequence_hash, chain_id in zip(
            components.sequence_sha256, components.chain_id
        )
    ]
    positions = (
        components.groupby("chain_sha256")["chain_position"]
        .agg(lambda values: sorted(set(map(int, values))))
        .to_dict()
    )
    assignments = components.drop_duplicates(["sample_id", "chain_sha256"])
    sample_counts = assignments.groupby("chain_sha256").size().to_dict()
    assay_sets = (
        components.groupby("chain_sha256")["source_assay"]
        .agg(lambda values: sorted(set(map(str, values))))
        .to_dict()
    )
    if args.prosst_cache_root is None:
        prosst_cache_root, prosst_run = load_prosst_cache_root(
            args.prosst_run, dataset["dataset_hash"]
        )
    else:
        prosst_run = json.loads(args.prosst_run.read_text())
        if prosst_run.get("dataset_hash") != dataset["dataset_hash"] or "run_config" not in prosst_run:
            raise ValueError("ProSST run provenance differs from the released dataset")
        prosst_cache_root = args.prosst_cache_root.resolve()
    prosst_run_config_sha256 = prosst_run.get(
        "run_config_sha256", sha256_json(prosst_run["run_config"])
    )
    rows = []
    for chain_sha256, required_positions in sorted(positions.items()):
        sequence = chain_sequences[chain_sha256]
        msa_row = msa_index.loc[chain_sha256]
        source_cache = prosst_cache_root / f"{chain_sha256}.npz"
        with np.load(source_cache, allow_pickle=False) as cache:
            if (
                str(cache["cache_config_sha256"].item())
                != prosst_run["run_config"]["cache_config_sha256"]
                or str(cache["context_sha256"].item()) != chain_sha256
            ):
                raise ValueError(f"{source_cache}: ProSST cache provenance differs")
            if int(cache["sequence_length"].item()) != len(sequence):
                raise ValueError(f"{source_cache}: ProSST chain length differs")
            structure_tokens = cache["structure_tokens"].astype(np.int32)
        if (
            structure_tokens.shape != (len(sequence),)
            or (structure_tokens < 0).any()
            or (structure_tokens >= 2048).any()
        ):
            raise ValueError(f"{source_cache}: invalid structure tokens")
        start, end = inference_window(len(sequence), required_positions)
        context_id = chain_sha256[:16]
        residue_fasta = args.output_dir / "aa_seq" / f"{context_id}.fasta"
        structure_fasta = args.output_dir / "struc_seq/2048" / f"{context_id}.fasta"
        alignment_file = args.output_dir / "aa_seq_aln_a2m" / f"{context_id}.a2m"
        source_alignment = resolve_dataset_path(args.dataset_dir, msa_row.a3m_path)
        if sha256_file(source_alignment) != msa_row.a3m_sha256:
            raise ValueError(f"{source_alignment}: source A3M differs from the frozen dataset")
        write_text(residue_fasta, f">{context_id}\n{sequence[start - 1:end]}\n")
        structure_window = ",".join(map(str, structure_tokens[start - 1 : end]))
        write_text(structure_fasta, f">{context_id}\n{structure_window}\n")
        msa_records, lowercase_removed, unknown = normalize_a3m(
            source_alignment, alignment_file, sequence, start, end
        )
        source_cache_hash = sha256_file(source_cache)
        hashes = {
            "residue_sha256": sha256_file(residue_fasta),
            "structure_sha256": sha256_file(structure_fasta),
            "aa_alignment_sha256": sha256_file(alignment_file),
        }
        context_hash = sha256_json(
            {
                "chain_sha256": chain_sha256,
                "window_start": start,
                "window_end": end,
                "source_alignment_sha256": msa_row.a3m_sha256,
                "source_prosst_cache_sha256": source_cache_hash,
                **hashes,
            }
        )
        parent_sha256, chain_id = representatives[chain_sha256]
        rows.append(
            {
                "context_id": context_id,
                "context_sha256": context_hash,
                "parent_sequence_sha256": parent_sha256,
                "chain_id": int(chain_id),
                "chain_sha256": chain_sha256,
                "chain_length": len(sequence),
                "window_start": start,
                "window_end": end,
                "window_length": end - start + 1,
                "required_position_count": len(required_positions),
                "required_position_min": required_positions[0],
                "required_position_max": required_positions[-1],
                "expected_samples": int(sample_counts[chain_sha256]),
                "source_assays_json": json.dumps(
                    assay_sets[chain_sha256], separators=(",", ":")
                ),
                "residue_fasta": str(residue_fasta.resolve()),
                "residue_sha256": hashes["residue_sha256"],
                "structure_fasta": str(structure_fasta.resolve()),
                "structure_sha256": hashes["structure_sha256"],
                "aa_alignment_file": str(alignment_file.resolve()),
                "aa_alignment_sha256": hashes["aa_alignment_sha256"],
                "aa_alignment_size_bytes": alignment_file.stat().st_size,
                "msa_records": msa_records,
                "msa_lowercase_insertions_removed": lowercase_removed,
                "msa_unknown_residues": unknown,
                "source_alignment": str(source_alignment.resolve()),
                "source_alignment_sha256": msa_row.a3m_sha256,
                "source_prosst_cache": str(source_cache.resolve()),
                "source_prosst_cache_sha256": source_cache_hash,
            }
        )
    contexts = pd.DataFrame(rows).sort_values("context_id")
    if contexts["chain_sha256"].duplicated().any() or int(
        contexts.expected_samples.sum()
    ) != len(assignments):
        raise ValueError("VenusREM context coverage differs")
    path = args.output_dir / "contexts.csv"
    atomic_write_csv(contexts, path)
    atomic_write_json(
        {
            "schema_version": 1,
            "model_id": "VenusREM_ProSST-2048_aa_seq_aln",
            "dataset_id": dataset["dataset_id"],
            "dataset_hash": dataset["dataset_hash"],
            "created_at_utc": utc_now(),
            "chain_contexts": len(contexts),
            "samples": len(samples),
            "chain_sample_assignments": int(contexts.expected_samples.sum()),
            "substitution_components": len(substitutions),
            "long_chain_contexts": int(contexts.chain_length.gt(MAX_RESIDUES).sum()),
            "max_residues": MAX_RESIDUES,
            "long_sequence_protocol": "required_interval_centered_synchronized_sequence_structure_msa_window",
            "multichain_protocol": "sum_independent_mutated_chain_log_odds",
            "msa_database": "UniRef100",
            "msa_method": "MMseqs2",
            "msa_normalization": "remove lowercase A3M insertions; retain aligned gaps; slice synchronized inference window",
            "msa_records": int(contexts.msa_records.sum()),
            "msa_lowercase_insertions_removed": int(
                contexts.msa_lowercase_insertions_removed.sum()
            ),
            "msa_unknown_residues": int(contexts.msa_unknown_residues.sum()),
            "structure_tokens": "full-chain ProSST-2048 tokens from the compatible finalized ProSST run",
            "prosst_run": str(args.prosst_run.resolve()),
            "prosst_run_config_sha256": prosst_run_config_sha256,
            "prosst_cache_root": str(prosst_cache_root.resolve()),
            "contexts_sha256": sha256_file(path),
        },
        args.output_dir / "inputs.json",
    )
    print(f"contexts={len(contexts)} chain_sample_assignments={int(contexts.expected_samples.sum())}")


if __name__ == "__main__":
    main()
