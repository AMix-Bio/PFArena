#!/usr/bin/env python3
"""Prepare full-chain EVE inputs for S3F-MSA."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common_io import atomic_write_csv, atomic_write_json, sha256_file, sha256_files, utc_now
from model_dataset_io import load_model_dataset
from alignment import convert_a3m
from dataset_io import load_dataset, resolve_dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"S3F-MSA inputs already exist: {args.output_dir}")
    dataset, proteins, samples, substitutions, groups = load_model_dataset(args.dataset_dir)
    *_, msa = load_dataset(args.dataset_dir)
    msa_index = msa.set_index("context_sha256")

    assay_metadata = groups[["source_assay", "taxon_domain"]].drop_duplicates()
    if assay_metadata.groupby("source_assay")["taxon_domain"].nunique().gt(1).any():
        raise ValueError("source assay has inconsistent taxon_domain")
    taxon = assay_metadata.drop_duplicates("source_assay").set_index("source_assay")["taxon_domain"].to_dict()

    chain_lookup, chain_sequences = {}, {}
    for protein in proteins.itertuples(index=False):
        for chain_id, sequence in enumerate(json.loads(protein.chain_sequences), start=1):
            digest = hashlib.sha256(sequence.encode()).hexdigest()
            chain_lookup[(protein.sequence_sha256, chain_id)] = digest
            chain_sequences[digest] = sequence
    components = substitutions.merge(samples[["sample_id", "sequence_sha256", "source_assay"]], on="sample_id", validate="many_to_one")
    components["chain_sha256"] = [chain_lookup[(sequence_hash, int(chain_id))] for sequence_hash, chain_id in zip(components.sequence_sha256, components.chain_id)]
    components["theta"] = [0.01 if str(taxon[assay]).strip().lower() == "virus" else 0.2 for assay in components.source_assay]

    context_specs = {}
    for (assay, parent_hash, chain_id), frame in components.groupby(["source_assay", "sequence_sha256", "chain_id"], sort=True):
        chain_hash = frame.chain_sha256.iloc[0]
        theta = float(frame.theta.iloc[0])
        sequence = chain_sequences[chain_hash]
        msa_row = msa_index.loc[chain_hash]
        candidate_id = chain_hash[:16]
        key = (chain_hash, str(msa_row.a3m_sha256), theta)
        context_specs.setdefault(key, {"context_id": candidate_id, "chain_hash": chain_hash, "sequence": sequence, "msa_path": resolve_dataset_path(args.dataset_dir, msa_row.a3m_path), "msa_sha256": str(msa_row.a3m_sha256), "theta": theta, "frames": []})["frames"].append(frame)

    context_rows, sample_rows, alignments, mutation_files = [], [], [], []
    for protein_index, key in enumerate(sorted(context_specs)):
        spec = context_specs[key]
        if sha256_file(spec["msa_path"]) != spec["msa_sha256"]:
            raise ValueError(f"{spec['msa_path']}: source A3M differs from the frozen dataset")
        alignment = args.output_dir / "alignments" / f"{spec['context_id']}.a2m"
        qc = convert_a3m(spec["msa_path"], alignment, spec["context_id"], spec["sequence"], 1, None)
        a2m_hash = sha256_file(alignment)
        context_id = spec["context_id"]
        combined = pd.concat(spec["frames"], ignore_index=True)
        combined["annotation"] = combined.wt_aa + combined.chain_position.astype(str) + combined.mut_aa
        eve_mutants = combined.sort_values(["sample_id", "component_index"]).groupby("sample_id")["annotation"].agg(":".join)
        mutation_frame = eve_mutants.drop_duplicates().sort_values().rename("mutant").reset_index(drop=True).to_frame()
        mutation_path = args.output_dir / "mutations" / f"{context_id}.csv"
        atomic_write_csv(mutation_frame, mutation_path)
        for sample_id, mutant in eve_mutants.items():
            sample_rows.append({"sample_id": sample_id, "context_id": context_id, "eve_mutant": mutant})
        assays = sorted(set(combined.source_assay.astype(str)))
        context_rows.append({
            "protein_index": protein_index, "DMS_id": context_id, "wt_id": context_id, "context_id": context_id,
            "parent_sequence_sha256": combined.sequence_sha256.iloc[0], "chain_id": int(combined.chain_id.iloc[0]),
            "sequence_sha256": spec["chain_hash"], "sequence_length": len(spec["sequence"]), "chain_sequence_length": len(spec["sequence"]),
            "focus_start": 1, "focus_end": len(spec["sequence"]), "MSA_filename": alignment.name, "DMS_filename": mutation_path.name,
            "MSA_theta": spec["theta"], "weight_file_name": f"{context_id}_theta_{spec['theta']}.npy",
            "msa_policy": "uniref100_mmseqs2_full_chain",
            "source_a3m_path": str(spec["msa_path"].resolve()), "source_a3m_sha256": spec["msa_sha256"], "a2m_sha256": a2m_hash,
            "mutation_sha256": sha256_file(mutation_path), "source_assays_json": json.dumps(assays), "source_assay_count": len(assays),
            "unique_mutants": len(mutation_frame), **qc,
        })
        alignments.append(alignment); mutation_files.append(mutation_path)

    mapping = pd.DataFrame(context_rows).sort_values("protein_index")
    sample_mapping = pd.DataFrame(sample_rows).merge(samples[["sample_id", "subset", "candidate_group_id", "source_assay", "sample_role", "mutant", "mutated_sequence"]], on="sample_id", validate="many_to_one")
    if sample_mapping.duplicated(["sample_id", "context_id"]).any() or set(sample_mapping.sample_id) != set(samples.sample_id):
        raise ValueError("S3F-MSA sample-chain mapping is incomplete")
    new_contexts = mapping[["protein_index", "context_id"]].reset_index(drop=True)
    new_contexts.insert(0, "task_index", range(len(new_contexts)))
    score_contexts = mapping[["protein_index", "context_id"]].reset_index(drop=True)
    score_contexts.insert(0, "task_index", range(len(score_contexts)))
    paths = {"mapping.csv": mapping, "sample_mapping.csv": sample_mapping, "new_contexts.csv": new_contexts, "score_contexts.csv": score_contexts}
    for name, frame in paths.items(): atomic_write_csv(frame, args.output_dir / name)
    metadata = {
        "schema_version": 3, "model_id": "S3F-MSA", "dataset_id": dataset["dataset_id"], "dataset_hash": dataset["dataset_hash"], "created_at_utc": utc_now(),
        "partial_input": False, "focus_column_gap_threshold": 1.0, "sequence_gap_threshold": 0.5, "viral_theta": 0.01, "other_theta": 0.2,
        "msa_policy": "UniRef100/MMseqs2 full-chain MSA for every context", "contexts": len(mapping),
        "new_contexts": len(new_contexts), "locally_scored_contexts": len(score_contexts), "samples": len(samples), "sample_chain_assignments": len(sample_mapping),
        "unique_context_mutants": int(mapping.unique_mutants.sum()),
        "mapping_sha256": sha256_file(args.output_dir / "mapping.csv"), "sample_mapping_sha256": sha256_file(args.output_dir / "sample_mapping.csv"),
        "new_contexts_sha256": sha256_file(args.output_dir / "new_contexts.csv"), "score_contexts_sha256": sha256_file(args.output_dir / "score_contexts.csv"),
        "input_hash": sha256_files([*(args.output_dir / name for name in paths), *alignments, *mutation_files]),
    }
    atomic_write_json(metadata, args.output_dir / "inputs.json")
    print(f"contexts={len(mapping)} trained={len(new_contexts)} sample_chain_assignments={len(sample_mapping)}")


if __name__ == "__main__":
    main()
