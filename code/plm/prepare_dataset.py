#!/usr/bin/env python3
"""Create validated PLM inputs from the published PFArena benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd

from common_io import atomic_write_csv, atomic_write_json, sha256_file, sha256_files, sha256_json, utc_now


TASKS = (
    "T1_single_mutant_generation",
    "T2_measurement_free_multi_mutant_ranking",
    "T3_anchor_informed_multi_mutant_ranking",
    "T4_single_mutant_informed_multi_mutant_ranking",
)
AA = set("ACDEFGHIKLMNPQRSTVWY")
SUBSTITUTION = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
PROTEIN_COLUMNS = ["wt_id", "sequence_sha256", "wildtype_sequence", "sequence_length", "n_chains", "chain_sequences", "chain_boundaries"]
SAMPLE_COLUMNS = ["sample_id", "task", "query_id", "source_assay", "sample_role", "mutant", "model_mutant", "mutation_count", "wt_id", "sequence_sha256", "mutated_sequence"]
SUBSTITUTION_COLUMNS = ["sample_id", "component_index", "chain_id", "chain_position", "global_position", "wt_aa", "mut_aa"]
QUERY_COLUMNS = ["task", "query_id", "source_assay", "source_csv", "expected_samples"]
MSA_COLUMNS = ["context_sha256", "sequence_length", "chain_sequence", "a3m_path", "a3m_sha256", "msa_method", "msa_database"]
PROVIDED_CONTEXT_COLUMNS = ["task", "query_id", "source_assay", "sample_role", "mutant", "visible_DMS_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default="pfarena_t1_t4")
    parser.add_argument("--structure-manifest", type=Path,
                        help="Optional released AF3 manifest; creates a validated local structures.csv without copying PDBs.")
    return parser.parse_args()


def chain_context(sequence: str) -> tuple[list[str], list[list[int]], str]:
    chains = sequence.split(":")
    if any(not chain or not set(chain).issubset(AA) for chain in chains):
        raise ValueError("invalid WT sequence")
    boundaries, start = [], 1
    for chain in chains:
        boundaries.append([start, start + len(chain) - 1])
        start += len(chain)
    return chains, boundaries, "".join(chains)


def parse_mutant(mutant: str, flat_wt: str) -> list[tuple[str, int, str]]:
    components = []
    for annotation in re.split(r"[+:]", mutant):
        match = SUBSTITUTION.fullmatch(annotation)
        if match is None:
            raise ValueError(f"invalid substitution annotation: {mutant}")
        wt, position_text, alt = match.groups()
        position = int(position_text)
        if not 1 <= position <= len(flat_wt) or flat_wt[position - 1] != wt or wt == alt:
            raise ValueError(f"invalid substitution mapping: {mutant}")
        components.append((wt, position, alt))
    positions = [component[1] for component in components]
    if len(positions) != len(set(positions)):
        raise ValueError(f"duplicate substitution position: {mutant}")
    return sorted(components, key=lambda component: component[1])


def mutated_sequence(flat_wt: str, chains: list[str], components: list[tuple[str, int, str]]) -> str:
    mutated = list(flat_wt)
    for _, position, alt in components:
        mutated[position - 1] = alt
    pieces, start = [], 0
    for chain in chains:
        pieces.append("".join(mutated[start : start + len(chain)]))
        start += len(chain)
    return ":".join(pieces)


def a3m_query(path: Path) -> str:
    lines = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if lines:
                    break
                continue
            lines.append(line)
    query = "".join(lines)
    return "".join(character for character in query if not character.islower() and character not in ".-")


def query_id(task: str, source_path: Path, frame: pd.DataFrame, assay_id: str) -> str:
    values = frame["candidate_group_id"].astype(str).unique()
    if len(values) != 1 or not values[0].strip():
        raise ValueError(f"{source_path}: expected one nonempty candidate_group_id")
    # The shared evaluator keys flat T1/T2 files by filename, and nested
    # T3/T4 files by candidate_group_id. Preserve that public contract.
    return source_path.stem if task in TASKS[:2] else values[0]


def add_sample(
    *, dataset_id: str, task: str, query: str, assay: str, role: str,
    mutant: str, source_mutated_sequence: str, sequence_sha256: str, wt_id: str,
    chains: list[str], boundaries: list[list[int]], flat_wt: str,
    samples: list[dict], substitutions: list[dict],
) -> None:
    components = parse_mutant(mutant, flat_wt)
    expected_sequence = mutated_sequence(flat_wt, chains, components)
    if source_mutated_sequence.strip().upper() != expected_sequence:
        raise ValueError(f"{task}/{query}: mutated_sequence mismatch for {mutant}")
    identity = {"dataset_id": dataset_id, "task": task, "query_id": query, "sample_role": role, "mutant": mutant}
    sample_id = sha256_json(identity)
    samples.append({
        "sample_id": sample_id, "task": task, "query_id": query,
        "source_assay": assay, "sample_role": role, "mutant": mutant,
        "model_mutant": ":".join(f"{wt}{position}{alt}" for wt, position, alt in components),
        "mutation_count": len(components), "wt_id": wt_id,
        "sequence_sha256": sequence_sha256, "mutated_sequence": expected_sequence,
    })
    for index, (wt, position, alt) in enumerate(components, start=1):
        for chain_id, (start, end) in enumerate(boundaries, start=1):
            if start <= position <= end:
                substitutions.append({
                    "sample_id": sample_id, "component_index": index, "chain_id": chain_id,
                    "chain_position": position - start + 1, "global_position": position,
                    "wt_aa": wt, "mut_aa": alt,
                })
                break


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    proteins, samples, substitutions, queries = {}, [], [], []
    context_samples, context_substitutions, provided_context, msa_contexts = [], [], [], {}
    source_manifest = []
    task_summary = {}

    for task in TASKS:
        task_dir = args.benchmark_root / task
        assay_path = task_dir / "assay.csv"
        assays = pd.read_csv(assay_path, keep_default_na=False)
        if assays["assay_id"].duplicated().any():
            raise ValueError(f"{task}: duplicate assay_id")
        source_manifest.append({"path": str(assay_path.relative_to(args.benchmark_root)), "sha256": sha256_file(assay_path)})
        task_sample_start, task_query_start = len(samples), len(queries)
        for assay_row in assays.sort_values("assay_id").itertuples(index=False):
            assay = str(assay_row.assay_id)
            sequence = str(assay_row.wildtype_sequence).strip().upper()
            chains, boundaries, flat_wt = chain_context(sequence)
            if int(float(assay_row.sequence_length)) != len(flat_wt):
                raise ValueError(f"{task}/{assay}: sequence_length mismatch")
            sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
            wt_id = sequence_sha256[:16]
            protein = {
                "wt_id": wt_id, "sequence_sha256": sequence_sha256,
                "wildtype_sequence": sequence, "sequence_length": len(flat_wt),
                "n_chains": len(chains),
                "chain_sequences": json.dumps(chains, separators=(",", ":")),
                "chain_boundaries": json.dumps(boundaries, separators=(",", ":")),
            }
            if proteins.setdefault(sequence_sha256, protein) != protein:
                raise ValueError(f"{task}/{assay}: inconsistent WT")

            msa_paths = [value for value in str(assay_row.a3m_relative_path).split(";") if value]
            if len(msa_paths) != len(chains):
                raise ValueError(f"{task}/{assay}: MSA path count differs from chain count")
            for chain, relative_msa in zip(chains, msa_paths):
                context_sha256 = hashlib.sha256(chain.encode()).hexdigest()
                path = args.benchmark_root / relative_msa
                if not path.is_file() or a3m_query(path) != chain:
                    raise ValueError(f"{task}/{assay}: invalid MSA query for {context_sha256[:16]}")
                record = {
                    "context_sha256": context_sha256, "sequence_length": len(chain),
                    "chain_sequence": chain, "a3m_path": os.path.relpath(path.resolve(), args.output_dir.resolve()),
                    "a3m_sha256": sha256_file(path), "msa_method": "MMseqs2",
                    "msa_database": "UniRef100",
                }
                previous = msa_contexts.setdefault(context_sha256, record)
                if previous["chain_sequence"] != chain or previous["a3m_sha256"] != record["a3m_sha256"]:
                    raise ValueError(f"{task}/{assay}: one chain maps to different MSA contents")

            source_path = task_dir / str(assay_row.DMS_filename)
            frame = pd.read_csv(source_path, keep_default_na=False)
            source_manifest.append({"path": str(source_path.relative_to(args.benchmark_root)), "sha256": sha256_file(source_path)})
            required = {"candidate_group_id", "mutant", "mutated_sequence"}
            if required - set(frame.columns) or frame.empty:
                raise ValueError(f"{source_path}: missing columns or empty")
            query = query_id(task, source_path, frame, assay)
            if frame["mutant"].astype(str).duplicated().any():
                raise ValueError(f"{source_path}: duplicate mutant")
            for row in frame.sort_values("mutant").itertuples(index=False):
                add_sample(
                    dataset_id=args.dataset_id, task=task, query=query, assay=assay,
                    role="evaluation_candidate", mutant=str(row.mutant),
                    source_mutated_sequence=str(row.mutated_sequence),
                    sequence_sha256=sequence_sha256, wt_id=wt_id, chains=chains,
                    boundaries=boundaries, flat_wt=flat_wt,
                    samples=samples, substitutions=substitutions,
                )
            queries.append({
                "task": task, "query_id": query, "source_assay": assay,
                "source_csv": str(source_path.relative_to(args.benchmark_root)),
                "expected_samples": len(frame),
            })

            if task == "T3_anchor_informed_multi_mutant_ranking":
                anchor_mutant = str(assay_row.anchor_mutant)
                anchor_sequence = mutated_sequence(flat_wt, chains, parse_mutant(anchor_mutant, flat_wt))
                add_sample(
                    dataset_id=args.dataset_id, task=task, query=query, assay=assay,
                    role="anchor_context", mutant=anchor_mutant,
                    source_mutated_sequence=anchor_sequence,
                    sequence_sha256=sequence_sha256, wt_id=wt_id, chains=chains,
                    boundaries=boundaries, flat_wt=flat_wt,
                    samples=context_samples, substitutions=context_substitutions,
                )
                provided_context.append({"task": task, "query_id": query, "source_assay": assay, "sample_role": "anchor_context", "mutant": anchor_mutant, "visible_DMS_score": float(assay_row.anchor_DMS_score)})
            elif task == "T4_single_mutant_informed_multi_mutant_ranking":
                context_path = args.benchmark_root / str(assay_row.single_context_data_path)
                context = pd.read_csv(context_path, keep_default_na=False)
                source_manifest.append({"path": str(context_path.relative_to(args.benchmark_root)), "sha256": sha256_file(context_path)})
                if context.empty or not {"mutant", "mutated_sequence", "DMS_score"}.issubset(context.columns):
                    raise ValueError(f"{context_path}: invalid visible single-mutant context")
                for row in context.sort_values("mutant").itertuples(index=False):
                    add_sample(
                        dataset_id=args.dataset_id, task=task, query=query, assay=assay,
                        role="visible_single_context", mutant=str(row.mutant),
                        source_mutated_sequence=str(row.mutated_sequence),
                        sequence_sha256=sequence_sha256, wt_id=wt_id, chains=chains,
                        boundaries=boundaries, flat_wt=flat_wt,
                        samples=context_samples, substitutions=context_substitutions,
                    )
                    provided_context.append({"task": task, "query_id": query, "source_assay": assay, "sample_role": "visible_single_context", "mutant": str(row.mutant), "visible_DMS_score": float(row.DMS_score)})

        task_summary[task] = {
            "source_assays": len(assays),
            "queries": len(queries) - task_query_start,
            "evaluation_samples": len(samples) - task_sample_start,
        }

    frames = {
        "proteins.csv": pd.DataFrame(proteins.values(), columns=PROTEIN_COLUMNS).sort_values("sequence_sha256"),
        "samples.csv": pd.DataFrame(samples, columns=SAMPLE_COLUMNS).sort_values(["task", "query_id", "mutant"]),
        "substitutions.csv": pd.DataFrame(substitutions, columns=SUBSTITUTION_COLUMNS).sort_values(["sample_id", "component_index"]),
        "queries.csv": pd.DataFrame(queries, columns=QUERY_COLUMNS).sort_values(["task", "query_id"]),
        "context_samples.csv": pd.DataFrame(context_samples, columns=SAMPLE_COLUMNS).sort_values(["task", "query_id", "sample_role", "mutant"]),
        "context_substitutions.csv": pd.DataFrame(context_substitutions, columns=SUBSTITUTION_COLUMNS).sort_values(["sample_id", "component_index"]),
        "provided_context.csv": pd.DataFrame(provided_context, columns=PROVIDED_CONTEXT_COLUMNS).sort_values(["task", "query_id", "mutant"]),
        "msa_contexts.csv": pd.DataFrame(msa_contexts.values(), columns=MSA_COLUMNS).sort_values("context_sha256"),
    }
    if frames["samples.csv"]["sample_id"].duplicated().any() or frames["context_samples.csv"]["sample_id"].duplicated().any():
        raise ValueError("sample ID collision")
    for name, frame in frames.items():
        atomic_write_csv(frame, args.output_dir / name)
    paths = [args.output_dir / name for name in frames]
    metadata = {
        "schema_version": 4, "status": "frozen", "dataset_id": args.dataset_id,
        "benchmark_root": str(args.benchmark_root.resolve()), "created_at_utc": utc_now(),
        "proteins": len(frames["proteins.csv"]), "samples": len(frames["samples.csv"]),
        "substitution_components": len(frames["substitutions.csv"]),
        "queries": len(frames["queries.csv"]), "context_samples": len(frames["context_samples.csv"]),
        "context_substitution_components": len(frames["context_substitutions.csv"]),
        "provided_context_rows": len(frames["provided_context.csv"]),
        "msa_contexts": len(frames["msa_contexts.csv"]), "task_summary": task_summary,
        "source_manifest": sorted(source_manifest, key=lambda item: item["path"]),
        "dataset_hash": sha256_files(paths),
    }
    atomic_write_json(metadata, args.output_dir / "dataset.json")
    from dataset_io import load_dataset
    load_dataset(args.output_dir)
    if args.structure_manifest:
        from prepare_structures import prepare_structures
        prepare_structures(args.output_dir, args.structure_manifest)
    print(json.dumps({key: metadata[key] for key in ("dataset_id", "dataset_hash", "proteins", "samples", "queries", "context_samples", "msa_contexts")}, indent=2))


if __name__ == "__main__":
    main()
