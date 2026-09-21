#!/usr/bin/env python3
"""Expose the frozen T1--T4 dataset in the model-facing table layout."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset_io import load_dataset


def load_model_dataset(
    dataset_dir: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    (
        metadata,
        proteins,
        samples,
        substitutions,
        queries,
        context_samples,
        context_substitutions,
        _,
    ) = load_dataset(dataset_dir)
    samples = pd.concat([samples, context_samples], ignore_index=True)
    substitutions = pd.concat(
        [substitutions, context_substitutions], ignore_index=True
    ).sort_values(["sample_id", "component_index"])
    samples["subset"] = samples["task"]
    samples["candidate_group_id"] = samples["query_id"]
    groups = queries.rename(columns={"task": "subset", "query_id": "candidate_group_id"})
    return metadata, proteins, samples, substitutions, groups
