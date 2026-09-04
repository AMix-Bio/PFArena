#!/usr/bin/env python3
"""Export evaluator-ready task tables from released model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common_io import atomic_write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")

    run = json.loads((args.result_dir / "run.json").read_text())
    schema = json.loads((args.result_dir / "output_schema.json").read_text())
    primary = schema["primary_score"]
    predictions = pd.read_csv(
        args.result_dir / "predictions.csv.gz", float_precision="round_trip"
    )
    required = {
        "sample_id", "task", "query_id", "mutant", "status", "sample_role", primary
    }
    if required - set(predictions.columns):
        raise ValueError("released predictions are missing required columns")
    if predictions["sample_id"].duplicated().any() or not predictions["status"].eq("ok").all():
        raise ValueError("released predictions contain duplicate or failed samples")
    if not predictions["sample_role"].eq("evaluation_candidate").all():
        raise ValueError("released predictions contain non-evaluation samples")
    if not np.isfinite(predictions[primary]).all():
        raise ValueError("primary model scores contain non-finite values")
    if len(predictions) != int(run["evaluation_samples"]):
        raise ValueError("released prediction count differs from run.json")
    if set(predictions["task"]) != set(run["tasks"]):
        raise ValueError("released prediction tasks differ from run.json")

    for task, expected in run["tasks"].items():
        selected = predictions[predictions["task"].eq(task)]
        if len(selected) != int(expected["samples"]) or selected["query_id"].nunique() != int(expected["queries"]):
            raise ValueError(f"{task}: released prediction coverage differs")
        evaluator = selected[["query_id", "mutant", primary]].rename(
            columns={"query_id": "assay", primary: "score"}
        )
        evaluator = evaluator.sort_values(["assay", "score"], ascending=[True, False])
        atomic_write_csv(evaluator, args.output_dir / task / "evaluator_predictions.csv")
        print(f"{task}: queries={expected['queries']} samples={len(evaluator)}")


if __name__ == "__main__":
    main()
