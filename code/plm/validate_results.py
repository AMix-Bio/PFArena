#!/usr/bin/env python3
"""Validate the released protein-model predictions and evaluation summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODELS = ("esm2", "progen2_base", "prosst_2048", "s3f", "s3f_msa", "venusrem")


def main() -> None:
    expected_dataset_hash = None
    leaderboard_rows = []
    for model in MODELS:
        result_dir = ROOT / "results" / model
        run = json.loads((result_dir / "run.json").read_text())
        schema = json.loads((result_dir / "output_schema.json").read_text())
        primary = schema["primary_score"]
        numeric = list(schema["sample_outputs"])
        if primary not in numeric:
            raise ValueError(f"{model}: primary score is absent from output schema")
        dataset_hash = run["dataset_hash"]
        if expected_dataset_hash is None:
            expected_dataset_hash = dataset_hash
        elif dataset_hash != expected_dataset_hash:
            raise ValueError(f"{model}: dataset hash differs")

        predictions = pd.read_csv(
            result_dir / "predictions.csv.gz", float_precision="round_trip"
        )
        if len(predictions) != run["evaluation_samples"]:
            raise ValueError(f"{model}: result count differs from run.json")
        if predictions.sample_id.duplicated().any() or not predictions.status.eq("ok").all():
            raise ValueError(f"{model}: duplicate or failed sample")
        if not predictions.sample_role.eq("evaluation_candidate").all():
            raise ValueError(f"{model}: non-evaluation sample in released predictions")
        if not np.isfinite(predictions[numeric].to_numpy()).all():
            raise ValueError(f"{model}: non-finite model output")

        manifest = json.loads((result_dir / "evaluation/evaluation_manifest.json").read_text())
        if set(predictions.task) != set(run["tasks"]) or set(manifest["tasks"]) != set(run["tasks"]):
            raise ValueError(f"{model}: released task set differs")
        for task, task_info in run["tasks"].items():
            selected = predictions[predictions.task.eq(task)]
            if len(selected) != task_info["samples"] or selected.query_id.nunique() != task_info["queries"]:
                raise ValueError(f"{model}/{task}: task coverage differs")
            summary = json.loads(
                (result_dir / "evaluation" / task / "summary_metrics.json").read_text()
            )
            per_query = pd.read_csv(
                result_dir / "evaluation" / task / "per_assay_metrics.csv"
            )
            if len(per_query) != int(task_info["queries"]) or per_query["assay"].duplicated().any():
                raise ValueError(f"{model}/{task}: per-query metric coverage differs")
            if set(per_query["assay"]) != set(selected["query_id"]):
                raise ValueError(f"{model}/{task}: per-query metric identities differ")
            if summary["quality"]["missing_assays"] or summary["quality"]["unknown_assays"]:
                raise ValueError(f"{model}/{task}: evaluation coverage is incomplete")
            if summary["metrics"] != manifest["tasks"][task]:
                raise ValueError(f"{model}/{task}: evaluation metrics differ")
            summary_csv = pd.read_csv(
                result_dir / "evaluation" / task / "summary_metrics.csv"
            )
            if len(summary_csv) != 1 or summary_csv.loc[0, "setting"] != task:
                raise ValueError(f"{model}/{task}: summary CSV identity differs")
            for metric, value in summary["metrics"].items():
                if not np.isclose(summary_csv.loc[0, metric], value, rtol=0, atol=1e-12):
                    raise ValueError(f"{model}/{task}: summary CSV metric differs")
            leaderboard_rows.extend(
                {
                    "model": model,
                    "model_id": run["model_id"],
                    "task": task,
                    "metric": metric,
                    "value": value,
                }
                for metric, value in summary["metrics"].items()
            )
        print(f"{model}: OK")

    expected_leaderboard = pd.DataFrame(leaderboard_rows)
    leaderboard = pd.read_csv(ROOT / "results/leaderboard.csv")
    pd.testing.assert_frame_equal(
        leaderboard.sort_values(["task", "metric", "model"]).reset_index(drop=True),
        expected_leaderboard.sort_values(["task", "metric", "model"]).reset_index(drop=True),
        check_dtype=False,
        rtol=0,
        atol=1e-12,
    )
    print("leaderboard.csv: OK")


if __name__ == "__main__":
    main()
