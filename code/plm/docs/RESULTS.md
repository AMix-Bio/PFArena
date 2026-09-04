# Released results

Each model directory under `results/` contains:

- `predictions.csv.gz`: all 607,269 evaluation candidates and every retained model output.
- `evaluation/<task>`: per-query and summary metrics from the unified evaluator.
- `run.json` and `output_schema.json`: model, dataset, score-direction, and provenance metadata.

Chain-level contributions, provided-context predictions, and evaluator-format
copies are deterministic intermediate or derived artifacts and are not
distributed. Their original row counts remain in `run.json` as inference
provenance. Recreate evaluator inputs from a model directory with:

```bash
python export_evaluator_predictions.py \
  --result-dir results/<model> \
  --output-dir /path/to/evaluator_inputs/<model>
```

`results/leaderboard.csv` is the normalized long-form table across all six models and four tasks. It is derived from, rather than substituted for, the per-model evaluation outputs.

Paths recorded inside historical run and evaluation metadata identify the
inputs used for the reported experiments; they are provenance fields, not
repository-relative runtime dependencies. New evaluations should use the
canonical dataset and evaluator paths supplied by the parent project.

T1 uses NMS@40 and Recall@40 to measure the quality and recovery of the top 40
single-mutation proposals. T2--T4 use Spearman correlation and NDCG for global
ranking agreement, plus NMS@5 and Recall@5 for top-candidate quality and
recovery. Metrics are computed independently for each query and macro-averaged;
the per-query values are retained in each evaluation directory.
