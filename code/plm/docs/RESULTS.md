# Released results

Each model directory under `results/` contains:

- `predictions.csv.gz`: all 607,269 evaluation candidates and every retained model output.
- `context_predictions.csv.gz`: all 28,676 provided-context samples.
- `chain_contributions.csv.gz`: pre-aggregation chain-level outputs for the five
  direct inference models; S3F-MSA stores its already aggregated sample outputs.
- `tasks/<task>/evaluator_predictions.csv`: the exact two-column ranking input plus mutant identity.
- `evaluation/<task>`: per-query and summary metrics from the unified evaluator.
- `run.json` and `output_schema.json`: model, dataset, score-direction, and provenance metadata.

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
