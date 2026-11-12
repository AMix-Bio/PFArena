"""Command-line entrypoint for benchmark evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.full import evaluate_full_setting, SUPPORTED_SETTINGS, setting_requires_full_ranking, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation.")
    parser.add_argument("--setting", choices=SUPPORTED_SETTINGS, default="T1_single_mutant_generation")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ground-truth-dir", default=None, type=Path)
    parser.add_argument("--lenient", type=int, choices=(0, 1, 2), nargs="?", const=1, default=1, help="0=fail on invalid/missing cases; 1=score provided rankings and randomly replace missing cases (default; T1 also replaces below-budget cases); 2=skip and report missing/incomplete cases.")
    parser.add_argument("--assay-column", default=None)
    parser.add_argument("--mutant-column", default=None)
    parser.add_argument("--rank-column", default="rank")
    parser.add_argument("--score-column", default="score")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_full_setting(
        setting=args.setting,
        ground_truth_dir=args.ground_truth_dir,
        predictions_path=args.predictions,
        lenient=args.lenient,
        assay_column=args.assay_column,
        mutant_column=args.mutant_column,
        rank_column=args.rank_column,
        score_column=args.score_column,
    )
    write_outputs(result, args.output_dir)

    metrics = result.summary["metrics"]
    metric_text = " ".join(f"{name}={value:.6f}" if value is not None else f"{name}=NA" for name, value in metrics.items())
    print(f"========== {args.setting} ==========")
    print(f"==> {metric_text}")
    incomplete_candidates_text = (
        f"incomplete_candidates={result.summary['quality']['assays_with_incomplete_candidate_ranking']} "
        if setting_requires_full_ranking(args.setting)
        else ""
    )
    print(
        f"[Completeness] "
        f"total_assays={result.summary['num_target_cases']} "
        f"evaluated_assays={result.summary['num_assays']} "
        f"missing_assays={result.summary['quality']['missing_assays']} "
        f"{incomplete_candidates_text}"
        f"random_replacements={result.summary['quality']['random_replacement_cases']} "
        f"skipped_cases={result.summary['quality']['skipped_cases']}"
    )
    print(f"[Output] wrote {args.output_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
