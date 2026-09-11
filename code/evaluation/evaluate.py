"""Command-line entrypoint for benchmark evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.full import evaluate_full_setting, SUPPORTED_SETTINGS, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation.")
    parser.add_argument("--setting", choices=SUPPORTED_SETTINGS, default="T1_single_mutant_generation")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ground-truth-dir", default=None, type=Path)
    parser.add_argument("--lenient", action="store_true", help="Score available assays instead of failing on coverage issues.")
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
        strict=not args.lenient,
        assay_column=args.assay_column,
        mutant_column=args.mutant_column,
        rank_column=args.rank_column,
        score_column=args.score_column,
    )
    write_outputs(result, args.output_dir)

    metrics = result.summary["metrics"]
    metric_text = " ".join(f"{name}={value:.6f}" for name, value in metrics.items())
    print(
        f"{args.setting} "
        f"num_assays={result.summary['num_assays']} "
        f"num_queries={result.summary['num_queries']} "
        f"{metric_text}"
    )
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
