"""CLI: assess a PR, collect a small dataset, or train.

Keep the original entry point:

    python run.py --repo owner/name --pr 123
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from src.github_client import GitHubAPIError
from src import config
from src.data_agent import build_dataset, ensure_data_dirs
from src.ml_agent import train
from src.orchestrator import FeaturePipelineError, run_pipeline
from src.synthetic import make_synthetic_dataset


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="PR bug-fix-signal association risk (ML + optional Groq explanation)"
    )
    parser.add_argument(
    "--repo",
    action="append",
    help="owner/name; can be provided multiple times, e.g. --repo django/django --repo numpy/numpy",
)
    parser.add_argument("--pr", type=int, help="PR number")
    parser.add_argument("--no-llm", action="store_true", help="Skip Groq; ML score only")
    parser.add_argument(
        "--model",
        help="Path to joblib bundle (default: models/risk_classifier.joblib)",
    )
    parser.add_argument(
        "--model-type",
        default="gboost",
        choices=list(config.SUPPORTED_MODEL_TYPES),
        help="Estimator for --train / --train-synthetic (default: gboost)",
    )
    parser.add_argument("--json", action="store_true", default=True, help="Print JSON (always on)")
    parser.add_argument("--collect", action="store_true", help="Collect merged PRs into a CSV")
    parser.add_argument("--limit", type=int, default=25, help="Merged PRs per repo for --collect")
    parser.add_argument(
        "--include-incomplete-lookahead",
        action="store_true",
        help="Keep PRs whose 30-day label window has not elapsed (not for training)",
    )
    parser.add_argument("--train", action="store_true", help="Train from processed CSV")
    parser.add_argument("--csv", help="Dataset CSV path")
    parser.add_argument("--train-synthetic", action="store_true", help="Train on synthetic data (no GitHub)")
    parser.add_argument(
        "--allow-synthetic-model",
        action="store_true",
        help="Allow scoring a real PR with a synthetic-trained bundle",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    ensure_data_dirs()

    if args.train_synthetic:
        df = make_synthetic_dataset(n=120)
        dest = args.model or str(config.SYNTHETIC_MODEL_PATH)
        result = train(
            df,
            model_type=args.model_type,
            model_path=dest,
            data_source="synthetic",
        )
        print(json.dumps(_public_train_result(result), indent=2, default=str))
        return 0

    if args.collect:
        if not args.repo:
            parser.error("--collect requires at least one --repo")
        out = args.csv or str(config.DATA_PROCESSED_DIR / "dataset.csv")
        df, stats = build_dataset(
        args.repo,
        limit_per_repo=args.limit,
        save_raw=True,
        require_complete_lookahead=not args.include_incomplete_lookahead,
    )
        if df.empty:
            print(json.dumps({"error": "No rows collected.", "stats": stats}, indent=2, default=str))
            return 1
        df.to_csv(out, index=False)
        print(
            json.dumps(
                {
                    "saved": out,
                    "rows": int(len(df)),
                    "label_counts": df["label"].value_counts().to_dict(),
                    "stats": stats,
                },
                indent=2,
                default=str,
            )
        )
        return 0

    if args.train:
        import pandas as pd

        path = args.csv or str(config.DATA_PROCESSED_DIR / "dataset.csv")
        df = pd.read_csv(path)
        result = train(
            df,
            model_type=args.model_type,
            model_path=args.model,
            data_source="github",
        )
        print(json.dumps(_public_train_result(result), indent=2, default=str))
        return 0

    if not args.repo or args.pr is None:
        parser.error("provide --repo and --pr, or --collect / --train / --train-synthetic")

    try:
        report = run_pipeline(
    args.repo[0],
    args.pr,
    use_llm=not args.no_llm,
    model_path=args.model,
    allow_synthetic=args.allow_synthetic_model,
)
    except (FeaturePipelineError, GitHubAPIError) as exc:
        print(json.dumps({"error": str(exc), "repo": args.repo, "pr_number": args.pr}, indent=2))
        return 1

    print(json.dumps(report, indent=2, default=str))
    return 0


def _public_train_result(result: dict) -> dict:
    skip = {"model"}
    return {k: v for k, v in result.items() if k not in skip}


if __name__ == "__main__":
    raise SystemExit(main())
