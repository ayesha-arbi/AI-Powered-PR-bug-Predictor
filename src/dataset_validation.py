"""Dataset sanity checks used before training."""
from __future__ import annotations

import pandas as pd

from . import config
from .data_agent import is_lookahead_complete, parse_github_datetime

FEATURE_COLS = config.FEATURE_COLS


def validate_dataset(df: pd.DataFrame, *, for_training: bool = True) -> dict:
    """Return {ok, errors, warnings, stats}.

    Inspection can use for_training=False so incomplete windows are warnings.
    Training treats incomplete 30-day windows as errors.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if df is None or df.empty:
        return {
            "ok": False,
            "errors": ["dataset is empty"],
            "warnings": [],
            "stats": {"n_rows": 0},
        }

    n = int(len(df))
    if "label" not in df.columns:
        errors.append("missing label column")
    for col in FEATURE_COLS:
        if col not in df.columns:
            errors.append(f"missing feature column: {col}")

    if "repo" in df.columns and "pr_number" in df.columns:
        dup = df.duplicated(subset=["repo", "pr_number"]).sum()
        if dup:
            errors.append(f"{int(dup)} duplicate repo+pr_number rows")
    full_dup = df.duplicated().sum()
    if full_dup:
        warnings.append(f"{int(full_dup)} fully duplicate rows")

    stats: dict = {"n_rows": n}

    incomplete_n = 0
    if "merged_at" in df.columns:
        bad_ts = 0
        for raw in df["merged_at"]:
            try:
                merged = parse_github_datetime(str(raw))
            except (ValueError, TypeError):
                bad_ts += 1
                continue
            if not is_lookahead_complete(merged):
                incomplete_n += 1
        if bad_ts:
            errors.append(f"{bad_ts} invalid merged_at timestamps")
        stats["incomplete_lookahead_rows"] = incomplete_n
        msg = (
            f"{incomplete_n} rows have an incomplete {config.LOOKAHEAD_DAYS}-day "
            "lookahead and cannot be treated as confirmed negatives"
        )
        if incomplete_n:
            if for_training:
                errors.append(msg)
            else:
                warnings.append(msg)
    else:
        errors.append("missing merged_at (required for temporal split and censoring checks)")

    if "label" in df.columns:
        vc = df["label"].value_counts(dropna=False).to_dict()
        stats["label_counts"] = {str(k): int(v) for k, v in vc.items()}
        if df["label"].isna().any():
            errors.append("label contains missing values")
        unique = set(df["label"].dropna().unique())
        if not unique.issubset({0, 1, 0.0, 1.0}):
            errors.append(f"label values must be 0/1, got {sorted(unique, key=str)}")
        pos = int((df["label"] == 1).sum())
        stats["positive_rate"] = round(pos / n, 4) if n else 0.0
        if pos == 0 or pos == n:
            warnings.append("label is constant; a classifier cannot learn a useful boundary")
        elif min(pos, n - pos) / n < 0.05:
            warnings.append("severe class imbalance (<5% minority)")

    present = [c for c in FEATURE_COLS if c in df.columns]
    for col in present:
        missing = int(df[col].isna().sum())
        if missing:
            errors.append(f"{col} has {missing} missing values")
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique(dropna=True) <= 1:
                warnings.append(f"constant feature: {col}")
            if (df[col] < 0).any() and col not in {"day_of_week"}:
                warnings.append(f"{col} has negative values")

    if "files_changed" in df.columns and (df["files_changed"] < 0).any():
        errors.append("files_changed must be >= 0")
    if "hour_of_day" in df.columns:
        if ((df["hour_of_day"] < 0) | (df["hour_of_day"] > 23)).any():
            errors.append("hour_of_day must be in 0..23")
    if "day_of_week" in df.columns:
        if ((df["day_of_week"] < 0) | (df["day_of_week"] > 6)).any():
            errors.append("day_of_week must be in 0..6")

    if n < 50:
        warnings.append(
            f"only {n} rows; metrics are unstable and not scientifically meaningful"
        )

    stats["n_errors"] = len(errors)
    stats["n_warnings"] = len(warnings)
    return {"ok": not errors, "errors": errors, "warnings": warnings, "stats": stats}
