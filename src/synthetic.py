"""Synthetic rows for local ML tests. Not a substitute for real GitHub data."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config


def make_synthetic_dataset(n: int = 120, seed: int = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed if seed is not None else config.RANDOM_SEED)
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        additions = int(rng.integers(1, 400))
        deletions = int(rng.integers(0, 200))
        files_changed = int(rng.integers(1, 25))
        n_test_files = int(rng.integers(0, 4))
        file_churn_count = int(rng.integers(0, 80))
        file_prior_bugfix_touch = int(rng.random() < 0.3)
        # Association, not causation: later bug-fix signal more likely with high churn and no tests.
        logit = (
            -1.2
            + 0.02 * file_churn_count
            + 0.8 * file_prior_bugfix_touch
            - 0.5 * (n_test_files > 0)
            + 0.002 * (additions + deletions)
        )
        p = 1 / (1 + np.exp(-logit))
        label = int(rng.random() < p)
        merged_at = start + timedelta(days=int(i * 3))
        rows.append(
            {
                "repo": "synthetic/demo",
                "pr_number": i + 1,
                "author": "synth",
                "merged_at": merged_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "additions": additions,
                "deletions": deletions,
                "files_changed": files_changed,
                "lines_changed": additions + deletions,
                "test_file_present": int(n_test_files > 0),
                "n_test_files": n_test_files,
                "source_file_touched": int(rng.random() < 0.85),
                "config_file_touched": int(rng.random() < 0.1),
                "dependency_file_touched": int(rng.random() < 0.1),
                "n_directories": int(rng.integers(1, 8)),
                "file_churn_count": file_churn_count,
                "file_prior_bugfix_touch": file_prior_bugfix_touch,
                "day_of_week": int(merged_at.weekday()),
                "hour_of_day": int(rng.integers(0, 24)),
                "label": label,
            }
        )
    return pd.DataFrame(rows)
