import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src.data_agent import (
    FeatureExtractionError,
    build_dataset,
    is_lookahead_complete,
    label_pr,
)
from src.github_client import PagedList
from src.dataset_validation import validate_dataset
from src.synthetic import make_synthetic_dataset


class TestRightCensoring(unittest.TestCase):
    def test_incomplete_recent_merge(self):
        now = datetime(2026, 8, 29, 20, 0, tzinfo=timezone.utc)
        recent = datetime(2026, 8, 11, 20, 19, 14, tzinfo=timezone.utc)
        old = datetime(2026, 5, 2, 3, 58, 57, tzinfo=timezone.utc)
        self.assertFalse(is_lookahead_complete(recent, now=now))
        self.assertTrue(is_lookahead_complete(old, now=now))

    def test_boundary_exactly_30_days(self):
        merged = datetime(2026, 7, 30, 17, 5, 5, tzinfo=timezone.utc)
        just_before = datetime(2026, 8, 29, 17, 5, 4, tzinfo=timezone.utc)
        just_on = datetime(2026, 8, 29, 17, 5, 5, tzinfo=timezone.utc)
        self.assertFalse(is_lookahead_complete(merged, now=just_before))
        self.assertTrue(is_lookahead_complete(merged, now=just_on))

    @patch("src.data_agent.featurize_pr")
    @patch("src.data_agent.label_pr")
    @patch("src.data_agent.fetch_merged_prs")
    def test_build_dataset_skips_incomplete(self, mock_fetch, mock_label, mock_feat):
        mock_fetch.return_value = (
            [
                {"number": 6133, "merged_at": "2026-08-11T20:19:14Z"},
                {"number": 6013, "merged_at": "2026-05-02T03:58:57Z"},
            ],
            {"closed_scanned": 2, "selected": 2},
        )
        mock_feat.return_value = {
            "pr_number": 6013,
            "files_changed": 1,
            "file_churn_count": 0,
            "file_churn_truncated": 0,
        }
        mock_label.return_value = 0
        now = datetime(2026, 8, 29, 20, 0, tzinfo=timezone.utc)
        df, stats = build_dataset(
            ["pallets/flask"],
            limit_per_repo=2,
            save_raw=False,
            now=now,
        )
        self.assertEqual(stats["skipped"]["incomplete_lookahead"], 1)
        self.assertEqual(len(df), 1)
        mock_feat.assert_called_once()

    def test_validate_rejects_incomplete_for_training(self):
        from datetime import timedelta

        df = make_synthetic_dataset(n=24)
        recent = (datetime.now(timezone.utc) - timedelta(days=10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        df.loc[0, "merged_at"] = recent
        training = validate_dataset(df, for_training=True)
        inspect = validate_dataset(df, for_training=False)
        self.assertFalse(training["ok"])
        self.assertTrue(inspect["ok"])
        self.assertTrue(any("incomplete" in e for e in training["errors"]))


class TestLabelTruncation(unittest.TestCase):
    @patch("src.data_agent.fetch_commits_for_path")
    def test_truncated_without_match_is_not_zero(self, mock_fetch):
        mock_fetch.return_value = PagedList(
            items=[{"sha": "x", "commit": {"message": "docs only"}}],
            truncated=True,
        )
        feats = {
            "merged_at": "2024-01-01T00:00:00Z",
            "filenames": ["src/app.py"],
            "exclude_shas": [],
            "pr_number": 1,
        }
        with self.assertRaises(FeatureExtractionError):
            label_pr("o/r", feats)


if __name__ == "__main__":
    unittest.main()
