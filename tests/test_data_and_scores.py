import unittest
from unittest.mock import patch

from src.data_agent import FeatureExtractionError, featurize_pr, label_pr
from src.github_client import GitHubAPIError
from src.orchestrator import is_borderline, label_from_score


class TestScoreBands(unittest.TestCase):
    def test_low_medium_high(self):
        self.assertEqual(label_from_score(0.0), "low")
        self.assertEqual(label_from_score(0.3999), "low")
        self.assertEqual(label_from_score(0.4), "medium")
        self.assertEqual(label_from_score(0.5), "medium")
        self.assertEqual(label_from_score(0.6), "medium")
        self.assertEqual(label_from_score(0.6001), "high")
        self.assertEqual(label_from_score(1.0), "high")

    def test_borderline_inclusive(self):
        self.assertTrue(is_borderline(0.4))
        self.assertTrue(is_borderline(0.6))
        self.assertFalse(is_borderline(0.3999))
        self.assertFalse(is_borderline(0.6001))


class TestLabelBoundaries(unittest.TestCase):
    def _features(self, exclude):
        return {
            "merged_at": "2024-01-01T00:00:00Z",
            "filenames": ["src/app.py"],
            "exclude_shas": exclude,
            "pr_number": 7,
        }

    @patch("src.data_agent.fetch_commits_for_path")
    def test_excludes_pr_and_merge_shas(self, mock_fetch):
        mock_fetch.return_value = [
            {"sha": "merge", "commit": {"message": "fix crash"}},
            {"sha": "later", "commit": {"message": "fix crash"}},
        ]
        self.assertEqual(label_pr("o/r", self._features(["merge"])), 1)
        self.assertEqual(label_pr("o/r", self._features(["merge", "later"])), 0)

    @patch("src.data_agent.fetch_commits_for_path")
    def test_label_api_failure_is_not_zero(self, mock_fetch):
        mock_fetch.side_effect = GitHubAPIError("boom")
        with self.assertRaises(FeatureExtractionError):
            label_pr("o/r", self._features([]))


class TestFeaturizeFailures(unittest.TestCase):
    def test_unmerged_pr(self):
        with self.assertRaises(FeatureExtractionError):
            featurize_pr("o/r", {"number": 1, "merged_at": None})

    @patch("src.data_agent.fetch_pr_commits")
    @patch("src.data_agent.fetch_pr_files")
    def test_churn_uses_until_merge_not_future(self, mock_files, mock_commits):
        mock_files.return_value = [{"filename": "src/a.py", "additions": 1, "deletions": 0}]
        mock_commits.return_value = [{"sha": "abc"}]
        captured = []

        def fake_commits(repo, path, since_iso, until_iso):
            captured.append((since_iso, until_iso))
            return []

        merged = "2024-06-01T12:00:00Z"
        with patch("src.data_agent.fetch_commits_for_path", side_effect=fake_commits):
            from src.data_agent import featurize_pr

            featurize_pr(
                "o/r",
                {
                    "number": 3,
                    "merged_at": merged,
                    "user": {"login": "a"},
                    "merge_commit_sha": "abc",
                    "head": {"sha": "abc"},
                },
            )
        self.assertTrue(captured)
        for since_iso, until_iso in captured:
            self.assertEqual(until_iso, merged)
            self.assertLess(since_iso, until_iso)

    @patch("src.data_agent.fetch_commits_for_path")
    def test_label_since_is_after_merge(self, mock_fetch):
        mock_fetch.return_value = []
        label_pr(
            "o/r",
            {
                "merged_at": "2024-01-01T00:00:00Z",
                "filenames": ["src/app.py"],
                "exclude_shas": [],
                "pr_number": 7,
            },
        )
        args, kwargs = mock_fetch.call_args
        since_iso = args[2]
        until_iso = args[3]
        self.assertGreater(since_iso, "2024-01-01T00:00:00Z")
        self.assertEqual(until_iso, "2024-01-31T00:00:00Z")

    @patch("src.data_agent.fetch_pr_commits")
    @patch("src.data_agent.fetch_pr_files")
    def test_churn_failure_not_zero(self, mock_files, mock_commits):
        mock_files.return_value = [{"filename": "src/a.py", "additions": 1, "deletions": 0}]
        mock_commits.return_value = [{"sha": "abc"}]
        with patch("src.data_agent.fetch_commits_for_path", side_effect=GitHubAPIError("rate")):
            with self.assertRaises(FeatureExtractionError):
                featurize_pr(
                    "o/r",
                    {
                        "number": 3,
                        "merged_at": "2024-06-01T12:00:00Z",
                        "user": {"login": "a"},
                        "merge_commit_sha": "abc",
                        "head": {"sha": "abc"},
                    },
                )


if __name__ == "__main__":
    unittest.main()
