import unittest
from unittest.mock import patch

from src.llm_agent import LLMUnavailableError, format_evidence
from src.orchestrator import run_pipeline


class TestEvidenceGrounding(unittest.TestCase):
    def test_evidence_contains_supplied_numbers(self):
        pr_meta = {
            "repo": "o/r",
            "pr_number": 9,
            "files_changed": 4,
            "additions": 10,
            "deletions": 2,
            "n_test_files": 1,
            "test_file_present": 1,
            "file_churn_count": 6,
            "file_prior_bugfix_touch": 0,
        }
        risk = {
            "score": 0.51,
            "feature_values": {"lines_changed": 12},
            "model_wide_importances": [{"feature": "file_churn_count", "importance": 0.2}],
            "importance_note": "model-wide",
        }
        text = format_evidence(pr_meta, risk)
        self.assertIn("0.51", text)
        self.assertIn("Files changed: 4", text)
        self.assertIn("model-wide", text.lower())


class TestPipelineNoLlm(unittest.TestCase):
    @patch("src.ml_agent.predict_risk")
    @patch("src.data_agent.featurize_pr")
    @patch("src.data_agent.fetch_pr")
    def test_no_llm_skips_groq(self, mock_pr, mock_feat, mock_pred):
        mock_pr.return_value = {"number": 1, "merged_at": "2024-01-01T00:00:00Z"}
        mock_feat.return_value = {"pr_number": 1, "merged_at": "2024-01-01T00:00:00Z"}
        mock_pred.return_value = {
            "score": 0.2,
            "top_features": [],
            "model_wide_importances": [],
            "importance_note": "x",
            "feature_values": {},
        }
        report = run_pipeline("o/r", 1, use_llm=False)
        self.assertEqual(report["llm_status"], "skipped")
        self.assertEqual(report["risk_label"], "low")
        self.assertIsNone(report["explanation"])

    @patch("src.llm_agent.explain", side_effect=LLMUnavailableError("no key"))
    @patch("src.ml_agent.predict_risk")
    @patch("src.data_agent.featurize_pr")
    @patch("src.data_agent.fetch_pr")
    def test_llm_failure_keeps_score(self, mock_pr, mock_feat, mock_pred, _explain):
        mock_pr.return_value = {"number": 1, "merged_at": "2024-01-01T00:00:00Z"}
        mock_feat.return_value = {"pr_number": 1, "merged_at": "2024-01-01T00:00:00Z"}
        mock_pred.return_value = {
            "score": 0.51,
            "top_features": [],
            "model_wide_importances": [],
            "importance_note": "x",
            "feature_values": {},
        }
        report = run_pipeline("o/r", 1, use_llm=True)
        self.assertEqual(report["risk_score"], 0.51)
        self.assertEqual(report["risk_label"], "medium")
        self.assertEqual(report["llm_status"], "unavailable")
        self.assertNotIn("contrarian", report)  # returned before contrarian

    @patch("src.llm_agent.contrarian_pass")
    @patch("src.llm_agent.explain", return_value="looks associated")
    @patch("src.ml_agent.predict_risk")
    @patch("src.data_agent.featurize_pr")
    @patch("src.data_agent.fetch_pr")
    def test_contrarian_does_not_change_score(
        self, mock_pr, mock_feat, mock_pred, _explain, mock_con
    ):
        mock_pr.return_value = {"number": 1, "merged_at": "2024-01-01T00:00:00Z"}
        mock_feat.return_value = {"pr_number": 1, "merged_at": "2024-01-01T00:00:00Z"}
        mock_pred.return_value = {
            "score": 0.51,
            "top_features": [],
            "model_wide_importances": [],
            "importance_note": "x",
            "feature_values": {},
            "trained_on": "github",
        }
        mock_con.return_value = {
            "counter_argument": "maybe not",
            "llm_synthesis": "high association-risk",
            "note": "does not change score",
        }
        report = run_pipeline("o/r", 1, use_llm=True)
        self.assertEqual(report["risk_score"], 0.51)
        self.assertEqual(report["risk_label"], "medium")
        self.assertEqual(report["contrarian"]["llm_synthesis"], "high association-risk")


if __name__ == "__main__":
    unittest.main()
