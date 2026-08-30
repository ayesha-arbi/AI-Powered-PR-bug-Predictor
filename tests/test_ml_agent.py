import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.dataset_validation import validate_dataset
from src.ml_agent import FEATURE_COLS, ModelError, predict_risk, train, temporal_split
from src.synthetic import make_synthetic_dataset


class TestDatasetValidation(unittest.TestCase):
    def test_empty(self):
        result = validate_dataset(pd.DataFrame())
        self.assertFalse(result["ok"])

    def test_duplicate_prs(self):
        df = make_synthetic_dataset(n=24)
        df.loc[1, "pr_number"] = df.loc[0, "pr_number"]
        result = validate_dataset(df)
        self.assertFalse(result["ok"])
        self.assertTrue(any("duplicate" in e for e in result["errors"]))

    def test_bad_hour(self):
        df = make_synthetic_dataset(n=24)
        df.loc[0, "hour_of_day"] = 99
        result = validate_dataset(df)
        self.assertFalse(result["ok"])


class TestTemporalSplit(unittest.TestCase):
    def test_older_then_newer(self):
        df = make_synthetic_dataset(n=40)
        train_df, test_df = temporal_split(df, test_size=0.25)
        self.assertLess(train_df["merged_at"].max(), test_df["merged_at"].min())


class TestTrainPredict(unittest.TestCase):
    def test_train_synthetic_and_predict_shape(self):
        df = make_synthetic_dataset(n=80)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.joblib"
            result = train(df, model_path=path, data_source="synthetic")
            self.assertEqual(result["primary_split"], "temporal")
            self.assertIn("f1", result["temporal"])
            self.assertEqual(len(result["temporal"]["confusion_matrix"]), 2)
            row = df.iloc[-1].to_dict()
            pred = predict_risk(row, model_path=path, allow_synthetic=True)
            self.assertIn("score", pred)
            self.assertGreaterEqual(pred["score"], 0.0)
            self.assertLessEqual(pred["score"], 1.0)
            self.assertEqual(list(pred["feature_values"].keys()), FEATURE_COLS)

    def test_missing_features_raise(self):
        df = make_synthetic_dataset(n=80)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.joblib"
            train(df, model_path=path, data_source="synthetic")
            with self.assertRaises(ModelError):
                predict_risk({"additions": 1}, model_path=path, allow_synthetic=True)

    def test_synthetic_model_blocked_for_real_scoring(self):
        df = make_synthetic_dataset(n=80)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.joblib"
            train(df, model_path=path, data_source="synthetic")
            row = df.iloc[-1].to_dict()
            with self.assertRaises(ModelError) as ctx:
                predict_risk(row, model_path=path, allow_synthetic=False)
            self.assertIn("synthetic", str(ctx.exception).lower())

    def test_unsupported_model_type(self):
        df = make_synthetic_dataset(n=80)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.joblib"
            with self.assertRaises(ModelError):
                train(df, model_type="xgboost", model_path=path, data_source="synthetic")

    def test_temporal_duplicate_timestamps_deterministic(self):
        df = make_synthetic_dataset(n=40)
        df.loc[10, "merged_at"] = df.loc[9, "merged_at"]
        a_train, a_test = temporal_split(df)
        b_train, b_test = temporal_split(df)
        self.assertListEqual(list(a_train["pr_number"]), list(b_train["pr_number"]))
        self.assertLessEqual(a_train["merged_at"].iloc[-1], a_test["merged_at"].iloc[0])


if __name__ == "__main__":
    unittest.main()
