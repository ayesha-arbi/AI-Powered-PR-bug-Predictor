from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


DATA = Path("data/processed/all_repos.csv")

FEATURE_COLS = [
    "additions",
    "deletions",
    "files_changed",
    "lines_changed",
    "test_file_present",
    "n_test_files",
    "source_file_touched",
    "config_file_touched",
    "dependency_file_touched",
    "n_directories",
    "file_churn_count",
    "file_prior_bugfix_touch",
    "day_of_week",
    "hour_of_day",
]


def evaluate(y_true, y_pred, y_prob):
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


df = pd.read_csv(DATA)

repos = sorted(df["repo"].unique())

print("=" * 80)
print("LEAVE-ONE-REPOSITORY-OUT EVALUATION")
print("=" * 80)

all_results = []

for held_out in repos:
    train_df = df[df["repo"] != held_out].copy()
    test_df = df[df["repo"] == held_out].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["label"]

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.45).astype(int)

    result = evaluate(y_test, y_pred, y_prob)
    result["repo"] = held_out
    result["n_train"] = len(train_df)
    result["n_test"] = len(test_df)
    result["test_positive_rate"] = float(y_test.mean())

    all_results.append(result)

    print(f"\nHELD OUT: {held_out}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows:  {len(test_df)}")
    print(f"Test positive rate: {y_test.mean():.3f}")
    print(f"F1:        {result['f1']:.3f}")
    print(f"Precision: {result['precision']:.3f}")
    print(f"Recall:    {result['recall']:.3f}")
    print(f"ROC-AUC:   {result['roc_auc']:.3f}")
    print(f"PR-AUC:    {result['pr_auc']:.3f}")
    print(f"Accuracy:  {result['accuracy']:.3f}")
    print(f"Confusion: {result['confusion_matrix']}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

summary = pd.DataFrame(all_results)

print(
    summary[
        [
            "repo",
            "n_test",
            "test_positive_rate",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "pr_auc",
            "accuracy",
        ]
    ].to_string(index=False)
)

print("\nMean across repositories:")
for metric in ["f1", "precision", "recall", "roc_auc", "pr_auc", "accuracy"]:
    print(f"{metric:10}: {summary[metric].mean():.3f}")