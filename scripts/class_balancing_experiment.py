"""
Controlled Class-Balancing Experiment (Z-Score Churn Baseline)
============================================================

This script evaluates the effect of training-derived class weights vs. no class balancing
on top of the z-score churn representation baseline across:
1. Four Leave-One-Repository-Out (LORO) folds:
   - django/django
   - kubernetes/kubernetes
   - numpy/numpy
   - pallets/flask
2. Temporal Split (within-repo temporal split)

Two Configurations:
1. No balancing (sample_weight=None)
2. Training-derived balanced weights:
   w_y = N / (len(classes) * count(y)) computed strictly from training split labels.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_PATH = Path("data/processed/all_repos.csv")

BASE_FEATURE_COLS = [
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


def compute_sample_weights(y_train):
    counts = pd.Series(y_train).value_counts()
    n = len(y_train)
    sample_weight = np.ones(n, dtype=float)
    if len(counts) > 1:
        sample_weight = np.array(
            [n / (len(counts) * counts[y]) for y in y_train], dtype=float
        )
    return sample_weight


def evaluate_predictions(y_true, y_pred, y_prob):
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def transform_churn_zscore_loro(train_df, test_df):
    train_out = train_df.copy()
    test_out = test_df.copy()

    repo_stats = {}
    for repo, grp in train_df.groupby("repo"):
        vals = grp["file_churn_count"].values
        repo_stats[repo] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)) + 1e-5,
        }
    pooled_vals = train_df["file_churn_count"].values
    pooled_stats = {
        "mean": float(np.mean(pooled_vals)),
        "std": float(np.std(pooled_vals)) + 1e-5,
    }

    train_vals = []
    for _, row in train_df.iterrows():
        r = row["repo"]
        c = row["file_churn_count"]
        st = repo_stats.get(r, pooled_stats)
        train_vals.append((c - st["mean"]) / st["std"])

    test_vals = []
    for _, row in test_df.iterrows():
        c = row["file_churn_count"]
        test_vals.append((c - pooled_stats["mean"]) / pooled_stats["std"])

    train_out["churn_zscore"] = np.array(train_vals, dtype=float)
    test_out["churn_zscore"] = np.array(test_vals, dtype=float)

    feature_cols = [c if c != "file_churn_count" else "churn_zscore" for c in BASE_FEATURE_COLS]
    return train_out, test_out, feature_cols


def transform_churn_zscore_temporal(train_df, test_df):
    train_out = train_df.copy()
    test_out = test_df.copy()

    repo_stats = {}
    for repo, grp in train_df.groupby("repo"):
        vals = grp["file_churn_count"].values
        repo_stats[repo] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)) + 1e-5,
        }
    pooled_vals = train_df["file_churn_count"].values
    pooled_stats = {
        "mean": float(np.mean(pooled_vals)),
        "std": float(np.std(pooled_vals)) + 1e-5,
    }

    def apply_transform(df_in):
        out_vals = []
        for _, row in df_in.iterrows():
            r = row["repo"]
            c = row["file_churn_count"]
            st = repo_stats.get(r, pooled_stats)
            out_vals.append((c - st["mean"]) / st["std"])
        return np.array(out_vals, dtype=float)

    train_out["churn_zscore"] = apply_transform(train_df)
    test_out["churn_zscore"] = apply_transform(test_df)

    feature_cols = [c if c != "file_churn_count" else "churn_zscore" for c in BASE_FEATURE_COLS]
    return train_out, test_out, feature_cols


def run_temporal(df, use_weighting=True):
    ordered = df.sort_values(["merged_at", "pr_number"], kind="mergesort").reset_index(drop=True)
    cut = int(len(ordered) * 0.75)
    train_raw = ordered.iloc[:cut].copy()
    test_raw = ordered.iloc[cut:].copy()

    train_df, test_df, feat_cols = transform_churn_zscore_temporal(train_raw, test_raw)

    X_train = train_df[feat_cols]
    y_train = train_df["label"].astype(int).values
    X_test = test_df[feat_cols]
    y_test = test_df["label"].astype(int).values

    model = GradientBoostingClassifier(random_state=42)
    sample_weight = compute_sample_weights(y_train) if use_weighting else None
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.45).astype(int)

    metrics = evaluate_predictions(y_test, y_pred, y_prob)
    metrics["n_train"] = len(train_df)
    metrics["n_test"] = len(test_df)
    return metrics


def run_loro(df, use_weighting=True):
    repos = sorted(df["repo"].unique())
    fold_results = {}

    for held_out in repos:
        train_raw = df[df["repo"] != held_out].copy()
        test_raw = df[df["repo"] == held_out].copy()

        train_df, test_df, feat_cols = transform_churn_zscore_loro(train_raw, test_raw)

        X_train = train_df[feat_cols]
        y_train = train_df["label"].astype(int).values
        X_test = test_df[feat_cols]
        y_test = test_df["label"].astype(int).values

        model = GradientBoostingClassifier(random_state=42)
        sample_weight = compute_sample_weights(y_train) if use_weighting else None
        model.fit(X_train, y_train, sample_weight=sample_weight)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.45).astype(int)

        res = evaluate_predictions(y_test, y_pred, y_prob)
        res["repo"] = held_out
        fold_results[held_out] = res

    f1_mean = float(np.mean([r["f1"] for r in fold_results.values()]))
    roc_auc_mean = float(np.mean([r["roc_auc"] for r in fold_results.values()]))
    pr_auc_mean = float(np.mean([r["pr_auc"] for r in fold_results.values()]))
    precision_mean = float(np.mean([r["precision"] for r in fold_results.values()]))
    recall_mean = float(np.mean([r["recall"] for r in fold_results.values()]))
    accuracy_mean = float(np.mean([r["accuracy"] for r in fold_results.values()]))

    return {
        "folds": fold_results,
        "mean_f1": f1_mean,
        "mean_roc_auc": roc_auc_mean,
        "mean_pr_auc": pr_auc_mean,
        "mean_precision": precision_mean,
        "mean_recall": recall_mean,
        "mean_accuracy": accuracy_mean,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {len(df)} rows across {df['repo'].nunique()} repositories.")

    # 1. No balancing
    print("\nRunning Configuration 1: No balancing...")
    no_bal_temporal = run_temporal(df, use_weighting=False)
    no_bal_loro = run_loro(df, use_weighting=False)

    # 2. Training-derived class weights
    print("Running Configuration 2: Training-derived class weights...")
    weighted_temporal = run_temporal(df, use_weighting=True)
    weighted_loro = run_loro(df, use_weighting=True)

    results = {
        "no_balancing": {
            "temporal": no_bal_temporal,
            "loro": no_bal_loro,
        },
        "training_weighted": {
            "temporal": weighted_temporal,
            "loro": weighted_loro,
        },
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/class_balancing_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved full results to results/class_balancing_results.json")

    # Format Output Tables
    def format_table(config_name, data):
        rows = []
        # Temporal row
        t = data["temporal"]
        rows.append({
            "Split / Repo": "Temporal Split",
            "F1": t["f1"],
            "Precision": t["precision"],
            "Recall": t["recall"],
            "ROC-AUC": t["roc_auc"],
            "PR-AUC": t["pr_auc"],
        })
        # LORO folds
        for repo in sorted(data["loro"]["folds"].keys()):
            f = data["loro"]["folds"][repo]
            rows.append({
                "Split / Repo": f"LORO: {repo}",
                "F1": f["f1"],
                "Precision": f["precision"],
                "Recall": f["recall"],
                "ROC-AUC": f["roc_auc"],
                "PR-AUC": f["pr_auc"],
            })
        # LORO Mean
        l = data["loro"]
        rows.append({
            "Split / Repo": "LORO: Mean",
            "F1": l["mean_f1"],
            "Precision": l["mean_precision"],
            "Recall": l["mean_recall"],
            "ROC-AUC": l["mean_roc_auc"],
            "PR-AUC": l["mean_pr_auc"],
        })
        df_out = pd.DataFrame(rows)
        print(f"\n--- {config_name} ---")
        print(df_out.to_string(index=False))
        return df_out

    df_no_bal = format_table("Configuration 1: No Balancing", results["no_balancing"])
    df_weighted = format_table("Configuration 2: Training-Derived Class Weights", results["training_weighted"])

    # Delta table (Weighted - No Balancing)
    delta_rows = []
    for i in range(len(df_no_bal)):
        name = df_no_bal.loc[i, "Split / Repo"]
        delta_rows.append({
            "Split / Repo": name,
            "Delta F1": df_weighted.loc[i, "F1"] - df_no_bal.loc[i, "F1"],
            "Delta Precision": df_weighted.loc[i, "Precision"] - df_no_bal.loc[i, "Precision"],
            "Delta Recall": df_weighted.loc[i, "Recall"] - df_no_bal.loc[i, "Recall"],
            "Delta ROC-AUC": df_weighted.loc[i, "ROC-AUC"] - df_no_bal.loc[i, "ROC-AUC"],
            "Delta PR-AUC": df_weighted.loc[i, "PR-AUC"] - df_no_bal.loc[i, "PR-AUC"],
        })
    df_delta = pd.DataFrame(delta_rows)
    print("\n--- Delta Table (Weighted minus No-Balancing) ---")
    print(df_delta.to_string(index=False))


if __name__ == "__main__":
    main()
