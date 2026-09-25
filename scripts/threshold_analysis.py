"""
Threshold-Sensitivity Analysis on Z-Score Churn Representation
=============================================================

This script performs threshold tuning and sensitivity analysis on the
z-score churn representation across:
1. Four Leave-One-Repository-Out (LORO) folds:
   - django/django
   - kubernetes/kubernetes
   - numpy/numpy
   - pallets/flask
2. Temporal Split (within-repo temporal test set)

Workflow & Leakage Guards:
- Z-score churn transformation and sample weights are computed strictly on training data.
- Training-Selected Threshold (T_train):
  Computed via Inner 5-fold Stratified Cross-Validation on the training split (OOF predictions),
  sweeping thresholds from 0.05 to 0.95 (step 0.01) to maximize training F1.
  Zero held-out test repo data or temporal test set data is seen during threshold selection.
- Fixed 0.45 Threshold:
  Evaluated as standard baseline.
- Oracle-Optimal Threshold (T_oracle) [Diagnostic Only]:
  Sweeps thresholds directly on actual test predictions to find the theoretical upper-bound F1.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
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


def evaluate_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def sweep_thresholds(y_true, y_prob, step=0.01):
    thresholds = np.arange(0.05, 0.96, step)
    results = []
    for t in thresholds:
        metrics = evaluate_at_threshold(y_true, y_prob, t)
        results.append(metrics)
    
    # Best threshold by F1 (tie break: highest precision, closest to 0.5)
    best = max(results, key=lambda m: (m["f1"], m["precision"], -abs(m["threshold"] - 0.5)))
    return best, results


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


def get_training_oof_probabilities(train_df, feature_cols, is_loro=True):
    """Compute out-of-fold training predictions to select threshold without leakage."""
    X = train_df[feature_cols].copy()
    y = train_df["label"].astype(int).values
    oof_probs = np.zeros(len(y), dtype=float)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val = X.iloc[val_idx]

        sw = compute_sample_weights(y_tr)
        m = GradientBoostingClassifier(random_state=42)
        m.fit(X_tr, y_tr, sample_weight=sw)
        oof_probs[val_idx] = m.predict_proba(X_val)[:, 1]

    return oof_probs


def run_fold_analysis(train_df, test_df, fold_name, is_loro=True):
    if is_loro:
        train_t, test_t, feat_cols = transform_churn_zscore_loro(train_df, test_df)
    else:
        train_t, test_t, feat_cols = transform_churn_zscore_temporal(train_df, test_df)

    X_train = train_t[feat_cols]
    y_train = train_t["label"].astype(int).values
    X_test = test_t[feat_cols]
    y_test = test_t["label"].astype(int).values

    # Fit final model on full training set
    sw_train = compute_sample_weights(y_train)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train, sample_weight=sw_train)

    test_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = float(roc_auc_score(y_test, test_prob))
    pr_auc = float(average_precision_score(y_test, test_prob))

    # 1. Fixed 0.45 Evaluation
    fixed_metrics = evaluate_at_threshold(y_test, test_prob, 0.45)

    # 2. Training-Only Threshold Selection (via OOF on training data)
    oof_train_probs = get_training_oof_probabilities(train_t, feat_cols, is_loro=is_loro)
    best_train_thresh_result, _ = sweep_thresholds(y_train, oof_train_probs, step=0.01)
    t_train_selected = best_train_thresh_result["threshold"]

    # Apply training-selected threshold to test set
    tuned_test_metrics = evaluate_at_threshold(y_test, test_prob, t_train_selected)

    # 3. Diagnostic Oracle Threshold (evaluated directly on test set)
    oracle_result, _ = sweep_thresholds(y_test, test_prob, step=0.01)
    t_oracle = oracle_result["threshold"]

    return {
        "fold": fold_name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_positives": int(sum(y_test)),
        "test_positive_rate": float(np.mean(y_test)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fixed_045": {
            "threshold": 0.45,
            "f1": fixed_metrics["f1"],
            "precision": fixed_metrics["precision"],
            "recall": fixed_metrics["recall"],
            "accuracy": fixed_metrics["accuracy"],
            "confusion_matrix": fixed_metrics["confusion_matrix"],
        },
        "training_selected": {
            "threshold": t_train_selected,
            "train_oof_f1": best_train_thresh_result["f1"],
            "test_f1": tuned_test_metrics["f1"],
            "test_precision": tuned_test_metrics["precision"],
            "test_recall": tuned_test_metrics["recall"],
            "test_accuracy": tuned_test_metrics["accuracy"],
            "confusion_matrix": tuned_test_metrics["confusion_matrix"],
        },
        "diagnostic_oracle": {
            "threshold": t_oracle,
            "oracle_f1": oracle_result["f1"],
            "oracle_precision": oracle_result["precision"],
            "oracle_recall": oracle_result["recall"],
            "oracle_accuracy": oracle_result["accuracy"],
            "confusion_matrix": oracle_result["confusion_matrix"],
        },
    }


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {len(df)} rows across {df['repo'].nunique()} repositories.")

    results = []

    # 1. LORO Folds
    repos = sorted(df["repo"].unique())
    print("\nRunning LORO Folds Threshold Analysis...")
    for held_out in repos:
        train_df = df[df["repo"] != held_out].copy()
        test_df = df[df["repo"] == held_out].copy()
        res = run_fold_analysis(train_df, test_df, held_out, is_loro=True)
        results.append(res)

    # 2. Temporal Split
    print("\nRunning Temporal Split Threshold Analysis...")
    ordered = df.sort_values(["merged_at", "pr_number"], kind="mergesort").reset_index(drop=True)
    cut = int(len(ordered) * 0.75)
    train_temporal = ordered.iloc[:cut].copy()
    test_temporal = ordered.iloc[cut:].copy()
    temp_res = run_fold_analysis(train_temporal, test_temporal, "Temporal Split", is_loro=False)
    results.append(temp_res)

    Path("results").mkdir(exist_ok=True)
    with open("results/threshold_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved full results to results/threshold_analysis_results.json")

    # Display Table
    print("\n" + "=" * 120)
    print("THRESHOLD SENSITIVITY ANALYSIS SUMMARY TABLE")
    print("=" * 120)

    rows = []
    for r in results:
        rows.append({
            "Fold / Split": r["fold"],
            "Fixed-0.45 F1 / P / R": f"{r['fixed_045']['f1']:.3f} / {r['fixed_045']['precision']:.3f} / {r['fixed_045']['recall']:.3f}",
            "T_train": f"{r['training_selected']['threshold']:.2f}",
            "Tuned F1 / P / R": f"{r['training_selected']['test_f1']:.3f} / {r['training_selected']['test_precision']:.3f} / {r['training_selected']['test_recall']:.3f}",
            "T_oracle (Diag)": f"{r['diagnostic_oracle']['threshold']:.2f}",
            "Oracle F1 / P / R": f"{r['diagnostic_oracle']['oracle_f1']:.3f} / {r['diagnostic_oracle']['oracle_precision']:.3f} / {r['diagnostic_oracle']['oracle_recall']:.3f}",
        })

    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
