import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_PATH = Path("data/processed/all_repos.csv")

FEATURE_FAMILIES = {
    "change-size": [
        "additions",
        "deletions",
        "files_changed",
        "lines_changed",
        "n_directories",
    ],
    "file-type": [
        "test_file_present",
        "n_test_files",
        "source_file_touched",
        "config_file_touched",
        "dependency_file_touched",
    ],
    "historical": [
        "file_churn_count",
        "file_prior_bugfix_touch",
    ],
    "temporal": [
        "day_of_week",
        "hour_of_day",
    ],
}

ALL_FEATURES = [col for fam in FEATURE_FAMILIES.values() for col in fam]


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
    }


def run_temporal_eval(df, feature_cols):
    ordered = df.sort_values(["merged_at", "pr_number"], kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    cut = int(n * 0.75)
    train_df = ordered.iloc[:cut]
    test_df = ordered.iloc[cut:]

    X_train = train_df[feature_cols]
    y_train = train_df["label"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["label"].astype(int)

    sample_weight = compute_sample_weights(y_train)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.45).astype(int)

    return evaluate_predictions(y_test, y_pred, y_prob)


def run_loro_eval(df, feature_cols):
    repos = sorted(df["repo"].unique())
    fold_results = []

    for held_out in repos:
        train_df = df[df["repo"] != held_out].copy()
        test_df = df[df["repo"] == held_out].copy()

        X_train = train_df[feature_cols]
        y_train = train_df["label"].astype(int)
        X_test = test_df[feature_cols]
        y_test = test_df["label"].astype(int)

        sample_weight = compute_sample_weights(y_train)
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train, sample_weight=sample_weight)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.45).astype(int)

        res = evaluate_predictions(y_test, y_pred, y_prob)
        res["repo"] = held_out
        fold_results.append(res)

    summary = pd.DataFrame(fold_results)
    means = {
        "f1": float(summary["f1"].mean()),
        "precision": float(summary["precision"].mean()),
        "recall": float(summary["recall"].mean()),
        "accuracy": float(summary["accuracy"].mean()),
        "roc_auc": float(summary["roc_auc"].mean()),
        "pr_auc": float(summary["pr_auc"].mean()),
        "folds": fold_results,
    }
    return means


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {len(df)} rows, {df['repo'].nunique()} repos")

    experiments = []

    # 1. Baseline: All features
    experiments.append(("all", "full-baseline", ALL_FEATURES))

    # 2. Leave-one-family-out
    for name, cols in FEATURE_FAMILIES.items():
        subset = [c for c in ALL_FEATURES if c not in cols]
        experiments.append((name, "leave-one-family-out", subset))

    # 3. Family-only
    for name, cols in FEATURE_FAMILIES.items():
        experiments.append((name, "family-only", cols))

    records = []

    print("\nRunning Ablation Experiments...")
    print("=" * 90)

    for fam_name, abl_type, feature_subset in experiments:
        # Temporal
        temp_res = run_temporal_eval(df, feature_subset)
        records.append({
            "feature_family": fam_name,
            "ablation_type": abl_type,
            "eval_mode": "temporal",
            "n_features": len(feature_subset),
            "features": feature_subset,
            "f1": temp_res["f1"],
            "precision": temp_res["precision"],
            "recall": temp_res["recall"],
            "roc_auc": temp_res["roc_auc"],
            "pr_auc": temp_res["pr_auc"],
            "accuracy": temp_res["accuracy"],
        })

        # LORO
        loro_res = run_loro_eval(df, feature_subset)
        records.append({
            "feature_family": fam_name,
            "ablation_type": abl_type,
            "eval_mode": "loro",
            "n_features": len(feature_subset),
            "features": feature_subset,
            "f1": loro_res["f1"],
            "precision": loro_res["precision"],
            "recall": loro_res["recall"],
            "roc_auc": loro_res["roc_auc"],
            "pr_auc": loro_res["pr_auc"],
            "accuracy": loro_res["accuracy"],
            "loro_folds": loro_res["folds"],
        })

    results_df = pd.DataFrame(records)

    display_cols = ["feature_family", "ablation_type", "eval_mode", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    print(results_df[display_cols].to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    with open("results/feature_ablation_results.json", "w") as f:
        json.dump(records, f, indent=2)

    print("\nSaved detailed results to results/feature_ablation_results.json")

    # Compute distribution stats across repos for each family
    print("\nPer-repo statistics across feature families:")
    for fam_name, cols in FEATURE_FAMILIES.items():
        print(f"\n--- Family: {fam_name} ---")
        for col in cols:
            grouped = df.groupby("repo")[col].agg(["count", "mean", "std", "median", "min", "max"])
            print(f"\nFeature: {col}")
            print(grouped.to_string())


if __name__ == "__main__":
    main()
