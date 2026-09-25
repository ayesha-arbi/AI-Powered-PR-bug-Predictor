"""
Controlled Representation Experiment for `file_churn_count`
===========================================================

Formulas & Leakage-Prevention Documentation:
--------------------------------------------
This experiment evaluates 5 representations of `file_churn_count` across both
Temporal (within-repo) and LORO (cross-repo, 4 folds) evaluation setups.
All models use GradientBoostingClassifier(random_state=42) with train-only sample weighting.

Feature Sets & Normalization Formulas:
1. Baseline:
   `file_churn_count` is unchanged (raw count).

2. Relative Churn:
   Churn is scaled relative to the repository mean churn:
   - Within-repo / Temporal:
     For each repo r in the training split D_train, calculate mean churn:
       mean_churn(r) = mean({c in D_train | repo == r})
     Both train and test PRs belonging to repo r are transformed as:
       churn_rel = c / (mean_churn(r) + 1e-5)
   - LORO (Cross-repo):
     For each training repo r in R_train:
       mean_churn(r) = mean({c in D_train | repo == r})
       Train PRs in repo r: churn_rel = c / (mean_churn(r) + 1e-5)
     For the held-out test repo R_test (never seen at train time):
       pooled_mean = mean({c in D_train})
       Held-out test PRs: churn_rel = c / (pooled_mean + 1e-5)
     (Zero statistics from the held-out repo are used).

3. Z-score:
   Standardize churn using training mean and std:
   - Within-repo / Temporal:
     For each repo r in D_train:
       mu(r) = mean({c in D_train | repo == r})
       sigma(r) = std({c in D_train | repo == r}) + 1e-5
     Train and test PRs in repo r: z = (c - mu(r)) / sigma(r)
   - LORO (Cross-repo):
     For training repos r in R_train:
       mu(r) = mean({c in D_train | repo == r})
       sigma(r) = std({c in D_train | repo == r}) + 1e-5
       Train PRs in repo r: z = (c - mu(r)) / sigma(r)
     For held-out test repo R_test:
       pooled_mu = mean({c in D_train})
       pooled_sigma = std({c in D_train}) + 1e-5
       Held-out test PRs: z = (c - pooled_mu) / pooled_sigma

4. Percentile:
   Transform churn into empirical percentile rank [0, 1] against training values:
   - Within-repo / Temporal:
     For each repo r in D_train, reference values V(r) = sorted({c in D_train | repo == r}):
     Train and test PRs in repo r: percentile = count(v in V(r) <= c) / |V(r)|
   - LORO (Cross-repo):
     For training repos r in R_train, reference values V(r) = sorted({c in D_train | repo == r}):
       Train PRs in repo r: percentile = count(v in V(r) <= c) / |V(r)|
     For held-out test repo R_test, reference values V_pool = sorted({c in D_train}):
       Held-out test PRs: percentile = count(v in V_pool <= c) / |V_pool|

5. Combined:
   Retains the raw `file_churn_count` AND appends the best-performing normalized
   feature from experiments 2-4 (15 features total).
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
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def transform_churn_temporal(train_df, test_df, mode):
    train_out = train_df.copy()
    test_out = test_df.copy()

    if mode == "baseline":
        return train_out, test_out, BASE_FEATURE_COLS

    # Compute per-repo stats from train_df only
    repo_stats = {}
    for repo, grp in train_df.groupby("repo"):
        vals = grp["file_churn_count"].values
        repo_stats[repo] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)) + 1e-5,
            "median": float(np.median(vals)),
            "values": np.sort(vals),
        }
    pooled_vals = train_df["file_churn_count"].values
    pooled_stats = {
        "mean": float(np.mean(pooled_vals)),
        "std": float(np.std(pooled_vals)) + 1e-5,
        "median": float(np.median(pooled_vals)),
        "values": np.sort(pooled_vals),
    }

    def apply_transform(df_in):
        out_vals = []
        for _, row in df_in.iterrows():
            r = row["repo"]
            c = row["file_churn_count"]
            st = repo_stats.get(r, pooled_stats)
            if mode == "relative":
                denom = st["mean"] if st["mean"] > 0 else 1.0
                out_vals.append(c / denom)
            elif mode == "zscore":
                out_vals.append((c - st["mean"]) / st["std"])
            elif mode == "percentile":
                rank = np.searchsorted(st["values"], c, side="right") / len(st["values"])
                out_vals.append(rank)
        return np.array(out_vals, dtype=float)

    feature_name = f"churn_{mode}"
    train_out[feature_name] = apply_transform(train_df)
    test_out[feature_name] = apply_transform(test_df)

    feature_cols = [c if c != "file_churn_count" else feature_name for c in BASE_FEATURE_COLS]
    return train_out, test_out, feature_cols


def transform_churn_loro(train_df, test_df, mode):
    train_out = train_df.copy()
    test_out = test_df.copy()

    if mode == "baseline":
        return train_out, test_out, BASE_FEATURE_COLS

    # Compute training repo stats from train_df ONLY
    repo_stats = {}
    for repo, grp in train_df.groupby("repo"):
        vals = grp["file_churn_count"].values
        repo_stats[repo] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)) + 1e-5,
            "median": float(np.median(vals)),
            "values": np.sort(vals),
        }
    pooled_vals = train_df["file_churn_count"].values
    pooled_stats = {
        "mean": float(np.mean(pooled_vals)),
        "std": float(np.std(pooled_vals)) + 1e-5,
        "median": float(np.median(pooled_vals)),
        "values": np.sort(pooled_vals),
    }

    # Training split: transform per training-repo stats
    train_vals = []
    for _, row in train_df.iterrows():
        r = row["repo"]
        c = row["file_churn_count"]
        st = repo_stats.get(r, pooled_stats)
        if mode == "relative":
            denom = st["mean"] if st["mean"] > 0 else 1.0
            train_vals.append(c / denom)
        elif mode == "zscore":
            train_vals.append((c - st["mean"]) / st["std"])
        elif mode == "percentile":
            rank = np.searchsorted(st["values"], c, side="right") / len(st["values"])
            train_vals.append(rank)

    # Held-out test split: transform using pooled training stats (strictly zero leakage)
    test_vals = []
    for _, row in test_df.iterrows():
        c = row["file_churn_count"]
        if mode == "relative":
            denom = pooled_stats["mean"] if pooled_stats["mean"] > 0 else 1.0
            test_vals.append(c / denom)
        elif mode == "zscore":
            test_vals.append((c - pooled_stats["mean"]) / pooled_stats["std"])
        elif mode == "percentile":
            rank = np.searchsorted(pooled_stats["values"], c, side="right") / len(pooled_stats["values"])
            test_vals.append(rank)

    feature_name = f"churn_{mode}"
    train_out[feature_name] = np.array(train_vals, dtype=float)
    test_out[feature_name] = np.array(test_vals, dtype=float)

    feature_cols = [c if c != "file_churn_count" else feature_name for c in BASE_FEATURE_COLS]
    return train_out, test_out, feature_cols


def run_temporal_experiment(df, mode, combined_best_mode=None):
    ordered = df.sort_values(["merged_at", "pr_number"], kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    cut = int(n * 0.75)
    train_raw = ordered.iloc[:cut].copy()
    test_raw = ordered.iloc[cut:].copy()

    if mode == "combined":
        train_df, test_df, feat_cols = transform_churn_temporal(train_raw, test_raw, combined_best_mode)
        # add raw file_churn_count back
        feat_cols = list(BASE_FEATURE_COLS) + [f"churn_{combined_best_mode}"]
    else:
        train_df, test_df, feat_cols = transform_churn_temporal(train_raw, test_raw, mode)

    X_train = train_df[feat_cols]
    y_train = train_df["label"].astype(int)
    X_test = test_df[feat_cols]
    y_test = test_df["label"].astype(int)

    sample_weight = compute_sample_weights(y_train)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.45).astype(int)

    metrics = evaluate_predictions(y_test, y_pred, y_prob)
    metrics["n_train"] = len(train_df)
    metrics["n_test"] = len(test_df)
    return metrics


def run_loro_experiment(df, mode, combined_best_mode=None):
    repos = sorted(df["repo"].unique())
    fold_results = {}

    for held_out in repos:
        train_raw = df[df["repo"] != held_out].copy()
        test_raw = df[df["repo"] == held_out].copy()

        if mode == "combined":
            train_df, test_df, feat_cols = transform_churn_loro(train_raw, test_raw, combined_best_mode)
            feat_cols = list(BASE_FEATURE_COLS) + [f"churn_{combined_best_mode}"]
        else:
            train_df, test_df, feat_cols = transform_churn_loro(train_raw, test_raw, mode)

        X_train = train_df[feat_cols]
        y_train = train_df["label"].astype(int)
        X_test = test_df[feat_cols]
        y_test = test_df["label"].astype(int)

        sample_weight = compute_sample_weights(y_train)
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train, sample_weight=sample_weight)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.45).astype(int)

        res = evaluate_predictions(y_test, y_pred, y_prob)
        res["repo"] = held_out
        fold_results[held_out] = res

    # Compute summary averages
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

    experiments = [
        ("Baseline (Raw)", "baseline"),
        ("Relative churn", "relative"),
        ("Z-score", "zscore"),
        ("Percentile", "percentile"),
    ]

    results = []

    # Run experiments 1 to 4
    for exp_label, mode in experiments:
        temp_res = run_temporal_experiment(df, mode)
        loro_res = run_loro_experiment(df, mode)
        results.append({
            "experiment": exp_label,
            "mode": mode,
            "temporal": temp_res,
            "loro": loro_res,
        })

    # Determine best performing normalization among 2-4 on LORO (by Mean LORO F1 / ROC-AUC)
    norm_exps = results[1:4]
    best_exp = max(norm_exps, key=lambda x: (x["loro"]["mean_f1"], x["loro"]["mean_roc_auc"]))
    best_mode = best_exp["mode"]
    best_label = best_exp["experiment"]
    print(f"\nBest LORO normalization among 2-4: {best_label} ({best_mode})")

    # Run Experiment 5: Combined
    comb_temp = run_temporal_experiment(df, "combined", combined_best_mode=best_mode)
    comb_loro = run_loro_experiment(df, "combined", combined_best_mode=best_mode)
    results.append({
        "experiment": f"Combined (Raw + {best_label})",
        "mode": "combined",
        "best_mode_used": best_mode,
        "temporal": comb_temp,
        "loro": comb_loro,
    })

    Path("results").mkdir(exist_ok=True)
    with open("results/churn_representation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print("F1 METRICS TABLE")
    print("=" * 100)
    f1_rows = []
    for r in results:
        f1_rows.append({
            "Experiment": r["experiment"],
            "Temporal F1": r["temporal"]["f1"],
            "LORO Django F1": r["loro"]["folds"]["django/django"]["f1"],
            "LORO Kubernetes F1": r["loro"]["folds"]["kubernetes/kubernetes"]["f1"],
            "LORO NumPy F1": r["loro"]["folds"]["numpy/numpy"]["f1"],
            "LORO Flask F1": r["loro"]["folds"]["pallets/flask"]["f1"],
            "Mean LORO F1": r["loro"]["mean_f1"],
        })
    df_f1 = pd.DataFrame(f1_rows)
    print(df_f1.to_string(index=False))

    print("\n" + "=" * 100)
    print("ROC-AUC METRICS TABLE")
    print("=" * 100)
    roc_rows = []
    for r in results:
        roc_rows.append({
            "Experiment": r["experiment"],
            "Temporal ROC-AUC": r["temporal"]["roc_auc"],
            "LORO Django ROC-AUC": r["loro"]["folds"]["django/django"]["roc_auc"],
            "LORO Kubernetes ROC-AUC": r["loro"]["folds"]["kubernetes/kubernetes"]["roc_auc"],
            "LORO NumPy ROC-AUC": r["loro"]["folds"]["numpy/numpy"]["roc_auc"],
            "LORO Flask ROC-AUC": r["loro"]["folds"]["pallets/flask"]["roc_auc"],
            "Mean LORO ROC-AUC": r["loro"]["mean_roc_auc"],
        })
    df_roc = pd.DataFrame(roc_rows)
    print(df_roc.to_string(index=False))

    print("\n" + "=" * 100)
    print("PR-AUC METRICS TABLE")
    print("=" * 100)
    pr_rows = []
    for r in results:
        pr_rows.append({
            "Experiment": r["experiment"],
            "Temporal PR-AUC": r["temporal"]["pr_auc"],
            "LORO Django PR-AUC": r["loro"]["folds"]["django/django"]["pr_auc"],
            "LORO Kubernetes PR-AUC": r["loro"]["folds"]["kubernetes/kubernetes"]["pr_auc"],
            "LORO NumPy PR-AUC": r["loro"]["folds"]["numpy/numpy"]["pr_auc"],
            "LORO Flask PR-AUC": r["loro"]["folds"]["pallets/flask"]["pr_auc"],
            "Mean LORO PR-AUC": r["loro"]["mean_pr_auc"],
        })
    df_pr = pd.DataFrame(pr_rows)
    print(df_pr.to_string(index=False))


if __name__ == "__main__":
    main()
