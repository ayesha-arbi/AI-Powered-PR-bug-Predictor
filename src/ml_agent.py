"""ML Agent: Gradient Boosting risk classifier with temporal evaluation."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from . import config
from .dataset_validation import validate_dataset

logger = logging.getLogger("bugpredict.ml")

FEATURE_COLS = list(config.FEATURE_COLS)


class ModelError(RuntimeError):
    """Raised when the model cannot be trained or used."""


def _metrics(y_true, y_pred, y_proba) -> dict:
    out = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0, labels=[0, 1]
        ),
        "n_test": int(len(y_true)),
        "test_positive_rate": float(np.mean(y_true)) if len(y_true) else None,
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            out["roc_auc"] = None
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            out["pr_auc"] = None
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def _predict_proba_positive(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X))


def temporal_split(df: pd.DataFrame, test_size: float = None):
    """Train on older PRs, test on newer PRs (by merged_at)."""
    test_size = config.TEMPORAL_TEST_SIZE if test_size is None else test_size
    ordered = df.sort_values(["merged_at", "pr_number"], kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    cut = int(n * (1.0 - test_size))
    if cut < 1 or cut >= n:
        raise ModelError(
            f"temporal split needs more rows (n={n}, test_size={test_size})"
        )
    train_df = ordered.iloc[:cut]
    test_df = ordered.iloc[cut:]
    return train_df, test_df


def _baseline_majority(y_train, n_test: int) -> np.ndarray:
    majority = int(pd.Series(y_train).mode().iloc[0]) if len(y_train) else 0
    return np.full(n_test, majority)


def _fit(model_type: str, X_train, y_train):
    if model_type not in config.SUPPORTED_MODEL_TYPES:
        raise ModelError(
            f"unsupported model_type={model_type!r}; "
            f"supported: {config.SUPPORTED_MODEL_TYPES}"
        )
    if model_type == "logreg":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=config.RANDOM_SEED,
        )
        model.fit(X_train, y_train)
        return model

    model = GradientBoostingClassifier(random_state=config.RANDOM_SEED)
    # GradientBoostingClassifier has no class_weight; balance via sample_weight
    # computed from the *training* labels only.
    counts = pd.Series(y_train).value_counts()
    n = len(y_train)
    sample_weight = np.ones(n, dtype=float)
    if len(counts) > 1:
        sample_weight = np.array(
            [n / (len(counts) * counts[y]) for y in y_train], dtype=float
        )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train(
    df: pd.DataFrame,
    model_type: str = "gboost",
    *,
    model_path: Path | None = None,
    data_source: str = "github",
) -> dict:
    """Train primary Gradient Boosting model. Temporal split is the main evaluation."""
    if data_source not in {"github", "synthetic"}:
        raise ModelError("data_source must be 'github' or 'synthetic'")

    validation = validate_dataset(df, for_training=True)
    if not validation["ok"]:
        raise ModelError("dataset validation failed: " + "; ".join(validation["errors"]))
    for warning in validation["warnings"]:
        logger.warning("dataset: %s", warning)

    if len(df) < config.MIN_TRAIN_ROWS:
        raise ModelError(
            f"need at least {config.MIN_TRAIN_ROWS} rows to train, got {len(df)}"
        )

    work = df.copy()
    train_df, test_df = temporal_split(work)
    X_train, y_train = train_df[FEATURE_COLS], train_df["label"].astype(int)
    X_test, y_test = test_df[FEATURE_COLS], test_df["label"].astype(int)

    train_classes = set(y_train.unique())
    test_classes = set(y_test.unique())
    eval_notes = []
    if len(train_df) < 40 or len(test_df) < 10:
        eval_notes.append(
            "evaluation is technically possible but not scientifically meaningful "
            f"(n_train={len(train_df)}, n_test={len(test_df)})"
        )
    if train_classes != {0, 1}:
        eval_notes.append(f"training fold is missing a class: {sorted(train_classes)}")
    if test_classes != {0, 1}:
        eval_notes.append(f"test fold is missing a class: {sorted(test_classes)}")
    for note in eval_notes:
        logger.warning("%s", note)

    model = _fit(model_type, X_train, y_train)
    y_proba = _predict_proba_positive(model, X_test)
    y_pred = (y_proba >= 0.45).astype(int)
    temporal = _metrics(y_test, y_pred, y_proba)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    base_pred = dummy.predict(X_test)
    baseline = _metrics(y_test, base_pred, None)
    majority_pred = _baseline_majority(y_train, len(y_test))
    baseline["majority_f1"] = float(f1_score(y_test, majority_pred, zero_division=0))

    random_metrics = None
    try:
        stratify = work["label"] if work["label"].nunique() > 1 else None
        rx_train, rx_test, ry_train, ry_test = train_test_split(
            work[FEATURE_COLS],
            work["label"].astype(int),
            test_size=config.RANDOM_SPLIT_TEST_SIZE,
            random_state=config.RANDOM_SEED,
            stratify=stratify,
        )
        rmodel = _fit(model_type, rx_train, ry_train)
        r_proba = _predict_proba_positive(rmodel, rx_test)
        r_pred = (r_proba >= 0.45).astype(int)
        random_metrics = _metrics(ry_test, r_pred, r_proba)
        random_metrics["note"] = (
            "Secondary comparison only. Random splits can leak temporally; "
            "do not treat as the main result."
        )
    except ValueError as exc:
        logger.warning("random split skipped: %s", exc)

    dest = Path(model_path) if model_path else Path(config.MODEL_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_cols": FEATURE_COLS,
            "model_type": model_type,
            "trained_on": data_source,
        },
        dest,
    )
    logger.info("saved model to %s trained_on=%s", dest, data_source)
    if data_source == "synthetic":
        logger.warning(
            "this model is synthetic-only; it must not be reported as GitHub performance"
        )

    return {
        "model": model,
        "model_path": str(dest),
        "model_type": model_type,
        "trained_on": data_source,
        "validation": validation,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_label_counts": {int(k): int(v) for k, v in train_df["label"].value_counts().items()},
        "test_label_counts": {int(k): int(v) for k, v in test_df["label"].value_counts().items()},
        "temporal": temporal,
        "baseline_most_frequent": baseline,
        "random_split": random_metrics,
        "primary_split": "temporal",
        "evaluation_notes": eval_notes,
        "f1": temporal["f1"],
        "confusion_matrix": temporal["confusion_matrix"],
        "report": temporal["classification_report"],
        "disclaimer": (
            "Synthetic metrics are pipeline tests only."
            if data_source == "synthetic"
            else "Small-sample metrics are unstable; compare against the majority baseline."
        ),
    }


def load_model(model_path: Path | None = None):
    path = Path(model_path) if model_path else Path(config.MODEL_PATH)
    if not path.exists():
        raise ModelError(
            f"no model at {path}; train with `python run.py --train` after collecting GitHub data"
        )
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload
    return {
        "model": payload,
        "feature_cols": FEATURE_COLS,
        "model_type": "unknown",
        "trained_on": "unknown",
    }


def predict_risk(
    pr_features: dict,
    model=None,
    *,
    model_path: Path | None = None,
    allow_synthetic: bool = False,
) -> dict:
    """Return {score, feature_values, model_wide_importances}.

    Importances are *global* (model-wide), not a causal explanation of this PR.
    Missing required features raise; they are not silently filled with 0.
    """
    bundle = None
    if model is None:
        bundle = load_model(model_path)
        estimator = bundle["model"]
        cols = list(bundle.get("feature_cols") or FEATURE_COLS)
        trained_on = bundle.get("trained_on") or "unknown"
        if trained_on != "github" and not allow_synthetic:
            raise ModelError(
                f"refusing to score with a {trained_on} model at "
                f"{model_path or config.MODEL_PATH}. Train on GitHub data "
                "(`python run.py --train`) or pass --allow-synthetic-model."
            )
    else:
        estimator = model
        cols = FEATURE_COLS

    missing = [c for c in cols if c not in pr_features]
    if missing:
        raise ModelError(f"missing required features: {missing}")

    ordered = {k: pr_features[k] for k in cols}
    x = pd.DataFrame([ordered], columns=cols)
    if x.isna().any().any():
        raise ModelError("feature vector contains missing values")

    score = float(_predict_proba_positive(estimator, x)[0])

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
        kind = "gradient_boosting_impurity_importance"
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.asarray(estimator.coef_[0], dtype=float))
        kind = "logistic_regression_abs_coefficient"
    else:
        importances = np.zeros(len(cols))
        kind = "none"

    ranked = sorted(zip(cols, importances), key=lambda t: -t[1])
    model_wide = [
        {
            "feature": f,
            "importance": round(float(imp), 4),
            "this_pr_value": pr_features.get(f),
        }
        for f, imp in ranked[:5]
    ]

    return {
        "score": round(score, 4),
        "feature_values": ordered,
        "model_wide_importances": model_wide,
        "importance_kind": kind,
        "importance_note": (
            "These importances describe the trained model overall, not a proof "
            "that a feature caused this PR's score."
        ),
        "top_features": model_wide[:3],
        "trained_on": (bundle or {}).get("trained_on") if bundle else None,
    }
