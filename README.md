# AI-Powered Pre-Merge PR Bug-Risk Predictor

A **multi-agent** machine-learning system that estimates whether a **merged GitHub pull request (PR)** will be associated with a subsequent bug-fix signal within 30 days.

The system combines GitHub repository history, pull-request characteristics, a Gradient Boosting classifier, and an optional Groq LLM explanation layer through a coordinated multi-agent architecture.

> **Important:** This is not a causal bug predictor. A positive label means that a later bug-fix-associated commit touched files from the original PR within the observation window. It does not prove that the original PR caused a bug.

## What It Does

```text
GitHub Pull Request
        ↓
Data Agent: PR + Files + Commits
        ↓
Data Agent: Feature Extraction + Labeling
        ↓
ML Agent: Gradient Boosting Classifier
        ↓
ML Agent: Risk Probability
        ↓
Orchestrator: Risk Band Assignment (Low/Medium/High)
        ↓
LLM Agent: Optional Explanation
        ↓
LLM Agent: Contrarian Analysis (Borderline Scores Only)
```

The ML model produces the numerical risk score.

The LLM only explains the result using supplied evidence. It does **not** modify the ML score.

## Multi-Agent Architecture

The system implements a multi-agent pattern with specialized components that coordinate through an orchestrator:

```text
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                        │
│              (Pipeline coordination & flow control)           │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Data Agent   │  │   ML Agent    │  │   LLM Agent   │
│               │  │               │  │               │
│ • GitHub API  │  │ • Model train│  │ • Explanation │
│ • Feature ext │  │ • Prediction  │  │ • Contrarian  │
│ • Labeling    │  │ • Evaluation │  │ • Synthesis   │
└───────────────┘  └───────────────┘  └───────────────┘
```

**Agent Responsibilities:**

- **Data Agent** (`data_agent.py`): Fetches GitHub PR data, extracts features, computes bug-fix association labels, manages dataset collection and validation
- **ML Agent** (`ml_agent.py`): Trains Gradient Boosting classifier, handles model persistence, performs risk prediction with feature importance analysis
- **LLM Agent** (`llm_agent.py`): Generates grounded explanations using Groq API, performs contrarian analysis for borderline cases, synthesizes counterarguments
- **Orchestrator** (`orchestrator.py`): Coordinates agent interactions, manages pipeline flow, handles error cases and fallbacks

This multi-agent design enables modularity, testability, and clear separation of concerns between data processing, machine learning, and natural language generation components.

## Dataset

The current dataset contains **547 merged pull requests** from four open-source repositories:

| Repository |     PRs |   Positive |
| ---------- | ------: | ---------: |
| Django     |     100 |      38.0% |
| Kubernetes |     150 |      29.3% |
| NumPy      |     150 |      38.0% |
| Flask      |     147 |      11.6% |
| **Total**  | **547** | **28.52%** |

Overall labels:

```text
Negative: 391
Positive: 156
Positive rate: 28.52%
```

All training examples have a complete 30-day lookahead window.

## Label Definition

A PR receives label `1` when a later commit within 30 days:

1. matches the project's bug-fix commit-subject pattern, and
2. touches at least one file associated with the original PR.

Otherwise it receives label `0`.

The PR's own commits and merge-related commits are excluded from the lookahead.

This is an **association signal**, not a ground-truth software defect label.

The labeling search is capped at **10 files per PR**, with documentation paths deprioritized. Historical churn and prior bug-fix lookups are capped at **5 files**.

PRs whose full 30-day observation window has not elapsed are excluded rather than treated as negative examples.

## Features

The classifier uses 14 features:

```text
additions
deletions
files_changed
lines_changed
test_file_present
n_test_files
source_file_touched
config_file_touched
dependency_file_touched
n_directories
file_churn_count
file_prior_bugfix_touch
day_of_week
hour_of_day
```

The strongest global Gradient Boosting feature importances are:

| Feature                 | Importance |
| ----------------------- | ---------: |
| file_prior_bugfix_touch |     20.19% |
| file_churn_count        |     18.03% |
| additions               |     14.13% |
| hour_of_day             |     10.50% |
| lines_changed           |      8.57% |

These are **model-wide feature importances**, not causal explanations for an individual PR.

## Machine Learning Model

The **ML Agent** manages model training, evaluation, and prediction:

Primary model:

```text
GradientBoostingClassifier(random_state=42)
```

The main training path uses training-set-derived sample weights to compensate for class imbalance.

A Logistic Regression model was also tested:

```text
LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
```

The reproduced temporal F1 for Logistic Regression was **0.595**, compared with **0.620** for Gradient Boosting. The Logistic Regression run also produced a convergence warning.

## Evaluation

The primary evaluation is a **temporal split**.

```text
Training: 410 PRs
Test:     137 PRs

Test negatives: 97
Test positives: 40
```

### Temporal Results

| Metric    |    Result |
| --------- | --------: |
| F1        | **0.620** |
| Precision | **0.517** |
| Recall    | **0.775** |
| Accuracy  | **0.723** |
| ROC-AUC   | **0.777** |
| PR-AUC    | **0.566** |

Confusion matrix:

```text
[[68, 29],
 [ 9, 31]]
```

The majority-class baseline has positive-class F1 = `0.0`.

### Random Split

The random split is included only as a secondary comparison because random splitting can introduce temporal leakage.

| Metric    | Result |
| --------- | -----: |
| F1        |  0.505 |
| Precision |  0.429 |
| Recall    |  0.615 |
| ROC-AUC   |  0.727 |
| PR-AUC    |  0.593 |

### Leave-One-Repository-Out

LORO evaluation tests cross-repository generalization by holding out one repository at a time.

| Held-out Repository |        F1 |
| ------------------- | --------: |
| Django              |     0.357 |
| Kubernetes          |     0.466 |
| NumPy               |     0.325 |
| Flask               |     0.111 |
| **Mean**            | **0.315** |

Overall LORO results:

```text
Mean F1        0.315
Mean Precision 0.400
Mean Recall    0.306
Mean Accuracy  0.662
Mean ROC-AUC   0.624
Mean PR-AUC    0.391
```

The large drop from temporal performance to LORO performance indicates **limited cross-repository generalization**.

The LORO implementation currently does not use the same class-balanced sample weighting as the primary training path. This is an important methodological limitation.

## Risk Bands

The system uses separate qualitative risk bands:

```text
score < 0.4       → low
0.4 ≤ score ≤ 0.6 → medium
score > 0.6      → high
```

The binary ML evaluation threshold is `0.45`.

The `0.45` threshold is currently configured in the implementation. A formal threshold-sweep artifact is not stored in the repository, so the project does not claim that the repository contains a reproducible sweep selecting `0.45`.

## LLM Explanation

The optional explanation layer is handled by the **LLM Agent** and uses:

```text
Provider: Groq
Model: openai/gpt-oss-120b
```

The LLM Agent receives evidence such as:

* ML risk score
* PR size
* files changed
* test-file information
* source/config/dependency changes
* historical file churn
* prior bug-fix touches
* temporal information
* model-wide feature importance

The prompt instructs the LLM to use only the supplied evidence and avoid causal claims.

The multi-agent architecture provides graceful degradation: if Groq is unavailable, the Orchestrator still returns the ML prediction with appropriate status flags.

## Contrarian Pass

For borderline predictions:

```text
0.4 ≤ score ≤ 0.6
```

the system performs an additional LLM pass.

The workflow is:

```text
Initial explanation
       ↓
Counterargument
       ↓
Qualitative synthesis
```

The contrarian stage is commentary only.

It **cannot change the numerical ML score**.

## Example Predictions

### Low Risk

```text
Repository: pallets/flask
PR: #6013

Risk score: 0.3861
Risk label: low
```

Multi-agent pipeline: Data Agent → ML Agent → LLM Agent
LLM explanation was successfully generated.
Contrarian pass was not triggered (score outside borderline range).

### Medium Risk

```text
Repository: numpy/numpy
PR: #31907

Risk score: 0.4363
Risk label: medium
```

Multi-agent pipeline: Data Agent → ML Agent → LLM Agent → Contrarian Pass
The explanation, counterargument, and synthesis stages successfully executed.
Demonstrates full multi-agent coordination for borderline cases.

### High Risk

```text
Repository: numpy/numpy
PR: #31856

Risk score: 0.9131
Risk label: high
```

Multi-agent pipeline: Data Agent → ML Agent → LLM Agent
LLM explanation was successfully generated.
Contrarian pass was not triggered (score outside borderline range).

## Testing

The project currently contains automated tests for:

```text
Censoring
Data processing
Feature/score behavior
GitHub client
Labeling
LLM agent behavior
ML agent behavior
Multi-agent pipeline coordination
```

Latest test run:

```text
37 passed
18 warnings
```

The warnings were NumPy/joblib deprecation warnings and were not test failures.

Tests verify individual agent behavior as well as multi-agent orchestration through the pipeline.

## Project Structure

```text
.
├── run.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── config.py
│   ├── data_agent.py           # GitHub fetch + feature extraction + labeling
│   ├── dataset_validation.py   # Data quality checks
│   ├── github_client.py        # REST API client with retry/rate-limit handling
│   ├── labeling.py             # Bug-fix commit pattern matching
│   ├── llm_agent.py            # Groq explanation + contrarian analysis
│   ├── ml_agent.py             # Model training + prediction + evaluation
│   ├── orchestrator.py         # Multi-agent coordination
│   └── synthetic.py            # Synthetic data generation for testing
│
├── scripts/
│   └── loro_eval.py            # Leave-one-repository-out evaluation
│
└── tests/
    ├── test_censoring.py
    ├── test_data_and_scores.py
    ├── test_github_client.py
    ├── test_labeling.py
    ├── test_llm_and_pipeline.py
    └── test_ml_agent.py
```

Generated datasets, models, and evaluation artifacts are kept outside the tracked source files.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Configure environment variables using `.env.example`.

A GitHub credential/configuration is required for live PR retrieval.

A Groq API key is required for LLM explanations.

## Train the Model

Train on the processed GitHub dataset:

```powershell
python run.py --train --csv data/processed/all_repos.csv --model-type gboost
```

The trained model is written locally to:

```text
models/risk_classifier.joblib
```

The saved bundle records:

```text
model
feature_cols
model_type
trained_on
```

Real GitHub PR scoring refuses synthetic-trained models unless explicitly allowed.

## Score a Real PR

ML prediction only:

```powershell
python run.py --repo numpy/numpy --pr 31856 --no-llm
```

ML + LLM explanation:

```powershell
python run.py --repo numpy/numpy --pr 31856
```

The output contains:

```text
risk_score
risk_label
feature_values
top_features
model_wide_importances
trained_on
explanation
```

## Run LORO Evaluation

```powershell
python scripts/loro_eval.py
```

## Run Tests

```powershell
python -m pytest
```

## Important Limitations

This project is experimental.

The target is a **subsequent bug-fix association**, not a verified software defect.

The label depends on commit-subject keyword matching and file overlap, so it can contain noise.

Feature searches are capped, meaning large pull requests may not have every touched file represented in some historical calculations.

Repository class distributions differ substantially, especially for Flask.

Temporal performance is considerably stronger than cross-repository LORO performance, showing that generalization remains a problem.

The classifier probability has not been independently calibrated, so the score should not be interpreted as a guaranteed empirical probability.

Model-wide feature importance is not causal and is not an instance-level explanation.

The LLM is an optional interpretation layer and depends on an external API.

## Research Summary

The current experiment shows that pull-request size, repository history, file churn, and prior bug-fix activity contain useful signals for predicting subsequent bug-fix association.

The primary temporal evaluation achieved:

```text
F1      = 0.620
ROC-AUC = 0.777
PR-AUC  = 0.566
```

However, the reproduced LORO evaluation achieved:

```text
Mean F1      = 0.315
Mean ROC-AUC = 0.624
Mean PR-AUC  = 0.391
```

The main conclusion is therefore that the approach shows promising within-dataset predictive signal, but **cross-repository generalization remains limited**.

## CV Summary

**Built BugPredict, a multi-agent PR risk system combining a trained gradient-boosting classifier (F1 = 0.620) with an LLM explanation agent, tested on 547 PRs from 4 public repositories.**

## Status

Current implementation:

```text
Dataset collection       ✅
Dataset validation       ✅
Feature engineering      ✅
Multi-agent architecture ✅
Gradient Boosting model  ✅
Temporal evaluation      ✅
Random evaluation        ✅
LORO evaluation          ✅
Real PR inference        ✅
Risk bands               ✅
Groq explanation         ✅
Contrarian pass          ✅
Automated tests          ✅
Research documentation   🔄
```

This repository contains an experimental multi-agent software-engineering ML system and its supporting evaluation code.
