# AI-Powered Pre-Merge Bug Risk Predictor

Predicts whether a **merged** GitHub pull request is **associated with a subsequent bug-fix signal** within 30 days (a later commit touching files from that PR, with a bug-fix/revert keyword in the commit subject).

This is **not** a causal model. It does **not** claim that a PR “causes” a bug.

**Experimental results on real GitHub data: not yet run.** The pipeline has local/synthetic tests only until you collect and train on real pull requests.

## Architecture

```
GitHub PR data
    → Data Agent (features + 30-day label)
    → Gradient Boosting classifier (scikit-learn)
    → Risk score (probability)
    → Groq LLM explanation (llama-3.3-70b-versatile)
    → Optional contrarian pass if 0.4 ≤ score ≤ 0.6
    → Structured JSON report
```

Local-only: no database, no web app, no deployment.

## Target / label

`label = 1` if, after merge and within `LOOKAHEAD_DAYS` (30), a commit that:

1. touches a file changed by the PR (up to 10 files; documentation paths skipped when any non-doc file exists), and
2. is **not** one of the original PR commits or the merge commit, and
3. has a **subject line** matching whole-word keywords: `fix`, `bug`, `hotfix`, `patch`, `revert` (and common inflections),

is observed.

`label = 0` if the full 30-day window elapsed and no such commit was found.

**Right-censoring:** PRs whose merge time is fewer than 30 days ago are **skipped during collection** (`REQUIRE_COMPLETE_LOOKAHEAD = True`). They are not stored as negatives. Keeping only early positives from an incomplete window would bias the sample toward quickly observed events, so those rows are skipped too. Use `--include-incomplete-lookahead` only for inspection, not training.

GitHub issue trackers and PR labels are **not** the training target. PR labels are stored as metadata (`pr_labels`) for later inspection only.

## Features

All features are computed from information at or before merge time. Historical commit queries **exclude this PR’s own SHAs**.

| Feature | Meaning |
| --- | --- |
| `additions` / `deletions` / `lines_changed` | Line stats from the PR files API |
| `files_changed` | Number of files in the PR (paginated) |
| `test_file_present` / `n_test_files` | Test path heuristics (`tests/`, `test_*.py`, etc.) |
| `source_file_touched` | Non-doc, non-test source extension |
| `config_file_touched` / `dependency_file_touched` | Config / lockfile heuristics |
| `n_directories` | Distinct parent directories |
| `file_churn_count` | Unique historical commits in the previous ~3 months on up to 5 files (source preferred). A **lower bound** if `file_churn_truncated=1` |
| `file_prior_bugfix_touch` | Any of those historical commits has a bug-fix subject |
| `day_of_week` / `hour_of_day` | Merge time in UTC |

Author identity is stored but **not** used as a model feature.

## ML methodology

- Primary model: `sklearn.ensemble.GradientBoostingClassifier`
- Optional baseline model: `LogisticRegression(class_weight="balanced")`
- **Primary evaluation: temporal split** (sort by `merged_at`, train on older PRs, test on newer PRs)
- Random stratified split is computed only as a **secondary** comparison (it can leak time)
- Also reported: majority-class baseline, F1, precision, recall, confusion matrix, class counts; ROC-AUC / PR-AUC when both classes appear in the test fold
- Tiny datasets can be **technically trainable** and still **not scientifically meaningful**; the trainer warns when folds are small or missing a class
- Feature importances from gradient boosting are **model-wide impurity importances**, not instance-level and not causal

## LLM explanation

Groq generates a short explanation from the **evidence block** (score and feature values). If the API key is missing or Groq fails, the ML score is still returned (`llm_status: unavailable`).

The contrarian pass runs only for borderline probabilities `[0.4, 0.6]`. Its `llm_synthesis` does **not** replace `risk_score` or `risk_label`.

## Setup

Python 3.14 (project default). Do not commit `.env`.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `GITHUB_TOKEN` and `GROQ_API_KEY` in `.env` (never commit that file).

## Commands

Score a merged PR (requires a **GitHub-trained** `models/risk_classifier.joblib`; a synthetic bundle is refused unless `--allow-synthetic-model`):

```bash
python run.py --repo pallets/flask --pr 123
python run.py --repo pallets/flask --pr 123 --no-llm
```

Collect a **small** dataset. Collection scans recently **created** closed PRs and keeps merged ones; it is **not** “the latest N merges”. Recent PRs with an incomplete 30-day window are skipped by default:

```bash
python run.py --collect --repo pallets/flask --limit 25
```

Train (temporal evaluation is the printed primary result). Training **fails** if the CSV still contains incomplete-lookahead rows (including the first 25-PR Flask file if it was collected before censoring):

```bash
python run.py --train
```

Local synthetic training (writes `models/synthetic_risk_classifier.joblib` by default; **not** a real-data result):

```bash
python run.py --train-synthetic
```

Tests (no live GitHub/Groq):

```bash
python -m unittest discover -s tests -v
```

## Limitations

- Keyword labels are a **noisy proxy** (style of commit messages, “fix typo”, incomplete follow-ups, unfixed bugs).
- A later commit on the same file is **association**, not proof the PR introduced the defect.
- Only files queried (capped) can produce a positive label; large PRs are under-sampled for labeling and churn.
- GitHub list-pulls cannot sort by `merged_at`; collection scans recently **created** closed PRs, then keeps merged ones.
- Commit windows use GitHub’s commits API (`since` / `until` / `path`) with pagination caps; truncated listings are flagged and are not silently treated as complete zeros.
- The existing 25-row Flask CSV includes PRs merged in August 2026 labeled 0 without a full 30-day window. **Do not train on that file as-is.** Re-collect with default censoring.
- Synthetic training is for pipeline tests only.
- No real-data performance claim until you collect a complete-lookahead dataset and run `--train`.
- Author identity is **not** in the current model.

## Example output shape

```json
{
  "repo": "pallets/flask",
  "pr_number": 123,
  "risk_score": 0.51,
  "risk_label": "medium",
  "explanation": null,
  "llm_status": "skipped",
  "contrarian": null
}
```

Numeric values above are illustrative of **shape**, not a measured experiment.
