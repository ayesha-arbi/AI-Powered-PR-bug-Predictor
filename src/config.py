"""Project constants and environment variables.

Secrets are read from the environment (or a local .env via python-dotenv).
Never log or print GITHUB_TOKEN / GROQ_API_KEY.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Optional. Unauthenticated GitHub requests have a much lower rate limit.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"

# Label keywords (matched as whole words on the commit subject line).
BUGFIX_KEYWORDS = ["fix", "bug", "hotfix", "patch", "revert"]

LOOKAHEAD_DAYS = 30
# Skip PRs whose merge_at + LOOKAHEAD_DAYS is still in the future.
# Incomplete-window negatives are not reliable; keeping only early positives
# would bias the sample toward quickly observed events.
REQUIRE_COMPLETE_LOOKAHEAD = True
CHURN_WINDOW_MONTHS = 3
# Approximate month length used only for the churn lookback window.
DAYS_PER_MONTH = 30

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "risk_classifier.joblib"
SYNTHETIC_MODEL_PATH = PROJECT_ROOT / "models" / "synthetic_risk_classifier.joblib"

BORDERLINE_LOW = 0.4
BORDERLINE_HIGH = 0.6

# GitHub HTTP
GITHUB_API = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 6
MAX_RATE_LIMIT_WAIT_SECONDS = 90
REQUEST_GAP_SECONDS = 0.25

# Bounds so collection stays reproducible and not unbounded.
PR_FILES_MAX_PAGES = 10
COMMITS_MAX_PAGES = 20
PR_COMMITS_MAX_PAGES = 10
CHURN_FILE_CAP = 5
LABEL_FILE_CAP = 10
CLOSED_PR_SCAN_MAX_PAGES = 40

# Dataset / training
MIN_TRAIN_ROWS = 20
TEMPORAL_TEST_SIZE = 0.25
RANDOM_SPLIT_TEST_SIZE = 0.25
RANDOM_SEED = 42
SUPPORTED_MODEL_TYPES = ("gboost", "logreg")

# Deterministic feature order for training and inference.
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
