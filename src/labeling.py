"""Bug-fix signal detection used for labels (and historical prior-bugfix features).

Target (unchanged in intent): a merged PR is labeled 1 if a *later* commit that
touches a file changed by the PR contains a bug-fix signal in its subject line
within LOOKAHEAD_DAYS.

Refinements vs naive substring matching:
- Whole-word matching so 'prefix' / 'fixture' / 'dispatch' / 'debug' do not match.
- Only the commit subject (first line) is inspected.
- 'revert' is treated as a bug-fix *signal* (often indicates the change was undone).
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath

# Explicit forms rather than a bare 'fix' substring.
_BUGFIX_RE = re.compile(
    r"\b("
    r"fix(?:es|ed|ing)?"
    r"|bug(?:s|fix)?"
    r"|hotfix(?:es|ed)?"
    r"|patch(?:es|ed|ing)?"
    r"|revert(?:s|ed|ing)?"
    r")\b",
    re.IGNORECASE,
)

_DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc", ".markdown"}
_SOURCE_EXTENSIONS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rb", ".rs",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".kt", ".scala", ".php", ".swift",
    ".m", ".mm", ".sh", ".sql",
}
_CONFIG_NAMES = {
    "pyproject.toml", "setup.cfg", "setup.py", "tox.ini", "pytest.ini",
    "makefile", "dockerfile", ".gitignore",
}
_CONFIG_EXTENSIONS = {".yml", ".yaml", ".toml", ".ini", ".cfg", ".json"}
_DEPENDENCY_NAMES = {
    "requirements.txt", "pipfile", "pipfile.lock", "poetry.lock", "package.json",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "gemfile", "gemfile.lock",
    "go.mod", "go.sum", "cargo.toml", "cargo.lock",
}


def commit_subject(message: str | None) -> str:
    text = (message or "").replace("\r\n", "\n").replace("\r", "\n")
    return text.split("\n", 1)[0].strip()


def is_bugfix_message(msg: str | None) -> bool:
    """True if the commit *subject* contains a bug-fix/revert keyword as a whole word."""
    return bool(_BUGFIX_RE.search(commit_subject(msg)))


def is_doc_path(filename: str) -> bool:
    path = PurePosixPath(filename.replace("\\", "/"))
    parts = {p.lower() for p in path.parts}
    if parts & {"docs", "documentation", "doc"}:
        return True
    return path.suffix.lower() in _DOC_EXTENSIONS


def is_test_path(filename: str) -> bool:
    path = PurePosixPath(filename.replace("\\", "/"))
    parts = [p.lower() for p in path.parts]
    if any(p in {"test", "tests", "spec", "specs"} for p in parts[:-1]):
        return True
    name = path.name.lower()
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    if ".test." in name or name.endswith(".spec.ts") or name.endswith(".spec.js"):
        return True
    if name.endswith("_test.go") or name.endswith("test.java"):
        return True
    return False


def is_source_path(filename: str) -> bool:
    if is_doc_path(filename) or is_test_path(filename):
        return False
    return PurePosixPath(filename.replace("\\", "/")).suffix.lower() in _SOURCE_EXTENSIONS


def is_config_path(filename: str) -> bool:
    name = PurePosixPath(filename.replace("\\", "/")).name.lower()
    suffix = PurePosixPath(filename.replace("\\", "/")).suffix.lower()
    return name in _CONFIG_NAMES or suffix in _CONFIG_EXTENSIONS


def is_dependency_path(filename: str) -> bool:
    name = PurePosixPath(filename.replace("\\", "/")).name.lower()
    return name in _DEPENDENCY_NAMES


def files_for_labeling(filenames: list[str], cap: int) -> list[str]:
    """Prefer non-documentation paths so typo-fixes in README are not the target."""
    preferred = [f for f in filenames if not is_doc_path(f)]
    pool = preferred if preferred else list(filenames)
    return pool[:cap]


def files_for_churn(filenames: list[str], cap: int) -> list[str]:
    """Bound historical commit queries; prefer source, then tests, then the rest."""
    source = [f for f in filenames if is_source_path(f)]
    tests = [f for f in filenames if is_test_path(f)]
    rest = [f for f in filenames if f not in source and f not in tests]
    ordered = source + tests + rest
    # Unique, original-first among each bucket.
    seen: set[str] = set()
    out: list[str] = []
    for name in ordered:
        if name not in seen:
            seen.add(name)
            out.append(name)
        if len(out) >= cap:
            break
    return out
