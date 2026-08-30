"""Data Agent: merged GitHub PRs → features + 30-day bug-fix-signal labels."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

import pandas as pd

from . import config
from .github_client import GitHubAPIError, PagedList, get_paginated, request_json
from .labeling import (
    files_for_churn,
    files_for_labeling,
    is_bugfix_message,
    is_config_path,
    is_dependency_path,
    is_source_path,
    is_test_path,
)

logger = logging.getLogger("bugpredict.data")


class FeatureExtractionError(RuntimeError):
    """Raised when a feature cannot be computed (do not treat as zero)."""


def parse_github_datetime(value: str) -> datetime:
    if not value or not isinstance(value, str):
        raise ValueError("merged_at is missing")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def lookahead_end(merged_at: datetime) -> datetime:
    return merged_at + timedelta(days=config.LOOKAHEAD_DAYS)


def is_lookahead_complete(merged_at: datetime, now: datetime | None = None) -> bool:
    """True iff the full LOOKAHEAD_DAYS window has elapsed (right-censoring gate)."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now >= lookahead_end(merged_at)


def _commit_items(result) -> list:
    if isinstance(result, PagedList):
        return result.items
    return list(result)


def _is_truncated(result) -> bool:
    return bool(getattr(result, "truncated", False))


def ensure_data_dirs() -> None:
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def fetch_merged_prs(
    repo: str,
    limit: int = 30,
    *,
    now: datetime | None = None,
) -> tuple[list[dict], dict]:
    """Fetch merged PRs with a completed lookahead window.

    GitHub cannot sort pulls by merged_at, so we scan recently created closed
    PRs until we have enough merged PRs whose full LOOKAHEAD_DAYS window has
    elapsed.

    This prevents the dataset from being dominated by right-censored PRs.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    cutoff = now - timedelta(days=config.LOOKAHEAD_DAYS)

    url = f"{config.GITHUB_API}/repos/{repo}/pulls"
    merged: list[dict] = []
    scanned = 0
    incomplete_merged = 0

    for page in range(1, config.CLOSED_PR_SCAN_MAX_PAGES + 1):
        batch = request_json(
            url,
            params={
                "state": "closed",
                "sort": "created",
                "direction": "desc",
                "per_page": 50,
                "page": page,
            },
            operation=f"list closed PRs {repo} page={page}",
        )

        if not isinstance(batch, list) or not batch:
            break

        scanned += len(batch)

        for pr in batch:
            merged_at_raw = pr.get("merged_at")
            if not merged_at_raw:
                continue

            try:
                merged_at = parse_github_datetime(merged_at_raw)
            except ValueError:
                continue

            # Skip right-censored PRs.
            if merged_at > cutoff:
                incomplete_merged += 1
                continue

            merged.append(pr)

            if len(merged) >= limit:
                break

        if len(merged) >= limit:
            break

        if len(batch) < 50:
            break

    # Sort by merge time so selection is deterministic.
    merged.sort(
        key=lambda pr: (
            parse_github_datetime(pr["merged_at"]),
            pr.get("number", 0),
        ),
        reverse=True,
    )

    selected = merged[:limit]

    stats = {
        "closed_scanned": scanned,
        "merged_found": len(merged),
        "selected": len(selected),
        "incomplete_merged_skipped": incomplete_merged,
        "lookahead_cutoff": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection": (
            "merged PRs with completed lookahead among recently created "
            "closed PRs"
        ),
    }

    logger.info(
        "repo=%s closed_scanned=%s merged_found=%s selected=%s "
        "incomplete_skipped=%s",
        repo,
        scanned,
        len(merged),
        len(selected),
        incomplete_merged,
    )

    return selected, stats

def fetch_pr(repo: str, pr_number: int) -> dict:
    logger.info("repo=%s pr=%s operation=get_pr", repo, pr_number)
    return request_json(
        f"{config.GITHUB_API}/repos/{repo}/pulls/{pr_number}",
        operation=f"get PR {repo}#{pr_number}",
    )


def fetch_pr_files(repo: str, pr_number: int) -> list[dict]:
    logger.info("repo=%s pr=%s operation=list_files", repo, pr_number)
    page = get_paginated(
        f"{config.GITHUB_API}/repos/{repo}/pulls/{pr_number}/files",
        operation=f"PR files {repo}#{pr_number}",
        per_page=100,
        max_pages=config.PR_FILES_MAX_PAGES,
    )
    if page.truncated:
        raise FeatureExtractionError(
            f"{repo}#{pr_number} file list truncated at {config.PR_FILES_MAX_PAGES} pages"
        )
    return page.items


def fetch_pr_commits(repo: str, pr_number: int) -> list[dict]:
    logger.info("repo=%s pr=%s operation=list_pr_commits", repo, pr_number)
    page = get_paginated(
        f"{config.GITHUB_API}/repos/{repo}/pulls/{pr_number}/commits",
        operation=f"PR commits {repo}#{pr_number}",
        per_page=100,
        max_pages=config.PR_COMMITS_MAX_PAGES,
    )
    if page.truncated:
        raise FeatureExtractionError(
            f"{repo}#{pr_number} PR commit list truncated; exclude-SHA set would be incomplete"
        )
    return page.items


def fetch_commits_for_path(
    repo: str,
    path: str,
    since_iso: str,
    until_iso: str,
) -> PagedList:
    logger.info(
        "repo=%s operation=commits path=%s since=%s until=%s",
        repo,
        path,
        since_iso,
        until_iso,
    )
    return get_paginated(
        f"{config.GITHUB_API}/repos/{repo}/commits",
        params={"since": since_iso, "until": until_iso, "path": path},
        operation=f"commits {repo} path={path}",
        per_page=100,
        max_pages=config.COMMITS_MAX_PAGES,
    )


def _pr_exclude_shas(pr: dict, pr_commits: list[dict]) -> set[str]:
    shas = {c["sha"] for c in pr_commits if c.get("sha")}
    merge_sha = pr.get("merge_commit_sha")
    if merge_sha:
        shas.add(merge_sha)
    head = (pr.get("head") or {}).get("sha")
    if head:
        shas.add(head)
    return shas


def _strip_file_payload(files: list[dict]) -> list[dict]:
    keep = ("filename", "status", "additions", "deletions", "changes", "sha")
    return [{k: f.get(k) for k in keep} for f in files]


def save_raw_pr(repo: str, pr: dict, files: list[dict]) -> Path | None:
    try:
        ensure_data_dirs()
        slug = repo.replace("/", "_")
        dest_dir = config.DATA_RAW_DIR / slug
        dest_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "repo": repo,
            "pr_number": pr.get("number"),
            "merged_at": pr.get("merged_at"),
            "title": pr.get("title"),
            "labels": [lb.get("name") for lb in (pr.get("labels") or [])],
            "merge_commit_sha": pr.get("merge_commit_sha"),
            "files": _strip_file_payload(files),
        }
        path = dest_dir / f"pr_{pr['number']}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
    except OSError as exc:
        logger.warning("repo=%s pr=%s failed to save raw JSON: %s", repo, pr.get("number"), exc)
        return None


def featurize_pr(repo: str, pr: dict, *, save_raw: bool = False) -> dict:
    """Features use only information at or before merge time.

    Historical churn/prior-bugfix queries use commits strictly before merge
    (GitHub `until=merged_at`) and exclude this PR's own commit SHAs.
    """
    if not pr.get("merged_at"):
        raise FeatureExtractionError(f"{repo}#{pr.get('number')} is not merged")

    pr_number = pr["number"]
    files = fetch_pr_files(repo, pr_number)
    filenames = [f["filename"] for f in files if f.get("filename")]

    additions = sum(int(f.get("additions") or 0) for f in files)
    deletions = sum(int(f.get("deletions") or 0) for f in files)
    files_changed = len(filenames)
    test_paths = [fn for fn in filenames if is_test_path(fn)]
    n_test_files = len(test_paths)
    test_file_present = int(n_test_files > 0)
    source_file_touched = int(any(is_source_path(fn) for fn in filenames))
    config_file_touched = int(any(is_config_path(fn) for fn in filenames))
    dependency_file_touched = int(any(is_dependency_path(fn) for fn in filenames))
    n_directories = len({str(PurePosixPath(fn.replace("\\", "/")).parent) for fn in filenames})

    merged_at = parse_github_datetime(pr["merged_at"])
    churn_since = (merged_at - timedelta(days=config.DAYS_PER_MONTH * config.CHURN_WINDOW_MONTHS))
    since_iso = churn_since.strftime("%Y-%m-%dT%H:%M:%SZ")
    until_iso = pr["merged_at"]

    try:
        pr_commits = fetch_pr_commits(repo, pr_number)
    except GitHubAPIError as exc:
        raise FeatureExtractionError(
            f"could not list commits for {repo}#{pr_number}: {exc}"
        ) from exc
    exclude_shas = _pr_exclude_shas(pr, pr_commits)

    churn_files = files_for_churn(filenames, config.CHURN_FILE_CAP)
    churn_shas: set[str] = set()
    file_had_bugfix = False
    churn_truncated = 0
    if churn_files:
        for fn in churn_files:
            try:
                page = fetch_commits_for_path(repo, fn, since_iso, until_iso)
            except GitHubAPIError as exc:
                logger.error(
                    "repo=%s pr=%s operation=churn path=%s status=failure error=%s",
                    repo,
                    pr_number,
                    fn,
                    exc,
                )
                raise FeatureExtractionError(
                    f"churn unavailable for {repo}#{pr_number}; not substituting 0"
                ) from exc
            if _is_truncated(page):
                churn_truncated = 1
                logger.warning(
                    "repo=%s pr=%s operation=churn path=%s status=truncated",
                    repo,
                    pr_number,
                    fn,
                )
            historical = [c for c in _commit_items(page) if c.get("sha") not in exclude_shas]
            for commit in historical:
                sha = commit.get("sha")
                if sha:
                    churn_shas.add(sha)
                if is_bugfix_message((commit.get("commit") or {}).get("message")):
                    file_had_bugfix = True
    churn_count = len(churn_shas)

    if save_raw:
        save_raw_pr(repo, pr, files)

    author = ((pr.get("user") or {}).get("login")) or ""
    labels = [lb.get("name") for lb in (pr.get("labels") or []) if lb.get("name")]

    return {
        "repo": repo,
        "pr_number": pr_number,
        "author": author,
        "merged_at": pr["merged_at"],
        "title": pr.get("title") or "",
        "pr_labels": ",".join(labels),
        "additions": additions,
        "deletions": deletions,
        "files_changed": files_changed,
        "lines_changed": additions + deletions,
        "test_file_present": test_file_present,
        "n_test_files": n_test_files,
        "source_file_touched": source_file_touched,
        "config_file_touched": config_file_touched,
        "dependency_file_touched": dependency_file_touched,
        "n_directories": n_directories,
        "file_churn_count": churn_count,
        "file_churn_truncated": churn_truncated,
        "file_prior_bugfix_touch": int(file_had_bugfix),
        "day_of_week": merged_at.weekday(),
        "hour_of_day": merged_at.hour,
        "filenames": filenames,
        "exclude_shas": sorted(exclude_shas),
    }


def label_pr(repo: str, features: dict) -> int:
    """1 if a later bug-fix-signal commit touches a PR file within LOOKAHEAD_DAYS.

    Window starts strictly after merge (`merged_at + 1s` as GitHub `since`).
    The original PR's own commits / merge commit are excluded by SHA.
    Documentation-only paths are skipped when any non-doc file exists.
    A truncated commit listing without a match is not treated as label 0.
    """
    merged_at = parse_github_datetime(features["merged_at"])
    until = lookahead_end(merged_at).strftime("%Y-%m-%dT%H:%M:%SZ")
    since = (merged_at + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    exclude = set(features.get("exclude_shas") or [])
    filenames = files_for_labeling(features.get("filenames") or [], config.LABEL_FILE_CAP)
    pr_number = features.get("pr_number")
    any_truncated = False

    for fn in filenames:
        try:
            page = fetch_commits_for_path(repo, fn, since, until)
        except GitHubAPIError as exc:
            logger.error(
                "repo=%s pr=%s operation=label path=%s status=failure error=%s",
                repo,
                pr_number,
                fn,
                exc,
            )
            raise FeatureExtractionError(
                f"labeling failed for {repo}#{pr_number} file {fn}: {exc}"
            ) from exc
        if _is_truncated(page):
            any_truncated = True
            logger.warning(
                "repo=%s pr=%s operation=label path=%s status=truncated",
                repo,
                pr_number,
                fn,
            )
        for commit in _commit_items(page):
            if commit.get("sha") in exclude:
                continue
            message = (commit.get("commit") or {}).get("message")
            if is_bugfix_message(message):
                logger.info(
                    "repo=%s pr=%s label=1 via path=%s sha=%s",
                    repo,
                    pr_number,
                    fn,
                    commit.get("sha"),
                )
                return 1
    if any_truncated:
        raise FeatureExtractionError(
            f"label search truncated for {repo}#{pr_number}; refusing to assign 0"
        )
    return 0


def build_dataset(
    repos: list[str],
    limit_per_repo: int = 20,
    *,
    save_raw: bool = True,
    require_complete_lookahead: bool | None = None,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Collect features+labels. Returns (dataframe, collection_stats).

    By default, PRs whose 30-day window has not elapsed are skipped entirely
    (right-censoring). That avoids treating recent merges as confirmed negatives.
    """
    ensure_data_dirs()
    require = (
        config.REQUIRE_COMPLETE_LOOKAHEAD
        if require_complete_lookahead is None
        else require_complete_lookahead
    )
    rows: list[dict] = []
    skipped: dict[str, int] = {
        "incomplete_lookahead": 0,
        "error": 0,
    }
    fetch_stats: dict = {}
    for repo in repos:
        logger.info("repo=%s operation=build_dataset limit=%s", repo, limit_per_repo)
        try:
            prs, fetch_stats = fetch_merged_prs(
    repo,
    limit=limit_per_repo,
    now=now,
)
        except GitHubAPIError as exc:
            logger.error("repo=%s operation=fetch_merged_prs status=failure error=%s", repo, exc)
            raise
        for pr in prs:
            pr_number = pr.get("number")
            try:
                merged_at = parse_github_datetime(pr["merged_at"])
                complete = is_lookahead_complete(merged_at, now=now)
                if require and not complete:
                    skipped["incomplete_lookahead"] += 1
                    logger.warning(
                        "repo=%s pr=%s status=skipped reason=incomplete_lookahead merged_at=%s",
                        repo,
                        pr_number,
                        pr.get("merged_at"),
                    )
                    continue
                feats = featurize_pr(repo, pr, save_raw=save_raw)
                feats["lookahead_complete"] = int(complete)
                feats["label"] = label_pr(repo, feats)
                rows.append(feats)
                logger.info(
                    "repo=%s pr=%s status=success label=%s files=%s churn=%s truncated=%s",
                    repo,
                    pr_number,
                    feats["label"],
                    feats["files_changed"],
                    feats["file_churn_count"],
                    feats.get("file_churn_truncated"),
                )
            except (GitHubAPIError, FeatureExtractionError, KeyError, ValueError) as exc:
                skipped["error"] += 1
                logger.warning(
                    "repo=%s pr=%s status=skipped reason=error error=%s",
                    repo,
                    pr_number,
                    exc,
                )
    stats = {
        "fetch": fetch_stats,
        "rows": len(rows),
        "skipped": skipped,
        "require_complete_lookahead": require,
    }
    if not rows:
        return pd.DataFrame(), stats
    df = pd.DataFrame(rows)
    if "exclude_shas" in df.columns:
        df = df.drop(columns=["exclude_shas"])
    if "filenames" in df.columns:
        df["filenames"] = df["filenames"].apply(
            lambda x: "|".join(x) if isinstance(x, list) else x
        )
    if "label" in df.columns:
        stats["label_counts"] = df["label"].value_counts().to_dict()
    return df, stats
