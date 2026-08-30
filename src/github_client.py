"""GitHub REST helper: pagination, retries, and rate-limit handling.

Does not log tokens or Authorization headers.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from . import config

logger = logging.getLogger("bugpredict.github")


class GitHubAPIError(RuntimeError):
    """Raised when a GitHub request fails after retries or is not retryable."""


@dataclass
class PagedList:
    """Paginated GitHub list plus whether the walk hit a cap."""

    items: list[Any]
    truncated: bool = False

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


def _headers() -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ai-powered-pr-risk-predictor",
    }
    if config.GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    return h


def _is_rate_limited(response: requests.Response) -> bool:
    if response.status_code == 429:
        return True
    if response.status_code != 403:
        return False
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining == "0":
        return True
    body = (response.text or "").lower()
    return "rate limit" in body or "secondary rate" in body


def _wait_seconds(response: requests.Response) -> float:
    """Uncapped delay suggested by GitHub. Caller decides whether it exceeds the cap."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    reset = response.headers.get("X-RateLimit-Reset")
    if reset:
        try:
            return max(0.0, float(reset) - time.time()) + 1.0
        except ValueError:
            pass
    return 2.0


def request_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    operation: str = "GET",
) -> Any:
    """GET JSON with retry/backoff. Raises GitHubAPIError on failure."""
    last_error: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=_headers(),
                params=params,
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            last_error = exc
            delay = min(2 ** (attempt - 1), 16)
            logger.warning(
                "operation=%s url=%s attempt=%s/%s transport_error=%s; retry in %ss",
                operation,
                url,
                attempt,
                config.MAX_RETRIES,
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)
            continue

        remaining = response.headers.get("X-RateLimit-Remaining")
        logger.debug(
            "operation=%s url=%s status=%s remaining=%s",
            operation,
            url,
            response.status_code,
            remaining,
        )

        if _is_rate_limited(response):
            wait = _wait_seconds(response)
            if wait > config.MAX_RATE_LIMIT_WAIT_SECONDS:
                raise GitHubAPIError(
                    f"GitHub rate-limited on {operation}; need to wait {wait:.0f}s "
                    f"(cap is {config.MAX_RATE_LIMIT_WAIT_SECONDS}s). Retry later."
                )
            logger.warning(
                "operation=%s rate-limited; sleeping %.1fs (attempt %s/%s)",
                operation,
                wait,
                attempt,
                config.MAX_RETRIES,
            )
            time.sleep(wait)
            last_error = GitHubAPIError(f"rate limited: HTTP {response.status_code}")
            continue

        if response.status_code >= 500:
            delay = min(2 ** (attempt - 1), 16)
            logger.warning(
                "operation=%s url=%s status=%s; retry in %ss",
                operation,
                url,
                response.status_code,
                delay,
            )
            time.sleep(delay)
            last_error = GitHubAPIError(f"HTTP {response.status_code} for {operation}")
            continue

        if response.status_code >= 400:
            raise GitHubAPIError(
                f"GitHub {operation} failed: HTTP {response.status_code} "
                f"({response.reason})"
            )

        if remaining is not None:
            try:
                if int(remaining) <= 5:
                    logger.info(
                        "GitHub primary rate limit low (%s remaining); pausing 2s",
                        remaining,
                    )
                    time.sleep(2.0)
            except ValueError:
                pass

        time.sleep(config.REQUEST_GAP_SECONDS)
        try:
            return response.json()
        except ValueError as exc:
            raise GitHubAPIError(f"Invalid JSON from {operation}") from exc

    raise GitHubAPIError(f"{operation} failed after {config.MAX_RETRIES} attempts: {last_error}")


def get_paginated(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    operation: str,
    per_page: int = 100,
    max_pages: int = 20,
    item_limit: int | None = None,
) -> PagedList:
    """Walk numeric pages until a short page, empty page, or limits."""
    params = dict(params or {})
    params["per_page"] = per_page
    items: list[Any] = []
    truncated = False
    for page in range(1, max_pages + 1):
        params["page"] = page
        batch = request_json(url, params, operation=f"{operation} page={page}")
        if not isinstance(batch, list):
            raise GitHubAPIError(f"{operation} expected a list, got {type(batch).__name__}")
        if not batch:
            break
        items.extend(batch)
        logger.info(
            "operation=%s page=%s batch=%s total=%s",
            operation,
            page,
            len(batch),
            len(items),
        )
        if item_limit is not None and len(items) >= item_limit:
            extra = len(items) > item_limit or len(batch) == per_page
            return PagedList(items=items[:item_limit], truncated=extra)
        if len(batch) < per_page:
            break
    else:
        truncated = True
        logger.warning(
            "operation=%s hit max_pages=%s; results may be truncated",
            operation,
            max_pages,
        )
    return PagedList(items=items, truncated=truncated)
