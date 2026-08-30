"""LLM Agent: Groq explanations grounded in supplied evidence only.

The numerical ML score is never overwritten. The contrarian pass is commentary.
"""
from __future__ import annotations

import logging

from . import config

logger = logging.getLogger("bugpredict.llm")


class LLMUnavailableError(RuntimeError):
    """Groq is missing, timed out, or returned an unusable response."""


def _client():
    if not config.GROQ_API_KEY:
        raise LLMUnavailableError("GROQ_API_KEY is not set")
    try:
        from groq import Groq
    except ImportError as exc:
        raise LLMUnavailableError("groq package is not installed") from exc
    return Groq(api_key=config.GROQ_API_KEY)


def _chat(system: str, user: str, max_tokens: int) -> str:
    try:
        client = _client()
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            max_tokens=max_tokens,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except LLMUnavailableError:
        raise
    except Exception as exc:
        raise LLMUnavailableError(f"Groq request failed: {type(exc).__name__}") from exc

    try:
        text = (resp.choices[0].message.content or "").strip()
    except (IndexError, AttributeError, TypeError) as exc:
        raise LLMUnavailableError("malformed Groq response") from exc
    if not text:
        raise LLMUnavailableError("empty Groq response")
    return text


def format_evidence(pr_meta: dict, risk_result: dict) -> str:
    values = risk_result.get("feature_values") or {}
    importances = risk_result.get("model_wide_importances") or risk_result.get("top_features")
    return (
        f"PR #{pr_meta.get('pr_number')} in {pr_meta.get('repo', '')}\n"
        f"ML risk probability (class=bug-fix signal within 30 days): {risk_result.get('score')}\n"
        f"Files changed: {pr_meta.get('files_changed')}\n"
        f"Additions: {pr_meta.get('additions')}; deletions: {pr_meta.get('deletions')}; "
        f"lines_changed: {pr_meta.get('lines_changed', values.get('lines_changed'))}\n"
        f"Test files changed: {pr_meta.get('n_test_files')} "
        f"(any test file: {bool(pr_meta.get('test_file_present'))})\n"
        f"Source file touched: {bool(pr_meta.get('source_file_touched'))}\n"
        f"Config file touched: {bool(pr_meta.get('config_file_touched'))}\n"
        f"Dependency file touched: {bool(pr_meta.get('dependency_file_touched'))}\n"
        f"Directories affected: {pr_meta.get('n_directories')}\n"
        f"Historical file churn (up to {config.CHURN_FILE_CAP} files, "
        f"last {config.CHURN_WINDOW_MONTHS} months, excluding this PR's commits): "
        f"{pr_meta.get('file_churn_count')}\n"
        f"Prior bug-fix touch on those files: {bool(pr_meta.get('file_prior_bugfix_touch'))}\n"
        f"Merge weekday (Mon=0): {pr_meta.get('day_of_week')}; "
        f"merge hour UTC: {pr_meta.get('hour_of_day')}\n"
        f"Model-wide feature importances (NOT instance-level or causal): {importances}\n"
        f"Importance note: {risk_result.get('importance_note', '')}\n"
    )


EXPLAIN_SYSTEM = (
    "You explain an ML risk score for a merged GitHub pull request. "
    "The score is the model's estimated probability that this PR is associated with a "
    "subsequent bug-fix signal (commit subject keywords) touching the same files within 30 days. "
    "It is NOT a claim that the PR caused a bug.\n"
    "Rules:\n"
    "- Use ONLY numbers and facts in the evidence block. Do not invent files, counts, or history.\n"
    "- Write 2-4 sentences.\n"
    "- Cite concrete evidence (score, files changed, additions/deletions, tests, churn, prior bugfix).\n"
    "- If you mention feature importances, say they are model-wide influences, not proof of causation "
    "and not necessarily why THIS pull request got its score.\n"
    "- No generic filler."
)

CONTRARIAN_SYSTEM = (
    "You are a skeptical reviewer. Using ONLY the evidence block, argue the opposite of the "
    "initial explanation (why the PR might be less risky than claimed, or more risky if the "
    "explanation called it safe). 2-3 sentences. Do not invent facts. Do not change the numerical score."
)

VERDICT_SYSTEM = (
    "You synthesize two arguments. Output exactly two sentences: "
    "(1) an LLM qualitative verdict of low, medium, or high association-risk based on the text, "
    "(2) a justification grounded in the evidence. "
    "Explicitly state that this verdict does not replace the numerical ML probability."
)


def explain(pr_meta: dict, risk_result: dict) -> str:
    evidence = format_evidence(pr_meta, risk_result)
    return _chat(EXPLAIN_SYSTEM, evidence, max_tokens=300)


def contrarian_pass(pr_meta: dict, risk_result: dict, explanation: str) -> dict:
    evidence = format_evidence(pr_meta, risk_result)
    user = (
        f"{evidence}\n"
        f"Original explanation:\n{explanation}\n"
    )
    counter = _chat(CONTRARIAN_SYSTEM, user, max_tokens=1000)
    verdict = _chat(
        VERDICT_SYSTEM,
        f"Argument for:\n{explanation}\n\nArgument against:\n{counter}\n\n{evidence}",
        max_tokens=600,
    )
    return {
        "counter_argument": counter,
        "llm_synthesis": verdict,
        "note": (
            "The LLM synthesis is qualitative commentary. It does not change "
            "risk_score or the classifier probability."
        ),
        # Backward-compatible alias
        "final_verdict": verdict,
    }
