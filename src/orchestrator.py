"""Orchestrator: GitHub PR → features → ML score → optional LLM commentary."""
from __future__ import annotations

import logging

from . import config, data_agent, llm_agent, ml_agent
from .llm_agent import LLMUnavailableError
from .ml_agent import ModelError

logger = logging.getLogger("bugpredict.orchestrator")


class FeaturePipelineError(RuntimeError):
    pass


def label_from_score(score: float) -> str:
    if score < config.BORDERLINE_LOW:
        return "low"
    if score > config.BORDERLINE_HIGH:
        return "high"
    return "medium"


def is_borderline(score: float) -> bool:
    return config.BORDERLINE_LOW <= score <= config.BORDERLINE_HIGH


def run_pipeline(
    repo: str,
    pr_number: int,
    *,
    use_llm: bool = True,
    model_path=None,
    allow_synthetic: bool = False,
) -> dict:
    logger.info("repo=%s pr=%s operation=pipeline", repo, pr_number)
    pr = data_agent.fetch_pr(repo, pr_number)
    if not pr.get("merged_at"):
        raise FeaturePipelineError(
            f"{repo}#{pr_number} is not merged; this pipeline scores merged PRs only"
        )

    features = data_agent.featurize_pr(repo, pr, save_raw=True)
    try:
        risk_result = ml_agent.predict_risk(
            features, model_path=model_path, allow_synthetic=allow_synthetic
        )
    except ModelError as exc:
        raise FeaturePipelineError(str(exc)) from exc

    score = risk_result["score"]
    risk_label = label_from_score(score)

    report = {
        "repo": repo,
        "pr_number": pr_number,
        "merged_at": features.get("merged_at"),
        "risk_score": score,
        "risk_label": risk_label,
        "risk_label_note": (
            "risk_label is a banding of the ML probability "
            f"(low < {config.BORDERLINE_LOW}, "
            f"medium {config.BORDERLINE_LOW}-{config.BORDERLINE_HIGH}, "
            f"high > {config.BORDERLINE_HIGH}). "
            "It is not a causal 'will cause a bug' claim."
        ),
        "top_features": risk_result["top_features"],
        "model_wide_importances": risk_result.get("model_wide_importances"),
        "importance_note": risk_result.get("importance_note"),
        "feature_values": risk_result.get("feature_values"),
        "trained_on": risk_result.get("trained_on"),
        "explanation": None,
        "llm_status": "skipped" if not use_llm else "pending",
    }

    if not use_llm:
        return report

    try:
        explanation = llm_agent.explain(features, risk_result)
        report["explanation"] = explanation
        report["llm_status"] = "ok"
    except LLMUnavailableError as exc:
        logger.warning("repo=%s pr=%s llm explain unavailable: %s", repo, pr_number, exc)
        report["explanation"] = None
        report["llm_status"] = "unavailable"
        report["llm_error"] = str(exc)
        return report

    if is_borderline(score):
        try:
            contrarian = llm_agent.contrarian_pass(features, risk_result, explanation)
            report["contrarian"] = {
                "counter_argument": contrarian["counter_argument"],
                "llm_synthesis": contrarian["llm_synthesis"],
                "note": contrarian["note"],
                "does_not_override_risk_score": True,
            }
        except LLMUnavailableError as exc:
            logger.warning("repo=%s pr=%s contrarian unavailable: %s", repo, pr_number, exc)
            report["contrarian"] = {"error": str(exc)}
    else:
        report["contrarian"] = None

    return report
