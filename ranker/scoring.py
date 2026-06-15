"""
scoring.py
==========
Hybrid Scoring Engine (Required Feature #5) + Bias Reduction (Feature #7).

Combines five 0-1 sub-scores into a final 0-1 score using config.WEIGHTS:

    semantic_similarity     (TF-IDF cosine vs. JD)
    skill_alignment          (required/preferred skill-category coverage,
                               weighted by redrob skill_assessment_scores)
    experience_relevance     (years-of-experience band fit + how much of the
                               career was actually applied-ML/AI work)
    domain_fit_screen        (the JD's explicit "what we do NOT want" list:
                               pure research, consulting-only, framework
                               tutorials, CV/speech/robotics-only,
                               title-chasing, architecture-only)
    behavioral_availability  (the 23 redrob_signals as an availability /
                               engagement multiplier, per the JD's closing
                               note about down-weighting unavailable
                               "perfect-on-paper" candidates)

Bias reduction: NONE of these sub-scores ever read anonymized_name, gender,
age, ethnicity, or any other identity field -- only skills, experience,
career-history shape, location/work-mode logistics, and platform-engagement
signals. `BIAS_NOTICE` is surfaced verbatim in the API/UI per Feature #7.
"""

from typing import Any, Dict

from . import config

BIAS_NOTICE = "Ranking generated using skill and experience signals only."


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _skill_alignment(features: Dict[str, Any]) -> float:
    cat_scores = features["skill_category_scores"]
    rp = config.ROLE_PROFILE

    required = rp["required_skill_categories"]
    preferred = rp["preferred_skill_categories"]

    required_vals = [cat_scores.get(c, 0.0) for c in required]
    preferred_vals = [cat_scores.get(c, 0.0) for c in preferred]

    required_score = sum(required_vals) / len(required_vals)
    preferred_score = sum(preferred_vals) / len(preferred_vals) if preferred_vals else 0.0

    # A candidate with literally none of the 3 most distinguishing required
    # categories (everything except plain "python") is very unlikely to be
    # a real fit, regardless of how strong their Python score is.
    distinguishing = [c for c in required if c != "python"]
    distinguishing_hit = any(cat_scores.get(c, 0.0) > 0 for c in distinguishing)
    if not distinguishing_hit:
        required_score *= 0.3

    return _clip01(0.7 * required_score + 0.3 * preferred_score)


def _experience_relevance(features: Dict[str, Any]) -> float:
    band_fit = features["experience_band_fit"]
    ai_fraction = features["ai_relevant_fraction"]
    title_bonus = 1.0 if features["is_relevant_title"] else 0.0
    return _clip01(0.4 * band_fit + 0.4 * ai_fraction + 0.2 * title_bonus)


def _domain_fit_screen(features: Dict[str, Any]) -> float:
    score = 1.0
    if features["is_unrelated_title"]:
        score -= 0.6
    if features["is_pure_research"]:
        score -= 0.5
    if features["is_consulting_only"]:
        score -= 0.4
    if features["is_architect_no_code"]:
        score -= 0.3
    if features["is_framework_enthusiast_only"]:
        score -= 0.3
    if features["is_cv_speech_only"]:
        score -= 0.3
    if features["is_closed_source_long_tenure"]:
        score -= 0.15
    score -= 0.3 * features["title_chaser_score"]
    return _clip01(score)


def _recency_score(last_active_date: str) -> float:
    """1.0 if active within the last week, tapering to 0 by ~6 months inactive."""
    from datetime import datetime
    if not last_active_date:
        return 0.0
    try:
        last = datetime.strptime(last_active_date, "%Y-%m-%d").date()
    except ValueError:
        return 0.0
    days_ago = (config.TODAY - last).days
    if days_ago <= 7:
        return 1.0
    if days_ago >= 180:
        return 0.0
    return _clip01(1.0 - (days_ago - 7) / (180 - 7))


def _notice_period_score(days: float) -> float:
    pref = config.ROLE_PROFILE["notice_period_preference_days"]
    if days <= pref:
        return 1.0
    if days >= 120:
        return 0.1
    return _clip01(1.0 - (days - pref) / (120 - pref))


def _behavioral_availability(features: Dict[str, Any]) -> float:
    open_to_work = 1.0 if features["open_to_work_flag"] else 0.3
    notice_score = _notice_period_score(features["notice_period_days"])
    recency_score = _recency_score(features["last_active_date"])
    location_score = features["location_score"]

    availability_component = (
        0.30 * open_to_work + 0.25 * notice_score + 0.25 * recency_score + 0.20 * location_score
    )

    response_rate = _clip01(features["recruiter_response_rate"])
    interview_rate = _clip01(features["interview_completion_rate"])
    offer_rate = features["offer_acceptance_rate"]
    offer_component = 0.5 if offer_rate is None or offer_rate < 0 else _clip01(offer_rate)

    engagement_component = (response_rate + interview_rate + offer_component) / 3.0

    return _clip01(0.6 * availability_component + 0.4 * engagement_component)


def score_candidate(features: Dict[str, Any], semantic_similarity: float, honeypot_result: Dict[str, Any]) -> Dict[str, Any]:
    sub_scores = {
        "semantic_similarity": _clip01(semantic_similarity),
        "skill_alignment": _skill_alignment(features),
        "experience_relevance": _experience_relevance(features),
        "domain_fit_screen": _domain_fit_screen(features),
        "behavioral_availability": _behavioral_availability(features),
    }

    final = sum(sub_scores[k] * config.WEIGHTS[k] for k in config.WEIGHTS)

    if honeypot_result["is_honeypot"]:
        final *= 0.05  # forces honeypots to the very bottom of the ranking

    return {"sub_scores": sub_scores, "final_score": _clip01(final), "fairness_notice": BIAS_NOTICE}
