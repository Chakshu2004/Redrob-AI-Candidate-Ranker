"""
features.py
============
Candidate Understanding Engine (Required Feature #2).

Turns a raw candidate JSON record into a flat, structured `CandidateFeatures`
dict that the scoring engine and reasoning generator consume. This is the
layer that understands relationships like:

    Team Lead          -> leadership / architecture-only risk
    Startup Founder     -> ownership
    Open-source contrib -> external validation
    IT-services lifer   -> consulting-only penalty

without relying on exact keyword matches alone -- categories, ratios, and
career-history shape are used instead of single-skill lookups.

New profile fields can be incorporated by extending `build_features` and
(if relevant) `SKILL_TAXONOMY` / the various keyword lists in config.py --
the rest of the pipeline (scoring, reasoning) reads from this dict by key,
so adding fields here doesn't require touching downstream code unless the
new field should affect scoring.
"""

from datetime import date, datetime
from typing import Any, Dict, List

from . import config


def _to_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _months_between(d1: date, d2: date) -> int:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def _match_skill_categories(skill_name: str) -> List[str]:
    """Case-insensitive substring match of a raw skill name against the taxonomy."""
    name = skill_name.lower().strip()
    cats = set()
    for key, categories in config.SKILL_TAXONOMY.items():
        if key in name or name in key:
            cats.update(categories)
    return list(cats)


def _skill_category_scores(skills: List[Dict[str, Any]], assessment_scores: Dict[str, float]):
    """
    For each taxonomy category, compute a 0-1 strength score combining:
      - presence of a matching skill
      - its proficiency level (beginner..expert -> 0.25..1.0)
      - duration_months used (caps the credit a freshly-claimed "expert" skill gets)
      - any redrob skill_assessment_scores for that skill name (0-100 -> 0-1)

    Returns (category_scores: dict, category_evidence: dict[list of (name, proficiency, months, assess)])
    """
    proficiency_weight = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}

    category_scores: Dict[str, float] = {}
    category_evidence: Dict[str, List[Dict[str, Any]]] = {}

    for skill in skills:
        name = skill.get("name", "")
        prof = skill.get("proficiency", "intermediate")
        months = skill.get("duration_months", 0) or 0
        cats = _match_skill_categories(name)
        if not cats:
            continue

        prof_w = proficiency_weight.get(prof, 0.5)
        # Discount unverified "expert" claims with little time behind them
        if prof == "expert" and months < 6:
            prof_w = 0.4
        elif months >= 24:
            prof_w = min(1.0, prof_w + 0.1)

        # redrob assessment score for this skill, if present (0-100 -> 0-1)
        assess = assessment_scores.get(name)
        assess_w = (assess / 100.0) if isinstance(assess, (int, float)) else None

        if assess_w is not None:
            combined = 0.5 * prof_w + 0.5 * assess_w
        else:
            combined = prof_w

        for cat in cats:
            category_scores[cat] = max(category_scores.get(cat, 0.0), combined)
            category_evidence.setdefault(cat, []).append(
                {"name": name, "proficiency": prof, "months": months, "assessment": assess}
            )

    return category_scores, category_evidence


def _career_history_analysis(career_history: List[Dict[str, Any]], years_of_experience: float):
    """
    Walks career_history (newest-first, per dataset convention) to compute:
      - total tracked months
      - fraction of months in roles that look AI/ML-relevant (by title)
      - fraction of months at consulting-only companies
      - title_chaser_score: short-tenure (<18mo, non-current) roles with
        escalating seniority titles
      - architect_no_code_months: months in current role if current title
        is architect/tech-lead/EM-type (proxy for "hasn't written code")
      - pure_research_only: every role's title looks like pure research,
        with no product/production-sounding company
    """
    total_months = 0
    ai_relevant_months = 0
    consulting_months = 0
    short_tenure_escalation_hits = 0
    architect_like_current_months = 0
    research_like_roles = 0
    n_roles = len(career_history) or 1

    seniority_ladder = ["junior", "associate", "engineer", "senior", "staff", "principal", "lead", "head", "director", "vp"]

    def seniority_rank(title: str) -> int:
        t = title.lower()
        rank = 0
        for i, word in enumerate(seniority_ladder):
            if word in t:
                rank = max(rank, i)
        return rank

    prev_rank = None
    for role in career_history:
        months = role.get("duration_months", 0) or 0
        total_months += months
        title = (role.get("title") or "").lower()
        company = (role.get("company") or "").lower()

        if any(t in title for t in config.RELEVANT_AI_TITLES):
            ai_relevant_months += months
        if any(c in company for c in config.CONSULTING_COMPANIES):
            consulting_months += months
        if any(t in title for t in config.PURE_RESEARCH_TITLES):
            research_like_roles += 1
        if role.get("is_current") and any(t in title for t in config.ARCHITECT_TL_TITLES):
            architect_like_current_months = months

        rank = seniority_rank(title)
        if prev_rank is not None and rank > prev_rank and months < 18 and not role.get("is_current"):
            short_tenure_escalation_hits += 1
        prev_rank = rank

    total_months = max(total_months, 1)
    return {
        "total_tracked_months": total_months,
        "ai_relevant_fraction": ai_relevant_months / total_months,
        "consulting_fraction": consulting_months / total_months,
        "title_chaser_score": min(1.0, short_tenure_escalation_hits / max(1, n_roles - 1)),
        "architect_like_current_months": architect_like_current_months,
        "pure_research_role_fraction": research_like_roles / n_roles,
        "n_roles": n_roles,
    }


def _location_score(location: str, country: str, willing_to_relocate: bool):
    loc = (location or "").lower()
    pref = config.ROLE_PROFILE["location_preference"]

    if country != pref["country_required_unless_relocate"]:
        return 0.3 if willing_to_relocate else 0.05

    if any(city.lower() in loc for city in pref["ideal"]):
        return 1.0
    if any(city.lower() in loc for city in pref["acceptable_india"]):
        return 0.75
    # elsewhere in India
    return 0.55 if willing_to_relocate else 0.45


def _experience_band_fit(years: float) -> float:
    band = config.ROLE_PROFILE["experience_band"]
    if band["min"] <= years <= band["max"]:
        return 1.0
    if band["soft_min"] <= years <= band["soft_max"]:
        # linear taper outside the ideal band but inside the soft band
        if years < band["min"]:
            span = band["min"] - band["soft_min"]
            return 0.5 + 0.5 * (years - band["soft_min"]) / max(span, 1e-6)
        else:
            span = band["soft_max"] - band["max"]
            return 0.5 + 0.5 * (band["soft_max"] - years) / max(span, 1e-6)
    return 0.1


def _eval_framework_text_score(text: str) -> float:
    """
    No candidate in this dataset lists "NDCG"/"MRR"/"A-B Testing" as a literal
    skill entry, so the eval_frameworks category would otherwise always be 0
    for everyone (uniformly deflating skill_alignment and producing an
    identical "no evidence of ranking evaluation" caveat on nearly every
    top-100 row). Instead, scan career-history descriptions / summary for
    concrete evaluation-methodology terms.
    """
    t = text.lower()
    strong_terms = ["ndcg", "mrr", "map@", "recall@", "precision@", "offline evaluation", "ranking evaluation", "online evaluation"]
    moderate_terms = ["a/b test", "ab test", "click-through", "ctr"]
    if any(term in t for term in strong_terms):
        return 1.0
    if any(term in t for term in moderate_terms):
        return 0.6
    return 0.0



def build_semantic_text(candidate: Dict[str, Any]) -> str:
    """Concatenate the free-text fields used for TF-IDF semantic similarity."""
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
    ]
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))
    for cert in candidate.get("certifications", []) or []:
        if isinstance(cert, str):
            parts.append(cert)
        elif isinstance(cert, dict):
            parts.append(cert.get("name", ""))
    return " ".join(p for p in parts if p)


def build_features(candidate: Dict[str, Any]) -> Dict[str, Any]:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career_history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    assessment_scores = signals.get("skill_assessment_scores", {}) or {}

    years_exp = float(profile.get("years_of_experience", 0) or 0)
    current_title = profile.get("current_title", "") or ""
    current_title_lower = current_title.lower()

    cat_scores, cat_evidence = _skill_category_scores(skills, assessment_scores)
    career = _career_history_analysis(career_history, years_exp)

    # Eval-framework signal: skill-name matching always returns 0 for this
    # dataset (see _eval_framework_text_score docstring), so fall back to
    # scanning free-text descriptions for NDCG/MRR/A-B-testing terminology.
    eval_text_blob = " ".join(
        [profile.get("summary", "")] + [r.get("description", "") for r in career_history]
    )
    eval_text_score = _eval_framework_text_score(eval_text_blob)
    if eval_text_score > cat_scores.get("eval_frameworks", 0.0):
        cat_scores["eval_frameworks"] = eval_text_score

    # --- domain-fit flags ---
    is_unrelated_title = any(t in current_title_lower for t in config.CLEARLY_UNRELATED_TITLES)
    is_relevant_title = any(t in current_title_lower for t in config.RELEVANT_AI_TITLES)
    is_pure_research = career["pure_research_role_fraction"] >= 0.5 and career["ai_relevant_fraction"] < 0.3
    is_consulting_only = career["consulting_fraction"] >= 0.9
    is_architect_no_code = career["architect_like_current_months"] >= 18

    has_framework_enthusiast_signal = "framework_enthusiast" in cat_scores
    has_required_depth = any(
        cat_scores.get(c, 0) > 0
        for c in ("embeddings_retrieval", "vector_db_hybrid_search", "eval_frameworks")
    )
    is_framework_enthusiast_only = has_framework_enthusiast_signal and not has_required_depth and years_exp < 4

    cv_speech_score = cat_scores.get("cv_speech_robotics", 0)
    nlp_score = cat_scores.get("nlp_domain", 0)
    is_cv_speech_only = cv_speech_score > 0 and nlp_score == 0 and not has_required_depth

    github_score = signals.get("github_activity_score", -1)
    is_closed_source_long_tenure = (github_score is None or github_score <= 0) and years_exp >= 5 and "open_source" not in cat_scores

    location_score = _location_score(profile.get("location", ""), profile.get("country", ""), bool(signals.get("willing_to_relocate", False)))
    experience_band_fit = _experience_band_fit(years_exp)

    return {
        "candidate_id": candidate.get("candidate_id"),
        "profile": profile,
        "years_of_experience": years_exp,
        "current_title": current_title,
        "current_company": profile.get("current_company", ""),
        "current_industry": profile.get("current_industry", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),

        "semantic_text": build_semantic_text(candidate),

        "skill_category_scores": cat_scores,
        "skill_category_evidence": cat_evidence,

        "experience_band_fit": experience_band_fit,
        "ai_relevant_fraction": career["ai_relevant_fraction"],
        "consulting_fraction": career["consulting_fraction"],
        "title_chaser_score": career["title_chaser_score"],
        "n_roles": career["n_roles"],
        "total_tracked_months": career["total_tracked_months"],

        "is_unrelated_title": is_unrelated_title,
        "is_relevant_title": is_relevant_title,
        "is_pure_research": is_pure_research,
        "is_consulting_only": is_consulting_only,
        "is_architect_no_code": is_architect_no_code,
        "is_framework_enthusiast_only": is_framework_enthusiast_only,
        "is_cv_speech_only": is_cv_speech_only,
        "is_closed_source_long_tenure": is_closed_source_long_tenure,

        "location_score": location_score,

        # behavioral / availability signals (raw, for scoring + reasoning)
        "open_to_work_flag": bool(signals.get("open_to_work_flag", False)),
        "last_active_date": signals.get("last_active_date"),
        "notice_period_days": signals.get("notice_period_days", 999),
        "recruiter_response_rate": signals.get("recruiter_response_rate", 0.0),
        "interview_completion_rate": signals.get("interview_completion_rate", 0.0),
        "offer_acceptance_rate": signals.get("offer_acceptance_rate", -1),
        "applications_submitted_30d": signals.get("applications_submitted_30d", 0),
        "profile_completeness_score": signals.get("profile_completeness_score", 0),
        "github_activity_score": github_score,
        "willing_to_relocate": bool(signals.get("willing_to_relocate", False)),
        "preferred_work_mode": signals.get("preferred_work_mode", "unspecified"),
        "verified_email": bool(signals.get("verified_email", False)),
        "verified_phone": bool(signals.get("verified_phone", False)),
        "linkedin_connected": bool(signals.get("linkedin_connected", False)),

        # raw, used by honeypot.py
        "_career_history": career_history,
        "_skills": skills,
    }
