"""
reasoning.py
============
Explainable AI Layer (Required Feature #6) + Skill Gap Analysis
(Required Feature #8).

Generates, for each candidate, a human-readable `reasoning` string built
entirely from facts present in that candidate's own feature dict -- no facts
are invented. This matters for the hackathon's Stage-4 manual review, which
explicitly penalizes:
  - hallucinated skills/employers not in the profile
  - all-identical or templated (name-swap-only) reasoning
  - reasoning whose tone contradicts the rank

To satisfy "variation" without an LLM, sentence structure is selected from a
small pool of templates using a deterministic hash of candidate_id (so reruns
are reproducible), while the *content* of each sentence is always derived
from that candidate's real years of experience, title, matched skill
categories (with concrete skill names), location/notice-period/activity
signals, and any honest gaps.

Also produces the richer `strengths` / `weaknesses` / `skill_gap` /
`recruiter_summary` objects for a future detail-page UI (Features #6 and #8),
even though only `reasoning` is required in the submission CSV.
"""

import hashlib
from typing import Any, Dict, List

from . import config


def _hash_index(candidate_id: str, n: int) -> int:
    h = hashlib.md5(candidate_id.encode("utf-8")).hexdigest()
    return int(h, 16) % n


def _top_categories(features: Dict[str, Any], categories: List[str], min_score: float = 0.01):
    scores = features["skill_category_scores"]
    hits = [(c, scores[c]) for c in categories if scores.get(c, 0) >= min_score]
    hits.sort(key=lambda x: -x[1])
    return hits


def _skill_names_for_category(features: Dict[str, Any], category: str, n: int = 2) -> List[str]:
    evidence = features["skill_category_evidence"].get(category, [])
    evidence = sorted(evidence, key=lambda e: -(e.get("assessment") or 0))
    return [e["name"] for e in evidence[:n]]


def _strength_phrase(features: Dict[str, Any]) -> str:
    rp = config.ROLE_PROFILE
    required_hits = _top_categories(features, rp["required_skill_categories"])
    preferred_hits = _top_categories(features, rp["preferred_skill_categories"])

    mentions = []
    for cat, _ in (required_hits + preferred_hits)[:2]:
        names = _skill_names_for_category(features, cat)
        label = config.CATEGORY_LABELS[cat]
        # Drop names that just repeat the category label (e.g. "Python (Python)")
        names = [n for n in names if n.lower() != label.lower()]
        if names:
            mentions.append(f"{label} ({', '.join(names)})")
        else:
            mentions.append(label)

    if mentions:
        return "background in " + " and ".join(mentions)
    if features["is_relevant_title"]:
        return "a directly relevant ML/AI title but limited matched skill evidence"
    return "adjacent experience without a clear retrieval/ranking specialization"


def _gap_phrase(features: Dict[str, Any]) -> str:
    rp = config.ROLE_PROFILE
    missing = [
        config.CATEGORY_LABELS[c]
        for c in rp["required_skill_categories"]
        if features["skill_category_scores"].get(c, 0) < 0.01
    ]
    gaps = []
    if missing:
        gaps.append("no evidence of " + " or ".join(missing))
    if features["consulting_fraction"] > 0.4 and not features["is_consulting_only"]:
        gaps.append("a meaningful chunk of career at IT-services firms")
    if features["notice_period_days"] > config.ROLE_PROFILE["notice_period_preference_days"]:
        gaps.append(f"a {int(features['notice_period_days'])}-day notice period")
    if features["country"] != "India" and not features["willing_to_relocate"]:
        gaps.append(f"based outside India ({features['country']}) and not flagged as willing to relocate")
    if features["title_chaser_score"] > 0.5:
        gaps.append("a history of short stints with escalating titles")
    return gaps


def _availability_phrase(features: Dict[str, Any]) -> str:
    bits = []
    if features["open_to_work_flag"]:
        bits.append("marked open to work")
    rr = features["recruiter_response_rate"]
    if rr >= 0.5:
        bits.append(f"a {rr:.0%} recruiter response rate")
    elif rr <= 0.15:
        bits.append(f"a low {rr:.0%} recruiter response rate")
    if features["location"] and features["country"] == "India":
        bits.append(f"based in {features['location']}")
    return bits


def build_reasoning(features: Dict[str, Any], honeypot_result: Dict[str, Any], rank: int) -> str:
    cid = features["candidate_id"]
    title = features["current_title"] or "candidate"
    company = features["current_company"]
    years = features["years_of_experience"]

    if honeypot_result["is_honeypot"]:
        reason = honeypot_result["reasons"][0] if honeypot_result["reasons"] else "an internally inconsistent profile"
        return f"{title} with {years:.1f} yrs claimed, but flagged: {reason}; excluded from serious contention despite raw signal strength."

    strength = _strength_phrase(features)
    gaps = _gap_phrase(features)
    avail = _availability_phrase(features)

    opening_templates = [
        f"{title} ({company}) with {years:.1f} years of experience, {strength}.",
        f"{years:.1f}-year {title}, {strength}.",
        f"Currently a {title} at {company} with {years:.1f} years total experience; {strength}.",
        f"{title} with {years:.1f} yrs; {strength}.",
    ]
    opening = opening_templates[_hash_index(cid, len(opening_templates))]

    extra_sentences = []
    if avail:
        avail_text = ", ".join(avail)
        extra_sentences.append(avail_text[0].upper() + avail_text[1:] + ".")
    if gaps:
        # For top ranks, phrase gaps as minor caveats; lower ranks, more blunt.
        label = "Worth probing" if rank <= 10 else "Concerns"
        extra_sentences.append(f"{label}: " + "; ".join(gaps[:2]) + ".")

    text = (opening + " " + " ".join(extra_sentences)).strip()

    # Keep CSV-friendly and within a reasonable length, cutting at a word
    # boundary so we never chop a token mid-word.
    max_len = 350
    if len(text) > max_len:
        truncated = text[:max_len].rsplit(" ", 1)[0]
        text = truncated.rstrip(",;") + "..."
    return text


def build_explanation_bundle(features: Dict[str, Any], scoring_result: Dict[str, Any], honeypot_result: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Richer explanation object for a future candidate-detail UI (Features #6/#8)."""
    rp = config.ROLE_PROFILE
    strengths = []
    for cat, score in _top_categories(features, rp["required_skill_categories"] + rp["preferred_skill_categories"]):
        names = _skill_names_for_category(features, cat)
        label = config.CATEGORY_LABELS[cat]
        strengths.append(f"{label}" + (f" ({', '.join(names)})" if names else ""))
    if features["ai_relevant_fraction"] >= 0.5:
        strengths.append(f"{features['ai_relevant_fraction']:.0%} of tracked career history in applied ML/AI roles")

    weaknesses = _gap_phrase(features)
    if not strengths:
        strengths = ["No strong matches against the JD's required skill categories"]

    missing_required = [
        config.CATEGORY_LABELS[c]
        for c in rp["required_skill_categories"]
        if features["skill_category_scores"].get(c, 0) < 0.01
    ]
    current_match_pct = round(scoring_result["sub_scores"]["skill_alignment"] * 100, 1)
    improvement_suggestions = []
    for c in rp["required_skill_categories"]:
        if features["skill_category_scores"].get(c, 0) < 0.01:
            improvement_suggestions.append(f"Gain hands-on experience with {config.CATEGORY_LABELS[c]}")
    for c in rp["preferred_skill_categories"]:
        if features["skill_category_scores"].get(c, 0) < 0.01:
            improvement_suggestions.append(f"Consider building exposure to {config.CATEGORY_LABELS[c]}")

    recruiter_summary = build_reasoning(features, honeypot_result, rank)

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "skill_gap": {
            "current_match_pct": current_match_pct,
            "missing_skills": missing_required,
            "improvement_suggestions": improvement_suggestions[:5],
        },
        "recruiter_summary": recruiter_summary,
    }
