"""
honeypot.py
===========
Detects "honeypot" candidates: profiles with subtly impossible claims
(per redrob_signals_doc / README: "8 years of experience at a company
founded 3 years ago; 'expert' proficiency in 10 skills with 0 years used").

We don't have external company-founding-date data, so detection is based on
internal-consistency checks within each candidate's own profile:

  1. Skill-claim inflation: several skills marked "expert" with near-zero
     duration_months -- the textbook example from the README.
  2. Experience-arithmetic mismatch: years_of_experience vs. the sum of
     career_history duration_months disagree by more than a tolerance.
  3. Single-role overrun: one role's duration_months alone exceeds the
     candidate's total claimed years_of_experience.
  4. Education timeline impossibilities: end_year < start_year, or
     end_year far beyond "today".
  5. Overlapping concurrent roles: two or more non-current roles with
     overlapping date ranges that aren't flagged as concurrent/part-time.

A candidate is marked a honeypot if it trips >= 2 independent checks, or
1 severe check (single-role overrun or a hard education-date impossibility).
This is intentionally conservative -- the goal is to keep the ~80 true
honeypots out of the top 100, not to penalize every minor data quirk.
"""

from datetime import date
from typing import Any, Dict, List, Tuple

from . import config


def _date_or_none(s):
    if not s:
        return None
    try:
        y, m, d = (int(x) for x in s.split("-"))
        return date(y, m, d)
    except Exception:
        return None


def _check_expert_inflation(features: Dict[str, Any]) -> Tuple[bool, str]:
    rules = config.HONEYPOT_RULES
    count = 0
    for skill in features["_skills"]:
        if skill.get("proficiency") == "expert" and (skill.get("duration_months", 0) or 0) <= rules["expert_low_duration_months_threshold"]:
            count += 1
    if count >= rules["min_expert_zero_duration_count"]:
        return True, f"{count} skills claimed as 'expert' with <= {rules['expert_low_duration_months_threshold']} months of use"
    return False, ""


def _check_experience_arithmetic(features: Dict[str, Any]) -> Tuple[bool, str]:
    rules = config.HONEYPOT_RULES
    years = features["years_of_experience"]
    tracked_years = features["total_tracked_months"] / 12.0
    if years <= 0:
        return False, ""
    rel_diff = abs(tracked_years - years) / years
    if rel_diff > rules["experience_mismatch_tolerance"]:
        return True, (
            f"profile claims {years:.1f} years of experience but career_history "
            f"sums to {tracked_years:.1f} years (mismatch {rel_diff:.0%})"
        )
    return False, ""


def _check_single_role_overrun(features: Dict[str, Any]) -> Tuple[bool, str]:
    rules = config.HONEYPOT_RULES
    years = features["years_of_experience"]
    max_allowed_months = years * 12 + rules["single_role_overrun_months"]
    for role in features["_career_history"]:
        months = role.get("duration_months", 0) or 0
        if months > max_allowed_months and years > 0:
            return True, (
                f"a single role ('{role.get('title')}' at '{role.get('company')}') "
                f"lasted {months} months, exceeding the candidate's total claimed "
                f"experience of {years:.1f} years"
            )
    return False, ""


def _check_education_dates(features: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[bool, str]:
    for edu in candidate.get("education", []) or []:
        start = edu.get("start_year")
        end = edu.get("end_year")
        if start and end:
            if end < start:
                return True, f"education end_year ({end}) is before start_year ({start}) at {edu.get('institution')}"
            if end > config.TODAY.year + 1:
                return True, f"education end_year ({end}) at {edu.get('institution')} is implausibly in the future"
    return False, ""


def _check_overlapping_roles(features: Dict[str, Any]) -> Tuple[bool, str]:
    ranges = []
    for role in features["_career_history"]:
        s = _date_or_none(role.get("start_date"))
        e = _date_or_none(role.get("end_date")) or config.TODAY
        if s:
            ranges.append((s, e, role.get("company"), role.get("is_current")))

    ranges.sort()
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            s1, e1, c1, cur1 = ranges[i]
            s2, e2, c2, cur2 = ranges[j]
            # overlap?
            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)
            overlap_months = (overlap_end.year - overlap_start.year) * 12 + (overlap_end.month - overlap_start.month)
            if overlap_months > 6 and c1 != c2:
                return True, f"overlapping full-time roles at '{c1}' and '{c2}' for {overlap_months} months"
    return False, ""


def assess_honeypot(candidate: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    """Return {'is_honeypot': bool, 'hit_count': int, 'reasons': [str], 'severe': bool}."""
    checks = [
        _check_expert_inflation(features),
        _check_experience_arithmetic(features),
        _check_single_role_overrun(features),
        _check_education_dates(features, candidate),
        _check_overlapping_roles(features),
    ]
    severe_idx = {2, 3}  # single-role overrun, education-date impossibility

    reasons = []
    hit_count = 0
    severe = False
    for idx, (hit, reason) in enumerate(checks):
        if hit:
            hit_count += 1
            reasons.append(reason)
            if idx in severe_idx:
                severe = True

    is_honeypot = severe or hit_count >= 2
    return {"is_honeypot": is_honeypot, "hit_count": hit_count, "reasons": reasons, "severe": severe}
