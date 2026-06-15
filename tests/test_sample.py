"""
tests/test_sample.py
=====================
Smoke tests for the ranker pipeline against the bundled 50-candidate sample.

Run with:
    python -m pytest tests/ -q
or simply:
    python tests/test_sample.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ranker.data_loader import iter_candidates
from ranker.features import build_features
from ranker.honeypot import assess_honeypot
from ranker.scoring import score_candidate
from ranker.semantic import compute_semantic_similarity
from ranker.reasoning import build_reasoning

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_candidates.json"


def test_pipeline_runs_end_to_end():
    candidates = list(iter_candidates(str(SAMPLE)))
    assert len(candidates) == 50

    features = [build_features(c) for c in candidates]
    sims = compute_semantic_similarity([f["semantic_text"] for f in features])
    assert len(sims) == len(candidates)
    assert all(0.0 <= s <= 1.0 for s in sims)

    for feats, sim, cand in zip(features, sims, candidates):
        hp = assess_honeypot(cand, feats)
        sc = score_candidate(feats, sim, hp)
        assert 0.0 <= sc["final_score"] <= 1.0
        reasoning = build_reasoning(feats, hp, rank=1)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert reasoning == reasoning.replace("\n", " ")  # CSV-safe, single line


def test_no_sensitive_fields_in_semantic_text():
    """Bias-reduction sanity check: name/gender/etc. must never enter scoring text."""
    candidates = list(iter_candidates(str(SAMPLE)))
    features = [build_features(c) for c in candidates]
    for feats, cand in zip(features, candidates):
        profile = cand.get("profile", {})
        for field in ("name", "gender", "age", "ethnicity", "date_of_birth"):
            val = profile.get(field)
            if val:
                assert str(val) not in feats["semantic_text"]


def test_ranking_is_deterministic():
    candidates = list(iter_candidates(str(SAMPLE)))
    features = [build_features(c) for c in candidates]
    sims = compute_semantic_similarity([f["semantic_text"] for f in features])

    def rank_once():
        scored = []
        for cand, feats, sim in zip(candidates, features, sims):
            hp = assess_honeypot(cand, feats)
            sc = score_candidate(feats, sim, hp)
            scored.append((feats["candidate_id"], sc["final_score"]))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [c for c, _ in scored]

    assert rank_once() == rank_once()


if __name__ == "__main__":
    test_pipeline_runs_end_to_end()
    test_no_sensitive_fields_in_semantic_text()
    test_ranking_is_deterministic()
    print("All smoke tests passed.")
