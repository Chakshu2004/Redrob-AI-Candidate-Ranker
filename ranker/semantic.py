"""
semantic.py
===========
Semantic Retrieval Layer (Required Feature #3), implemented with a local
TF-IDF vectorizer + cosine similarity rather than a downloaded sentence-
embedding model.

Why TF-IDF instead of sentence-transformers/BGE/FAISS here:
  - The ranking step must run with NO network access (submission_spec
    Section 3), so any model that needs to be downloaded at runtime is a
    non-starter unless it's vendored into the repo.
  - TF-IDF over candidate semantic_text vs. the JD text is fast and scales
    cleanly to 100K candidates within the 5-minute / 16GB CPU budget
    (fit + transform + cosine similarity on a 100k x ~20k sparse matrix
    takes seconds).
  - It is intentionally given a MODEST weight (see config.WEIGHTS) because
    the dataset deliberately contains keyword-stuffed "trap" profiles that
    would otherwise dominate a pure cosine-similarity ranking. The
    structured `features.py` signals carry most of the weight.

If a vendored local embedding model is later added to the repo (so it loads
from disk with no network call), it can be dropped in here as an additional
similarity column without changing the scoring engine's interface --
just add another key to the returned dict and a corresponding weight.
"""

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from . import config


def build_jd_text() -> str:
    """Construct a representative text blob for the JD from the structured role profile."""
    rp = config.ROLE_PROFILE
    pieces = [
        rp["role"],
        rp["company"],
        "Required skills: " + ", ".join(config.CATEGORY_LABELS[c] for c in rp["required_skill_categories"]),
        "Preferred skills: " + ", ".join(config.CATEGORY_LABELS[c] for c in rp["preferred_skill_categories"]),
        "Behavioral traits: " + ", ".join(rp["behavioral_traits"]),
        "Domain expertise: " + ", ".join(rp["domain_expertise"]),
        "Industry preference: " + ", ".join(rp["industry_preferences"]),
        "Communication: " + ", ".join(rp["communication_requirements"]),
        "Leadership: " + ", ".join(rp["leadership_expectations"]),
        "Seniority: " + rp["seniority"],
        (
            "own the intelligence layer ranking retrieval matching systems "
            "embeddings hybrid retrieval LLM re-ranking evaluation NDCG MRR MAP "
            "production deployment scaled search recommendation system applied "
            "machine learning product company shipping working ranker"
        ),
    ]
    return " ".join(pieces)


def compute_semantic_similarity(candidate_texts: List[str]) -> List[float]:
    """
    Fit a TF-IDF vectorizer over [JD_text] + candidate_texts and return the
    cosine similarity of each candidate text to the JD text, scaled to 0-1.
    """
    jd_text = build_jd_text()
    corpus = [jd_text] + list(candidate_texts)

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    tfidf = vectorizer.fit_transform(corpus)

    jd_vec = tfidf[0:1]
    cand_vecs = tfidf[1:]

    sims = linear_kernel(jd_vec, cand_vecs).flatten()

    # Cosine similarity from TF-IDF is already in [0, 1] for non-negative
    # weights, but clip defensively.
    return [float(max(0.0, min(1.0, s))) for s in sims]
