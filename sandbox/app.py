"""
sandbox/app.py
===============
Lightweight Streamlit sandbox demo for the Redrob ranker.

It runs the same pipeline as rank.py (ranker.features, ranker.semantic,
ranker.honeypot, ranker.scoring, ranker.reasoning) with zero network calls,
CPU-only inference, and a small uploaded sample or bundled candidate file.

Run locally with:
    streamlit run sandbox/app.py
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the `ranker` package importable when run as `streamlit run sandbox/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ranker import config
from ranker.data_loader import iter_candidates
from ranker.features import build_features
from ranker.honeypot import assess_honeypot
from ranker.scoring import score_candidate, BIAS_NOTICE
from ranker.semantic import compute_semantic_similarity
from ranker.reasoning import build_reasoning, build_explanation_bundle

MAX_CANDIDATES = 200
DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_candidates.json"

st.set_page_config(page_title="Redrob AI Candidate Ranker - Sandbox", layout="wide")

st.title("🎯 Redrob AI Candidate Ranker — Sandbox")
st.caption(
    "Senior AI Engineer (Founding Team) role demo — local TF-IDF + hybrid scoring, "
    "no LLM calls / no network during ranking."
)

st.info(f"🛡️ {BIAS_NOTICE}")

with st.sidebar:
    st.header("Candidate pool")
    uploaded = st.file_uploader(
        "Upload a candidate file",
        type=["json", "jsonl"],
        help="Upload a JSON array (.json) or JSON Lines file (.jsonl) with candidate records.",
    )
    top_n = st.slider("Show top N ranked candidates", 5, 100, 20)
    st.markdown("---")
    st.markdown("**Scoring weights**")
    for k, v in config.WEIGHTS.items():
        st.write(f"- {k}: {v:.0%}")
    st.markdown("---")
    st.write(
        "Built for demo and review: upload a file with up to 200 candidates or use the "
        "bundled sample data. The leaderboard updates automatically."
    )


@st.cache_data(show_spinner=False)
def parse_candidates(file_bytes, filename):
    if file_bytes is None:
        return list(iter_candidates(str(DEFAULT_SAMPLE_PATH)))

    suffix = Path(filename).suffix.lower()
    text = file_bytes.decode("utf-8")

    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            raise ValueError("Uploaded JSON must be an array of candidate objects.")
    elif suffix == ".jsonl":
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raise ValueError("Only .json and .jsonl uploads are supported.")

    if not isinstance(data, list):
        raise ValueError("Uploaded file must contain a list of candidate records.")
    if not data:
        raise ValueError("Uploaded file contains no candidate records.")
    if len(data) > MAX_CANDIDATES:
        raise ValueError(
            f"Uploaded file contains {len(data)} candidates, but the demo supports up to {MAX_CANDIDATES}."
        )
    return data


@st.cache_data(show_spinner=False)
def load_and_rank(candidates, top_n):
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(f"Candidate count exceeds the demo maximum of {MAX_CANDIDATES}.")

    feature_list = [build_features(c) for c in candidates]
    sims = compute_semantic_similarity([f["semantic_text"] for f in feature_list])

    results = []
    for cand, feats, sim in zip(candidates, feature_list, sims):
        hp = assess_honeypot(cand, feats)
        sc = score_candidate(feats, sim, hp)
        explanation = build_explanation_bundle(feats, sc, hp, rank=0)
        results.append((feats, sc, hp, explanation))

    results.sort(key=lambda x: (-x[1]["final_score"], x[0]["candidate_id"]))

    rows = []
    for rank, (feats, sc, hp, explanation) in enumerate(results[:top_n], start=1):
        rows.append({
            "rank": rank,
            "candidate_id": feats["candidate_id"],
            "score": round(sc["final_score"], 4),
            "title": feats["current_title"],
            "years_exp": feats["years_of_experience"],
            "location": feats["location"],
            "honeypot": "⚠️" if hp["is_honeypot"] else "",
            "reasoning": build_reasoning(feats, hp, rank),
        })
    return rows, results[:top_n], len(candidates), sum(1 for _, _, hp, _ in results if hp["is_honeypot"])


def format_download(rows):
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


file_bytes = uploaded.read() if uploaded else None
filename = uploaded.name if uploaded else DEFAULT_SAMPLE_PATH.name

try:
    candidates = parse_candidates(file_bytes, filename)
    rows, full_results, n_total, honeypot_count = load_and_rank(candidates, top_n)
except ValueError as exc:
    st.error(str(exc))
    st.stop()


top_count = len(rows)

st.subheader(f"Leaderboard — top {top_count} of {n_total} candidates")
col_a, col_b, col_c = st.columns([1, 1, 1])
col_a.metric("Candidates evaluated", n_total)
col_b.metric("Top candidates shown", top_count)
col_c.metric("Honeypots flagged", honeypot_count)

# --- Interactive controls: search / score filter / sort
with st.expander("Controls", expanded=False):
    search_text = st.text_input("Search by candidate id or title", value="")
    min_score = st.slider("Minimum score to include", 0.0, 1.0, 0.0, 0.01)
    sort_by = st.selectbox("Sort results by", options=["rank", "score", "years_exp"], index=0)
    show_reasoning_col = st.checkbox("Show reasoning column in table", value=False)

leaderboard_df = pd.DataFrame(rows)[["rank", "candidate_id", "score", "title", "years_exp", "location", "honeypot", "reasoning"]]

# Apply search and score filters
if search_text:
    mask = (
        leaderboard_df["candidate_id"].str.contains(search_text, case=False, na=False)
        | leaderboard_df["title"].fillna("").str.contains(search_text, case=False, na=False)
    )
    leaderboard_df = leaderboard_df[mask]
leaderboard_df = leaderboard_df[leaderboard_df["score"] >= min_score]

# Add simple badges for top 3
def _badge(rank):
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")

leaderboard_df["badge"] = leaderboard_df["rank"].apply(_badge)

if sort_by == "score":
    leaderboard_df = leaderboard_df.sort_values(by=["score", "rank"], ascending=[False, True])
elif sort_by == "years_exp":
    leaderboard_df = leaderboard_df.sort_values(by=["years_exp", "rank"], ascending=[False, True])
else:
    leaderboard_df = leaderboard_df.sort_values(by=["rank"]) 

# Choose displayed columns
display_cols = ["badge", "rank", "candidate_id", "score", "title", "years_exp", "location", "honeypot"]
if show_reasoning_col:
    display_cols.append("reasoning")

st.dataframe(leaderboard_df[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

csv_bytes = format_download(rows)
st.download_button(
    label="Download ranked CSV",
    data=csv_bytes,
    file_name="ranked_candidates.csv",
    mime="text/csv",
)

st.markdown("---")

st.subheader("Candidate detail")
selected_id = st.selectbox("Select a candidate to inspect", leaderboard_df["candidate_id"].tolist())
selected_idx = leaderboard_df.index[leaderboard_df["candidate_id"] == selected_id][0]
feats, sc, hp, explanation = full_results[selected_idx]

score_col, detail_col = st.columns([1, 2])
with score_col:
    st.metric("Overall match score", f"{sc['final_score']:.2%}")
    st.write("**Score breakdown**")
    for k, v in sc["sub_scores"].items():
        st.progress(v, text=f"{k}: {v:.2f} (weight {config.WEIGHTS[k]:.0%})")

with detail_col:
    st.write("**Recruiter summary**")
    st.write(explanation["recruiter_summary"])
    st.write("**Strengths**")
    for s in explanation["strengths"]:
        st.write(f"- {s}")
    st.write("**Weaknesses / gaps**")
    if explanation["weaknesses"]:
        for w in explanation["weaknesses"]:
            st.write(f"- {w}")
    else:
        st.write("- None identified")

with st.expander("Skill gap analysis", expanded=True):
    gap = explanation["skill_gap"]
    st.write(f"Current match: {gap['current_match_pct']} %")
    if gap["missing_skills"]:
        st.write("Missing required skill categories:")
        for missing in gap["missing_skills"]:
            st.write(f"- {missing}")
    if gap["improvement_suggestions"]:
        st.write("Improvement suggestions:")
        for suggestion in gap["improvement_suggestions"]:
            st.write(f"- {suggestion}")

if hp["is_honeypot"]:
    st.error("⚠️ Flagged as a honeypot / internally-inconsistent profile: " + "; ".join(hp["reasons"]))

with st.expander("Raw candidate reasoning and explanation bundle", expanded=False):
    st.write("**Reasoning:**")
    st.write(build_reasoning(feats, hp, selected_idx + 1))
    st.write("**Explanation bundle:**")
    st.json(explanation)
