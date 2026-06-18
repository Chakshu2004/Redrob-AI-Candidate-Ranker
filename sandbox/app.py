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

# --- Summary metrics dashboard
st.subheader("📊 Dashboard")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("📋 Evaluated", n_total)
col_b.metric("🏆 Top candidates", top_count)
col_c.metric("⚠️ Honeypots", honeypot_count)
avg_score = round(pd.DataFrame(rows)["score"].mean(), 3)
col_d.metric("📈 Avg score", f"{avg_score:.3f}")

# --- Interactive controls: search / score filter / sort
with st.expander("🔧 Controls", expanded=False):
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

# --- Tabbed interface
tab_leaderboard, tab_stats, tab_compare = st.tabs(["🏅 Leaderboard", "📊 Statistics", "⚖️ Compare"])

with tab_leaderboard:
    st.dataframe(leaderboard_df[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
    csv_bytes = format_download(rows)
    st.download_button(
        label="📥 Download ranked CSV",
        data=csv_bytes,
        file_name="ranked_candidates.csv",
        mime="text/csv",
    )

with tab_stats:
    stats_col1, stats_col2 = st.columns(2)
    
    with stats_col1:
        st.subheader("Score Distribution")
        score_data = pd.DataFrame(rows)["score"]
        st.bar_chart(score_data.value_counts(bins=10).sort_index(), use_container_width=True)
        
        st.subheader("Experience Distribution")
        exp_data = pd.DataFrame(rows)["years_exp"].value_counts().sort_index()
        st.bar_chart(exp_data, use_container_width=True)
    
    with stats_col2:
        st.subheader("Top Locations")
        loc_data = pd.DataFrame(rows)["location"].value_counts().head(10)
        st.bar_chart(loc_data, use_container_width=True)
        
        st.subheader("Quick Stats")
        stats_df = pd.DataFrame(rows)
        st.write(f"- **Min score:** {stats_df['score'].min():.3f}")
        st.write(f"- **Max score:** {stats_df['score'].max():.3f}")
        st.write(f"- **Median score:** {stats_df['score'].median():.3f}")
        st.write(f"- **Std dev:** {stats_df['score'].std():.3f}")
        st.write(f"- **Avg experience:** {stats_df['years_exp'].mean():.1f} years")

with tab_compare:
    st.subheader("Compare Candidates")
    compare_df = leaderboard_df[["badge", "candidate_id", "score", "title", "years_exp", "location"]]
    selected_candidates = st.multiselect(
        "Select up to 3 candidates to compare",
        compare_df["candidate_id"].tolist(),
        max_selections=3,
    )
    
    if selected_candidates:
        comparison_rows = []
        for cid in selected_candidates:
            row_data = compare_df[compare_df["candidate_id"] == cid].iloc[0]
            comparison_rows.append(row_data)
        
        compare_table = pd.DataFrame(comparison_rows)
        st.dataframe(compare_table, use_container_width=True, hide_index=True)
        
        # Side-by-side score bars
        st.subheader("Score Comparison")
        for cid in selected_candidates:
            row = compare_df[compare_df["candidate_id"] == cid].iloc[0]
            score_val = row["score"]
            title_val = row["title"]
            st.progress(score_val, text=f"{cid} — {title_val} — {score_val:.3f}")
    else:
        st.info("👆 Select 1-3 candidates above to compare their profiles side by side.")

st.markdown("---")

st.subheader("🔍 Candidate Deep Dive")
selected_id = st.selectbox("👤 Select a candidate to inspect", leaderboard_df["candidate_id"].tolist())
selected_idx = leaderboard_df.index[leaderboard_df["candidate_id"] == selected_id][0]
feats, sc, hp, explanation = full_results[selected_idx]

# Create a styled header card
score_val = sc['final_score']
score_pct = f"{score_val:.1%}"
score_color = "🟢" if score_val >= 0.7 else "🟡" if score_val >= 0.5 else "🔴"

col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
with col_header1:
    st.markdown(f"### {feats['current_title']} @ {feats['current_company']}")
    st.write(f"📍 {feats['location']} | 📅 {feats['years_of_experience']:.1f} yrs exp | {feats['candidate_id']}")
with col_header2:
    st.metric("Match Score", score_pct)
with col_header3:
    if hp["is_honeypot"]:
        st.error("⚠️ Honeypot")
    else:
        st.success("✅ Valid")

# Two-column detail layout
score_col, detail_col = st.columns([1, 2])
with score_col:
    st.write("**Score Breakdown**")
    for k, v in sc["sub_scores"].items():
        color_bar = "🟢" if v >= 0.7 else "🟡" if v >= 0.5 else "🔴"
        st.progress(v, text=f"{color_bar} {k}: {v:.2f}")

with detail_col:
    st.write("**📋 Recruiter Summary**")
    st.write(explanation["recruiter_summary"])
    
    st.write("**✨ Strengths**")
    for s in explanation["strengths"]:
        st.write(f"✅ {s}")
    
    st.write("**⚠️ Weaknesses / Gaps**")
    if explanation["weaknesses"]:
        for w in explanation["weaknesses"]:
            st.write(f"❌ {w}")
    else:
        st.write("✅ None identified")

# Skill gap section with expandable detail
with st.expander("🎯 Skill Gap Analysis", expanded=True):
    gap = explanation["skill_gap"]
    gap_cols1, gap_cols2 = st.columns(2)
    
    with gap_cols1:
        st.metric("Current Match", f"{gap['current_match_pct']}%")
        if gap["missing_skills"]:
            st.write("**Missing required skills:**")
            for missing in gap["missing_skills"]:
                st.write(f"- 🔴 {missing}")
    
    with gap_cols2:
        if gap["improvement_suggestions"]:
            st.write("**Recommendations:**")
            for i, suggestion in enumerate(gap["improvement_suggestions"], 1):
                st.write(f"{i}. 💡 {suggestion}")

if hp["is_honeypot"]:
    st.error(f"🚨 **Flagged Profile:** {'; '.join(hp['reasons'])}")

with st.expander("📄 Raw Artifacts (Debug)", expanded=False):
    st.write("**Reasoning:**")
    st.code(build_reasoning(feats, hp, selected_idx + 1))
    st.write("**Full Explanation Bundle:**")
    st.json(explanation)
