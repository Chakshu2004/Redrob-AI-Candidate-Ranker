# Redrob AI — Candidate Ranking System

A recruiter-style ranker for the **"Senior AI Engineer – Founding Team"** role
at Redrob AI, built for the *India Runs Data & AI Challenge*.

It ranks the candidate pool and outputs the top 100 with a per-candidate
`reasoning` string, fully **offline, CPU-only, no GPU, no LLM/network calls**
during ranking — per `submission_spec.md` Section 3.

## TL;DR — reproduce the submission

Create and activate a Python virtual environment first:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows cmd
.\.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate
```

Install dependencies and run the ranker:

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
```

On the bundled 50-candidate dev sample:

```bash
python rank.py --candidates data/sample_candidates.json --out submission_sample.csv --top-n 50
```

Runtime on the full 100K pool: **~65 seconds, ~3.3GB peak RAM** (CPU-only,
measured on the real `candidates.jsonl`). Comfortably inside the 5-minute /
16GB budget.

## Sandbox demo

If you want to run the Streamlit frontend, activate the same virtual environment and install the sandbox requirements:

```bash
# activate the same .venv created above
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows cmd
.\.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate

pip install -r sandbox/requirements.txt
streamlit run sandbox/app.py
```

Runs the *identical* pipeline (no LLM, no network) on the bundled sample or an uploaded JSON/JSONL file (≤200 candidates), with a ranked table and a per-candidate detail view (score breakdown, strengths/weaknesses, skill-gap analysis, honeypot flag).

---

## Architecture

```
redrob-ranker/
├── rank.py                  # entry point: candidates -> submission.csv
├── ranker/
│   ├── config.py            # structured Role Profile (Feature #1), skill
│   │                         # taxonomy, scoring weights, location/honeypot rules
│   ├── data_loader.py        # streaming .jsonl / .jsonl.gz / .json loader
│   ├── features.py            # Candidate Understanding Engine (Feature #2)
│   ├── semantic.py            # Semantic Retrieval Layer via local TF-IDF (Feature #3)
│   ├── honeypot.py             # Honeypot / implausible-profile detection
│   ├── scoring.py               # Hybrid Scoring Engine + Bias Reduction (Features #5, #7)
│   └── reasoning.py             # Explainable AI layer + Skill Gap Analysis (Features #6, #8)
├── sandbox/app.py             # Streamlit demo (Feature: UI dashboard, scaled down)
├── tests/test_sample.py       # smoke tests + bias/determinism checks
└── data/sample_candidates.json
```

### 1. Role Profile (Intelligent Job Understanding)

`ranker/config.ROLE_PROFILE` is a structured extraction of `job_description.md`:
required vs. preferred skill *categories* (not literal strings), the
experience band (5–9y core, 3–12y soft), behavioral traits, location
preferences, and notice-period preference. Because the JD is fixed for this
challenge and no LLM call is allowed at ranking time, this extraction is done
once as an editable config object rather than re-derived at runtime — but the
rest of the pipeline reads it generically, so a different JD just means
editing this one file.

### 2. Candidate Understanding Engine (`features.py`)

Converts each raw candidate record into structured features:
- **Skill-category scores** — raw `skills[]` entries are mapped to taxonomy
  categories (`embeddings_retrieval`, `vector_db_hybrid_search`, `python`,
  `eval_frameworks`, `llm_finetuning`, `learning_to_rank`,
  `distributed_systems_scale`, `hr_recruiting_tech`, `open_source`,
  `nlp_domain`, `cv_speech_robotics`, `framework_enthusiast`), blended with
  `proficiency`, `duration_months`, and `redrob_signals.skill_assessment_scores`
  so a freshly-claimed "expert" with 0 months behind it gets discounted.
- **Career-history shape** — fraction of tracked months in AI/ML-relevant
  roles, fraction at pure IT-services/consulting firms, "title-chaser" score
  (short non-current stints with escalating seniority titles), and whether
  the *current* role is an architecture/tech-lead-type title held ≥18 months
  (a proxy for "hasn't shipped code recently").
- **Disqualifier flags**, mirroring the JD's explicit "what we do NOT want"
  section: pure-research-only, consulting-only, framework-tutorial-only
  (LangChain/LlamaIndex without real retrieval/eval depth + <4y experience),
  CV/speech/robotics-only, long-tenure-with-zero-external-validation.
- **Location/relocation fit** vs. Pune/Noida (ideal) → other India metros →
  rest of India → outside India (only credited if `willing_to_relocate`).

### 3. Semantic Retrieval Layer (`semantic.py`)

A **local TF-IDF vectorizer + cosine similarity** between each candidate's
concatenated profile text and a JD text built from the structured role
profile. This deliberately replaces sentence-transformers/BGE/FAISS because:
- the ranking step must run with **no network access**, so any model that
  needs to be downloaded at runtime is disqualifying;
- TF-IDF on a ~30K-feature vocabulary over 100K candidates runs in ~25s and
  fits easily in memory;
- it is given a **modest weight (15%)** because the dataset deliberately
  contains keyword-stuffed "trap" profiles that would dominate a pure
  cosine-similarity ranking — structured signals from `features.py` carry
  most of the weight instead.

### 4. Honeypot detection (`honeypot.py`)

Flags internally-inconsistent profiles using only within-profile checks (no
external company data needed):
- ≥3 skills marked `"expert"` with ≤6 months of `duration_months`
  ("expert proficiency in 10 skills with 0 years used");
- `years_of_experience` vs. the sum of `career_history.duration_months`
  disagreeing by >35%;
- a single role's duration exceeding the candidate's total claimed experience
  ("8 years of experience at a company founded 3 years ago"-style claims —
  detected as an arithmetic impossibility rather than via external data);
- impossible education date ranges;
- overlapping full-time roles at different companies for >6 months.

A candidate is flagged if it trips ≥2 checks, or one *severe* check (single-
role overrun / impossible education dates). Flagged candidates have their
final score multiplied by 0.05, pushing them out of the top 100.

### 5. Hybrid Scoring Engine (`scoring.py`)

```
final_score = 0.15 * semantic_similarity
             + 0.25 * skill_alignment            (required > preferred, 70/30)
             + 0.25 * experience_relevance        (band fit + AI-role fraction + title)
             + 0.20 * domain_fit_screen           (disqualifier penalties)
             + 0.15 * behavioral_availability     (redrob_signals)
```

`behavioral_availability` folds in the 23 `redrob_signals` fields
(open-to-work flag, notice period, last-active recency, recruiter response
rate, interview-completion rate, offer-acceptance rate, location/relocation)
per the JD's closing note about not over-weighting a "perfect-on-paper"
candidate who is unreachable.

**Bias reduction**: none of the scoring code reads `name`, `gender`, `age`,
`ethnicity`, or any other identity field — only skills, career-history shape,
and platform-engagement signals. `scoring.BIAS_NOTICE` ("Ranking generated
using skill and experience signals only.") is surfaced in the sandbox UI.

### 6. Explainable AI / Skill Gap (`reasoning.py`)

For each top-100 candidate, `build_reasoning()` produces a 1–2 sentence,
**fact-grounded** explanation built only from that candidate's own feature
values — years of experience, current title/company, matched skill
categories *with the actual skill names*, location, notice period,
recruiter-response rate, and honest gaps (missing required skill categories,
long notice period, consulting-heavy background, title-chasing pattern,
non-India location without relocation flag). Sentence structure is varied via
a deterministic hash of `candidate_id` so reruns are reproducible but
phrasing isn't identical across rows. Honeypot-flagged candidates get an
explicit "flagged: ..." reasoning naming the failed check.

`build_explanation_bundle()` additionally produces `strengths`,
`weaknesses`, and a `skill_gap` object (`current_match_pct`,
`missing_skills`, `improvement_suggestions`) for the sandbox detail view.

---

## Tuning

All weights, skill-taxonomy mappings, location lists, and honeypot
thresholds live in `ranker/config.py`. To adapt this to a different JD:
1. Edit `ROLE_PROFILE` (required/preferred skill categories, experience band,
   location preferences, behavioral traits).
2. Extend `SKILL_TAXONOMY` / `CATEGORY_LABELS` for any new skill categories.
3. Adjust `WEIGHTS` if a different balance between semantic similarity,
   skills, experience, domain-fit, and availability is desired (must sum to 1.0).

## Known limitations / next steps

- `_career_history_analysis`'s title-chaser and architect-no-code heuristics
  are keyword-based; a learned model over career-history sequences would be
  more robust.
- TF-IDF semantic similarity has no notion of synonymy beyond the explicit
  `SKILL_TAXONOMY` mapping (e.g. it won't know "BGE" and "E5" are both
  embedding models unless they're in the taxonomy — they are, but new
  unlisted tools won't be recognized).
- Honeypot detection is internal-consistency-only; it cannot catch a
  profile that is self-consistent but references a real company whose
  founding date contradicts the claimed tenure.
