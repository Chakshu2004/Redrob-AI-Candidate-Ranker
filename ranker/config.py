"""
config.py
=========
Static configuration for the Redrob "Senior AI Engineer - Founding Team" ranker.

This file encodes the *structured role profile* extracted from job_description.md
through careful reading (Required Feature #1 of the brief: "Intelligent Job
Understanding"). Because the JD is fixed for this challenge, the extraction is
done once, here, as an editable structured object -- not re-derived at runtime
via an LLM (which is disallowed during the ranking step anyway).

Everything that drives scoring lives in this file so weights / taxonomies can
be tuned without touching pipeline logic.
"""

from datetime import date

# ----------------------------------------------------------------------
# Reference "today" used for recency calculations (last_active_date, etc.)
# The dataset's redrob_signals dates are set in a near-future window
# (signup/last_active up to mid-2026), so we anchor to that.
# ----------------------------------------------------------------------
TODAY = date(2026, 6, 1)

# ----------------------------------------------------------------------
# Structured Role Profile (Required Feature #1)
# Extracted from job_description.md
# ----------------------------------------------------------------------
ROLE_PROFILE = {
    "role": "Senior AI Engineer - Founding Team",
    "company": "Redrob AI",
    "seniority": "Senior (5-9 years, flexible band)",
    "experience_band": {"min": 5, "max": 9, "soft_min": 3, "soft_max": 12},
    "required_skill_categories": [
        "embeddings_retrieval",
        "vector_db_hybrid_search",
        "python",
        "eval_frameworks",
    ],
    "preferred_skill_categories": [
        "llm_finetuning",
        "learning_to_rank",
        "hr_recruiting_tech",
        "distributed_systems_scale",
        "open_source",
    ],
    "behavioral_traits": [
        "Ownership / shipper mindset",
        "Async-first written communication",
        "Direct, low-ego disagreement",
        "Comfort with ambiguity / fast iteration",
        "Mentorship (growing team 4 -> 12)",
    ],
    "domain_expertise": ["NLP", "Information Retrieval", "Search / Ranking / Recommendation"],
    "industry_preferences": ["Product company (not pure IT services/consulting)"],
    "communication_requirements": [
        "Writes a lot (async-first culture)",
        "Can defend technical design choices",
    ],
    "leadership_expectations": [
        "Drives long-term architecture",
        "Mentors junior hires as team scales 4 -> 12",
    ],
    "location_preference": {
        "ideal": ["Pune", "Noida"],
        "acceptable_india": ["Hyderabad", "Mumbai", "Delhi", "Delhi NCR", "Gurugram", "Gurgaon", "Bengaluru", "Bangalore"],
        "country_required_unless_relocate": "India",
    },
    "notice_period_preference_days": 30,  # <=30 ideal, can buy out up to 30
}

# ----------------------------------------------------------------------
# Skill taxonomy: maps raw skill-name strings (as they appear in
# candidate.skills[].name) to one or more taxonomy categories.
# Matching is case-insensitive substring matching against this dict's keys.
# ----------------------------------------------------------------------
SKILL_TAXONOMY = {
    # --- embeddings / retrieval (REQUIRED) ---
    "sentence-transformers": ["embeddings_retrieval"],
    "sentence transformers": ["embeddings_retrieval"],
    "bge": ["embeddings_retrieval"],
    "e5": ["embeddings_retrieval"],
    "openai embeddings": ["embeddings_retrieval"],
    "embeddings": ["embeddings_retrieval"],
    "dense retrieval": ["embeddings_retrieval"],
    "semantic search": ["embeddings_retrieval"],
    "retrieval": ["embeddings_retrieval"],
    "rag": ["embeddings_retrieval", "llm_finetuning"],
    "information retrieval": ["embeddings_retrieval"],

    # --- vector db / hybrid search (REQUIRED) ---
    "pinecone": ["vector_db_hybrid_search"],
    "weaviate": ["vector_db_hybrid_search"],
    "qdrant": ["vector_db_hybrid_search"],
    "milvus": ["vector_db_hybrid_search"],
    "opensearch": ["vector_db_hybrid_search"],
    "elasticsearch": ["vector_db_hybrid_search"],
    "faiss": ["vector_db_hybrid_search"],
    "bm25": ["vector_db_hybrid_search"],
    "hybrid search": ["vector_db_hybrid_search"],
    "vector database": ["vector_db_hybrid_search"],
    "vector search": ["vector_db_hybrid_search"],

    # --- python (REQUIRED) ---
    "python": ["python"],

    # --- evaluation frameworks (REQUIRED) ---
    "ndcg": ["eval_frameworks"],
    "mrr": ["eval_frameworks"],
    "map": ["eval_frameworks"],
    "a/b testing": ["eval_frameworks"],
    "ab testing": ["eval_frameworks"],
    "offline evaluation": ["eval_frameworks"],
    "ranking evaluation": ["eval_frameworks"],
    "experimentation": ["eval_frameworks"],

    # --- LLM fine-tuning (PREFERRED) ---
    "lora": ["llm_finetuning"],
    "qlora": ["llm_finetuning"],
    "peft": ["llm_finetuning"],
    "fine-tuning llms": ["llm_finetuning"],
    "fine tuning": ["llm_finetuning"],
    "llm fine-tuning": ["llm_finetuning"],

    # --- learning to rank (PREFERRED) ---
    "xgboost": ["learning_to_rank"],
    "lambdamart": ["learning_to_rank"],
    "learning to rank": ["learning_to_rank"],
    "ltr": ["learning_to_rank"],
    "lightgbm": ["learning_to_rank"],

    # --- distributed systems / scale (PREFERRED) ---
    "spark": ["distributed_systems_scale"],
    "kafka": ["distributed_systems_scale"],
    "distributed systems": ["distributed_systems_scale"],
    "kubernetes": ["distributed_systems_scale"],
    "ray": ["distributed_systems_scale"],
    "airflow": ["distributed_systems_scale"],

    # --- general NLP/ML signal (used for domain_expertise / cv-speech screen) ---
    "nlp": ["nlp_domain"],
    "natural language processing": ["nlp_domain"],
    "transformers": ["nlp_domain"],
    "llm": ["nlp_domain"],
    "large language models": ["nlp_domain"],
    "text classification": ["nlp_domain"],
    "named entity recognition": ["nlp_domain"],
    "topic modeling": ["nlp_domain"],

    # --- computer vision / speech / robotics (used for negative screen) ---
    "computer vision": ["cv_speech_robotics"],
    "image classification": ["cv_speech_robotics"],
    "object detection": ["cv_speech_robotics"],
    "speech recognition": ["cv_speech_robotics"],
    "robotics": ["cv_speech_robotics"],
    "ros": ["cv_speech_robotics"],
    "slam": ["cv_speech_robotics"],

    # --- framework-enthusiast signal words (used for negative screen) ---
    "langchain": ["framework_enthusiast"],
    "llamaindex": ["framework_enthusiast"],

    # --- HR / recruiting / marketplace (PREFERRED industry) ---
    "ats": ["hr_recruiting_tech"],
    "applicant tracking": ["hr_recruiting_tech"],
    "recruiting": ["hr_recruiting_tech"],
    "talent acquisition": ["hr_recruiting_tech"],

    # --- open source signal ---
    "open source": ["open_source"],
    "open-source": ["open_source"],
}

# Category display names used in reasoning text
CATEGORY_LABELS = {
    "embeddings_retrieval": "embeddings/retrieval",
    "vector_db_hybrid_search": "vector DB / hybrid search",
    "python": "Python",
    "eval_frameworks": "ranking evaluation (NDCG/MRR/MAP)",
    "llm_finetuning": "LLM fine-tuning",
    "learning_to_rank": "learning-to-rank",
    "hr_recruiting_tech": "HR-tech / recruiting",
    "distributed_systems_scale": "distributed systems at scale",
    "open_source": "open-source contributions",
    "nlp_domain": "NLP",
    "cv_speech_robotics": "CV/speech/robotics",
    "framework_enthusiast": "framework tutorials",
}

# ----------------------------------------------------------------------
# Title-based heuristics
# ----------------------------------------------------------------------
RELEVANT_AI_TITLES = [
    "ai engineer", "ml engineer", "machine learning engineer", "applied scientist",
    "data scientist", "nlp engineer", "search engineer", "ranking engineer",
    "recommendation", "research engineer", "ai researcher", "ml researcher",
    "deep learning engineer", "mle", "applied ml",
]

PURE_RESEARCH_TITLES = [
    "research scientist", "postdoc", "phd researcher", "academic researcher",
    "research fellow", "professor", "lecturer",
]

ARCHITECT_TL_TITLES = [
    "architect", "tech lead", "technical lead", "engineering manager",
    "director of engineering", "vp engineering", "head of engineering",
]

CLEARLY_UNRELATED_TITLES = [
    "marketing manager", "hr manager", "content writer", "accountant",
    "graphic designer", "customer support", "operations manager",
    "business analyst", "mechanical engineer", "civil engineer",
    "project manager", "sales", "recruiter", "office manager",
]

# Companies considered "pure IT services / consulting"
CONSULTING_COMPANIES = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
    "ltimindtree", "mphasis", "l&t infotech", "ltts",
]

# ----------------------------------------------------------------------
# Hybrid scoring weights (must sum to 1.0)
# Rationale (documented further in README):
#  - semantic_similarity kept modest (15%) since the dataset deliberately
#    contains keyword-stuffed profiles that would dominate a pure
#    cosine-similarity ranking.
#  - skill_alignment (25%) rewards REQUIRED category coverage, weighted by
#    redrob skill_assessment_scores where available (so "expert, 0 months
#    used" honeypot-style claims don't get full credit).
#  - experience_relevance (25%) checks years-of-experience band fit AND
#    what fraction of career history is actually applied-ML/AI at product
#    companies (not just total tenure).
#  - domain_fit_screen (20%) implements the JD's explicit "what we do NOT
#    want" disqualifier/derisking logic (pure research, consulting-only,
#    framework-tutorial-only, CV/speech/robotics-only, title-chasing).
#  - behavioral_availability (15%) folds in the 23 redrob_signals as an
#    availability/engagement multiplier per the JD's final note: "a
#    perfect-on-paper candidate ... not actually available ... down-weight."
# ----------------------------------------------------------------------
WEIGHTS = {
    "semantic_similarity": 0.15,
    "skill_alignment": 0.25,
    "experience_relevance": 0.25,
    "domain_fit_screen": 0.20,
    "behavioral_availability": 0.15,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

# ----------------------------------------------------------------------
# Honeypot detection thresholds
# ----------------------------------------------------------------------
HONEYPOT_RULES = {
    # if >= this many skills are "expert" with <= this many months used -> flag
    "min_expert_zero_duration_count": 3,
    "expert_low_duration_months_threshold": 6,
    # career_history total months vs years_of_experience relative tolerance
    "experience_mismatch_tolerance": 0.35,
    # single role duration cannot exceed total claimed experience
    "single_role_overrun_months": 6,  # grace period
}

# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
TOP_N = 100
RANDOM_SEED = 1337
