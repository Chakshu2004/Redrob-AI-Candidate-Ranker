#!/usr/bin/env python3
"""
rank.py
=======
Single-command entry point for producing the Redrob hackathon submission CSV.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./data/sample_candidates.json --out ./submission_sample.csv --top-n 50

Pipeline (see ranker/ package for each stage):
    1. data_loader  -> stream-load candidate records
    2. features     -> Candidate Understanding Engine: structured features per candidate
    3. semantic      -> TF-IDF cosine similarity vs. the JD (local, no network)
    4. honeypot      -> internal-consistency / implausibility checks
    5. scoring       -> Hybrid Scoring Engine (5 weighted sub-scores)
    6. reasoning     -> Explainable AI layer: per-candidate reasoning string
    7. write CSV     -> top-N rows, format per submission_spec.md Section 2

Designed to satisfy the compute constraints in submission_spec.md Section 3:
CPU-only, no network calls, <= 5 minutes wall-clock and <= 16GB RAM for the
full 100,000-candidate pool.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

from ranker import config
from ranker.data_loader import iter_candidates
from ranker.features import build_features
from ranker.honeypot import assess_honeypot
from ranker.scoring import score_candidate
from ranker.semantic import compute_semantic_similarity
from ranker.reasoning import build_reasoning


def run(candidates_path: str, out_path: str, top_n: int = config.TOP_N, verbose: bool = True):
    t0 = time.time()

    # Check if candidates file exists
    candidates_file = Path(candidates_path)
    if not candidates_file.exists():
        print(f"ERROR: Candidates file not found: {candidates_path}", file=sys.stderr)
        print(f"       Available files in current directory:", file=sys.stderr)
        print(f"       - data/sample_candidates.json (50 candidates, for testing)", file=sys.stderr)
        print(f"       Please provide a valid path to candidates.jsonl, candidates.jsonl.gz, or sample_candidates.json", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"[1/5] Loading candidates from {candidates_path} ...")
    candidates = list(iter_candidates(candidates_path))
    n_total = len(candidates)
    if verbose:
        print(f"      loaded {n_total} candidates in {time.time() - t0:.1f}s")

    if verbose:
        print("[2/5] Extracting structured features (Candidate Understanding Engine) ...")
    t1 = time.time()
    feature_list = [build_features(c) for c in candidates]
    if verbose:
        print(f"      done in {time.time() - t1:.1f}s")

    if verbose:
        print("[3/5] Computing TF-IDF semantic similarity vs. JD (Semantic Retrieval Layer) ...")
    t1 = time.time()
    semantic_texts = [f["semantic_text"] for f in feature_list]
    semantic_sims = compute_semantic_similarity(semantic_texts)
    if verbose:
        print(f"      done in {time.time() - t1:.1f}s")

    if verbose:
        print("[4/5] Honeypot screening + hybrid scoring (Bias-Reduction-aware) ...")
    t1 = time.time()
    scored = []
    honeypot_flags = 0
    for cand, feats, sim in zip(candidates, feature_list, semantic_sims):
        hp = assess_honeypot(cand, feats)
        if hp["is_honeypot"]:
            honeypot_flags += 1
        sc = score_candidate(feats, sim, hp)
        scored.append((feats, sc, hp))
    if verbose:
        print(f"      done in {time.time() - t1:.1f}s ({honeypot_flags} candidates flagged as honeypots out of {n_total})")

    if verbose:
        print(f"[5/5] Ranking + generating explanations for top {top_n} ...")
    t1 = time.time()
    # Sort: highest score first; tie-break by candidate_id ascending (per spec
    # Section 3). Tie-breaking must be done on the *rounded* score, since
    # that's what's written to the CSV and what the validator compares.
    scored.sort(key=lambda x: (-round(x[1]["final_score"], 4), x[0]["candidate_id"]))

    top = scored[:top_n]
    honeypots_in_top = sum(1 for _, _, hp in top if hp["is_honeypot"])

    rows = []
    for rank, (feats, sc, hp) in enumerate(top, start=1):
        reasoning = build_reasoning(feats, hp, rank)
        rows.append({
            "candidate_id": feats["candidate_id"],
            "rank": rank,
            "score": round(sc["final_score"], 4),
            "reasoning": reasoning,
        })
    if verbose:
        print(f"      done in {time.time() - t1:.1f}s")
        print(f"      honeypots in top {top_n}: {honeypots_in_top} ({honeypots_in_top / top_n:.1%})")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    total_time = time.time() - t0
    if verbose:
        print(f"\nWrote {len(rows)} rows to {out_path}")
        print(f"Total runtime: {total_time:.1f}s")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Redrob hackathon candidate ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl / .jsonl.gz / sample_candidates.json")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=config.TOP_N, help="Number of ranked rows to output (default: 100)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run(args.candidates, args.out, top_n=args.top_n, verbose=not args.quiet)


if __name__ == "__main__":
    main()
