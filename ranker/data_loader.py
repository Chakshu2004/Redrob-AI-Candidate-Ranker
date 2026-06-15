"""
data_loader.py
===============
Streaming loaders for the candidate pool.

Supports:
  - candidates.jsonl            (one JSON object per line, 100K rows)
  - candidates.jsonl.gz         (gzipped variant)
  - sample_candidates.json      (pretty-printed JSON array, for dev/testing)

Designed to iterate lazily so the full 100K-candidate pool never needs to be
materialized as parsed Python objects all at once if memory is tight --
though in practice 100K small dicts comfortably fit in 16GB.
"""

import gzip
import json
from pathlib import Path
from typing import Iterator, Dict, Any


def iter_candidates(path: str) -> Iterator[Dict[str, Any]]:
    """Yield candidate dicts one at a time from a .jsonl, .jsonl.gz, or .json file."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".gz":
        opener = lambda: gzip.open(p, "rt", encoding="utf-8")
    else:
        opener = lambda: open(p, "r", encoding="utf-8")

    if suffix == ".json":
        # Pretty-printed JSON array (e.g. sample_candidates.json)
        with opener() as f:
            data = json.load(f)
        for rec in data:
            yield rec
        return

    # .jsonl or .jsonl.gz -> one JSON object per non-empty line
    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_candidates(path: str):
    """Materialize all candidates into a list. Fine for up to ~100K records."""
    return list(iter_candidates(path))
