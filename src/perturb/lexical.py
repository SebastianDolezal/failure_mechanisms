"""Conservative lexical substitution (Section 7.C).

Uses a small, hand-curated dictionary of mechanically safe, meaning-
preserving synonym swaps (quantity/unit phrasing only). Deliberately avoids
unrestricted LLM paraphrasing in the confirmatory dataset, per spec.
"""
from __future__ import annotations

import re

# (pattern, replacement) pairs; applied with word boundaries, case-preserving
# on the first letter. Kept intentionally small and unambiguous.
SAFE_SYNONYMS = [
    (r"\bhow many\b", "what is the number of"),
    (r"\bwhat is the number of\b", "how many"),
    (r"\bin total\b", "altogether"),
    (r"\baltogether\b", "in total"),
    (r"\bleft over\b", "remaining"),
    (r"\bremaining\b", "left over"),
    (r"\btwice as many\b", "two times as many"),
    (r"\btwo times as many\b", "twice as many"),
    (r"\bhalf as many\b", "one-half as many"),
    (r"\bone-half as many\b", "half as many"),
    (r"\beach\b", "every"),
    (r"\bevery\b", "each"),
    (r"\bcombined\b", "in total"),
    (r"\bper\b", "for each"),
]


def _match_case(src_word: str, repl: str) -> str:
    if src_word[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def apply_lexical_conservative(text: str, rng, tokenizer=None) -> tuple[str, dict]:
    applicable = []
    for pattern, repl in SAFE_SYNONYMS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            applicable.append((pattern, repl))
    if not applicable:
        return text, {"applied": False, "n_changes": 0}

    pattern, repl = applicable[rng.randrange(len(applicable))]

    def _sub(m):
        return _match_case(m.group(0), repl)

    new_text, k = re.subn(pattern, _sub, text, count=1, flags=re.IGNORECASE)
    return new_text, {"applied": k > 0, "n_changes": k, "rule": pattern}
