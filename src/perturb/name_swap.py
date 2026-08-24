"""Name / entity substitution perturbation (Section 7.A).

Conservative and mechanical: known proper names in the problem text are
replaced by a name drawn from a fixed pool, holding the mathematical content
fixed. When a tokenizer is supplied, candidate replacement names are ranked
by how closely they preserve the clean problem's token count so that
clean/fail realizations stay token-aligned (Section 25).
"""
from __future__ import annotations

import re

# A fixed, deliberately small pool so re-runs are deterministic given a seed.
NAME_POOL = [
    "Maria", "James", "Linda", "Carlos", "Priya", "Kevin", "Aisha", "Tom",
    "Elena", "Marcus", "Sofia", "David", "Nina", "Omar", "Grace", "Leo",
    "Hannah", "Victor", "Wendy", "Ravi", "Chloe", "Adam", "Fiona", "Ben",
]

# GSM8K names are proper nouns capitalized mid-sentence; detect capitalized
# word tokens that are not the first word of a sentence and not a common
# capitalized non-name (Month names, weekday names, unit words) to avoid
# false positives.
_STOPWORDS = {
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "I", "If", "How", "What", "The", "A", "An", "In", "On", "At", "After",
    "Before", "Each", "Every", "This", "That", "There", "He", "She", "They",
}

_WORD_RE = re.compile(r"\b[A-Z][a-zA-Z']+\b")


def find_candidate_names(text: str) -> list[str]:
    seen = []
    for m in _WORD_RE.finditer(text):
        w = m.group(0)
        if w in _STOPWORDS or w in seen:
            continue
        seen.append(w)
    return seen


def _token_len(tok, s: str) -> int:
    if tok is None:
        return len(s.split())
    return len(tok.encode(s, add_special_tokens=False))


def apply_name_swap(text: str, rng, tokenizer=None) -> tuple[str, dict]:
    """Replace every distinct candidate proper name with a pool name.

    Returns (new_text, meta) where meta records the substitution map and
    whether a token-length-preserving replacement was found for every name.
    """
    names = find_candidate_names(text)
    if not names:
        return text, {"applied": False, "substitutions": {}}

    substitutions = {}
    new_text = text
    all_aligned = True
    for name in names:
        pool = [n for n in NAME_POOL if n not in names and n not in substitutions.values()]
        rng.shuffle(pool)
        target_len = _token_len(tokenizer, name)
        pool_sorted = sorted(pool, key=lambda n: abs(_token_len(tokenizer, n) - target_len))
        replacement = pool_sorted[0] if pool_sorted else name
        if _token_len(tokenizer, replacement) != target_len:
            all_aligned = False
        substitutions[name] = replacement
        new_text = re.sub(rf"\b{re.escape(name)}\b", replacement, new_text)

    return new_text, {
        "applied": True,
        "substitutions": substitutions,
        "token_aligned": all_aligned,
    }
