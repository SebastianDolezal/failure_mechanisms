"""Number-format variation perturbation (Section 7.B).

Mechanically rewrites the surface form of currency and percentage
quantities without changing their value: "$20" <-> "20 dollars",
"50%" <-> "50 percent". Purely a string-level, value-preserving rewrite.
"""
from __future__ import annotations

import re

_DOLLAR_PREFIX_RE = re.compile(r"\$(\d+(?:\.\d+)?)")
_DOLLAR_SUFFIX_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*dollars\b")
_PERCENT_SYMBOL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_PERCENT_WORD_RE = re.compile(r"(\d+(?:\.\d+)?)\s*percent\b")


def apply_number_format(text: str, rng, tokenizer=None) -> tuple[str, dict]:
    new_text = text
    n_changes = 0

    # $20 -> 20 dollars (direction chosen per-call, deterministically from rng
    # so repeated calls on the same seed are reproducible)
    to_words = rng.random() < 0.5

    if to_words and _DOLLAR_PREFIX_RE.search(new_text):
        new_text, k = _DOLLAR_PREFIX_RE.subn(lambda m: f"{m.group(1)} dollars", new_text)
        n_changes += k
    elif _DOLLAR_SUFFIX_RE.search(new_text):
        new_text, k = _DOLLAR_SUFFIX_RE.subn(lambda m: f"${m.group(1)}", new_text)
        n_changes += k
    elif _DOLLAR_PREFIX_RE.search(new_text):
        new_text, k = _DOLLAR_PREFIX_RE.subn(lambda m: f"{m.group(1)} dollars", new_text)
        n_changes += k

    if _PERCENT_SYMBOL_RE.search(new_text) and (n_changes == 0 or rng.random() < 0.5):
        new_text, k = _PERCENT_SYMBOL_RE.subn(lambda m: f"{m.group(1)} percent", new_text)
        n_changes += k
    elif _PERCENT_WORD_RE.search(new_text):
        new_text, k = _PERCENT_WORD_RE.subn(lambda m: f"{m.group(1)}%", new_text)
        n_changes += k

    return new_text, {"applied": n_changes > 0, "n_changes": n_changes}
