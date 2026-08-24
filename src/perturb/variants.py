"""Build the meaning-preserving surface-variant bank V_i for each problem
(Section 7). Every variant must preserve the exact mathematical solution;
only the conservative, mechanical transformations in this package are used.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .name_swap import apply_name_swap
from .number_format import apply_number_format
from .lexical import apply_lexical_conservative

PERTURBATION_FUNCS = {
    "name_swap": apply_name_swap,
    "number_format": apply_number_format,
    "lexical_conservative": apply_lexical_conservative,
}


@dataclass
class Variant:
    variant_id: str
    base_id: str
    text: str
    perturbation_family: str
    meta: dict = field(default_factory=dict)


def token_edit_distance(a: str, b: str, tokenizer=None) -> int:
    """Levenshtein distance over tokens (or whitespace tokens as a fallback)."""
    ta = tokenizer.encode(a, add_special_tokens=False) if tokenizer else a.split()
    tb = tokenizer.encode(b, add_special_tokens=False) if tokenizer else b.split()
    n, m = len(ta), len(tb)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ta[i - 1] == tb[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def build_variant_bank(
    base_id: str,
    question: str,
    families: list[str],
    k: int = 8,
    seed: int = 0,
    tokenizer=None,
) -> list[Variant]:
    """Generate up to k meaning-preserving variants of `question`.

    Family choice is cycled deterministically (seeded by base_id + index) so
    the bank is reproducible. Variant 0 is always the untouched original so
    downstream pair selection can treat it as a baseline realization.
    """
    variants = [Variant(
        variant_id=f"{base_id}_v0",
        base_id=base_id,
        text=question,
        perturbation_family="none",
        meta={"is_original": True},
    )]

    for i in range(1, k):
        rng = random.Random(f"{base_id}_{seed}_{i}")
        family = families[(i - 1) % len(families)]
        func = PERTURBATION_FUNCS[family]
        text, meta = func(question, rng, tokenizer=tokenizer)
        meta["family"] = family
        meta["is_original"] = False
        meta["token_edit_distance_from_original"] = token_edit_distance(question, text, tokenizer)
        variants.append(Variant(
            variant_id=f"{base_id}_v{i}",
            base_id=base_id,
            text=text,
            perturbation_family=family,
            meta=meta,
        ))
    return variants
