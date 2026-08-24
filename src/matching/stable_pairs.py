"""Stable-pair (correct -> correct) control construction (Section 8).

For every perturbation family, find a closely matched pair of realizations
of the SAME underlying problem where the surface form changes but
correctness does not. These estimate the causal effect of the perturbation
itself, independent of failure, and are subtracted out to form D_i
(Section 23).
"""
from __future__ import annotations

from .pairs import PairCandidate, _aligned_changed_positions
from ..perturb.variants import token_edit_distance


def select_stable_pairs(
    base_id: str,
    gold_answer: str,
    variant_generations: list[dict],
    tokenizer=None,
    max_pairs_per_family: int = 2,
) -> list[PairCandidate]:
    correct = [v for v in variant_generations if v["predicted_answer"] == gold_answer]
    if len(correct) < 2:
        return []

    by_family: dict[str, list[dict]] = {}
    for v in correct:
        by_family.setdefault(v.get("perturbation_family", "unknown"), []).append(v)

    original = next((v for v in correct if v.get("is_original")), None)
    pairs = []
    for family, items in by_family.items():
        if family == "none":
            continue
        anchor = original if original is not None else items[0]
        candidates = sorted(
            (v for v in items if v["variant_id"] != anchor["variant_id"]),
            key=lambda v: token_edit_distance(anchor["text"], v["text"], tokenizer),
        )
        for v in candidates[:max_pairs_per_family]:
            ted = token_edit_distance(anchor["text"], v["text"], tokenizer)
            len_a = len(tokenizer.encode(anchor["text"], add_special_tokens=False)) if tokenizer else len(anchor["text"].split())
            len_b = len(tokenizer.encode(v["text"], add_special_tokens=False)) if tokenizer else len(v["text"].split())
            clean_pos, fail_pos = _aligned_changed_positions(tokenizer, anchor["text"], v["text"])
            pairs.append(PairCandidate(
                base_id=base_id,
                clean_variant_id=anchor["variant_id"],
                fail_variant_id=v["variant_id"],
                clean_text=anchor["text"],
                fail_text=v["text"],
                clean_answer=anchor["predicted_answer"],
                fail_answer=v["predicted_answer"],
                gold_answer=gold_answer,
                perturbation_family=family,
                token_edit_distance=ted,
                same_token_length=(len_a == len_b),
                changed_clean_tokens=clean_pos,
                changed_fail_tokens=fail_pos,
                pair_type="stable_control",
            ))
    return pairs
