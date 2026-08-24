"""Clean/failure pair selection (Section 10, 11).

Given all variant generations for one underlying problem, find the closest
correct/incorrect realization pair, preferring (in order):
  1. identical token sequence length
  2. one-to-one alignment of changed tokens
  3. lowest token edit distance
  4. lowest semantic edit magnitude (embedding distance, tie-break only)

Only one primary pair is kept per underlying benchmark problem so the unit
of statistical independence stays at the original-problem level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..perturb.variants import token_edit_distance


@dataclass
class PairCandidate:
    base_id: str
    clean_variant_id: str
    fail_variant_id: str
    clean_text: str
    fail_text: str
    clean_answer: str
    fail_answer: str
    gold_answer: str
    perturbation_family: str
    token_edit_distance: int
    same_token_length: bool
    changed_clean_tokens: list[int]
    changed_fail_tokens: list[int]
    pair_type: str  # "induced_failure" | "rescued_failure"


def _aligned_changed_positions(tok, clean_text: str, fail_text: str) -> tuple[list[int], list[int]]:
    """Best-effort one-to-one changed-token alignment: tokenizes both texts
    and returns the (clean_positions, fail_positions) indices of the first
    and last differing tokens under a common prefix/suffix trim. Exact for
    single-span substitutions like name_swap / number_format; approximate
    otherwise (flagged via length mismatch upstream)."""
    a = tok.encode(clean_text, add_special_tokens=False) if tok else clean_text.split()
    b = tok.encode(fail_text, add_special_tokens=False) if tok else fail_text.split()

    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    j = 0
    while j < len(a) - i and j < len(b) - i and a[-1 - j] == b[-1 - j]:
        j += 1

    clean_positions = list(range(i, len(a) - j))
    fail_positions = list(range(i, len(b) - j))
    return clean_positions, fail_positions


def select_clean_fail_pair(
    base_id: str,
    gold_answer: str,
    variant_generations: list[dict],
    tokenizer=None,
) -> PairCandidate | None:
    """`variant_generations`: list of dicts with keys variant_id, text,
    predicted_answer, perturbation_family (as produced by script 02/03)."""
    correct = [v for v in variant_generations if v["predicted_answer"] == gold_answer]
    incorrect = [v for v in variant_generations if v["predicted_answer"] != gold_answer
                 and v["predicted_answer"] is not None]
    if not correct or not incorrect:
        return None

    original = next((v for v in variant_generations if v.get("is_original")), None)
    pair_type = "induced_failure"
    if original is not None and original["predicted_answer"] != gold_answer:
        pair_type = "rescued_failure"
        # original is the incorrect realization; we need a correct variant to pair with it
        if original in incorrect:
            pass

    best = None
    best_key = None
    for c in correct:
        for f in incorrect:
            if c["variant_id"] == f["variant_id"]:
                continue
            ted = token_edit_distance(c["text"], f["text"], tokenizer)
            len_c = len(tokenizer.encode(c["text"], add_special_tokens=False)) if tokenizer else len(c["text"].split())
            len_f = len(tokenizer.encode(f["text"], add_special_tokens=False)) if tokenizer else len(f["text"].split())
            same_len = len_c == len_f
            sem_dist = 0.0
            if "embedding" in c and "embedding" in f and c["embedding"] is not None:
                ec, ef = np.array(c["embedding"]), np.array(f["embedding"])
                sem_dist = 1 - float(ec @ ef / (np.linalg.norm(ec) * np.linalg.norm(ef) + 1e-8))
            key = (0 if same_len else 1, ted, sem_dist)
            if best_key is None or key < best_key:
                best_key = key
                clean_pos, fail_pos = _aligned_changed_positions(tokenizer, c["text"], f["text"])
                best = PairCandidate(
                    base_id=base_id,
                    clean_variant_id=c["variant_id"],
                    fail_variant_id=f["variant_id"],
                    clean_text=c["text"],
                    fail_text=f["text"],
                    clean_answer=c["predicted_answer"],
                    fail_answer=f["predicted_answer"],
                    gold_answer=gold_answer,
                    perturbation_family=f.get("perturbation_family", "unknown"),
                    token_edit_distance=ted,
                    same_token_length=same_len,
                    changed_clean_tokens=clean_pos,
                    changed_fail_tokens=fail_pos,
                    pair_type=pair_type,
                )
    return best
