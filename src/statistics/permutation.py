"""Primary RQ2 statistic and its permutation inference (Section 30-31).

    delta_i = S_i^same - S_i^diff
    Delta   = mean_i delta_i

S_i^same / S_i^diff are computed only over a pre-matched candidate pool per
item (typically the covariate-nearest neighbors within the same benchmark /
stratum, see src.matching.covariates) so that both same- and different-
category comparisons are drawn from equally "hard" surrounding examples -
this is what makes different-category comparisons hard negatives rather
than arbitrary unrelated failures.
"""
from __future__ import annotations

import numpy as np


def compute_delta(
    sim_matrix: np.ndarray,
    labels: list[str],
    candidate_pool: list[list[int]],
) -> tuple[np.ndarray, float]:
    n = len(labels)
    delta_i = np.full(n, np.nan)
    for i in range(n):
        pool = [j for j in candidate_pool[i] if j != i]
        same = [j for j in pool if labels[j] == labels[i]]
        diff = [j for j in pool if labels[j] != labels[i]]
        if not same or not diff:
            continue
        s_same = float(np.mean([sim_matrix[i, j] for j in same]))
        s_diff = float(np.mean([sim_matrix[i, j] for j in diff]))
        delta_i[i] = s_same - s_diff
    valid = delta_i[~np.isnan(delta_i)]
    Delta = float(valid.mean()) if valid.size else float("nan")
    return delta_i, Delta


def _stratified_permute(labels: list[str], strata: list[str], rng: np.random.Generator) -> list[str]:
    """Shuffle labels within each stratum independently, preserving both
    per-stratum category-size composition and the stratification variable
    (Section 31: 'preserving category sizes, benchmark, important matching
    strata')."""
    labels = np.array(labels)
    strata = np.array(strata)
    out = labels.copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        shuffled = labels[idx].copy()
        rng.shuffle(shuffled)
        out[idx] = shuffled
    return out.tolist()


def permutation_test_delta(
    sim_matrix: np.ndarray,
    labels: list[str],
    candidate_pool: list[list[int]],
    strata: list[str] | None = None,
    n_permutations: int = 10000,
    seed: int = 0,
) -> dict:
    n = len(labels)
    strata = strata if strata is not None else ["all"] * n
    rng = np.random.default_rng(seed)

    _, Delta_obs = compute_delta(sim_matrix, labels, candidate_pool)

    null = np.zeros(n_permutations)
    for p in range(n_permutations):
        perm_labels = _stratified_permute(labels, strata, rng)
        _, d = compute_delta(sim_matrix, perm_labels, candidate_pool)
        null[p] = d if d == d else 0.0  # NaN-safe

    p_value = float(np.mean(null >= Delta_obs))
    return {
        "Delta_observed": Delta_obs,
        "null_distribution": null,
        "p_value": p_value,
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
    }
