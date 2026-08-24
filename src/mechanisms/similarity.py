"""Mechanistic similarity metrics (Section 28). Cosine is preregistered as
primary; Spearman rank correlation and top-k overlap are robustness checks
only and must never be substituted for cosine post hoc based on results.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def spearman_sim(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    rho, _ = spearmanr(a, b)
    return float(rho) if rho == rho else 0.0  # guard against NaN


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int = 8) -> float:
    ka = set(np.argsort(-np.abs(a))[:k].tolist())
    kb = set(np.argsort(-np.abs(b))[:k].tolist())
    if not ka and not kb:
        return 0.0
    return len(ka & kb) / len(ka | kb)


def pairwise_similarity_matrix(signatures: list[np.ndarray], metric: str = "cosine") -> np.ndarray:
    n = len(signatures)
    M = np.zeros((n, n))
    fn = {"cosine": cosine_sim, "spearman": spearman_sim, "topk_jaccard": topk_jaccard}[metric]
    for i in range(n):
        for j in range(i, n):
            v = fn(signatures[i], signatures[j])
            M[i, j] = M[j, i] = v
    return M
