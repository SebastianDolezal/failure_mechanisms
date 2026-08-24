"""Failure-excess causal signature D_i = R_i - B_i (Section 23).

B_i estimates the causal-profile change normally produced by the same kind
of surface perturbation even when the model stays correct, conditional on
covariates X_i (perturbation family, edit size, problem type, ...). This is
what lets D_i be read as *failure-associated* rather than merely
*perturbation-associated* (Section 24).
"""
from __future__ import annotations

import numpy as np

from ..matching.covariates import _numeric_vector


def aggregate_stable_baseline(
    target_covariates: dict,
    stable_records: list[dict],
    k: int = 8,
) -> np.ndarray:
    """B_i = E[R^stable | X_i], estimated by a covariate-matched local
    average: restrict to stable pairs sharing perturbation_family (falling
    back to problem_type, then the full pool if too few match), then average
    the k nearest by numeric covariate distance.

    `stable_records`: list of {"covariates": dict, "R": np.ndarray}.
    """
    if not stable_records:
        return np.zeros(0)

    same_family = [s for s in stable_records
                    if s["covariates"]["perturbation_family"] == target_covariates["perturbation_family"]]
    pool = same_family if len(same_family) >= 3 else stable_records

    tv = _numeric_vector(target_covariates)
    scored = sorted(pool, key=lambda s: float(np.linalg.norm(_numeric_vector(s["covariates"]) - tv)))
    neighbors = scored[:k] if len(scored) > k else scored
    stacked = np.stack([s["R"] for s in neighbors], axis=0)
    return stacked.mean(axis=0)


def compute_failure_excess_signature(R_i: np.ndarray, B_i: np.ndarray) -> np.ndarray:
    if B_i.size == 0:
        return R_i.copy()
    return R_i - B_i
