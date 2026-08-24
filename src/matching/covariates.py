"""Nonmechanistic covariate construction and hard-negative matching
(Section 29). These covariates are what the primary RQ2 test conditions on
before asking whether *semantic* similarity carries any additional signal.
"""
from __future__ import annotations

import numpy as np

from ..datasets.schema import PairRecord

COVARIATE_KEYS = [
    "problem_type", "perturbation_family", "edit_size", "problem_length",
    "trace_length", "reasoning_step_count", "error_step_index", "difficulty",
]


def _step_count(trace: str | None) -> int:
    if not trace:
        return 0
    return sum(1 for l in trace.splitlines() if l.strip())


def build_covariates(record: PairRecord, problem_embedding: np.ndarray | None = None) -> dict:
    """Returns a flat covariate dict for one failure record. `problem_embedding`
    is a precomputed sentence embedding of the (clean) problem text, used only
    for nearest-neighbor matching, not as a regression feature by itself."""
    x = {
        "problem_type": record.meta.get("problem_type", "unknown"),
        "perturbation_family": record.perturbation,
        "edit_size": record.token_edit_distance,
        "problem_length": len(record.fail_question.split()),
        "trace_length": len((record.fail_trace or "").split()),
        "reasoning_step_count": _step_count(record.fail_trace),
        "error_step_index": record.first_observable_error_step or -1,
        "difficulty": record.meta.get("difficulty", _step_count(record.fail_trace)),
    }
    if problem_embedding is not None:
        x["_embedding"] = problem_embedding
    return x


def _numeric_vector(cov: dict) -> np.ndarray:
    return np.array([
        cov["edit_size"], cov["problem_length"], cov["trace_length"],
        cov["reasoning_step_count"], cov["error_step_index"], cov["difficulty"],
    ], dtype=float)


def covariate_distance(a: dict, b: dict) -> float:
    """Mixed categorical + numeric distance used to find hard negatives:
    same perturbation family / problem type strongly preferred, then
    Euclidean distance in standardized numeric covariate space."""
    penalty = 0.0
    if a["perturbation_family"] != b["perturbation_family"]:
        penalty += 2.0
    if a["problem_type"] != b["problem_type"]:
        penalty += 2.0
    num_dist = float(np.linalg.norm(_numeric_vector(a) - _numeric_vector(b)))
    return penalty + num_dist


def hard_negative_matches(
    index: int,
    covariates: list[dict],
    labels: list[str],
    n_negatives: int = 5,
) -> list[int]:
    """For item `index`, return indices of the `n_negatives` closest items
    (by covariate_distance) whose semantic label differs from item `index`'s
    label. These are the "good negatives" of Section 29 (e.g. same problem
    type / perturbation family, different failure semantics)."""
    target = covariates[index]
    target_label = labels[index]
    scored = []
    for j, (cov, lab) in enumerate(zip(covariates, labels)):
        if j == index or lab == target_label:
            continue
        scored.append((covariate_distance(target, cov), j))
    scored.sort(key=lambda t: t[0])
    return [j for _, j in scored[:n_negatives]]


def same_category_matches(index: int, labels: list[str]) -> list[int]:
    target_label = labels[index]
    return [j for j, lab in enumerate(labels) if j != index and lab == target_label]


def label_agnostic_candidate_pool(
    index: int,
    covariates: list[dict],
    base_ids: list[str],
    n_neighbors: int = 20,
) -> list[int]:
    """Covariate-nearest neighbors of `index`, excluding any item sharing its
    base_id (a hard negative must be a *different* underlying problem). Used
    to build the matched candidate pool that the primary RQ2 statistic
    (Section 30) and its permutation test (Section 31) both draw from, so
    same- and different-category comparisons are equally "hard".
    """
    target = covariates[index]
    target_base = base_ids[index]
    scored = []
    for j, cov in enumerate(covariates):
        if j == index or base_ids[j] == target_base:
            continue
        scored.append((covariate_distance(target, cov), j))
    scored.sort(key=lambda t: t[0])
    return [j for _, j in scored[:n_neighbors]]
