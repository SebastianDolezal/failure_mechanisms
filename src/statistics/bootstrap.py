"""Bootstrap confidence intervals over underlying benchmark problems
(Section 31: "Construct confidence intervals by bootstrapping underlying
benchmark problems rather than individual pair rows.")
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci_by_problem(
    base_ids: list[str],
    statistic_fn,
    n_bootstrap: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """`statistic_fn(selected_base_ids: list[str]) -> float` computes the
    statistic (e.g. Delta) restricted to the given base_ids (with
    repetition). Resampling is done at the base_id level, not the row level,
    so within-problem pairs never get treated as independent observations.
    """
    unique_ids = sorted(set(base_ids))
    rng = np.random.default_rng(seed)
    stats = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        stats[b] = statistic_fn(sample.tolist())
    # statistic_fn returns NaN for resamples with too few unique problems
    # (see callers' "len(idx) < 5" guards) - use the nan-aware percentile/mean
    # so those degenerate draws are excluded rather than poisoning the whole
    # CI (plain np.percentile/np.mean propagate a single NaN to every output).
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(np.nanmean(stats)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_nan_draws": int(np.isnan(stats).sum()),
        "bootstrap_distribution": stats,
    }
