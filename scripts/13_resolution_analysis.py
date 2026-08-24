#!/usr/bin/env python
"""H5 - semantic-resolution analysis (Section 34, 47 Fig 5).

Recomputes the primary Delta statistic at each taxonomy resolution level
(coarse / mid / fine) to test whether mechanistic coherence peaks at an
intermediate level of semantic abstraction. Empirical question - no
assumption of where the peak (if any) will be.

Usage:
    python scripts/13_resolution_analysis.py --benchmark gsm8k
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_experiment_config, load_benchmark_config, read_jsonl, _abs
from src.matching.covariates import build_covariates, label_agnostic_candidate_pool
from src.mechanisms.similarity import pairwise_similarity_matrix
from src.statistics.permutation import compute_delta, permutation_test_delta
from src.statistics.bootstrap import bootstrap_ci_by_problem
from src.plots.figures import fig5_resolution_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("resolution_analysis")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    exp_cfg = load_experiment_config(args.config)
    bench_cfg = load_benchmark_config(args.benchmark)
    n_neighbors = exp_cfg["matching"]["n_hard_negatives_per_item"] * 4

    records = read_jsonl(_abs(bench_cfg["pairs_path"]) / "pairs_annotated.jsonl")
    records = [r for r in records if r.failure_excess_signature is not None]
    base_ids = [r.base_id for r in records]
    covariates = [build_covariates(r) for r in records]
    D = [np.array(r.failure_excess_signature) for r in records]
    C_mech = pairwise_similarity_matrix(D, metric="cosine")
    candidate_pool = [label_agnostic_candidate_pool(i, covariates, base_ids, n_neighbors=n_neighbors)
                        for i in range(len(records))]
    strata = [c["perturbation_family"] for c in covariates]

    base_id_to_idx = {}
    for i, b in enumerate(base_ids):
        base_id_to_idx.setdefault(b, i)

    levels, deltas, cis, pvals, ns = [], [], [], [], []
    for level in exp_cfg["taxonomy"]["hierarchy_levels"]:
        labels = [getattr(r, f"semantic_{level}") for r in records]
        valid = [i for i, l in enumerate(labels) if l is not None]
        if len(valid) < 10:
            log.warning("Skipping level=%s: only %d labeled items", level, len(valid))
            continue
        sub_sim = C_mech[np.ix_(valid, valid)]
        sub_labels = [labels[i] for i in valid]
        sub_pool_map = {g: p for p, g in enumerate(valid)}
        sub_pool = [[sub_pool_map[j] for j in candidate_pool[g] if j in sub_pool_map] for g in valid]
        sub_strata = [strata[i] for i in valid]
        sub_base_ids = [base_ids[i] for i in valid]

        _, Delta = compute_delta(sub_sim, sub_labels, sub_pool)
        perm = permutation_test_delta(sub_sim, sub_labels, sub_pool, strata=sub_strata,
                                       n_permutations=exp_cfg["statistics"]["n_permutations"] // 5,
                                       seed=exp_cfg["random_seed"])

        def statistic_fn(sample_base_ids, sub_base_ids=sub_base_ids, sub_sim=sub_sim, sub_labels=sub_labels, sub_pool=sub_pool):
            local_base_to_pos = {}
            for p, b in enumerate(sub_base_ids):
                local_base_to_pos.setdefault(b, p)
            idx = sorted({local_base_to_pos[b] for b in set(sample_base_ids) if b in local_base_to_pos})
            if len(idx) < 5:
                return float("nan")
            pos = {g: p for p, g in enumerate(idx)}
            s2 = sub_sim[np.ix_(idx, idx)]
            l2 = [sub_labels[g] for g in idx]
            pool2 = [[pos[j] for j in sub_pool[g] if j in pos] for g in idx]
            _, d = compute_delta(s2, l2, pool2)
            return d if d == d else float("nan")

        boot = bootstrap_ci_by_problem(sub_base_ids, statistic_fn,
                                        n_bootstrap=exp_cfg["statistics"]["n_bootstrap"] // 5,
                                        seed=exp_cfg["random_seed"], alpha=exp_cfg["statistics"]["alpha"])

        levels.append(level)
        deltas.append(Delta)
        cis.append((boot["ci_low"], boot["ci_high"]))
        pvals.append(perm["p_value"])
        ns.append(len(valid))
        log.info("level=%s n=%d Delta=%.4f CI=[%.4f,%.4f] p=%.4g",
                  level, len(valid), Delta, boot["ci_low"], boot["ci_high"], perm["p_value"])

    out_dir = _abs(args.out_dir)
    report = {"benchmark": args.benchmark, "levels": levels, "delta": deltas,
              "ci": cis, "p_value": pvals, "n_items": ns}
    out_path = out_dir / "reports" / f"resolution_analysis_{args.benchmark}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Figure is generated last, after the report is already on disk, so a
    # plotting issue can never discard a computed result.
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig5_resolution_curve(levels, deltas, cis, str(out_dir / "figures" / f"fig5_resolution_curve_{args.benchmark}.png"))


if __name__ == "__main__":
    main()
