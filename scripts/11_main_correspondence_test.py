#!/usr/bin/env python
"""Primary RQ2 analysis (Section 2, 28-32, 47 Fig 3-4).

Three things happen here, in order:
  1. Continuous semantic correspondence: C_ij^mech ~ b0 + b1*S_ij^sem + b2*X_ij
     (Section 32) - the cleanest direct test, independent of any clustering.
  2. Matched same-vs-different category analysis: Delta = E[C|T_i=T_j] -
     E[C|T_i!=T_j] over a covariate-matched hard-negative candidate pool
     (Section 29-30), with a stratified block-permutation test (Section 31)
     and a problem-level bootstrap CI.
  3. Everything is computed on the failure-EXCESS signature D_i (i.e.
     already stable-pair-adjusted, Section 23), so the result cannot simply
     be re-deriving "how the model processes this perturbation family".

Usage:
    python scripts/11_main_correspondence_test.py --benchmark gsm8k --permutations 10000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_experiment_config, load_benchmark_config, read_jsonl, _abs
from src.matching.covariates import build_covariates, label_agnostic_candidate_pool, _numeric_vector
from src.mechanisms.similarity import cosine_sim, pairwise_similarity_matrix
from src.statistics.permutation import compute_delta, permutation_test_delta
from src.statistics.bootstrap import bootstrap_ci_by_problem
from src.statistics.regression import continuous_correspondence_regression
from src.plots.figures import fig3_continuous_correspondence, fig4_primary_category_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("main_correspondence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--permutations", type=int, default=None)
    ap.add_argument("--category-level", default="mid", choices=["coarse", "mid", "fine"])
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    exp_cfg = load_experiment_config(args.config)
    bench_cfg = load_benchmark_config(args.benchmark)
    n_perm = args.permutations or exp_cfg["statistics"]["n_permutations"]
    n_neighbors = exp_cfg["matching"]["n_hard_negatives_per_item"] * 4  # pool size, negatives drawn from it

    records = read_jsonl(_abs(bench_cfg["pairs_path"]) / "pairs_annotated.jsonl")
    records = [r for r in records if r.failure_excess_signature is not None and r.semantic_embedding is not None
                and getattr(r, f"semantic_{args.category_level}") is not None]
    log.info("%d failures with valid D_i and semantic labels", len(records))

    base_ids = [r.base_id for r in records]
    labels = [getattr(r, f"semantic_{args.category_level}") for r in records]
    D = [np.array(r.failure_excess_signature) for r in records]
    S = np.array([r.semantic_embedding for r in records])
    covariates = [build_covariates(r) for r in records]

    # --- mechanistic + semantic pairwise similarity ------------------------
    C_mech = pairwise_similarity_matrix(D, metric="cosine")
    S_sem = S @ S.T / (np.linalg.norm(S, axis=1, keepdims=True) @ np.linalg.norm(S, axis=1, keepdims=True).T + 1e-8)

    candidate_pool = [label_agnostic_candidate_pool(i, covariates, base_ids, n_neighbors=n_neighbors)
                        for i in range(len(records))]

    # === 1. Continuous correspondence (Section 32) =========================
    rows = []
    for i in range(len(records)):
        for j in candidate_pool[i]:
            if j <= i:
                continue
            rows.append({
                "i": i, "j": j,
                "S_sem": float(S_sem[i, j]), "C_mech": float(C_mech[i, j]),
                "edit_size_diff": abs(covariates[i]["edit_size"] - covariates[j]["edit_size"]),
                "problem_length_diff": abs(covariates[i]["problem_length"] - covariates[j]["problem_length"]),
                "trace_length_diff": abs(covariates[i]["trace_length"] - covariates[j]["trace_length"]),
                "same_perturbation": float(covariates[i]["perturbation_family"] == covariates[j]["perturbation_family"]),
                "cluster_group": base_ids[i],  # cluster SEs by one member's underlying problem
            })
    df = pd.DataFrame(rows)
    continuous_result = continuous_correspondence_regression(
        C_mech=df["C_mech"].values, S_sem=df["S_sem"].values,
        X_covariates=df[["edit_size_diff", "problem_length_diff", "trace_length_diff", "same_perturbation"]],
        cluster_groups=df["cluster_group"].values,
    )
    log.info("Continuous correspondence: beta_sem=%.4f (p=%.4g), n=%d",
              continuous_result["beta_sem"], continuous_result["p_sem"], continuous_result["n_obs"])

    out_dir = _abs(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    # === 2. Categorical Delta + permutation + bootstrap (Section 30-31) ====
    delta_i, Delta_obs = compute_delta(C_mech, labels, candidate_pool)
    strata = [c["perturbation_family"] for c in covariates]
    perm_result = permutation_test_delta(C_mech, labels, candidate_pool, strata=strata,
                                          n_permutations=n_perm, seed=exp_cfg["random_seed"])

    base_id_to_idx = {}
    for i, b in enumerate(base_ids):
        base_id_to_idx.setdefault(b, i)  # one primary pair per base problem (Section 10)

    def statistic_fn(sample_base_ids):
        idx_set = sorted({base_id_to_idx[b] for b in set(sample_base_ids) if b in base_id_to_idx})
        if len(idx_set) < 5:
            return float("nan")
        pos = {g: p for p, g in enumerate(idx_set)}
        sub_sim = C_mech[np.ix_(idx_set, idx_set)]
        sub_labels = [labels[g] for g in idx_set]
        sub_pool = [[pos[j] for j in candidate_pool[g] if j in pos] for g in idx_set]
        _, d = compute_delta(sub_sim, sub_labels, sub_pool)
        return d if d == d else float("nan")

    boot = bootstrap_ci_by_problem(base_ids, statistic_fn,
                                    n_bootstrap=exp_cfg["statistics"]["n_bootstrap"],
                                    seed=exp_cfg["random_seed"], alpha=exp_cfg["statistics"]["alpha"])

    same_cat_sims = np.array([C_mech[i, j] for i in range(len(records)) for j in candidate_pool[i]
                                if labels[j] == labels[i]])
    diff_cat_sims = np.array([C_mech[i, j] for i in range(len(records)) for j in candidate_pool[i]
                                if labels[j] != labels[i]])

    report = {
        "benchmark": args.benchmark,
        "category_level": args.category_level,
        "n_failures": len(records),
        "continuous_correspondence": {k: v for k, v in continuous_result.items() if k != "summary"},
        "Delta_observed": Delta_obs,
        "Delta_permutation_p_value": perm_result["p_value"],
        "Delta_bootstrap_ci": [boot["ci_low"], boot["ci_high"]],
        "Delta_bootstrap_mean": boot["mean"],
        "n_permutations": n_perm,
    }
    out_path = out_dir / "reports" / f"main_correspondence_{args.benchmark}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("Delta=%.4f, 95%% bootstrap CI=[%.4f, %.4f], permutation p=%.4g",
              Delta_obs, boot["ci_low"], boot["ci_high"], perm_result["p_value"])

    # Figures are generated last, after every statistical result is already
    # on disk, so a plotting issue can never discard a computed result.
    fig3_continuous_correspondence(df["S_sem"].values, df["C_mech"].values,
                                    str(out_dir / "figures" / f"fig3_continuous_correspondence_{args.benchmark}.png"),
                                    beta1=continuous_result["beta_sem"])
    fig4_primary_category_result(same_cat_sims, diff_cat_sims, Delta_obs, (boot["ci_low"], boot["ci_high"]),
                                  str(out_dir / "figures" / f"fig4_primary_category_{args.benchmark}.png"))


if __name__ == "__main__":
    main()
