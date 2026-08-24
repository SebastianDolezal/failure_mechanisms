#!/usr/bin/env python
"""RQ1b - mechanistic reliability (Section 26, Gate C).

Compares exact activation patching against Attribution Patching on the same
failures (rho_i = Spearman(D_i^exact, D_i^AtP)), and tests the crucial
condition:

    sim(D_i^exact, D_i^AtP)  >  sim(D_i, D_j)   for i != j

i.e. two independent causal-measurement methods must agree on the SAME
failure more than exact-patching profiles agree across DIFFERENT failures.
If not, the causal instrument is too unreliable to support a strong RQ2
interpretation (Section 41, Gate C).

Usage:
    python scripts/10_mechanistic_reliability.py --benchmark gsm8k
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_experiment_config, load_benchmark_config, read_jsonl, _abs
from src.mechanisms.similarity import cosine_sim, spearman_sim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mech_reliability")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out", default="results/reports/mechanistic_reliability.json")
    ap.add_argument("--n-different-pairs", type=int, default=500)
    args = ap.parse_args()

    exp_cfg = load_experiment_config(args.config)
    bench_cfg = load_benchmark_config(args.benchmark)
    gates = exp_cfg["gates"]

    atp_dir = _abs("data/mechanisms") / args.benchmark
    subset = read_jsonl(atp_dir / "attribution_patching_subset.jsonl")
    subset = [r for r in subset if r.causal_profile_atp is not None and r.failure_excess_signature is not None]

    rhos, coss = [], []
    for r in subset:
        d_exact = np.array(r.failure_excess_signature)
        d_atp = np.array(r.causal_profile_atp)
        rhos.append(spearman_sim(d_exact, d_atp))
        coss.append(cosine_sim(d_exact, d_atp))

    all_fail = read_jsonl(_abs(bench_cfg["pairs_path"]) / "pairs_annotated.jsonl")
    all_fail = [r for r in all_fail if r.failure_excess_signature is not None]

    rng = random.Random(exp_cfg["random_seed"])
    diff_pair_sims = []
    n = len(all_fail)
    for _ in range(min(args.n_different_pairs, n * (n - 1) // 2)):
        i, j = rng.sample(range(n), 2)
        di = np.array(all_fail[i].failure_excess_signature)
        dj = np.array(all_fail[j].failure_excess_signature)
        diff_pair_sims.append(cosine_sim(di, dj))

    same_failure_mean = float(np.mean(coss)) if coss else float("nan")
    diff_failure_mean = float(np.mean(diff_pair_sims)) if diff_pair_sims else float("nan")
    reliability_ratio = same_failure_mean - diff_failure_mean

    mean_rho = float(np.nanmean(rhos)) if rhos else float("nan")
    gate_c_passed = (reliability_ratio >= gates["gate_c_min_reliability_ratio"]
                       if reliability_ratio == reliability_ratio else False)

    report = {
        "n_reliability_subset": len(subset),
        "mean_spearman_exact_vs_atp": mean_rho,
        "mean_cosine_exact_vs_atp_same_failure": same_failure_mean,
        "mean_cosine_exact_different_failures": diff_failure_mean,
        "reliability_margin_same_minus_diff": reliability_ratio,
        "gate_c_passed": gate_c_passed,
        "gate_c_threshold": gates["gate_c_min_reliability_ratio"],
        "per_item_rho": rhos,
        "per_item_cosine": coss,
    }

    out_path = _abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("mean rho(exact,AtP)=%.3f | same-failure cos=%.3f vs different-failure cos=%.3f -> Gate C %s",
              mean_rho, same_failure_mean, diff_failure_mean, "PASSED" if gate_c_passed else "FAILED")


if __name__ == "__main__":
    main()
