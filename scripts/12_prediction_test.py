#!/usr/bin/env python
"""H4 - incremental mechanistic information (Section 33).

Tests whether P(T_i | X_i, D_i) beats P(T_i | X_i) under stratified
cross-validated multinomial logistic regression, i.e. whether the causal
signature carries information about semantic failure type that isn't
already recoverable from nonmechanistic covariates alone.

Usage:
    python scripts/12_prediction_test.py --benchmark gsm8k --category-level mid
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_experiment_config, load_benchmark_config, read_jsonl, _abs
from src.matching.covariates import build_covariates
from src.statistics.regression import incremental_prediction_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prediction_test")


def _covariate_matrix(covariates: list[dict]) -> np.ndarray:
    numeric_keys = ["edit_size", "problem_length", "trace_length", "reasoning_step_count",
                     "error_step_index", "difficulty"]
    numeric = np.array([[c[k] for k in numeric_keys] for c in covariates], dtype=float)
    numeric = StandardScaler().fit_transform(numeric)

    cat = np.array([[c["perturbation_family"], c["problem_type"]] for c in covariates])
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_encoded = enc.fit_transform(cat)
    return np.concatenate([numeric, cat_encoded], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--category-level", default="mid", choices=["coarse", "mid", "fine"])
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()

    exp_cfg = load_experiment_config(args.config)
    bench_cfg = load_benchmark_config(args.benchmark)

    records = read_jsonl(_abs(bench_cfg["pairs_path"]) / "pairs_annotated.jsonl")
    records = [r for r in records if r.failure_excess_signature is not None
                and getattr(r, f"semantic_{args.category_level}") is not None]

    labels = [getattr(r, f"semantic_{args.category_level}") for r in records]
    label_counts = {l: labels.count(l) for l in set(labels)}
    # Stratified CV needs every class to have >= n_splits members.
    n_splits = 5
    keep_labels = {l for l, c in label_counts.items() if c >= n_splits}
    filtered = [(r, l) for r, l in zip(records, labels) if l in keep_labels]
    if len(filtered) < len(records):
        log.warning("Dropped %d items whose category has < %d members (can't stratify-CV)",
                    len(records) - len(filtered), n_splits)
    records = [r for r, _ in filtered]
    labels = [l for _, l in filtered]

    covariates = [build_covariates(r) for r in records]
    X = _covariate_matrix(covariates)
    D = np.array([r.failure_excess_signature for r in records])

    result = incremental_prediction_test(X, D, labels, n_splits=n_splits, seed=exp_cfg["random_seed"])

    out_path = _abs("results/reports") / f"prediction_test_{args.benchmark}_{args.category_level}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"n_items": len(records), "n_classes": len(set(labels)), **result}, f, indent=2)

    log.info("Baseline macro-F1=%.3f, X+D macro-F1=%.3f, delta=%.3f",
              result["baseline"]["macro_f1"], result["mechanistic"]["macro_f1"], result["delta_macro_f1"])


if __name__ == "__main__":
    main()
