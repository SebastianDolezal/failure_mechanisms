#!/usr/bin/env python
"""H6 - independent mechanistic clustering, splits and merges
(Section 35-36, 47 Fig 6).

Clusters D_1..D_N purely from the causal signatures (no semantic labels),
then contrasts the resulting mechanistic groups against the frozen semantic
taxonomy. Reports global ARI/NMI (not the headline result) plus an explicit
per-category split/merge table and a Sankey diagram - which single symptoms
correspond to multiple mechanistic subgroups (splits) and which distinct
semantic concepts collapse onto a shared mechanistic group (merges).

Usage:
    python scripts/14_split_merge_analysis.py --benchmark gsm8k --category-level mid
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_experiment_config, load_benchmark_config, read_jsonl, _abs
from src.plots.figures import fig6_splits_merges_sankey

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("split_merge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--category-level", default="mid", choices=["coarse", "mid", "fine"])
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--pca-dims", type=int, default=10)
    args = ap.parse_args()

    exp_cfg = load_experiment_config(args.config)
    bench_cfg = load_benchmark_config(args.benchmark)

    records = read_jsonl(_abs(bench_cfg["pairs_path"]) / "pairs_annotated.jsonl")
    records = [r for r in records if r.failure_excess_signature is not None
                and getattr(r, f"semantic_{args.category_level}") is not None]

    D = np.array([r.failure_excess_signature for r in records])
    D_norm = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    sem_labels = [getattr(r, f"semantic_{args.category_level}") for r in records]

    n_dims = min(args.pca_dims, D_norm.shape[0] - 1, D_norm.shape[1])
    D_reduced = PCA(n_components=n_dims, random_state=exp_cfg["random_seed"]).fit_transform(D_norm)

    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(3, len(records) // 30), min_samples=2)
    mech_labels = clusterer.fit_predict(D_reduced)
    mech_labels_named = [f"mech_{l}" if l != -1 else "mech_noise" for l in mech_labels]

    ari = adjusted_rand_score(sem_labels, mech_labels)
    nmi = normalized_mutual_info_score(sem_labels, mech_labels)
    log.info("Global ARI=%.3f, NMI=%.3f (secondary result, see splits/merges below)", ari, nmi)

    # --- splits: one semantic category -> multiple mechanistic groups ------
    sem_to_mech = defaultdict(Counter)
    mech_to_sem = defaultdict(Counter)
    for s, m in zip(sem_labels, mech_labels_named):
        sem_to_mech[s][m] += 1
        mech_to_sem[m][s] += 1

    splits = {s: dict(counts) for s, counts in sem_to_mech.items() if len(counts) > 1}
    merges = {m: dict(counts) for m, counts in mech_to_sem.items() if len(counts) > 1 and m != "mech_noise"}

    out_dir = _abs(args.out_dir)
    report = {
        "benchmark": args.benchmark, "category_level": args.category_level,
        "n_items": len(records), "n_mechanistic_clusters": len(set(mech_labels_named) - {"mech_noise"}),
        "global_ari": ari, "global_nmi": nmi,
        "splits_semantic_to_multiple_mechanistic": splits,
        "merges_multiple_semantic_to_one_mechanistic": merges,
    }
    out_path = out_dir / "reports" / f"split_merge_{args.benchmark}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Figure is generated last, after the report is already on disk, so a
    # plotting issue can never discard a computed result. Uses the same
    # mech_labels_named strings as the report above (not the raw numeric
    # cluster ids) so the diagram's labels match splits/merges/mech_noise.
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig6_splits_merges_sankey(sem_labels, mech_labels_named,
                                str(out_dir / "figures" / f"fig6_splits_merges_{args.benchmark}.html"))

    log.info("%d semantic categories split across multiple mechanistic groups; %d mechanistic groups merge multiple semantic categories",
              len(splits), len(merges))


if __name__ == "__main__":
    main()
