#!/usr/bin/env python
"""RQ1a - semantic reliability (Section 19).

Compares annotations from >=2 independent annotators on the same failures:
  - first-error-step localization agreement
  - description/category agreement (Cohen's kappa, Krippendorff's alpha,
    macro-F1), using a *pilot* joint clustering of all annotators' raw
    descriptions purely to obtain comparable category labels - this
    clustering is NOT the frozen taxonomy (that happens in script 07, on the
    primary annotator only, after this gate passes).
  - raw-description embedding stability (cosine similarity between
    independent descriptions of the same failure).

Writes results/reports/semantic_reliability.json and prints the Gate B
pass/fail verdict (Section 41).

Usage:
    python scripts/06_semantic_reliability.py --benchmark gsm8k \
        --annotators model_judge annotator_1 annotator_2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_experiment_config, _abs
from src.taxonomy.embed import DescriptionEmbedder
from src.taxonomy.cluster import cluster_descriptions
from src.statistics.reliability import (
    cohens_kappa, krippendorffs_alpha, macro_f1, first_error_step_agreement, embedding_stability,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("semantic_reliability")


def load_annotations(benchmark: str, annotator_id: str) -> dict[str, dict]:
    path = _abs("data/annotations") / benchmark / f"{annotator_id}.jsonl"
    out = {}
    if not path.exists():
        # An annotator whose every item failed to parse (script 05) never
        # gets its output file created at all - treat that the same as zero
        # annotations rather than crashing, so this stage can report Gate B
        # as a clean, honest FAILED instead of an unhandled traceback.
        return out
    for line in open(path):
        d = json.loads(line)
        out[d["pair_id"]] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--annotators", nargs="+", required=True)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out", default="results/reports/semantic_reliability.json")
    args = ap.parse_args()

    exp_cfg = load_experiment_config(args.config)
    gates = exp_cfg["gates"]

    ann_by_rater = {a: load_annotations(args.benchmark, a) for a in args.annotators}
    shared_ids = set.intersection(*[set(d.keys()) for d in ann_by_rater.values()])
    shared_ids = sorted(shared_ids)
    log.info("%d shared annotated pair_ids across %d annotators", len(shared_ids), len(args.annotators))

    if not shared_ids:
        log.warning("No pair_ids are shared across all annotators (%s) - at least one annotator produced "
                    "zero parseable annotations (see data/annotations/%s/*.jsonl). Gate B cannot be "
                    "computed; reporting it as FAILED rather than crashing.", args.annotators, args.benchmark)
        report = {
            "n_shared_items": 0, "pairwise": {}, "krippendorffs_alpha_category": None,
            "mean_cohens_kappa": None, "gate_b_passed": False, "gate_b_threshold": gates["gate_b_min_kappa"],
            "error": "no pair_ids shared across all annotators - at least one annotator produced zero "
                     "parseable annotations",
        }
        out_path = _abs(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info("Gate B FAILED (no shared annotated items).")
        return

    all_descriptions = []
    owner = []
    for a in args.annotators:
        for pid in shared_ids:
            all_descriptions.append(ann_by_rater[a][pid]["description"] or "")
            owner.append((a, pid))

    embedder = DescriptionEmbedder(exp_cfg["taxonomy"]["embedding_model"])
    embeddings = embedder.embed(all_descriptions)
    _, cluster_labels = cluster_descriptions(
        embeddings,
        min_cluster_size=max(3, exp_cfg["taxonomy"]["hdbscan"]["min_cluster_size"] // 2),
        min_samples=2,
        metric=exp_cfg["taxonomy"]["hdbscan"]["metric"],
    )

    label_by_rater_pid = {(a, pid): int(lab) for (a, pid), lab in zip(owner, cluster_labels)}

    rater_pairs = [(args.annotators[i], args.annotators[j])
                    for i in range(len(args.annotators)) for j in range(i + 1, len(args.annotators))]

    pairwise_results = {}
    for a, b in rater_pairs:
        labels_a = [label_by_rater_pid[(a, pid)] for pid in shared_ids]
        labels_b = [label_by_rater_pid[(b, pid)] for pid in shared_ids]
        steps_a = [ann_by_rater[a][pid]["first_error_step"] for pid in shared_ids]
        steps_b = [ann_by_rater[b][pid]["first_error_step"] for pid in shared_ids]
        emb_a = embedder.embed([ann_by_rater[a][pid]["description"] or "" for pid in shared_ids])
        emb_b = embedder.embed([ann_by_rater[b][pid]["description"] or "" for pid in shared_ids])

        pairwise_results[f"{a}_vs_{b}"] = {
            "cohens_kappa_category": cohens_kappa([str(l) for l in labels_a], [str(l) for l in labels_b]),
            "macro_f1_category": macro_f1([str(l) for l in labels_a], [str(l) for l in labels_b]),
            "first_error_step_exact_agreement": first_error_step_agreement(steps_a, steps_b, tolerance=0),
            "first_error_step_agreement_tol1": first_error_step_agreement(steps_a, steps_b, tolerance=1),
            "raw_description_embedding_stability": embedding_stability(emb_a, emb_b),
        }

    rater_label_lists = [[str(label_by_rater_pid[(a, pid)]) for pid in shared_ids] for a in args.annotators]
    alpha = krippendorffs_alpha(rater_label_lists)

    mean_kappa = float(np.mean([r["cohens_kappa_category"] for r in pairwise_results.values()]))
    gate_b_passed = mean_kappa >= gates["gate_b_min_kappa"]

    report = {
        "n_shared_items": len(shared_ids),
        "pairwise": pairwise_results,
        "krippendorffs_alpha_category": alpha,
        "mean_cohens_kappa": mean_kappa,
        "gate_b_passed": gate_b_passed,
        "gate_b_threshold": gates["gate_b_min_kappa"],
    }

    out_path = _abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("Mean Cohen's kappa=%.3f, Krippendorff's alpha=%.3f -> Gate B %s",
              mean_kappa, alpha, "PASSED" if gate_b_passed else "FAILED")


if __name__ == "__main__":
    main()
