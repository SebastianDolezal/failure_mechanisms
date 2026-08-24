#!/usr/bin/env python
"""Build and freeze the semantic taxonomy (Section 18). Only run after
script 06 reports Gate B passed. Uses ONLY the raw free-form descriptions
from one designated primary annotator (or an adjudicated merge) - no
activation / mechanistic information is used anywhere in this script
(Section 3, 18).

After freezing, merges the annotation (first_observable_error_step,
wrong_span, corrected_span, semantic_description, semantic_{coarse,mid,fine},
semantic_embedding, judge_confidence) back onto the pairs, producing
data/pairs/<benchmark>/pairs_annotated.jsonl - the frozen, immutable input to
every confirmatory analysis from script 08 onward.

Usage:
    python scripts/07_freeze_taxonomy.py --benchmark gsm8k --primary-annotator annotator_1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_experiment_config, load_benchmark_config, read_jsonl, write_jsonl, _abs
from src.taxonomy.embed import DescriptionEmbedder
from src.taxonomy.cluster import cluster_descriptions, build_hierarchy
from src.taxonomy.freeze import freeze_taxonomy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("freeze_taxonomy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--primary-annotator", required=True)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()

    exp_cfg = load_experiment_config(args.config)
    bench_cfg = load_benchmark_config(args.benchmark)
    tax_cfg = exp_cfg["taxonomy"]

    ann_path = _abs("data/annotations") / args.benchmark / f"{args.primary_annotator}.jsonl"
    annotations = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ann_path)}

    pair_ids = list(annotations.keys())
    descriptions = [annotations[pid]["description"] or "" for pid in pair_ids]

    embedder = DescriptionEmbedder(tax_cfg["embedding_model"])
    embeddings = embedder.embed(descriptions)

    clusterer, labels = cluster_descriptions(
        embeddings, min_cluster_size=tax_cfg["hdbscan"]["min_cluster_size"],
        min_samples=tax_cfg["hdbscan"]["min_samples"], metric=tax_cfg["hdbscan"]["metric"],
    )
    n_clusters = len(set(labels.tolist()) - {-1})
    n_noise = int((labels == -1).sum())
    log.info("HDBSCAN found %d clusters (%d noise points out of %d)", n_clusters, n_noise, len(labels))

    hierarchy = build_hierarchy(embeddings, descriptions, clusterer,
                                 min_cluster_size=tax_cfg["hdbscan"]["min_cluster_size"])

    base_ids = [ann_pid.rsplit("_pair", 1)[0] for ann_pid in pair_ids]
    freeze_info = freeze_taxonomy(hierarchy, descriptions, base_ids, embeddings,
                                   out_path=tax_cfg["frozen_path"], version=args.version)
    log.info("Froze taxonomy to %s (sha256=%s)", freeze_info["path"], freeze_info["sha256"])

    # --- merge annotation + taxonomy assignment back onto the pair records ---
    pairs = read_jsonl(_abs(bench_cfg["pairs_path"]) / "pairs_primary.jsonl")
    pair_by_id = {p.pair_id: p for p in pairs}

    for i, pid in enumerate(pair_ids):
        if pid not in pair_by_id:
            continue
        rec = pair_by_id[pid]
        ann = annotations[pid]
        rec.first_observable_error_step = ann.get("first_error_step")
        rec.wrong_span = ann.get("wrong_span")
        rec.corrected_span = ann.get("minimal_corrected_span")
        rec.semantic_description = ann.get("description")
        rec.judge_confidence = ann.get("confidence")
        rec.semantic_embedding = embeddings[i].tolist()
        for level in ("coarse", "mid", "fine"):
            info = hierarchy[level]["assignment"][i]
            setattr(rec, f"semantic_{level}", info["label"])

    annotated_out = _abs(bench_cfg["pairs_path"]) / "pairs_annotated.jsonl"
    write_jsonl(list(pair_by_id.values()), annotated_out)
    log.info("Wrote %d annotated pairs (with frozen taxonomy labels) to %s", len(pair_by_id), annotated_out)


if __name__ == "__main__":
    main()
