#!/usr/bin/env python
"""Construct matched correct->correct perturbation control pairs
(Section 8). Writes data/pairs/<benchmark>/pairs_stable.jsonl using the same
PairRecord schema with pair_type == "stable_control".

Usage:
    python scripts/04_make_stable_controls.py --benchmark gsm8k --model qwen2.5-3b-instruct
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_benchmark_config, load_model_config, PairRecord, write_jsonl, read_jsonl, _abs
from src.matching.stable_pairs import select_stable_pairs
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stable_controls")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-per-family", type=int, default=2)
    args = ap.parse_args()

    bench_cfg = load_benchmark_config(args.benchmark)
    model_cfg = load_model_config(args.model)

    gens_path = _abs(bench_cfg["generations_path"]) / model_cfg["name"] / "generations.jsonl"
    gens = [json.loads(l) for l in open(gens_path)]

    # Restrict to base_ids that already have a primary failure pair, so every
    # stable control has a matched failure pair sharing the same underlying
    # problem and covariate context.
    primary_pairs = read_jsonl(_abs(bench_cfg["pairs_path"]) / "pairs_primary.jsonl")
    target_base_ids = {p.base_id for p in primary_pairs}

    by_base: dict[str, list[dict]] = defaultdict(list)
    gold_by_base: dict[str, str] = {}
    for g in gens:
        by_base[g["base_id"]].append(g)
        gold_by_base[g["base_id"]] = g["gold_answer"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_id"])

    stable_records = []
    for base_id in target_base_ids:
        variant_gens = by_base.get(base_id, [])
        candidates = select_stable_pairs(base_id, gold_by_base[base_id], variant_gens,
                                          tokenizer=tokenizer, max_pairs_per_family=args.max_per_family)
        for c in candidates:
            clean_g = next(g for g in variant_gens if g["variant_id"] == c.clean_variant_id)
            fail_g = next(g for g in variant_gens if g["variant_id"] == c.fail_variant_id)
            stable_records.append(PairRecord(
                pair_id=f"{base_id}_stable_{c.perturbation_family}_{c.fail_variant_id}",
                benchmark=args.benchmark,
                base_id=base_id,
                clean_question=c.clean_text,
                fail_question=c.fail_text,
                pair_type="stable_control",
                perturbation=c.perturbation_family,
                token_edit_distance=c.token_edit_distance,
                gold_answer=c.gold_answer,
                clean_answer=c.clean_answer,
                fail_answer=c.fail_answer,
                clean_trace=clean_g["raw_output"],
                fail_trace=fail_g["raw_output"],
                changed_clean_tokens=c.changed_clean_tokens,
                changed_fail_tokens=c.changed_fail_tokens,
                meta={"model": model_cfg["name"], "same_token_length": c.same_token_length},
            ))

    out_path = _abs(bench_cfg["pairs_path"]) / "pairs_stable.jsonl"
    write_jsonl(stable_records, out_path)
    log.info("Wrote %d stable control pairs (correct->correct) to %s", len(stable_records), out_path)


if __name__ == "__main__":
    main()
