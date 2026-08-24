#!/usr/bin/env python
"""Select the closest clean/failure realization pair for every underlying
problem (Section 10, 11). Writes data/pairs/<benchmark>/pairs.jsonl using the
canonical PairRecord schema (Section 42), with pair_type in
{induced_failure, rescued_failure}.

Usage:
    python scripts/03_select_pairs.py --benchmark gsm8k --model qwen2.5-3b-instruct \
        --prefer-token-aligned --target-n 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_benchmark_config, load_model_config, PairRecord, write_jsonl, _abs
from src.matching.pairs import select_clean_fail_pair
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("select_pairs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--target-n", type=int, default=None)
    ap.add_argument("--prefer-token-aligned", action="store_true",
                     help="drop pairs whose changed-token counts do not match 1:1 (secondary robustness set instead)")
    args = ap.parse_args()

    bench_cfg = load_benchmark_config(args.benchmark)
    model_cfg = load_model_config(args.model)
    target_n = args.target_n or bench_cfg["target_n_pairs"]

    gens_path = _abs(bench_cfg["generations_path"]) / model_cfg["name"] / "generations.jsonl"
    gens = [json.loads(l) for l in open(gens_path)]

    by_base: dict[str, list[dict]] = defaultdict(list)
    gold_by_base: dict[str, str] = {}
    for g in gens:
        by_base[g["base_id"]].append(g)
        gold_by_base[g["base_id"]] = g["gold_answer"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_id"])

    primary, secondary = [], []
    for base_id, variant_gens in by_base.items():
        candidate = select_clean_fail_pair(base_id, gold_by_base[base_id], variant_gens, tokenizer=tokenizer)
        if candidate is None:
            continue
        clean_g = next(g for g in variant_gens if g["variant_id"] == candidate.clean_variant_id)
        fail_g = next(g for g in variant_gens if g["variant_id"] == candidate.fail_variant_id)

        record = PairRecord(
            pair_id=f"{base_id}_pair",
            benchmark=args.benchmark,
            base_id=base_id,
            clean_question=candidate.clean_text,
            fail_question=candidate.fail_text,
            pair_type=candidate.pair_type,
            perturbation=candidate.perturbation_family,
            token_edit_distance=candidate.token_edit_distance,
            gold_answer=candidate.gold_answer,
            clean_answer=candidate.clean_answer,
            fail_answer=candidate.fail_answer,
            clean_trace=clean_g["raw_output"],
            fail_trace=fail_g["raw_output"],
            changed_clean_tokens=candidate.changed_clean_tokens,
            changed_fail_tokens=candidate.changed_fail_tokens,
            meta={"model": model_cfg["name"], "same_token_length": candidate.same_token_length},
        )

        aligned = len(candidate.changed_clean_tokens) == len(candidate.changed_fail_tokens) and len(candidate.changed_clean_tokens) > 0
        if aligned:
            primary.append(record)
        else:
            secondary.append(record)

        if args.prefer_token_aligned and len(primary) >= target_n:
            break
        if not args.prefer_token_aligned and (len(primary) + len(secondary)) >= target_n:
            break

    out_dir = _abs(bench_cfg["pairs_path"])
    write_jsonl(primary, out_dir / "pairs_primary.jsonl")
    write_jsonl(secondary, out_dir / "pairs_secondary_unaligned.jsonl")

    log.info("Selected %d primary (token-aligned) pairs, %d secondary (robustness) pairs. base problems seen=%d",
              len(primary), len(secondary), len(by_base))
    if len(primary) < bench_cfg["min_n_pairs"]:
        log.warning("Primary pair count %d is below the minimum reasonable sample (%d) - see Section 12.",
                    len(primary), bench_cfg["min_n_pairs"])


if __name__ == "__main__":
    main()
