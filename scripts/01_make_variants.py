#!/usr/bin/env python
"""Generate meaning-preserving surface variants for every problem in a
benchmark (Section 7). Writes data/variants/<benchmark>/variants.jsonl.

Usage:
    python scripts/01_make_variants.py --benchmark gsm8k --variants 8
    python scripts/01_make_variants.py --benchmark svamp --variants 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_benchmark_config, _abs
from src.datasets.loaders import load_gsm8k, load_svamp
from src.perturb.variants import build_variant_bank

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_variants")

LOADERS = {"gsm8k": load_gsm8k, "svamp": load_svamp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--variants", type=int, default=None, help="override n_variants_per_problem")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap number of base problems (debug)")
    args = ap.parse_args()

    cfg = load_benchmark_config(args.benchmark)
    k = args.variants or cfg["n_variants_per_problem"]
    problems = LOADERS[args.benchmark](cfg)
    if args.limit:
        problems = problems[: args.limit]

    out_path = _abs(cfg["variants_path"]) / "variants.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(out_path, "w") as f:
        for problem in problems:
            bank = build_variant_bank(
                problem["base_id"], problem["question"], cfg["perturbation_families"],
                k=k, seed=args.seed, tokenizer=None,
            )
            for v in bank:
                rec = {
                    "variant_id": v.variant_id, "base_id": v.base_id, "text": v.text,
                    "perturbation_family": v.perturbation_family, "meta": v.meta,
                    "gold_answer": problem["gold_answer"],
                }
                f.write(json.dumps(rec) + "\n")
                n_written += 1
        f.flush()

    log.info("Wrote %d variants (%d base problems x up to %d variants) to %s",
              n_written, len(problems), k, out_path)


if __name__ == "__main__":
    main()
