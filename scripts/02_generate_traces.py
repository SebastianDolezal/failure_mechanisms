#!/usr/bin/env python
"""Run deterministic model inference over every surface variant to produce
reasoning traces (Section 9). Writes
data/generations/<benchmark>/<model>/generations.jsonl.

Usage:
    python scripts/02_generate_traces.py --benchmark gsm8k --model qwen2.5-3b-instruct
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_benchmark_config, load_model_config, _abs
from src.inference.model_wrapper import ModelWrapper
from src.inference.trace import render_prompt, parse_final_answer, trace_is_well_structured
from src.utils import ErrorTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate_traces")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--model", required=True, help="model config name, e.g. qwen2.5-3b-instruct")
    ap.add_argument("--resume", action="store_true", help="skip variant_ids already in the output file")
    args = ap.parse_args()

    bench_cfg = load_benchmark_config(args.benchmark)
    model_cfg = load_model_config(args.model)

    variants_path = _abs(bench_cfg["variants_path"]) / "variants.jsonl"
    variants = [json.loads(l) for l in open(variants_path)]

    out_dir = _abs(bench_cfg["generations_path"]) / model_cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generations.jsonl"

    done_ids = set()
    if args.resume and out_path.exists():
        done_ids = {json.loads(l)["variant_id"] for l in open(out_path)}

    log.info("Loading model %s ...", model_cfg["hf_id"])
    wrapper = ModelWrapper(model_cfg)

    errors_path = _abs("results/logs") / f"02_generate_traces_{args.benchmark}_{model_cfg['name']}_errors.jsonl"
    tracker = ErrorTracker(errors_path)

    mode = "a" if args.resume else "w"
    n_ok, n_parsed = 0, 0
    with open(out_path, mode) as f:
        for i, v in enumerate(variants):
            if v["variant_id"] in done_ids:
                continue
            with tracker.guard(v["variant_id"], context="generate_trace"):
                prompt = render_prompt(bench_cfg["reasoning_prompt"], v["text"])
                raw_output = wrapper.generate(prompt)
                answer = parse_final_answer(raw_output)
                well_structured = trace_is_well_structured(raw_output)
                n_ok += 1
                n_parsed += int(answer is not None)

                rec = {
                    "variant_id": v["variant_id"], "base_id": v["base_id"], "text": v["text"],
                    "perturbation_family": v["perturbation_family"], "gold_answer": v["gold_answer"],
                    "is_original": v["meta"].get("is_original", False),
                    "raw_output": raw_output, "predicted_answer": answer,
                    "well_structured": well_structured, "model": model_cfg["name"],
                }
                f.write(json.dumps(rec) + "\n")
                log.info("[%d/%d] %s correct=%s", i + 1, len(variants), v["variant_id"], answer == v["gold_answer"])
            if (i + 1) % 50 == 0:
                f.flush()
                log.info("-- progress checkpoint: %d/%d processed, parse_rate so far=%.3f, failures so far=%d --",
                          i + 1, len(variants), n_parsed / max(n_ok, 1), tracker.n_failed)

    tracker.close()
    log.info("Done. %d generations written to %s (answer-parse rate=%.3f)",
              n_ok, out_path, n_parsed / max(n_ok, 1))


if __name__ == "__main__":
    main()
