#!/usr/bin/env python
"""Attribution Patching reliability subset (Section 26, 44 Phase 3).

Computes the gradient-based Attribution Patching approximation D_i^AtP on a
stratified subset of ~50-60 failures (stratified by semantic_coarse category
so every major category is represented), for comparison against exact
patching in script 10.

Checkpoints each computed profile to results/logs/ as soon as it's produced;
a re-run of the same command skips pair_ids already checkpointed.

Usage:
    python scripts/09_compute_attribution_patching.py --benchmark gsm8k --model qwen2.5-3b-instruct
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_experiment_config, load_benchmark_config, load_model_config, read_jsonl, write_jsonl, _abs
from src.inference.model_wrapper import ModelWrapper
from src.inference.trace import render_prompt, split_into_steps
from src.mechanisms.scoring import build_prefix, question_token_offset, offset_positions
from src.mechanisms.attribution import attribution_patching_profile
from src.utils import ErrorTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("attribution_patching")


def stratified_subset(records, n_target: int, seed: int) -> list:
    by_cat = defaultdict(list)
    for r in records:
        if r.raw_causal_profile is not None and r.wrong_span and r.corrected_span:
            by_cat[r.semantic_coarse or "unlabeled"].append(r)
    rng = random.Random(seed)
    for cat in by_cat:
        rng.shuffle(by_cat[cat])

    cats = list(by_cat.keys())
    subset = []
    i = 0
    while len(subset) < n_target and any(by_cat.values()):
        cat = cats[i % len(cats)]
        if by_cat[cat]:
            subset.append(by_cat[cat].pop())
        i += 1
        if i > 10000:
            break
    return subset


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    done = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            done[row["pair_id"]] = row
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--n-subset", type=int, default=None)
    args = ap.parse_args()

    exp_cfg = load_experiment_config(args.config)
    bench_cfg = load_benchmark_config(args.benchmark)
    model_cfg = load_model_config(args.model)
    n_target = args.n_subset or exp_cfg["mechanisms"]["reliability_subset_n"]

    fail_pairs = read_jsonl(_abs(bench_cfg["pairs_path"]) / "pairs_annotated.jsonl")
    subset = stratified_subset(fail_pairs, n_target, exp_cfg["random_seed"])
    log.info("Selected %d failures for the AtP reliability subset", len(subset))

    log.info("Loading model %s ...", model_cfg["hf_id"])
    wrapper = ModelWrapper(model_cfg)
    log.info("Model loaded.")

    errors_path = _abs("results/logs") / f"09_compute_attribution_patching_{args.benchmark}_{model_cfg['name']}_errors.jsonl"
    tracker = ErrorTracker(errors_path)

    ckpt_path = _abs("results/logs") / f"09_checkpoint_{args.benchmark}_{model_cfg['name']}.jsonl"
    done = _load_checkpoint(ckpt_path)
    if done:
        log.info("Resuming: %d/%d items already checkpointed.", len(done), len(subset))

    # changed_clean_tokens / changed_fail_tokens are stored relative to the bare
    # question text (Section 10-11); rebase them onto build_prefix(...)'s full
    # rendered text before using them to index its activations.
    tok_offset = question_token_offset(bench_cfg["reasoning_prompt"], wrapper.tokenizer)

    n_resumed = 0
    for i, rec in enumerate(subset):
        if rec.pair_id in done:
            rec.causal_profile_atp = done[rec.pair_id]["causal_profile_atp"]
            n_resumed += 1
            log.info("[%d/%d] %s resumed from checkpoint", i + 1, len(subset), rec.pair_id)
            continue

        with tracker.guard(rec.pair_id, context="attribution_patching"):
            error_step = rec.first_observable_error_step
            clean_steps = split_into_steps(rec.clean_trace or "")
            fail_steps = split_into_steps(rec.fail_trace or "")
            clean_prefix = build_prefix(render_prompt(bench_cfg["reasoning_prompt"], rec.clean_question), clean_steps, error_step)
            fail_prefix = build_prefix(render_prompt(bench_cfg["reasoning_prompt"], rec.fail_question), fail_steps, error_step)

            D_atp = attribution_patching_profile(
                wrapper, clean_prefix, fail_prefix, rec.corrected_span, rec.wrong_span,
                offset_positions(rec.changed_clean_tokens, tok_offset),
                offset_positions(rec.changed_fail_tokens, tok_offset),
            )
            rec.causal_profile_atp = D_atp.tolist()
            with open(ckpt_path, "a") as f:
                f.write(json.dumps({"pair_id": rec.pair_id, "causal_profile_atp": rec.causal_profile_atp}) + "\n")
                f.flush()
        log.info("[%d/%d] %s done (resumed=%d, failed=%d)", i + 1, len(subset), rec.pair_id, n_resumed, tracker.n_failed)

    tracker.close()
    out_dir = _abs("data/mechanisms") / args.benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(subset, out_dir / "attribution_patching_subset.jsonl")
    log.info("Wrote %d attribution-patching profiles to %s (%d resumed, %d failed, see error log)",
              sum(1 for r in subset if r.causal_profile_atp is not None),
              out_dir / "attribution_patching_subset.jsonl", n_resumed, tracker.n_failed)


if __name__ == "__main__":
    main()
