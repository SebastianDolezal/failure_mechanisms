#!/usr/bin/env python
"""Exact clean->failed activation patching (Section 22-23, 27).

For every annotated failure pair, computes the layer-level recovery profile
R_i using the frozen first_observable_error_step / wrong_span /
corrected_span annotations as (P_i, c_i^-, c_i^+). Also computes R for every
stable (correct->correct) control pair using the SAME error-step index (the
matched original problem's error step) as the local target, so stable and
failure profiles are computed with an identical procedure. Finally estimates
the failure-excess signature D_i = R_i - B_i (Section 23).

Each computed profile is checkpointed to results/logs/ as soon as it's
produced, and a re-run of the same command skips pair_ids already
checkpointed - the same resumability pattern scripts 02 and 05 already use.
This is the most expensive stage in the pipeline (one forward pass per layer
per pair), so a crash or interruption partway through only costs the time
since the last item, not the whole stage.

Usage:
    python scripts/08_compute_exact_patching.py --benchmark gsm8k \
        --model qwen2.5-3b-instruct --site resid_pre --metric error_boundary_margin
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_benchmark_config, load_model_config, read_jsonl, write_jsonl, _abs
from src.inference.model_wrapper import ModelWrapper
from src.inference.trace import render_prompt, split_into_steps
from src.mechanisms.scoring import build_prefix, question_token_offset, offset_positions
from src.mechanisms.patching import exact_layer_patching_profile
from src.mechanisms.signature import aggregate_stable_baseline, compute_failure_excess_signature
from src.matching.covariates import build_covariates
from src.utils import ErrorTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("exact_patching")

FAIL_META_KEYS = ["patch_site", "metric", "m_clean", "m_fail", "token_aligned", "error_step_used"]
STABLE_META_KEYS = ["patch_site", "error_step_used"]


def _continuation_pair(rec) -> tuple[str | None, str | None]:
    correct = rec.corrected_span
    error = rec.wrong_span
    if not correct or not error:
        return None, None
    return correct, error


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    done = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            done[row["pair_id"]] = row
    return done


def _append_checkpoint(path: Path, pair_id: str, raw_causal_profile: list, meta: dict) -> None:
    row = {"pair_id": pair_id, "raw_causal_profile": raw_causal_profile, **meta}
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def _apply_checkpoint_row(rec, row: dict, meta_keys: list[str]) -> None:
    rec.raw_causal_profile = row["raw_causal_profile"]
    for k in meta_keys:
        rec.meta[k] = row[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--site", default="resid_pre")
    ap.add_argument("--metric", default="error_boundary_margin")
    args = ap.parse_args()

    bench_cfg = load_benchmark_config(args.benchmark)
    model_cfg = load_model_config(args.model)

    log.info("Loading model %s ...", model_cfg["hf_id"])
    wrapper = ModelWrapper(model_cfg)
    log.info("Model loaded.")

    fail_pairs = read_jsonl(_abs(bench_cfg["pairs_path"]) / "pairs_annotated.jsonl")
    stable_pairs = read_jsonl(_abs(bench_cfg["pairs_path"]) / "pairs_stable.jsonl")

    errors_path = _abs("results/logs") / f"08_compute_exact_patching_{args.benchmark}_{model_cfg['name']}_errors.jsonl"
    tracker = ErrorTracker(errors_path)

    fail_ckpt_path = _abs("results/logs") / f"08_checkpoint_fail_{args.benchmark}_{model_cfg['name']}.jsonl"
    stable_ckpt_path = _abs("results/logs") / f"08_checkpoint_stable_{args.benchmark}_{model_cfg['name']}.jsonl"
    done_fail = _load_checkpoint(fail_ckpt_path)
    done_stable = _load_checkpoint(stable_ckpt_path)
    if done_fail or done_stable:
        log.info("Resuming: %d/%d failure pairs and %d/%d stable pairs already checkpointed.",
                  len(done_fail), len(fail_pairs), len(done_stable), len(stable_pairs))

    # changed_clean_tokens / changed_fail_tokens are stored relative to the bare
    # question text (Section 10-11); rebase them onto build_prefix(...)'s full
    # rendered text before using them to index its activations.
    tok_offset = question_token_offset(bench_cfg["reasoning_prompt"], wrapper.tokenizer)

    # --- failure pairs -------------------------------------------------
    n_skipped = 0
    n_resumed = 0
    stage_start = time.monotonic()
    for i, rec in enumerate(fail_pairs):
        if rec.pair_id in done_fail:
            _apply_checkpoint_row(rec, done_fail[rec.pair_id], FAIL_META_KEYS)
            n_resumed += 1
            continue

        correct_cont, error_cont = _continuation_pair(rec)
        error_step = rec.first_observable_error_step
        if correct_cont is None or error_step is None:
            n_skipped += 1
            continue

        with tracker.guard(rec.pair_id, context="exact_patching_failure_pair"):
            clean_steps = split_into_steps(rec.clean_trace or "")
            fail_steps = split_into_steps(rec.fail_trace or "")
            clean_prefix = build_prefix(render_prompt(bench_cfg["reasoning_prompt"], rec.clean_question), clean_steps, error_step)
            fail_prefix = build_prefix(render_prompt(bench_cfg["reasoning_prompt"], rec.fail_question), fail_steps, error_step)

            result = exact_layer_patching_profile(
                wrapper, clean_prefix, fail_prefix, correct_cont, error_cont,
                offset_positions(rec.changed_clean_tokens, tok_offset),
                offset_positions(rec.changed_fail_tokens, tok_offset),
            )
            rec.raw_causal_profile = result["R"].tolist()
            rec.meta["patch_site"] = args.site
            rec.meta["metric"] = args.metric
            rec.meta["m_clean"] = result["m_clean"]
            rec.meta["m_fail"] = result["m_fail"]
            rec.meta["token_aligned"] = result["aligned"]
            rec.meta["error_step_used"] = error_step
            _append_checkpoint(fail_ckpt_path, rec.pair_id, rec.raw_causal_profile,
                                {k: rec.meta[k] for k in FAIL_META_KEYS})

        elapsed = time.monotonic() - stage_start
        log.info("[failure pairs %d/%d] %s done (resumed=%d, skipped=%d, failed=%d, elapsed=%.0fs)",
                  i + 1, len(fail_pairs), rec.pair_id, n_resumed, n_skipped, tracker.n_failed, elapsed)

    n_failed_fail_pairs = tracker.n_failed
    log.info("Computed raw causal profiles for %d/%d failure pairs (%d resumed from checkpoint, %d skipped: "
              "missing annotation, %d failed: see error log)",
              len(fail_pairs) - n_skipped - n_failed_fail_pairs, len(fail_pairs), n_resumed, n_skipped, n_failed_fail_pairs)

    # --- stable pairs: use the matched failure pair's error step for the
    # same base_id as the local target index, so R^stable is measured with
    # an identical procedure to R^fail (Section 23). -----------------------
    error_step_by_base = {p.base_id: p.first_observable_error_step for p in fail_pairs
                            if p.first_observable_error_step is not None}
    wrong_span_by_base = {p.base_id: p.wrong_span for p in fail_pairs if p.wrong_span}
    corrected_span_by_base = {p.base_id: p.corrected_span for p in fail_pairs if p.corrected_span}

    n_stable_skipped = 0
    n_stable_resumed = 0
    stage_start = time.monotonic()
    for i, rec in enumerate(stable_pairs):
        if rec.pair_id in done_stable:
            _apply_checkpoint_row(rec, done_stable[rec.pair_id], STABLE_META_KEYS)
            n_stable_resumed += 1
            continue

        error_step = error_step_by_base.get(rec.base_id)
        correct_cont = corrected_span_by_base.get(rec.base_id)
        error_cont = wrong_span_by_base.get(rec.base_id)
        if error_step is None or correct_cont is None or error_cont is None:
            n_stable_skipped += 1
            continue

        with tracker.guard(rec.pair_id, context="exact_patching_stable_pair"):
            clean_steps = split_into_steps(rec.clean_trace or "")
            fail_steps = split_into_steps(rec.fail_trace or "")
            clean_prefix = build_prefix(render_prompt(bench_cfg["reasoning_prompt"], rec.clean_question), clean_steps, error_step)
            fail_prefix = build_prefix(render_prompt(bench_cfg["reasoning_prompt"], rec.fail_question), fail_steps, error_step)

            result = exact_layer_patching_profile(
                wrapper, clean_prefix, fail_prefix, correct_cont, error_cont,
                offset_positions(rec.changed_clean_tokens, tok_offset),
                offset_positions(rec.changed_fail_tokens, tok_offset),
            )
            rec.raw_causal_profile = result["R"].tolist()
            rec.meta["patch_site"] = args.site
            rec.meta["error_step_used"] = error_step
            _append_checkpoint(stable_ckpt_path, rec.pair_id, rec.raw_causal_profile,
                                {k: rec.meta[k] for k in STABLE_META_KEYS})

        elapsed = time.monotonic() - stage_start
        log.info("[stable pairs %d/%d] %s done (resumed=%d, skipped=%d, failed=%d, elapsed=%.0fs)",
                  i + 1, len(stable_pairs), rec.pair_id, n_stable_resumed, n_stable_skipped,
                  tracker.n_failed - n_failed_fail_pairs, elapsed)

    n_failed_stable_pairs = tracker.n_failed - n_failed_fail_pairs
    log.info("Computed raw causal profiles for %d/%d stable pairs (%d resumed from checkpoint, %d skipped: "
              "no matched failure annotation, %d failed: see error log)",
              len(stable_pairs) - n_stable_skipped - n_failed_stable_pairs, len(stable_pairs),
              n_stable_resumed, n_stable_skipped, n_failed_stable_pairs)

    # --- failure-excess signature D_i = R_i - B_i (Section 23) ------------
    stable_records = []
    for rec in stable_pairs:
        if rec.raw_causal_profile is None:
            continue
        cov = build_covariates(rec)
        stable_records.append({"covariates": cov, "R": np.array(rec.raw_causal_profile)})

    n_signatures = sum(1 for r in fail_pairs if r.raw_causal_profile is not None)
    log.info("Computing failure-excess signatures for %d failures against %d stable baselines...",
              n_signatures, len(stable_records))
    n_sig_done = 0
    for rec in fail_pairs:
        if rec.raw_causal_profile is None:
            continue
        with tracker.guard(rec.pair_id, context="failure_excess_signature"):
            cov = build_covariates(rec)
            B_i = aggregate_stable_baseline(cov, stable_records, k=8)
            D_i = compute_failure_excess_signature(np.array(rec.raw_causal_profile), B_i)
            rec.stable_pair_baseline = B_i.tolist()
            rec.failure_excess_signature = D_i.tolist()
        n_sig_done += 1
        if n_sig_done % 20 == 0 or n_sig_done == n_signatures:
            log.info("[failure-excess signatures %d/%d]", n_sig_done, n_signatures)

    tracker.close()
    out_dir = _abs("data/mechanisms") / args.benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing outputs to %s and %s ...", _abs(bench_cfg["pairs_path"]), out_dir)
    write_jsonl(fail_pairs, _abs(bench_cfg["pairs_path"]) / "pairs_annotated.jsonl")
    write_jsonl(stable_pairs, _abs(bench_cfg["pairs_path"]) / "pairs_stable.jsonl")
    write_jsonl(fail_pairs, out_dir / "failure_signatures.jsonl")
    write_jsonl(stable_pairs, out_dir / "stable_profiles.jsonl")

    log.info("Wrote failure-excess signatures for %d failures to %s",
              sum(1 for r in fail_pairs if r.failure_excess_signature is not None), out_dir)


if __name__ == "__main__":
    main()
