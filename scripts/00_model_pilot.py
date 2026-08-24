#!/usr/bin/env python
"""Phase 0 - model-measurement pilot (Section 4, 44 Phase 0).

Evaluates each candidate model in configs/experiment.yaml:pilot.models on a
small number of GSM8K flips and reports whether it clears the measurement
gates (flip availability, trace structure, non-degenerate patching profiles,
exact-vs-attribution-patching reliability). Does NOT look at the semantic
correspondence hypothesis - selection is purely on measurement quality
(Section 44: "Do not inspect the semantic correspondence hypothesis during
model selection.").

Usage:
    python scripts/00_model_pilot.py --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_experiment_config, load_benchmark_config, load_model_config, _abs
from src.datasets.loaders import load_gsm8k
from src.perturb.variants import build_variant_bank
from src.inference.model_wrapper import ModelWrapper
from src.inference.trace import render_prompt, parse_final_answer, split_into_steps, trace_is_well_structured
from src.matching.pairs import select_clean_fail_pair
from src.matching.stable_pairs import select_stable_pairs
from src.mechanisms.scoring import build_prefix, question_token_offset, offset_positions
from src.mechanisms.patching import exact_layer_patching_profile
from src.mechanisms.attribution import attribution_patching_profile
from src.mechanisms.similarity import spearman_sim
from src.utils import ErrorTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pilot")


def _pick_error_step(trace_text: str) -> int:
    """Pilot heuristic (no human annotation yet): treat the step just before
    the midpoint of the trace as the first observable error. Replaced by
    annotated first_observable_error_step everywhere downstream of script 05.
    """
    steps = split_into_steps(trace_text)
    if not steps:
        return 1
    return steps[max(0, len(steps) // 2 - 1)]["index"]


def run_pilot_for_model(model_cfg: dict, gsm8k_cfg: dict, n_pairs: int, seed: int) -> dict:
    log.info("Loading model %s", model_cfg["name"])
    wrapper = ModelWrapper(model_cfg)
    rng = random.Random(seed)

    problems = load_gsm8k(gsm8k_cfg)
    rng.shuffle(problems)

    n_attempted = 0
    n_flips = 0
    n_parsed = 0
    n_traces = 0
    pairs = []
    stable_pairs_all = []

    gen_errors_path = _abs("results/logs") / f"00_model_pilot_{model_cfg['name']}_generation_errors.jsonl"
    gen_tracker = ErrorTracker(gen_errors_path)

    for problem in problems:
        if len(pairs) >= n_pairs:
            break
        n_attempted += 1
        with gen_tracker.guard(problem["base_id"], context="pilot_generation"):
            bank = build_variant_bank(
                problem["base_id"], problem["question"], gsm8k_cfg["perturbation_families"],
                k=gsm8k_cfg["n_variants_per_problem"], seed=seed, tokenizer=wrapper.tokenizer,
            )
            gens = []
            for v in bank:
                prompt = render_prompt(gsm8k_cfg["reasoning_prompt"], v.text)
                out = wrapper.generate(prompt)
                n_traces += 1
                ans = parse_final_answer(out)
                if ans is not None:
                    n_parsed += 1
                log.info("[pilot generation, pair %d/%d attempted] %s correct=%s",
                          n_attempted, n_pairs, v.variant_id, ans == problem["gold_answer"])
                gens.append({
                    "variant_id": v.variant_id, "text": v.text, "raw_output": out,
                    "predicted_answer": ans, "perturbation_family": v.perturbation_family,
                    "is_original": v.meta.get("is_original", False),
                })

            pair = select_clean_fail_pair(problem["base_id"], problem["gold_answer"], gens, tokenizer=wrapper.tokenizer)
            if pair is not None:
                n_flips += 1
                clean_gen = next(g for g in gens if g["variant_id"] == pair.clean_variant_id)
                fail_gen = next(g for g in gens if g["variant_id"] == pair.fail_variant_id)
                pairs.append((pair, clean_gen, fail_gen, problem))
                stable_pairs_all.extend(
                    select_stable_pairs(problem["base_id"], problem["gold_answer"], gens, tokenizer=wrapper.tokenizer)
                )

    gen_tracker.close()
    flip_rate = n_flips / max(n_attempted, 1)
    trace_parse_rate = n_parsed / max(n_traces, 1)

    log.info("model=%s attempted=%d flips=%d flip_rate=%.3f parse_rate=%.3f",
              model_cfg["name"], n_attempted, n_flips, flip_rate, trace_parse_rate)

    patch_errors_path = _abs("results/logs") / f"00_model_pilot_{model_cfg['name']}_patching_errors.jsonl"
    patch_tracker = ErrorTracker(patch_errors_path)

    # changed_clean_tokens / changed_fail_tokens are stored relative to the bare
    # question text (Section 10-11); rebase them onto build_prefix(...)'s full
    # rendered text before using them to index its activations.
    tok_offset = question_token_offset(gsm8k_cfg["reasoning_prompt"], wrapper.tokenizer)

    nondeg_fracs = []
    exact_profiles = []
    atp_profiles = []
    pilot_patch_items = pairs[: min(len(pairs), 20)]
    for pi, (pair, clean_gen, fail_gen, problem) in enumerate(pilot_patch_items):
        with patch_tracker.guard(pair.base_id, context="pilot_patching"):
            error_step = _pick_error_step(fail_gen["raw_output"])
            fail_steps = split_into_steps(fail_gen["raw_output"])
            clean_steps = split_into_steps(clean_gen["raw_output"])
            clean_prefix = build_prefix(render_prompt(gsm8k_cfg["reasoning_prompt"], pair.clean_text), clean_steps, error_step)
            fail_prefix = build_prefix(render_prompt(gsm8k_cfg["reasoning_prompt"], pair.fail_text), fail_steps, error_step)

            error_step_obj = next((s for s in fail_steps if s["index"] == error_step), None)
            correct_step_obj = next((s for s in clean_steps if s["index"] == error_step), None)
            if error_step_obj is None or correct_step_obj is None:
                continue
            correct_cont = correct_step_obj["text"]
            error_cont = error_step_obj["text"]

            result = exact_layer_patching_profile(
                wrapper, clean_prefix, fail_prefix, correct_cont, error_cont,
                offset_positions(pair.changed_clean_tokens, tok_offset),
                offset_positions(pair.changed_fail_tokens, tok_offset),
            )
            R = result["R"]
            nondeg_fracs.append(float(np.mean(np.abs(R) > 0.05)))
            exact_profiles.append(R)

            atp = attribution_patching_profile(
                wrapper, clean_prefix, fail_prefix, correct_cont, error_cont,
                offset_positions(pair.changed_clean_tokens, tok_offset),
                offset_positions(pair.changed_fail_tokens, tok_offset),
            )
            atp_profiles.append(atp)
        log.info("[pilot patching %d/%d] %s done (failed=%d)",
                  pi + 1, len(pilot_patch_items), pair.base_id, patch_tracker.n_failed)

    patch_tracker.close()
    mean_nondeg = float(np.mean(nondeg_fracs)) if nondeg_fracs else 0.0
    rhos = [spearman_sim(e, a) for e, a in zip(exact_profiles, atp_profiles)]
    mean_rho = float(np.nanmean(rhos)) if rhos else float("nan")

    return {
        "model": model_cfg["name"],
        "n_attempted": n_attempted,
        "n_flips": n_flips,
        "flip_rate": flip_rate,
        "trace_parse_rate": trace_parse_rate,
        "mean_nondegenerate_fraction": mean_nondeg,
        "mean_exact_atp_spearman": mean_rho,
        "n_reliability_examples": len(rhos),
        "generation_errors": gen_tracker.summary(),
        "patching_errors": patch_tracker.summary(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--gsm8k-config", default="configs/gsm8k.yaml")
    ap.add_argument("--out", default="results/reports/pilot_report.json")
    args = ap.parse_args()

    exp_cfg = load_experiment_config(args.config)
    gsm8k_cfg = load_benchmark_config(args.gsm8k_config)
    pilot_cfg = exp_cfg["pilot"]
    gates = pilot_cfg["gates"]

    reports = []
    for model_path in pilot_cfg["models"]:
        model_cfg = load_model_config(model_path)
        report = run_pilot_for_model(model_cfg, gsm8k_cfg, pilot_cfg["n_pairs_per_model"], exp_cfg["random_seed"])
        report["gates_passed"] = {
            "flip_rate": report["flip_rate"] >= gates["min_flip_rate"],
            "trace_parse_rate": report["trace_parse_rate"] >= gates["min_trace_parse_rate"],
            "nondegenerate": report["mean_nondegenerate_fraction"] >= gates["min_nondegenerate_frac"],
            "exact_atp_reliability": (report["mean_exact_atp_spearman"] >= gates["min_exact_atp_spearman"]
                                        if report["mean_exact_atp_spearman"] == report["mean_exact_atp_spearman"] else False),
        }
        report["all_gates_passed"] = all(report["gates_passed"].values())
        reports.append(report)

    eligible = [r for r in reports if r["all_gates_passed"]]
    selection = eligible[0]["model"] if eligible else None

    out_path = _abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"reports": reports, "selected_model": selection}, f, indent=2)

    log.info("Pilot complete. Selected model: %s", selection)
    if selection is None:
        log.warning("No candidate model cleared all measurement gates - inspect %s before proceeding.", out_path)


if __name__ == "__main__":
    main()
