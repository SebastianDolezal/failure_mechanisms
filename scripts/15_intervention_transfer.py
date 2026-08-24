#!/usr/bin/env python
"""H7 - functional intervention transfer (Section 37-39, 47 Fig 7).

For sampled failure pairs (A, B): identify A's top-K |D_A| layers, then
intervene on B using B's OWN clean activations but only at the locations
selected from A. A tells us WHERE to intervene; B's clean counterpart
supplies WHAT to insert - so no activation vector needs to transfer across
prompts. Compares against controls (Section 38): same-category source,
different-category source, random layer sets, and B's own top layers as an
approximate upper bound. Then fits the functional-validity regression
(Section 39): R_A->B ~ b0 + b1*C_AB + b2*1[T_A=T_B] + b3*X_AB.

Usage:
    python scripts/15_intervention_transfer.py --benchmark gsm8k --model qwen2.5-3b-instruct \
        --n-pairs 150 --top-k 5
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_experiment_config, load_benchmark_config, load_model_config, read_jsonl, _abs
from src.inference.model_wrapper import ModelWrapper
from src.inference.trace import render_prompt, split_into_steps
from src.mechanisms.scoring import build_prefix, question_token_offset, offset_positions
from src.mechanisms.patching import top_k_layers, multi_layer_patch_recovery
from src.mechanisms.similarity import cosine_sim
from src.matching.covariates import build_covariates, label_agnostic_candidate_pool
from src.statistics.regression import functional_validity_model
from src.utils import ErrorTracker
from src.plots.figures import fig7_intervention_transfer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("intervention_transfer")


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    done = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            done[row["pair_key"]] = row
    return done


def _prefix_and_continuations(bench_cfg, rec):
    error_step = rec.first_observable_error_step
    clean_steps = split_into_steps(rec.clean_trace or "")
    fail_steps = split_into_steps(rec.fail_trace or "")
    clean_prefix = build_prefix(render_prompt(bench_cfg["reasoning_prompt"], rec.clean_question), clean_steps, error_step)
    fail_prefix = build_prefix(render_prompt(bench_cfg["reasoning_prompt"], rec.fail_question), fail_steps, error_step)
    return clean_prefix, fail_prefix, rec.corrected_span, rec.wrong_span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--category-level", default="mid", choices=["coarse", "mid", "fine"])
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--n-pairs", type=int, default=150)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    exp_cfg = load_experiment_config(args.config)
    bench_cfg = load_benchmark_config(args.benchmark)
    model_cfg = load_model_config(args.model)

    records = read_jsonl(_abs(bench_cfg["pairs_path"]) / "pairs_annotated.jsonl")
    records = [r for r in records if r.failure_excess_signature is not None and r.wrong_span and r.corrected_span
                and getattr(r, f"semantic_{args.category_level}") is not None]
    base_ids = [r.base_id for r in records]
    covariates = [build_covariates(r) for r in records]
    D = [np.array(r.failure_excess_signature) for r in records]
    labels = [getattr(r, f"semantic_{args.category_level}") for r in records]
    n_neighbors = exp_cfg["matching"]["n_hard_negatives_per_item"] * 4
    candidate_pool = [label_agnostic_candidate_pool(i, covariates, base_ids, n_neighbors=n_neighbors)
                        for i in range(len(records))]

    rng = random.Random(exp_cfg["random_seed"])
    b_indices = [i for i in range(len(records)) if candidate_pool[i]]
    rng.shuffle(b_indices)
    b_indices = b_indices[: args.n_pairs]
    sampled_pairs = []
    for b_idx in b_indices:
        a_idx = rng.choice(candidate_pool[b_idx])
        sampled_pairs.append((a_idx, b_idx))

    log.info("Loading model %s ...", model_cfg["hf_id"])
    wrapper = ModelWrapper(model_cfg)
    log.info("Model loaded.")
    n_layers = model_cfg["n_layers"]

    errors_path = _abs("results/logs") / f"15_intervention_transfer_{args.benchmark}_{model_cfg['name']}_errors.jsonl"
    tracker = ErrorTracker(errors_path)

    ckpt_path = _abs("results/logs") / f"15_checkpoint_{args.benchmark}_{model_cfg['name']}.jsonl"
    done = _load_checkpoint(ckpt_path)
    if done:
        log.info("Resuming: %d/%d transfer pairs already checkpointed.", len(done), len(sampled_pairs))

    # changed_clean_tokens / changed_fail_tokens are stored relative to the bare
    # question text (Section 10-11); rebase them onto build_prefix(...)'s full
    # rendered text before using them to index its activations.
    tok_offset = question_token_offset(bench_cfg["reasoning_prompt"], wrapper.tokenizer)

    rows = []
    n_resumed = 0
    for pair_num, (a_idx, b_idx) in enumerate(sampled_pairs):
        pair_key = f"{base_ids[a_idx]}->{base_ids[b_idx]}"
        if pair_key in done:
            row = dict(done[pair_key])
            row.pop("pair_key", None)
            rows.append(row)
            n_resumed += 1
            log.info("[transfer %d/%d] %s resumed from checkpoint", pair_num + 1, len(sampled_pairs), pair_key)
            continue

        with tracker.guard(pair_key, context="intervention_transfer"):
            rec_a, rec_b = records[a_idx], records[b_idx]
            layers_A = top_k_layers(D[a_idx], args.top_k)
            random_layers = rng.sample(range(n_layers), min(args.top_k, n_layers))
            layers_B_own = top_k_layers(D[b_idx], args.top_k)

            clean_prefix_b, fail_prefix_b, correct_b, error_b = _prefix_and_continuations(bench_cfg, rec_b)

            def transfer(layers):
                result = multi_layer_patch_recovery(
                    wrapper, clean_prefix_b, fail_prefix_b, correct_b, error_b,
                    offset_positions(rec_b.changed_clean_tokens, tok_offset),
                    offset_positions(rec_b.changed_fail_tokens, tok_offset),
                    layers,
                )
                return result["R"]

            R_A_to_B = transfer(layers_A)
            R_random = transfer(random_layers)
            R_B_own_upper_bound = transfer(layers_B_own)

            C_AB = cosine_sim(D[a_idx], D[b_idx])
            same_cat = labels[a_idx] == labels[b_idx]

            row = {
                "a_idx": a_idx, "b_idx": b_idx, "a_base_id": base_ids[a_idx], "b_base_id": base_ids[b_idx],
                "R_transfer": R_A_to_B, "R_random_layers": R_random, "R_own_top_layers_upper_bound": R_B_own_upper_bound,
                "C_AB": C_AB, "same_category": same_cat,
                "edit_size_diff": abs(covariates[a_idx]["edit_size"] - covariates[b_idx]["edit_size"]),
                "problem_length_diff": abs(covariates[a_idx]["problem_length"] - covariates[b_idx]["problem_length"]),
                "cluster_group": base_ids[b_idx],
            }
            rows.append(row)
            with open(ckpt_path, "a") as f:
                f.write(json.dumps({"pair_key": pair_key, **row}) + "\n")
                f.flush()
        log.info("[transfer %d/%d] %s done (resumed=%d, failed=%d)",
                  pair_num + 1, len(sampled_pairs), pair_key, n_resumed, tracker.n_failed)

    tracker.close()
    if not rows:
        raise RuntimeError(f"All {len(sampled_pairs)} sampled transfer pairs failed - see {errors_path}")
    df = pd.DataFrame(rows)
    model_result = functional_validity_model(
        R_transfer=df["R_transfer"].values, C_AB=df["C_AB"].values,
        same_category=df["same_category"].values,
        X_AB=df[["edit_size_diff", "problem_length_diff"]],
        cluster_groups=df["cluster_group"].values,
    )

    out_dir = _abs(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "tables" / f"intervention_transfer_{args.benchmark}.csv", index=False)

    report = {
        "benchmark": args.benchmark, "n_pairs": len(df), "top_k": args.top_k,
        "mean_R_transfer": float(df["R_transfer"].mean()),
        "mean_R_random_layers": float(df["R_random_layers"].mean()),
        "mean_R_own_top_layers_upper_bound": float(df["R_own_top_layers_upper_bound"].mean()),
        "mean_R_same_category": float(df.loc[df["same_category"], "R_transfer"].mean()),
        "mean_R_diff_category": float(df.loc[~df["same_category"], "R_transfer"].mean()),
        "functional_validity_model": {k: v for k, v in model_result.items() if k != "summary"},
        "n_pairs_failed": tracker.n_failed,
        "error_log_path": str(errors_path),
    }
    out_path = out_dir / "reports" / f"intervention_transfer_{args.benchmark}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("mean R_transfer=%.3f (random=%.3f, own-top upper bound=%.3f); beta_C_AB=%.4f (p=%.4g)",
              report["mean_R_transfer"], report["mean_R_random_layers"], report["mean_R_own_top_layers_upper_bound"],
              model_result["beta_C_AB"], model_result["p_C_AB"])

    # Figure is generated last, after every statistical result and table is
    # already on disk, so a plotting issue can never discard a computed result.
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig7_intervention_transfer(df["C_AB"].values, df["R_transfer"].values, df["same_category"].values,
                                 str(out_dir / "figures" / f"fig7_intervention_transfer_{args.benchmark}.png"))


if __name__ == "__main__":
    main()
