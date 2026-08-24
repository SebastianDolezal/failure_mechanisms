#!/usr/bin/env python
"""Frozen SVAMP replication (Section 6, 40, 44 Phase 6).

Prerequisite: scripts 01-04 have already been run with --benchmark svamp to
produce data/variants/svamp, data/generations/svamp/<model>, and
data/pairs/svamp/{pairs_primary,pairs_stable}.jsonl. This script must not be
run before the GSM8K taxonomy, causal metrics, matching criteria, and
statistical tests are all frozen (Section 6).

Pipeline (all frozen from the GSM8K discovery phase - nothing here is
re-tuned):
  1. Annotate SVAMP failures with the frozen annotation prompt (model judge).
  2. Map each raw description onto the FROZEN GSM8K taxonomy via nearest-
     centroid matching (src.taxonomy.freeze.map_description_to_taxonomy);
     descriptions that don't fit any existing cluster are labeled
     "novel/unmapped" rather than force-fit (Section 40).
  3. Compute exact-patching failure-excess signatures with the same
     pipeline as script 08.
  4. Recompute the same continuous + categorical correspondence statistics
     as script 11, restricted to items that mapped onto the frozen taxonomy.

Usage:
    python scripts/16_replicate_svamp.py --model qwen2.5-3b-instruct
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_experiment_config, load_benchmark_config, load_model_config, PairRecord, read_jsonl, write_jsonl, _abs
from src.inference.model_wrapper import ModelWrapper
from src.inference.trace import render_prompt, split_into_steps
from src.mechanisms.scoring import build_prefix, question_token_offset, offset_positions
from src.mechanisms.patching import exact_layer_patching_profile
from src.mechanisms.signature import aggregate_stable_baseline, compute_failure_excess_signature
from src.mechanisms.similarity import pairwise_similarity_matrix
from src.matching.covariates import build_covariates, label_agnostic_candidate_pool
from src.taxonomy.embed import DescriptionEmbedder
from src.taxonomy.freeze import load_taxonomy, map_description_to_taxonomy
from src.statistics.permutation import compute_delta, permutation_test_delta
from src.statistics.bootstrap import bootstrap_ci_by_problem
from src.statistics.regression import continuous_correspondence_regression

from importlib import import_module

annotate_mod = import_module("05_annotate_failures")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("replicate_svamp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                     help="TARGET model under study - used for trace generation (already run via "
                          "scripts 01-04) and for exact activation patching here. Must match whichever "
                          "model produced data/generations/svamp/<model>/.")
    ap.add_argument("--judge-model", default="configs/models/smollm2-1.7b-instruct.yaml",
                     help="Semantic-annotation judge - deliberately NOT --model, so SVAMP annotation "
                          "doesn't reintroduce the self-judging problem the GSM8K phase avoided "
                          "(Section 15). Defaults to the same judge_a used on GSM8K for consistency.")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--category-level", default="mid", choices=["coarse", "mid", "fine"])
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    exp_cfg = load_experiment_config(args.config)
    bench_cfg = load_benchmark_config("svamp")
    model_cfg = load_model_config(args.model)
    judge_cfg = load_model_config(args.judge_model)

    pairs = read_jsonl(_abs(bench_cfg["pairs_path"]) / "pairs_primary.jsonl")
    stable_pairs = read_jsonl(_abs(bench_cfg["pairs_path"]) / "pairs_stable.jsonl")
    fail_pairs = [p for p in pairs if p.pair_type in ("induced_failure", "rescued_failure")]
    log.info("Loaded %d SVAMP failure pairs, %d stable controls", len(fail_pairs), len(stable_pairs))

    # --- 1. annotate with the frozen judge prompt, using the SAME judge
    # model (family) as GSM8K, never the target model (Section 15). ---------
    judge_wrapper = ModelWrapper(judge_cfg)
    annotations = {}
    for idx, p in enumerate(fail_pairs):
        prompt = annotate_mod.ANNOTATION_PROMPT.format(
            question=p.fail_question, gold_solution="", correct_answer=p.gold_answer,
            trace=p.fail_trace, model_answer=p.fail_answer,
        )
        raw = judge_wrapper.generate(prompt, max_new_tokens=300)
        parsed = annotate_mod._parse_json_annotation(raw)
        log.info("[%d/%d] %s parsed=%s", idx + 1, len(fail_pairs), p.pair_id, parsed is not None)
        if parsed is None:
            continue
        annotations[p.pair_id] = parsed
    del judge_wrapper  # free the judge model's memory before loading the target model below

    # --- target model, used only from here on for activation patching ------
    wrapper = ModelWrapper(model_cfg)

    # --- 2. map descriptions onto the frozen GSM8K taxonomy -----------------
    taxonomy = load_taxonomy(bench_cfg["taxonomy_path"])
    embedder = DescriptionEmbedder(exp_cfg["taxonomy"]["embedding_model"])

    mapped = []
    for p in fail_pairs:
        ann = annotations.get(p.pair_id)
        if ann is None:
            continue
        p.first_observable_error_step = ann.get("first_error_step")
        p.wrong_span = ann.get("wrong_span")
        p.corrected_span = ann.get("minimal_corrected_span")
        p.semantic_description = ann.get("description")
        p.judge_confidence = ann.get("confidence")
        if not p.semantic_description:
            continue
        emb = embedder.embed_one(p.semantic_description)
        p.semantic_embedding = emb.tolist()
        for level in ("coarse", "mid", "fine"):
            mapping = map_description_to_taxonomy(emb, taxonomy, level=level,
                                                    novelty_threshold=exp_cfg["taxonomy"].get("novelty_threshold", 0.35))
            setattr(p, f"semantic_{level}", mapping["label"])
        mapped.append(p)

    n_novel = sum(1 for p in mapped if getattr(p, f"semantic_{args.category_level}") == "novel/unmapped")
    log.info("%d/%d SVAMP failures mapped onto the frozen taxonomy (%d novel/unmapped at level=%s)",
              len(mapped) - n_novel, len(mapped), n_novel, args.category_level)

    # --- 3. exact-patching failure-excess signatures (same pipeline as 08) --
    # changed_clean_tokens / changed_fail_tokens are stored relative to the bare
    # question text (Section 10-11); rebase them onto build_prefix(...)'s full
    # rendered text before using them to index its activations.
    tok_offset = question_token_offset(bench_cfg["reasoning_prompt"], wrapper.tokenizer)
    for rec in mapped:
        if not (rec.wrong_span and rec.corrected_span and rec.first_observable_error_step):
            continue
        clean_steps = split_into_steps(rec.clean_trace or "")
        fail_steps = split_into_steps(rec.fail_trace or "")
        clean_prefix = build_prefix(render_prompt(bench_cfg["reasoning_prompt"], rec.clean_question), clean_steps, rec.first_observable_error_step)
        fail_prefix = build_prefix(render_prompt(bench_cfg["reasoning_prompt"], rec.fail_question), fail_steps, rec.first_observable_error_step)
        result = exact_layer_patching_profile(
            wrapper, clean_prefix, fail_prefix, rec.corrected_span, rec.wrong_span,
            offset_positions(rec.changed_clean_tokens, tok_offset),
            offset_positions(rec.changed_fail_tokens, tok_offset),
        )
        rec.raw_causal_profile = result["R"].tolist()

    error_step_by_base = {p.base_id: p.first_observable_error_step for p in mapped if p.first_observable_error_step}
    corrected_by_base = {p.base_id: p.corrected_span for p in mapped if p.corrected_span}
    wrong_by_base = {p.base_id: p.wrong_span for p in mapped if p.wrong_span}
    for rec in stable_pairs:
        es = error_step_by_base.get(rec.base_id)
        cc, ec = corrected_by_base.get(rec.base_id), wrong_by_base.get(rec.base_id)
        if es is None or cc is None or ec is None:
            continue
        clean_steps = split_into_steps(rec.clean_trace or "")
        fail_steps = split_into_steps(rec.fail_trace or "")
        clean_prefix = build_prefix(render_prompt(bench_cfg["reasoning_prompt"], rec.clean_question), clean_steps, es)
        fail_prefix = build_prefix(render_prompt(bench_cfg["reasoning_prompt"], rec.fail_question), fail_steps, es)
        result = exact_layer_patching_profile(
            wrapper, clean_prefix, fail_prefix, cc, ec,
            offset_positions(rec.changed_clean_tokens, tok_offset),
            offset_positions(rec.changed_fail_tokens, tok_offset),
        )
        rec.raw_causal_profile = result["R"].tolist()

    stable_records = [{"covariates": build_covariates(r), "R": np.array(r.raw_causal_profile)}
                        for r in stable_pairs if r.raw_causal_profile is not None]
    for rec in mapped:
        if rec.raw_causal_profile is None:
            continue
        B_i = aggregate_stable_baseline(build_covariates(rec), stable_records, k=8)
        rec.failure_excess_signature = compute_failure_excess_signature(np.array(rec.raw_causal_profile), B_i).tolist()

    write_jsonl(mapped, _abs(bench_cfg["pairs_path"]) / "pairs_annotated.jsonl")

    # --- 4. confirmatory correspondence test on frozen structure ------------
    valid = [r for r in mapped if r.failure_excess_signature is not None
              and getattr(r, f"semantic_{args.category_level}") not in (None, "novel/unmapped")]
    log.info("%d SVAMP failures usable for the confirmatory correspondence test", len(valid))

    if len(valid) < 10:
        log.warning("Too few mapped, signed failures for a meaningful replication test; stopping.")
        return

    base_ids = [r.base_id for r in valid]
    labels = [getattr(r, f"semantic_{args.category_level}") for r in valid]
    covariates = [build_covariates(r) for r in valid]
    D = [np.array(r.failure_excess_signature) for r in valid]
    S = np.array([r.semantic_embedding for r in valid])

    C_mech = pairwise_similarity_matrix(D, metric="cosine")
    S_sem = S @ S.T / (np.linalg.norm(S, axis=1, keepdims=True) @ np.linalg.norm(S, axis=1, keepdims=True).T + 1e-8)
    n_neighbors = exp_cfg["matching"]["n_hard_negatives_per_item"] * 4
    candidate_pool = [label_agnostic_candidate_pool(i, covariates, base_ids, n_neighbors=n_neighbors)
                        for i in range(len(valid))]

    delta_i, Delta_obs = compute_delta(C_mech, labels, candidate_pool)
    strata = [c["perturbation_family"] for c in covariates]
    perm = permutation_test_delta(C_mech, labels, candidate_pool, strata=strata,
                                    n_permutations=exp_cfg["statistics"]["n_permutations"], seed=exp_cfg["random_seed"])

    base_id_to_idx = {}
    for i, b in enumerate(base_ids):
        base_id_to_idx.setdefault(b, i)

    def statistic_fn(sample_base_ids):
        idx = sorted({base_id_to_idx[b] for b in set(sample_base_ids) if b in base_id_to_idx})
        if len(idx) < 5:
            return float("nan")
        pos = {g: p for p, g in enumerate(idx)}
        s2 = C_mech[np.ix_(idx, idx)]
        l2 = [labels[g] for g in idx]
        pool2 = [[pos[j] for j in candidate_pool[g] if j in pos] for g in idx]
        _, d = compute_delta(s2, l2, pool2)
        return d if d == d else float("nan")

    boot = bootstrap_ci_by_problem(base_ids, statistic_fn, n_bootstrap=exp_cfg["statistics"]["n_bootstrap"],
                                    seed=exp_cfg["random_seed"], alpha=exp_cfg["statistics"]["alpha"])

    rows = []
    for i in range(len(valid)):
        for j in candidate_pool[i]:
            if j <= i:
                continue
            rows.append({"S_sem": float(S_sem[i, j]), "C_mech": float(C_mech[i, j]),
                          "edit_size_diff": abs(covariates[i]["edit_size"] - covariates[j]["edit_size"]),
                          "problem_length_diff": abs(covariates[i]["problem_length"] - covariates[j]["problem_length"]),
                          "trace_length_diff": abs(covariates[i]["trace_length"] - covariates[j]["trace_length"]),
                          "same_perturbation": float(covariates[i]["perturbation_family"] == covariates[j]["perturbation_family"]),
                          "cluster_group": base_ids[i]})
    df = pd.DataFrame(rows)
    continuous_result = continuous_correspondence_regression(
        C_mech=df["C_mech"].values, S_sem=df["S_sem"].values,
        X_covariates=df[["edit_size_diff", "problem_length_diff", "trace_length_diff", "same_perturbation"]],
        cluster_groups=df["cluster_group"].values,
    )

    report = {
        "benchmark": "svamp", "role": "frozen_replication", "category_level": args.category_level,
        "n_failures": len(valid), "n_novel_unmapped": n_novel,
        "Delta_observed": Delta_obs, "Delta_permutation_p_value": perm["p_value"],
        "Delta_bootstrap_ci": [boot["ci_low"], boot["ci_high"]],
        "continuous_beta_sem": continuous_result["beta_sem"], "continuous_p_sem": continuous_result["p_sem"],
    }
    out_dir = _abs(args.out_dir)
    out_path = out_dir / "reports" / "replication_svamp.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("SVAMP replication: Delta=%.4f (p=%.4g), beta_sem=%.4f (p=%.4g)",
              Delta_obs, perm["p_value"], continuous_result["beta_sem"], continuous_result["p_sem"])


if __name__ == "__main__":
    main()
