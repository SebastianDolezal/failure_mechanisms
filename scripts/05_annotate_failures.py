#!/usr/bin/env python
"""Collect first-observable-error annotations (Section 13-17).

Supports two annotator types:
  - model:  a small local instruction model judges each failure automatically
            using the exact prompt in Section 13 (never "why did it fail?").
  - human:  an interactive CLI loop for a human annotator; resumable, and
            safe to run multiple independent passes under different
            --annotator-id values (needed for RQ1a inter-annotator agreement,
            script 06).

Each output record: {pair_id, first_error_step, wrong_span,
minimal_corrected_span, description, confidence, annotator_id, annotator_type}
written to data/annotations/<benchmark>/<annotator_id>.jsonl

Usage:
    python scripts/05_annotate_failures.py --benchmark gsm8k \
        --annotator-type model --annotator-id model_judge \
        --model configs/models/qwen2.5-3b-instruct.yaml

    python scripts/05_annotate_failures.py --benchmark gsm8k \
        --annotator-type human --annotator-id annotator_1

    python scripts/05_annotate_failures.py --benchmark gsm8k \
        --annotator-type human --annotator-id annotator_2 --subset-frac 0.3
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_benchmark_config, load_model_config, read_jsonl, append_jsonl, _abs
from src.utils import ErrorTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("annotate")

ANNOTATION_PROMPT = """You are reviewing a language model's step-by-step solution to a math word \
problem. Identify the first externally observable error in the reasoning. Describe what went \
wrong at that point without speculating about the model's internal mechanism.

Problem:
{question}

Gold solution:
{gold_solution}

Correct final answer: {correct_answer}

Model's reasoning trace:
{trace}

Model's final answer: {model_answer}

Return ONLY a JSON object with exactly these keys:
{{
  "first_error_step": <integer step number where the error first appears>,
  "wrong_span": "<verbatim text of the erroneous step or phrase>",
  "minimal_corrected_span": "<the smallest edit that would fix that step>",
  "description": "<short, phenomenological description of what went wrong, e.g. \
'applied the percentage to the wrong quantity'>",
  "confidence": <float between 0 and 1>
}}
"""

JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_annotation(text: str) -> dict | None:
    m = JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def annotate_with_model(pairs: list, model_cfg: dict, annotator_id: str, out_path: Path,
                          resume: bool = True) -> dict:
    """Writes each annotation to `out_path` as soon as it's produced (rather
    than buffering everything in memory and writing once at the end) so a
    crash - a bad example, an OOM, the process getting killed outright -
    loses only the items after the crash point, not the whole run. Skips
    pair_ids already present in `out_path` so a re-run after a crash resumes
    instead of redoing completed work.
    """
    from src.inference.model_wrapper import ModelWrapper

    done_ids = set()
    if resume and out_path.exists():
        done_ids = {json.loads(l)["pair_id"] for l in open(out_path)}
        if done_ids:
            log.info("Resuming: %d/%d items already annotated in %s", len(done_ids), len(pairs), out_path)

    wrapper = ModelWrapper(model_cfg)
    errors_path = out_path.parent / f"{annotator_id}_errors.jsonl"
    tracker = ErrorTracker(errors_path)
    n_unparsed = 0

    for idx, p in enumerate(pairs):
        if p.pair_id in done_ids:
            continue
        with tracker.guard(p.pair_id, context="model_annotation"):
            prompt = ANNOTATION_PROMPT.format(
                question=p.fail_question, gold_solution=p.meta.get("gold_solution", ""),
                correct_answer=p.gold_answer, trace=p.fail_trace, model_answer=p.fail_answer,
            )
            raw = wrapper.generate(prompt, max_new_tokens=300)
            parsed = _parse_json_annotation(raw)
            log.info("[%d/%d] %s parsed=%s", idx + 1, len(pairs), p.pair_id, parsed is not None)
            if parsed is None:
                n_unparsed += 1
                log.warning("Failed to parse model annotation for %s", p.pair_id)
                continue
            record = {
                "pair_id": p.pair_id, "annotator_id": annotator_id, "annotator_type": "model",
                "first_error_step": parsed.get("first_error_step"),
                "wrong_span": parsed.get("wrong_span"),
                "minimal_corrected_span": parsed.get("minimal_corrected_span"),
                "description": parsed.get("description"),
                "confidence": parsed.get("confidence"),
            }
            append_jsonl(record, out_path)

    tracker.close()
    return {"n_unparsed": n_unparsed, **tracker.summary()}


def annotate_interactively(pairs: list, annotator_id: str, out_path: Path) -> None:
    done_ids = set()
    if out_path.exists():
        done_ids = {json.loads(l)["pair_id"] for l in open(out_path)}

    remaining = [p for p in pairs if p.pair_id not in done_ids]
    print(f"{len(remaining)} of {len(pairs)} items remaining for annotator '{annotator_id}'. "
          f"Progress is saved after every item; Ctrl-C is safe.")

    for p in remaining:
        print("\n" + "=" * 80)
        print(f"pair_id: {p.pair_id}")
        print(f"\nPROBLEM:\n{p.fail_question}")
        print(f"\nGOLD SOLUTION:\n{p.meta.get('gold_solution', '(not provided)')}")
        print(f"\nCORRECT ANSWER: {p.gold_answer}")
        print(f"\nMODEL TRACE:\n{p.fail_trace}")
        print(f"\nMODEL FINAL ANSWER: {p.fail_answer}")
        print("\nIdentify the first externally observable error. Do NOT speculate about internal mechanism.")
        try:
            step = int(input("first_error_step (integer): ").strip())
            wrong_span = input("wrong_span (verbatim text): ").strip()
            corrected = input("minimal_corrected_span: ").strip()
            description = input("description (short, phenomenological): ").strip()
            confidence = float(input("confidence [0-1]: ").strip() or "0.8")
        except (ValueError, KeyboardInterrupt):
            print("\nStopping; progress saved.")
            break

        rec = {
            "pair_id": p.pair_id, "annotator_id": annotator_id, "annotator_type": "human",
            "first_error_step": step, "wrong_span": wrong_span,
            "minimal_corrected_span": corrected, "description": description,
            "confidence": confidence,
        }
        append_jsonl(rec, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    ap.add_argument("--annotator-type", required=True, choices=["model", "human"])
    ap.add_argument("--annotator-id", required=True)
    ap.add_argument("--model", help="model config path, required for --annotator-type model")
    ap.add_argument("--subset-frac", type=float, default=1.0,
                     help="annotate only a random subset (e.g. 0.3 for a stratified second-human pass)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bench_cfg = load_benchmark_config(args.benchmark)
    pairs = read_jsonl(_abs(bench_cfg["pairs_path"]) / "pairs_primary.jsonl")
    fail_only = [p for p in pairs if p.pair_type in ("induced_failure", "rescued_failure")]

    if args.subset_frac < 1.0:
        rng = random.Random(args.seed)
        fail_only = rng.sample(fail_only, k=max(1, int(len(fail_only) * args.subset_frac)))

    out_dir = _abs("data/annotations") / args.benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.annotator_id}.jsonl"

    if args.annotator_type == "model":
        if not args.model:
            raise ValueError("--model is required for --annotator-type model")
        model_cfg = load_model_config(args.model)
        summary = annotate_with_model(fail_only, model_cfg, args.annotator_id, out_path)
        log.info("Model annotation done: %d succeeded, %d failed (see error log), %d unparsed, written to %s",
                  summary["n_succeeded"] - summary["n_unparsed"], summary["n_failed"], summary["n_unparsed"], out_path)
    else:
        annotate_interactively(fail_only, args.annotator_id, out_path)


if __name__ == "__main__":
    main()
