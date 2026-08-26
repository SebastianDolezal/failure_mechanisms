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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_benchmark_config, load_model_config, read_jsonl, append_jsonl, _abs
from src.utils import ErrorTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("annotate")

ANNOTATION_PROMPT = """You are reviewing a language model's step-by-step solution to a math word \
problem. Identify the first externally observable error in the reasoning: the first step that \
introduces a computation, quantity, or logical move that is objectively wrong given the problem \
and the (correct) steps before it. Do not pick a later step that merely repeats or carries \
forward an error already present earlier. Describe what went wrong at that point without \
speculating about the model's internal mechanism.

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
  "wrong_span": "<verbatim text of ONLY the specific erroneous phrase or sub-expression within \
that step - not the whole step, unless the whole step is wrong>",
  "minimal_corrected_span": "<the same span, edited to be correct - this must differ from \
wrong_span; if you cannot find a sub-span that isolates the error, widen wrong_span until the \
two differ>",
  "description": "<start with EXACTLY ONE of these category labels, then a colon, then a short \
specific detail - e.g. 'wrong computation: divided instead of multiplying the discount rate'. \
These three categories are grouped by WHICH STAGE of solving the problem broke, not by surface \
arithmetic behavior - that's deliberate: 'divided instead of multiplied', 'used the wrong ratio', \
and 'rounded down without basis' all look different on the surface but are the same underlying \
failure (the model understood the problem correctly and had the right numbers, but the computation \
it then did was wrong). Grouping by that shared underlying failure, rather than by surface \
behavior, is meant to make it easier for two different annotators to land on the same label \
instead of splitting between equally-valid surface descriptions of the same mistake. Check the \
categories in this exact order and stop at the first one that applies: \
(1) wrong quantity - the model correctly understood what to compute, but plugged in a wrong number \
or variable value from the problem. The input to the computation is wrong, regardless of whether \
the computation itself would otherwise have been correct. \
(2) wrong computation - the model correctly understood what to compute and used the right numbers, \
but the computation applied to them was wrong: wrong arithmetic operation, wrong formula/ratio/rate, \
wrong sign/direction, or an unjustified rounding/truncation of a fractional remainder. All of these \
are the same underlying failure - right inputs, wrong processing - so do not split them into \
separate labels. \
(3) wrong problem understanding - the model misunderstood what the problem is asking, misread a \
stated condition, or skipped a necessary step entirely because it did not realize that step was \
needed. This is a failure in understanding the task itself, not in the arithmetic that followed. \
(4) other (only if none of the above fit).>",
  "confidence": <float between 0 and 1>
}}

Output ONLY that JSON object. Do not include any reasoning, analysis, or explanation before or \
after it - not even a short lead-in sentence. Your entire response must be parseable as JSON on \
its own.

Example (a different problem, showing the expected format and level of detail - do not reuse any \
of its content):

Problem:
A store had 84 apples. They sold 37 apples in the morning and 19 more in the afternoon. How many \
apples are left?

Gold solution:
84 - 37 = 47 apples left after the morning. 47 - 19 = 28 apples left after the afternoon.

Correct final answer: 28

Model's reasoning trace:
Step 1: 84 - 37 = 47 apples left after the morning.
Step 2: 47 - 19 = 38 apples left after the afternoon.

Model's final answer: 38

Correct output for that example:
{{"first_error_step": 2, "wrong_span": "47 - 19 = 38", "minimal_corrected_span": "47 - 19 = 28", \
"description": "wrong computation: subtracted 19 from 47 incorrectly, getting 38 instead of 28", \
"confidence": 0.95}}

Now annotate the actual problem above. Output ONLY the JSON object for it.
"""

def _extract_json_objects(text: str) -> list[str]:
    """Finds every top-level {...} substring via brace-depth counting rather
    than a single greedy first-to-last regex. The naive regex breaks as soon
    as the model adds any preamble/reasoning before the JSON despite being
    told not to, or writes a draft attempt followed by a "final" one - a
    first-brace-to-last-brace match then spans both and is not valid JSON on
    its own, even though one of the individual objects usually is. Tracks
    string literals so braces inside a quoted value don't affect depth.
    """
    objects = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start:i + 1])
    return objects


def _parse_json_annotation(text: str) -> dict | None:
    # Prefer the LAST candidate object: if the model ignored the
    # output-only-JSON instruction and wrote reasoning or a draft attempt
    # first, the final one is the intended answer.
    for candidate in reversed(_extract_json_objects(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
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
            raw = wrapper.generate(prompt, max_new_tokens=500)
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
