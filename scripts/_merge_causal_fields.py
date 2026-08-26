#!/usr/bin/env python
"""One-off helper for the Gate B/C fix re-run.

Stage 07 (07_freeze_taxonomy.py) rebuilds data/pairs/<benchmark>/pairs_annotated.jsonl
from scratch out of pairs_primary.jsonl + the (freshly re-annotated) judge_a
descriptions - it has no knowledge of raw_causal_profile, stable_pair_baseline,
or failure_excess_signature, since those are only ever added later by stage 08
(exact patching), which we are deliberately NOT re-running here (it's the
~13-hour, expensive stage, and none of the Gate B/C fixes touch it).

Run this once, right after re-running stages 05 -> 06 -> 07 with the new
annotation prompt, and before stage 09 (attribution patching) or 11-15
(correspondence/prediction/resolution/split-merge/transfer), all of which
need raw_causal_profile / failure_excess_signature present on
pairs_annotated.jsonl to do anything.

Source of truth for the causal fields: data/mechanisms/<benchmark>/
failure_signatures.jsonl, which stage 08 already wrote (mirroring
pairs_annotated.jsonl as it stood right after the ORIGINAL annotation run)
and which this script does not modify.

Also prints how many pair_ids got a different first_observable_error_step /
wrong_span / corrected_span from the new annotation pass vs. the old one
that the exact-patching profiles were actually computed against - exact
patching truncates its prefix at that step, so a large number here would
mean the preserved raw_causal_profile values are no longer measuring quite
the same thing as the newly-annotated error location, and 08 would need a
real re-run rather than a field-merge. A handful of drifted items is
expected and fine; if it's most of the 102, stop and re-run 08 instead of
trusting this merge.

Usage:
    python scripts/_merge_causal_fields.py --benchmark gsm8k
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import load_benchmark_config, read_jsonl, write_jsonl, _abs

CAUSAL_FIELDS = ["raw_causal_profile", "stable_pair_baseline", "failure_excess_signature"]
DRIFT_CHECK_FIELDS = ["first_observable_error_step", "wrong_span", "corrected_span"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gsm8k", "svamp"])
    args = ap.parse_args()

    bench_cfg = load_benchmark_config(args.benchmark)
    pairs_path = _abs(bench_cfg["pairs_path"]) / "pairs_annotated.jsonl"
    old_path = _abs("data/mechanisms") / args.benchmark / "failure_signatures.jsonl"

    new_records = read_jsonl(pairs_path)
    old_by_id = {r.pair_id: r for r in read_jsonl(old_path)}

    n_merged = 0
    n_drifted = 0
    n_missing_old = 0
    for rec in new_records:
        old = old_by_id.get(rec.pair_id)
        if old is None:
            n_missing_old += 1
            continue
        drifted = any(getattr(rec, f) != getattr(old, f) for f in DRIFT_CHECK_FIELDS)
        if drifted:
            n_drifted += 1
        for f in CAUSAL_FIELDS:
            setattr(rec, f, getattr(old, f))
        n_merged += 1

    write_jsonl(new_records, pairs_path)

    print(f"Merged causal fields onto {n_merged}/{len(new_records)} pairs "
          f"({n_missing_old} had no matching old record - these get no causal profile).")
    print(f"{n_drifted}/{n_merged} pairs have a DIFFERENT first_observable_error_step, "
          f"wrong_span, or corrected_span under the new annotation pass than the one the "
          f"preserved exact-patching profile was actually computed against.")
    if n_merged and n_drifted / n_merged > 0.15:
        print("WARNING: that's a large fraction (>15%). The preserved raw_causal_profile "
              "values may no longer match the newly-annotated error location closely enough "
              "to trust - consider re-running stage 08 instead of relying on this merge.")


if __name__ == "__main__":
    main()
