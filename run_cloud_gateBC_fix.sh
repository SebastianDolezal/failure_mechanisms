#!/usr/bin/env bash
# Targeted re-run for the Gate B (annotation category scheme) and Gate C
# (two-point attribution-patching correction) fixes ONLY.
#
# Does NOT touch stage 00 (pilot), 01-04 (variant generation / pair
# isolation / stable controls), or 08 (exact activation patching) - those
# are untouched by either fix and 08 in particular is the ~13-hour, most
# expensive stage. This script re-runs only what the two fixes actually
# affect: annotation -> Gate B -> taxonomy -> (causal-field merge-back) ->
# attribution patching -> Gate C -> every downstream test that reads the
# taxonomy or the AtP profile.
#
# Prerequisite: this repo's data/ and results/ must already have the FIRST
# post-fix run's output in place (pairs_primary.jsonl, pairs_stable.jsonl,
# the stage-08 checkpoints, data/mechanisms/gsm8k/failure_signatures.jsonl
# and stable_profiles.jsonl) - i.e. pull the latest git state before running
# this, don't run it on a fresh clone.
#
# Usage (from the repo root, on the cloud instance):
#   bash run_cloud_gateBC_fix.sh
#
# Progress: tail -f logs/gateBC_fix_*.log
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set. Downloads will be unauthenticated and rate-limited."
  echo "Set it first if you have one:  export HF_TOKEN=hf_..."
  echo "Continuing in 10 seconds anyway (Ctrl-C to abort and set it)..."
  sleep 10
fi

echo "Installing dependencies..."
pip install -q -r requirements.txt

python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU visible to torch.'"
echo "CUDA check passed: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

mkdir -p logs
LOGFILE="logs/gateBC_fix_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOGFILE"

BENCH=gsm8k
MODEL=qwen2.5-7b-instruct
JUDGE_A_ID=judge_a
JUDGE_B_ID=judge_b
JUDGE_A_CFG=configs/models/mistral-7b-instruct-v0.3.yaml
JUDGE_B_CFG=configs/models/phi-3.5-mini-instruct.yaml

{
  echo "=== 05 annotate (judge_a, new merged-category prompt) ==="
  python3 -u scripts/05_annotate_failures.py --benchmark "$BENCH" --annotator-type model \
      --annotator-id "$JUDGE_A_ID" --model "$JUDGE_A_CFG"

  echo "=== 05 annotate (judge_b, new merged-category prompt) ==="
  python3 -u scripts/05_annotate_failures.py --benchmark "$BENCH" --annotator-type model \
      --annotator-id "$JUDGE_B_ID" --model "$JUDGE_B_CFG"

  echo "=== 06 Gate B ==="
  python3 -u scripts/06_semantic_reliability.py --benchmark "$BENCH" --annotators "$JUDGE_A_ID" "$JUDGE_B_ID"
  python3 -c "import json; r=json.load(open('results/reports/semantic_reliability.json')); print('GATE B:', 'PASSED' if r['gate_b_passed'] else 'FAILED', '- kappa=%.3f (threshold %.2f)' % (r['mean_cohens_kappa'], r['gate_b_threshold']))"

  echo "=== 07 freeze taxonomy (rebuilds pairs_annotated.jsonl from scratch) ==="
  python3 -u scripts/07_freeze_taxonomy.py --benchmark "$BENCH" --primary-annotator "$JUDGE_A_ID"

  echo "=== merge back the preserved exact-patching fields stage 07 doesn't know about ==="
  python3 -u scripts/_merge_causal_fields.py --benchmark "$BENCH"

  echo "=== 09 attribution patching (two-point corrected version, 55-item subset) ==="
  python3 -u scripts/09_compute_attribution_patching.py --benchmark "$BENCH" --model "$MODEL"

  echo "=== 10 Gate C ==="
  python3 -u scripts/10_mechanistic_reliability.py --benchmark "$BENCH"
  python3 -c "import json; r=json.load(open('results/reports/mechanistic_reliability.json')); print('GATE C:', 'PASSED' if r['gate_c_passed'] else 'FAILED', '- margin=%.3f (threshold %.2f)' % (r['reliability_margin_same_minus_diff'], r['gate_c_threshold']))"

  echo "=== 11-14 correspondence / prediction / resolution / split-merge (re-run: taxonomy changed) ==="
  python3 -u scripts/11_main_correspondence_test.py --benchmark "$BENCH"
  python3 -u scripts/12_prediction_test.py --benchmark "$BENCH"
  python3 -u scripts/13_resolution_analysis.py --benchmark "$BENCH"
  python3 -u scripts/14_split_merge_analysis.py --benchmark "$BENCH"

  echo "=== 15 intervention transfer (re-run: taxonomy + wrong_span/corrected_span changed) ==="
  python3 -u scripts/15_intervention_transfer.py --benchmark "$BENCH" --model "$MODEL" --n-pairs 150 --top-k 5

  echo "Gate B/C fix re-run complete. Updated reports are under results/reports/."
} > "$LOGFILE" 2>&1 &

PID=$!
echo "Running in the background, PID $PID. Survives SSH disconnects."
echo "Monitor with:  tail -f $LOGFILE"
echo "Check if still running with:  ps -p $PID"
