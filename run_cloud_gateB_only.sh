#!/usr/bin/env bash
# Third-pass Gate B fix ONLY (broad stage-of-reasoning categories, matched-
# capability judge_b). Deliberately does NOT re-run stage 09 (attribution
# patching) or 10 (Gate C) - two prior attempts at a legitimate Gate C fix
# (full-margin gradient, then a two-point path correction) both left it
# failing at roughly the same margin, and there is no further low-risk,
# literature-backed fix left to try before the deadline. Skipping it here
# saves the compute/time of re-testing an approximation we have no more
# legitimate way to improve - it does NOT mean Gate C is deleted from the
# study. Its earlier result (FAILED, margin=0.061) still stands and still
# belongs in the writeup, correctly framed as not affecting the exact-
# patching-based main results, which this script's downstream stages
# (11-15) still rely on entirely.
#
# Does NOT touch stage 00 (pilot), 01-04 (variant generation / pair
# isolation / stable controls), or 08 (exact activation patching) - those
# are untouched by this fix and 08 in particular is the ~13-hour, most
# expensive stage. Re-runs only: annotation -> Gate B -> taxonomy ->
# (causal-field merge-back) -> every downstream test that reads the
# taxonomy (correspondence, prediction, resolution, split-merge, transfer).
#
# Prerequisite: this repo's data/ and results/ must already have the FIRST
# post-fix run's output in place (pairs_primary.jsonl, pairs_stable.jsonl,
# the stage-08 checkpoints, data/mechanisms/gsm8k/failure_signatures.jsonl
# and stable_profiles.jsonl) - i.e. pull the latest git state before running
# this, don't run it on a fresh clone.
#
# Usage (from the repo root, on the cloud instance):
#   bash run_cloud_gateB_only.sh
#
# Progress: tail -f logs/gateB_only_*.log
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
LOGFILE="logs/gateB_only_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOGFILE"

BENCH=gsm8k
MODEL=qwen2.5-7b-instruct
JUDGE_A_ID=judge_a
JUDGE_B_ID=judge_b
JUDGE_A_CFG=configs/models/mistral-7b-instruct-v0.3.yaml
JUDGE_B_CFG=configs/models/olmo2-7b-instruct.yaml

{
  echo "=== 05 annotate (judge_a, broad stage-of-reasoning categories, upgraded judge_b) ==="
  python3 -u scripts/05_annotate_failures.py --benchmark "$BENCH" --annotator-type model \
      --annotator-id "$JUDGE_A_ID" --model "$JUDGE_A_CFG"

  echo "=== 05 annotate (judge_b, broad stage-of-reasoning categories, upgraded judge_b) ==="
  python3 -u scripts/05_annotate_failures.py --benchmark "$BENCH" --annotator-type model \
      --annotator-id "$JUDGE_B_ID" --model "$JUDGE_B_CFG"

  echo "=== 06 Gate B ==="
  python3 -u scripts/06_semantic_reliability.py --benchmark "$BENCH" --annotators "$JUDGE_A_ID" "$JUDGE_B_ID"
  python3 -c "import json; r=json.load(open('results/reports/semantic_reliability.json')); print('GATE B:', 'PASSED' if r['gate_b_passed'] else 'FAILED', '- kappa=%.3f (threshold %.2f)' % (r['mean_cohens_kappa'], r['gate_b_threshold']))"

  echo "=== 07 freeze taxonomy (rebuilds pairs_annotated.jsonl from scratch) ==="
  python3 -u scripts/07_freeze_taxonomy.py --benchmark "$BENCH" --primary-annotator "$JUDGE_A_ID"

  echo "=== merge back the preserved exact-patching fields stage 07 doesn't know about ==="
  python3 -u scripts/_merge_causal_fields.py --benchmark "$BENCH"

  echo "=== skipping stage 09 (attribution patching) and 10 (Gate C) - see script header ==="
  echo "=== Gate C's last real result stands: FAILED, margin=0.061 (threshold 1.00) - unchanged, not re-tested ==="

  echo "=== 11-14 correspondence / prediction / resolution / split-merge (re-run: taxonomy changed) ==="
  python3 -u scripts/11_main_correspondence_test.py --benchmark "$BENCH"
  python3 -u scripts/12_prediction_test.py --benchmark "$BENCH"
  python3 -u scripts/13_resolution_analysis.py --benchmark "$BENCH"
  python3 -u scripts/14_split_merge_analysis.py --benchmark "$BENCH"

  echo "=== 15 intervention transfer (re-run: taxonomy + wrong_span/corrected_span changed) ==="
  python3 -u scripts/15_intervention_transfer.py --benchmark "$BENCH" --model "$MODEL" --n-pairs 150 --top-k 5

  echo "Gate B fix re-run complete (Gate C intentionally not re-tested - see header). Updated reports are under results/reports/."
} > "$LOGFILE" 2>&1 &

PID=$!
echo "Running in the background, PID $PID. Survives SSH disconnects."
echo "Monitor with:  tail -f $LOGFILE"
echo "Check if still running with:  ps -p $PID"
