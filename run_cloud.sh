#!/usr/bin/env bash
# Resumes the GSM8K discovery pipeline from stage 05 onward, after the
# annotation-prompt (Gate B), attribution-patching (Gate C), and
# intervention-transfer layer-selection (H7) fixes. Stages 00-04 (pilot,
# variants, traces, pairs, stable controls) are untouched by those fixes and
# already completed in the prior run - re-running them would just redo the
# ~13-hour trace-generation stage for no reason. Only run this against a
# repo whose data/annotations and results/logs/{08,09,15}_checkpoint* files
# have been cleared of the pre-fix run's data (already done as of this
# script's last edit - see the archived _pre_fix_stale_state_* folder).
#
# Usage (from the repo root, on the cloud instance):
#   bash run_cloud.sh
#
# Progress: tail -f logs/cloud_run_*.log
# If it stops for any reason, just re-run this script - every expensive
# stage (05 annotate, 08 patching, 09 attribution, 15 transfer) checkpoints
# its own progress and resumes automatically.
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

python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU visible to torch - check the instance/driver before spending money.'"
echo "CUDA check passed: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

mkdir -p logs
LOGFILE="logs/cloud_run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOGFILE"

nohup python3 -u scripts/run_pipeline.py \
    --model qwen2.5-7b-instruct \
    --target-n 200 \
    --start-stage 05_annotate_judge_a \
    --force \
    > "$LOGFILE" 2>&1 &

PID=$!
echo "Pipeline started in the background, PID $PID."
echo "This will keep running even if your SSH session disconnects."
echo "Monitor with:  tail -f $LOGFILE"
echo "Check if still running with:  ps -p $PID"
