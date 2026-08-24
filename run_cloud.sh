#!/usr/bin/env bash
# Launches the full GSM8K discovery pipeline for the cloud-scale run
# (Qwen2.5-7B-Instruct target, Mistral-7B-Instruct-v0.3 / Phi-3.5-mini-instruct
# judges, target-n=200), and makes sure it survives an SSH disconnect.
#
# Usage (from the repo root, on the cloud instance):
#   bash run_cloud.sh
#
# Progress: tail -f logs/cloud_run_*.log
# If it stops for any reason, just re-run this script - every expensive
# stage (02 traces, 05 annotate, 08 patching, 09 attribution, 15 transfer)
# checkpoints its own progress and resumes automatically.
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
    --force \
    > "$LOGFILE" 2>&1 &

PID=$!
echo "Pipeline started in the background, PID $PID."
echo "This will keep running even if your SSH session disconnects."
echo "Monitor with:  tail -f $LOGFILE"
echo "Check if still running with:  ps -p $PID"
