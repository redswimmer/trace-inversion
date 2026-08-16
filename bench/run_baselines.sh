#!/usr/bin/env bash
# Zero-shot baselines: student candidates + surrogate.
# Keeps vLLM's progress bar in a per-model log so the run is monitorable.
set -uo pipefail
cd /home/asavala/Development/papers/trace-inversion
export PATH="$PWD/.venv-vllm/bin:$PATH"
mkdir -p bench/results bench/logs
STATUS=bench/logs/STATUS.txt
: > "$STATUS"

note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

# model | temperature | top_p | top_k   (per-family recommended sampling)
RUNS=(
  "Qwen/Qwen3.5-0.8B|1.0|0.95|20"
  "Qwen/Qwen3.5-2B|1.0|0.95|20"
  "Qwen/Qwen3.5-4B|1.0|0.95|20"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|0.6|0.95|-1"
)

note "START — ${#RUNS[@]} models"
for spec in "${RUNS[@]}"; do
  IFS='|' read -r M T P K <<< "$spec"
  TAG="${M##*/}"
  if [[ -f "bench/results/${TAG}.jsonl" ]]; then note "SKIP ${TAG} (already done)"; continue; fi
  note "BEGIN ${TAG}  (temp=${T} top_p=${P} top_k=${K})"

  # full output kept, including the \r progress bar, so progress is greppable
  .venv-vllm/bin/python bench/eval_baseline.py \
      --model "$M" --out "bench/results/${TAG}.jsonl" \
      --max-tokens 32768 --max-len 40960 --gpu-frac 0.90 \
      --temperature "$T" --top-p "$P" --top-k "$K" \
      > "bench/logs/${TAG}.log" 2>&1
  rc=$?

  if [[ $rc -ne 0 ]]; then
    note "FAILED ${TAG} (rc=${rc}) — see bench/logs/${TAG}.log"
    tail -5 "bench/logs/${TAG}.log" | tee -a "$STATUS"
  else
    note "DONE ${TAG}"
    grep -E "acc=" "bench/logs/${TAG}.log" | tee -a "$STATUS"
  fi
  sleep 10   # let VRAM drain between models
done
note "ALL DONE"
