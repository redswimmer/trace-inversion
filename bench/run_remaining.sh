#!/usr/bin/env bash
# All remaining baselines under the PAPER's sampling (0.7 / 0.9 / rep 1.05).
# Order puts the cheap 2B first so the new protocol is validated before a ~10h run.
set -uo pipefail
cd /home/asavala/Development/papers/trace-inversion
export PATH="$PWD/.venv-vllm/bin:$PATH"
mkdir -p bench/logs bench/results

echo "[$(date +%H:%M:%S)] === 2B student re-measure (paper sampling) ==="
.venv-vllm/bin/python bench/eval_baseline.py \
  --model Qwen/Qwen3.5-2B --out bench/results/Qwen3.5-2B.jsonl \
  --max-tokens 32768 --max-len 40960 --gpu-frac 0.90 \
  > bench/logs/Qwen3.5-2B.log 2>&1
grep -E "acc=|WARNING" bench/logs/Qwen3.5-2B.log | tail -4

echo "[$(date +%H:%M:%S)] === victim full ==="
bash bench/run_victim_full.sh

echo "[$(date +%H:%M:%S)] === 7B surrogate ==="
bash bench/run_surrogate_7b.sh

echo "[$(date +%H:%M:%S)] === ALL REMAINING BASELINES DONE ==="
