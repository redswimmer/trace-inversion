#!/usr/bin/env bash
# Smoke test: does the victim truncate at 32k? ~30 problems, before committing 8h.
set -uo pipefail
cd /home/asavala/Development/papers/trace-inversion
M="${HOME}/trace-inversion-bench/models/Qwen3.8-27B-IQ4_XS.gguf"
mkdir -p bench/logs bench/results
echo "[$(date +%H:%M:%S)] starting server (6 slots x 32768)"
llama-server -m "$M" -ngl 999 -c $((32768*6)) -np 6 -fa on -ctk q8_0 -ctv q8_0 \
  --port 8077 --host 127.0.0.1 --jinja > bench/logs/server-victim-smoke.log 2>&1 &
SRV=$!
for _ in $(seq 1 90); do curl -sf http://127.0.0.1:8077/health >/dev/null 2>&1 && break; sleep 5; done
if ! curl -sf http://127.0.0.1:8077/health >/dev/null 2>&1; then
  echo "SERVER FAILED"; tail -8 bench/logs/server-victim-smoke.log; kill -9 $SRV; exit 1
fi
echo "[$(date +%H:%M:%S)] server up, running 15/bench"
.venv-vllm/bin/python bench/eval_victim_gguf.py \
  --url http://127.0.0.1:8077 --out bench/results/victim-smoke.jsonl \
  --n-per-bench 15 --concurrency 6 --max-tokens 32768 \
  --temperature 1.0 --top-p 0.95 --top-k 20 2>&1 | tail -12
kill -9 $SRV 2>/dev/null
echo "[$(date +%H:%M:%S)] smoke done"
