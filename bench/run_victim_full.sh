#!/usr/bin/env bash
# Victim baseline, FULL 1015 tasks. Cheap because the victim is terse:
# median 4,295 tok (JEEBench) / 615 (MATH500) => ~2.5M tokens total.
set -uo pipefail
cd /home/asavala/Development/papers/trace-inversion
M="${HOME}/trace-inversion-bench/models/Qwen3.8-27B-IQ4_XS.gguf"
TAG="Qwen3.8-27B-IQ4_XS-full"; PORT=8079; URL="http://127.0.0.1:${PORT}"
SLOTS=6; CTX=32768
STATUS=bench/logs/STATUS-victim.txt
mkdir -p bench/logs bench/results; : > "$STATUS"
note(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

note "waiting for the 1.5B surrogate run to finish"
while pgrep -f run_gguf_baselines.sh > /dev/null; do sleep 120; done
sleep 30
note "GPU free: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

note "BEGIN ${TAG} ctx=${CTX} slots=${SLOTS} FULL 1015"
llama-server -m "$M" -ngl 999 -c $((CTX*SLOTS)) -np $SLOTS -fa on \
  -ctk q8_0 -ctv q8_0 --port $PORT --host 127.0.0.1 --jinja \
  > "bench/logs/server-${TAG}.log" 2>&1 &
SRV=$!
for _ in $(seq 1 120); do curl -sf "${URL}/health" >/dev/null 2>&1 && break; sleep 5; done
if ! curl -sf "${URL}/health" >/dev/null 2>&1; then
  note "SERVER FAILED — try fewer slots"
  grep -iE "error|failed|alloc|out of" "bench/logs/server-${TAG}.log" | tail -6 | tee -a "$STATUS"
  kill -9 $SRV 2>/dev/null; exit 1
fi
note "server up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"

.venv-vllm/bin/python bench/eval_victim_gguf.py \
  --url "$URL" --out "bench/results/${TAG}.jsonl" \
  --n-per-bench 0 --concurrency $SLOTS --max-tokens 32768 \
  --temperature 1.0 --top-p 0.95 --top-k 20 > "bench/logs/${TAG}.log" 2>&1
rc=$?
kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null
if [[ $rc -ne 0 ]]; then note "FAILED (rc=$rc)"; tail -8 "bench/logs/${TAG}.log" | tee -a "$STATUS"
else note "DONE"; grep -E "acc=" "bench/logs/${TAG}.log" | tee -a "$STATUS"; fi
note "VICTIM FULL COMPLETE"
