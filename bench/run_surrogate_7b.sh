#!/usr/bin/env bash
# Baseline the 7B surrogate candidate, full 1015 tasks, F16 on llama.cpp.
#
# Full precision to match the 1.5B BF16 run — comparing a BF16 1.5B against a
# 4-bit 7B would confound quantization damage with model strength.
#
# Sizing: 23.4 GiB usable - 14.19 weights = ~9.2 GiB free. R1-Distill-Qwen-7B is
# a standard transformer (28 layers, 4 KV heads, head_dim 128) => 28 KiB/token
# at q8_0 KV, so 32768 ctx costs ~0.875 GiB/slot. 8 slots ~= 7.0 GiB. Fits.
set -uo pipefail
cd /home/asavala/Development/papers/trace-inversion

MODELS="${HOME}/trace-inversion-bench/models"
FILE="DeepSeek-R1-Distill-Qwen-7B-F16.gguf"
TAG="DeepSeek-R1-Distill-Qwen-7B-gguf"
PORT=8078
URL="http://127.0.0.1:${PORT}"
SLOTS=8
CTX=32768
STATUS=bench/logs/STATUS-7b.txt
mkdir -p bench/logs bench/results
: > "$STATUS"
note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

note "GPU free: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

[[ -f "${MODELS}/${FILE}" ]] || { note "MISSING ${FILE}"; exit 1; }

# Measure optimal slots instead of guessing. Arithmetic sized the 1.5B at 12
# slots and it never exceeded 55% GPU utilization for 4 hours.
note "sweeping concurrency before committing to a ~10h run"
SWEEP=$(bench/sweep_concurrency.sh "${MODELS}/${FILE}" "${CTX}" 4 8 12 16 2>&1 | tail -20)
echo "$SWEEP" | tee -a "$STATUS"
BEST=$(echo "$SWEEP" | grep -oE "BEST: [0-9]+ slots" | grep -oE "[0-9]+" | head -1)
if [[ -n "$BEST" && "$BEST" -gt 0 ]]; then
  SLOTS="$BEST"; note "sweep chose ${SLOTS} slots"
else
  note "sweep inconclusive, keeping ${SLOTS} slots"
fi

note "BEGIN ${TAG}  ctx=${CTX} slots=${SLOTS} (full 1015 tasks)"
llama-server -m "${MODELS}/${FILE}" -ngl 999 \
    -c $(( CTX * SLOTS )) -np "${SLOTS}" -fa on -ctk q8_0 -ctv q8_0 \
    --port "${PORT}" --host 127.0.0.1 --jinja \
    > "bench/logs/server-${TAG}.log" 2>&1 &
SRV=$!

for _ in $(seq 1 120); do
  curl -sf "${URL}/health" > /dev/null 2>&1 && break
  sleep 5
done
if ! curl -sf "${URL}/health" > /dev/null 2>&1; then
  # most likely cause is KV overcommit; report it rather than dying silently
  note "SERVER FAILED TO LOAD — try fewer slots"
  grep -iE "error|failed|out of|alloc" "bench/logs/server-${TAG}.log" | tail -6 | tee -a "$STATUS"
  kill -9 $SRV 2>/dev/null
  exit 1
fi
note "server up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"

# R1-Distill sampling, same as the 1.5B run
.venv-vllm/bin/python bench/eval_victim_gguf.py \
    --url "$URL" --out "bench/results/${TAG}.jsonl" \
    --n-per-bench 0 --concurrency "${SLOTS}" \
    --max-tokens 32768 --temperature 0.7 --top-p 0.9 --top-k -1 --repetition-penalty 1.05 \
    > "bench/logs/${TAG}.log" 2>&1
rc=$?

kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null

if [[ $rc -ne 0 ]]; then
  note "FAILED (rc=${rc})"; tail -8 "bench/logs/${TAG}.log" | tee -a "$STATUS"
else
  note "DONE"; grep -E "acc=" "bench/logs/${TAG}.log" | tee -a "$STATUS"
fi
note "7B SURROGATE COMPLETE"
