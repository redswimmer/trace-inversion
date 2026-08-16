#!/usr/bin/env bash
# Baselines 0.4 (surrogate) and 0.5 (victim) — both GGUF on llama.cpp.
#
# Waits for the vLLM queue to release the GPU, then runs each in turn:
# start llama-server, evaluate over its OpenAI endpoint, shut it down.
set -uo pipefail
cd /home/asavala/Development/papers/trace-inversion

MODELS="${HOME}/trace-inversion-bench/models"
PORT=8077
URL="http://127.0.0.1:${PORT}"
STATUS=bench/logs/STATUS-gguf.txt
mkdir -p bench/logs bench/results
: > "$STATUS"
note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

# The skip-marker that kept the vLLM runner from claiming 0.4 is not a result.
if [[ -f bench/results/DeepSeek-R1-Distill-Qwen-1.5B.jsonl ]] && \
   grep -q PLACEHOLDER bench/results/DeepSeek-R1-Distill-Qwen-1.5B.jsonl 2>/dev/null; then
  rm -f bench/results/DeepSeek-R1-Distill-Qwen-1.5B.jsonl
  note "removed placeholder skip-marker"
fi

note "waiting for the vLLM queue to release the GPU"
while pgrep -f eval_baseline.py > /dev/null; do sleep 60; done
sleep 30   # let VRAM drain
note "GPU free: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

# model_file | tag | n_per_bench (0 = all) | temp | top_p | top_k | ctx | slots
RUNS=(
  "DeepSeek-R1-Distill-Qwen-1.5B-BF16.gguf|DeepSeek-R1-Distill-Qwen-1.5B|0|0.6|0.95|-1|40960|16"
  "Qwen3.8-27B-IQ4_XS.gguf|Qwen3.8-27B-IQ4_XS|250|1.0|0.95|20|40960|8"
)

for spec in "${RUNS[@]}"; do
  IFS='|' read -r FILE TAG N T P K CTX SLOTS <<< "$spec"
  if [[ -f "bench/results/${TAG}.jsonl" ]]; then note "SKIP ${TAG} (done)"; continue; fi
  if [[ ! -f "${MODELS}/${FILE}" ]]; then note "MISSING ${FILE}"; continue; fi

  note "BEGIN ${TAG}  ctx=${CTX} slots=${SLOTS} n_per_bench=${N}"
  # total KV budget is shared across slots, so scale -c by slot count
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
    note "SERVER FAILED for ${TAG}"; tail -15 "bench/logs/server-${TAG}.log" | tee -a "$STATUS"
    kill -9 $SRV 2>/dev/null; continue
  fi
  note "server up for ${TAG}"

  .venv-vllm/bin/python bench/eval_victim_gguf.py \
      --url "$URL" --out "bench/results/${TAG}.jsonl" \
      --n-per-bench "$N" --concurrency "$SLOTS" \
      --max-tokens 32768 --temperature "$T" --top-p "$P" --top-k "$K" \
      > "bench/logs/${TAG}.log" 2>&1
  rc=$?

  kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 20

  if [[ $rc -ne 0 ]]; then
    note "FAILED ${TAG} (rc=${rc})"; tail -8 "bench/logs/${TAG}.log" | tee -a "$STATUS"
  else
    note "DONE ${TAG}"; grep -E "acc=" "bench/logs/${TAG}.log" | tee -a "$STATUS"
  fi
done

note "GGUF BASELINES COMPLETE"
