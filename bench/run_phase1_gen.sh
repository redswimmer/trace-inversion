#!/usr/bin/env bash
# Phase 1 generation driver — one surrogate arm, start to finish.
#
# Usage: bench/run_phase1_gen.sh <model.gguf> <tag> <slots> [per_slot_ctx] [port]
#
# Per-slot context must cover the 8192-token cap plus the prompt; 10240 is what the
# sweeps assume. -c is the TOTAL KV budget across -np slots, so total = ctx * slots.
#
# The generation itself stops at --target-kept clean rows rather than a fixed row
# count, because the cap-hit rate for a distill is unmeasured.
set -uo pipefail
cd /home/asavala/Development/papers/trace-inversion

MODEL="${1:?usage: run_phase1_gen.sh <model.gguf> <tag> <slots> [ctx] [port]}"
TAG="${2:?tag}"
SLOTS="${3:?slots}"
CTX="${4:-10240}"
PORT="${5:-8078}"
MODELS="${HOME}/trace-inversion-bench/models"
[[ -f "$MODEL" ]] || MODEL="${MODELS}/${MODEL}"
[[ -f "$MODEL" ]] || { echo "no such model: $MODEL"; exit 1; }

URL="http://127.0.0.1:${PORT}"
OUT="bench/results/phase1/traces-${TAG}.jsonl"
SRVLOG="bench/logs/server-phase1-${TAG}.log"
RUNLOG="bench/logs/phase1-${TAG}.log"
mkdir -p bench/logs bench/results/phase1

note() { echo "[$(date +%H:%M:%S)] $*"; }

note "df: $(df -h / | tail -1 | awk '{print $4" free"}')"
note "starting server: ${TAG}  slots=${SLOTS}  ctx/slot=${CTX}  total=$(( CTX * SLOTS ))"
llama-server -m "$MODEL" -ngl 999 \
    -c $(( CTX * SLOTS )) -np "${SLOTS}" -fa on -ctk q8_0 -ctv q8_0 \
    --port "${PORT}" --host 127.0.0.1 --jinja \
    > "$SRVLOG" 2>&1 &
SRV=$!

for _ in $(seq 1 120); do
  curl -sf "${URL}/health" > /dev/null 2>&1 && break
  sleep 5
done
if ! curl -sf "${URL}/health" > /dev/null 2>&1; then
  # a silent OOM at load exits non-zero and leaves the GPU idle; say so loudly
  note "SERVER FAILED TO LOAD — try fewer slots"
  grep -iE "error|failed|out of|alloc" "$SRVLOG" | tail -6
  kill -9 $SRV 2>/dev/null
  exit 1
fi
note "server up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"

.venv-vllm/bin/python bench/phase1_generate.py \
    --url "$URL" --out "$OUT" \
    --target-kept 5000 --concurrency "${SLOTS}" \
    > "$RUNLOG" 2>&1
rc=$?

kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null

if [[ $rc -ne 0 ]]; then
  note "FAILED (rc=${rc})"; tail -12 "$RUNLOG"; exit $rc
fi
note "generation complete"
tail -12 "$RUNLOG"
note "measuring"
.venv-vllm/bin/python bench/phase1_stats.py "$OUT" --mode traces 2>/dev/null
