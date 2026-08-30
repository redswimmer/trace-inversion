#!/usr/bin/env bash
# Phase 3 generation driver — the victim over split B, start to finish (docs/14 §4–5).
#
# Usage: bench/run_phase3_gen.sh <slots> [extra phase1_generate.py args, e.g. --max-rows 200]
#
# Copied from run_phase1_gen.sh. What differs is fixed by docs/14 §4: the victim
# (Qwen3.8-27B IQ4_XS) on Phase 0's port 8079, NO system prompt, 16,384 ctx per slot,
# max_new_tokens 14,336 (= 16,384 − 2,048, so every kept row fits the student's context),
# --timeout 3600, prompts over 5,000 chars skipped, stop at 5,040 kept rows. The generator
# resumes from --out, so the 200-row probe and the full run share one output file and one
# pair of logs (appended; one dated header per launch).
#
# Runs THIS checkout's code (a worktree session edits bench/*.py here) with the main
# checkout's interpreter, and writes data and logs under the main checkout, where every
# other phase's gitignored artifacts live. Phase 4 reads victimB-attack.jsonl from there.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MAIN=/home/asavala/Development/papers/trace-inversion
PY="${PY:-$MAIN/.venv-vllm/bin/python}"
DATA="${DATA:-$MAIN/bench/results/phase3}"
LOGS="${LOGS:-$MAIN/bench/logs}"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

SLOTS="${1:?usage: run_phase3_gen.sh <slots> [generator args]}"; shift
CTX=16384; PORT=8079; URL="http://127.0.0.1:${PORT}"
MODEL="${HOME}/trace-inversion-bench/models/Qwen3.8-27B-IQ4_XS.gguf"
[[ -f "$MODEL" ]] || { echo "no such model: $MODEL"; exit 1; }
[[ -f "$DATA/promptsB.jsonl" ]] || { echo "no prompts at $DATA/promptsB.jsonl — run phase1_split.py --only-b"; exit 1; }

OUT="$DATA/victim-traces-ORACLE.jsonl"
SRVLOG="$LOGS/server-phase3-victim.log"
RUNLOG="$LOGS/phase3-victim.log"
mkdir -p "$LOGS" "$DATA"

note() { echo "[$(date '+%F %T')] $*" | tee -a "$RUNLOG"; }

note "df: $(df -h / | tail -1 | awk '{print $4" free"}')  code=${ROOT}  data=${DATA}  generator args: $*"
note "starting server: victim  slots=${SLOTS}  ctx/slot=${CTX}  total=$(( CTX * SLOTS ))"
echo "===== [$(date '+%F %T')] launch: slots=${SLOTS} ctx/slot=${CTX} =====" >> "$SRVLOG"
llama-server -m "$MODEL" -ngl 999 \
    -c $(( CTX * SLOTS )) -np "${SLOTS}" -fa on -ctk q8_0 -ctv q8_0 \
    --port "${PORT}" --host 127.0.0.1 --jinja \
    >> "$SRVLOG" 2>&1 &
SRV=$!

for _ in $(seq 1 120); do
  curl -sf "${URL}/health" > /dev/null 2>&1 && break
  sleep 5
done
if ! curl -sf "${URL}/health" > /dev/null 2>&1; then
  # a silent OOM at load exits non-zero and leaves the GPU idle; say so loudly
  note "SERVER FAILED TO LOAD — try fewer slots"
  grep -iE "error|failed|out of|alloc" "$SRVLOG" | tail -6 | tee -a "$RUNLOG"
  kill -9 $SRV 2>/dev/null
  exit 1
fi
note "server up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"

"$PY" bench/phase1_generate.py \
    --url "$URL" --prompts "$DATA/promptsB.jsonl" --out "$OUT" \
    --no-system --max-new-tokens 14336 --timeout 3600 --max-prompt-chars 5000 \
    --target-kept 5040 --concurrency "${SLOTS}" "$@" \
    >> "$RUNLOG" 2>&1
rc=$?

kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null

if [[ $rc -ne 0 ]]; then
  note "FAILED (rc=${rc})"; tail -12 "$RUNLOG"; exit $rc
fi
note "generation complete"
tail -12 "$RUNLOG"
# docs/14 §4.2: a prompt that did not fit its slot is truncated silently; count it here
note "server log: 'truncated' $(grep -c truncated "$SRVLOG")  'context shift' $(grep -ci 'context shift' "$SRVLOG")  'context' errors $(grep -ci 'context capacity\|context is full\|exceeds' "$SRVLOG")"
note "measuring (docs/14 §5 step 2)"
"$PY" bench/phase1_stats.py "$OUT" --mode traces --cap 14336 --cap-hit-band 0 15 \
    --trace-tokenizer Qwen/Qwen3.5-4B --paired --paired-gap-tol 100 \
    --vs-r1 --vs-r1-out "$ROOT/bench/results/phase3/victimB-vs-r1.json" 2>&1 | tee -a "$RUNLOG"
exit "${PIPESTATUS[0]}"
