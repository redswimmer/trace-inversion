#!/usr/bin/env bash
# Measure optimal slot count for a GGUF model before committing to a long run.
#
# Why this is not optional: slot count is model-size dependent and arithmetic
# alone gets it badly wrong. The 1.5B surrogate ran 4 h at 12 slots and never
# exceeded 55% GPU utilization — a 1.5B is latency-bound and needs dozens of
# concurrent sequences to saturate a 4090, while a 27B saturates on a handful.
#
# Constraint: in llama.cpp, -c is the TOTAL KV budget shared across -np slots,
# so slots and per-slot context trade directly. Per-slot context must still
# cover the longest generation you expect, or long requests fail.
#
# Usage: bench/sweep_concurrency.sh <model.gguf> <per_slot_ctx> [slots...]
set -uo pipefail

MODEL="${1:?usage: sweep_concurrency.sh <model.gguf> <per_slot_ctx> [slots...]}"
CTX="${2:?per-slot context}"
shift 2
SLOTS=("${@:-1 4 8 16 24 32}")
[[ $# -eq 0 ]] && SLOTS=(1 4 8 16 24 32)

MODELS="${HOME}/trace-inversion-bench/models"
[[ -f "$MODEL" ]] || MODEL="${MODELS}/${MODEL}"
[[ -f "$MODEL" ]] || { echo "no such model: $MODEL"; exit 1; }

TAG=$(basename "$MODEL" .gguf)
OUT="${HOME}/Development/papers/trace-inversion/docs/results/sweeps/${TAG}-ctx${CTX}.md"
mkdir -p "$(dirname "$OUT")"

{
  echo "# Concurrency sweep — ${TAG}"
  echo
  echo "Per-slot context: ${CTX} · KV cache q8_0 · flash-attn on · $(date '+%Y-%m-%d %H:%M')"
  echo
  echo "| Slots | Total KV | Gen t/s | Total t/s | Result |"
  echo "|---:|---:|---:|---:|---|"
} > "$OUT"

BEST_S=0; BEST_TPS=0
for S in "${SLOTS[@]}"; do
  TOTAL=$(( CTX * S ))
  echo "[$(date +%H:%M:%S)] probing ${S} slots (total ctx ${TOTAL})…"
  # short generations keep the probe quick; we want relative scaling, not absolutes
  RES=$(timeout 1800 llama-batched-bench -m "$MODEL" -ngl 999 -c "$TOTAL" \
        -npp 512 -ntg 512 -npl "$S" -fa on -ctk q8_0 -ctv q8_0 2>&1)

  ROW=$(echo "$RES" | grep -E "^\|" | tail -1)
  if [[ -z "$ROW" ]]; then
    REASON="OOM/failed"
    echo "$RES" | grep -qi "rs cache"  && REASON="OOM (recurrent-state cache)"
    echo "$RES" | grep -qi "kv cache"  && REASON="OOM (KV cache)"
    echo "| ${S} | ${TOTAL} | — | — | ❌ ${REASON} |" >> "$OUT"
    echo "   ${S} slots: ${REASON} — stopping sweep"
    break
  fi

  TG=$(echo "$ROW"  | awk -F'|' '{gsub(/ /,"",$9);  print $9}')
  TOT=$(echo "$ROW" | awk -F'|' '{gsub(/ /,"",$11); print $11}')
  echo "| ${S} | ${TOTAL} | ${TG} | ${TOT} | ok |" >> "$OUT"
  echo "   ${S} slots: gen ${TG} t/s, total ${TOT} t/s"

  # track the best generation throughput seen
  if awk "BEGIN{exit !(${TG:-0} > ${BEST_TPS})}"; then BEST_TPS=$TG; BEST_S=$S; fi
  sleep 5
done

{
  echo
  echo "**Best: ${BEST_S} slots at ${BEST_TPS} gen t/s.**"
  echo
  echo "Use \`-np ${BEST_S} -c $(( CTX * BEST_S ))\`. If throughput was still climbing at the"
  echo "highest slot count that fit, the ceiling is memory, not compute — consider reducing"
  echo "per-slot context if the workload's tail allows it."
} >> "$OUT"

echo
echo "=== BEST: ${BEST_S} slots @ ${BEST_TPS} gen t/s ==="
echo "written to ${OUT}"
