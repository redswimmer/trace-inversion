#!/usr/bin/env bash
# Phase 4 inversion driver — ONE inverter over split B, start to finish (docs/15 §4–5):
#   merge (the epoch-2 adapter -> bf16) -> draw 1 (seed 1234) -> redraw the capped rows at 1235,
#   then 1236 -> assemble the final file (still-capped rows dropped and reported) -> stats, with
#   the phase's one read of the ORACLE file (paired lengths, read-only) -> delete the merged weights.
#
# Usage:  bench/run_phase4_invert.sh <arm: 7b|1.5b> <setting: sum|nosum>
#         LIMIT=30 bench/run_phase4_invert.sh 7b sum
#             the probe: the first 30 rows -> forged-<tag>-probe.jsonl + stats on them; the merged
#             weights are KEPT so the full run that follows reuses them (its merge-check is re-read).
#
# Copied from run_phase2_invert.sh. Runs THIS checkout's code with the main checkout's venvs and
# writes data and logs under the main checkout (docs/15 §0.1); the small committed JSONs are copied
# into this checkout's bench/results/phase4/. Sampling is invert.py's defaults, passed explicitly so
# the log states them. Every path handed to invert.py passes no_oracle(); the stats step is the one
# permitted reader of the ORACLE file. Never edit this file while it runs.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
MAIN=/home/asavala/Development/papers/trace-inversion
PY="$MAIN/.venv-vllm/bin/python"; PYT="$MAIN/.venv/bin/python"
DATA="$MAIN/bench/results"; LOGS="$MAIN/bench/logs"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

ARM="${1:?arm 7b|1.5b}"; SET="${2:?setting sum|nosum}"; LIMIT="${LIMIT:-0}"
TAG="${ARM}-${SET}"
case "$ARM" in 7b) CK=checkpoint-402;; 1.5b) CK=checkpoint-404;; *) echo "arm must be 7b or 1.5b"; exit 1;; esac
ADAPTER="$DATA/phase2/inverter-${TAG}/${CK}"          # epoch 2, decided 2026-08-30 (docs/10 Phase 4)
MERGED="bench/results/phase2/merged-${TAG}"           # relative: phase2_train.py writes it under $ROOT
PROMPTS="$DATA/phase4/attack-${SET}.jsonl"
FINAL="$DATA/phase4/forged-${TAG}.jsonl"
ORACLE="$DATA/phase3/victimB-ORACLE.jsonl"            # the stats step ONLY
LOG="$LOGS/phase4-${TAG}.log"
SAMPLING=(--temperature 0.7 --top-p 0.9 --repetition-penalty 1.05 --max-tokens 8192 --gpu-frac 0.90 --holdout '')
mkdir -p "$LOGS" "$DATA/phase4"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
no_oracle() { for p in "$@"; do [[ "${p,,}" == *oracle* ]] && { note "REFUSED: ${p} on the generation path"; exit 2; }; done; return 0; }

invert() {   # invert <seed> <prompt file> <out> [extra invert.py args]
  local seed="$1" data="$2" out="$3"; shift 3
  no_oracle "$data" "$out" "$MERGED"
  note "invert seed=${seed}  data=$(basename "$data")  out=$(basename "$out")  $* ${SAMPLING[*]}"
  PATH="$MAIN/.venv-vllm/bin:$PATH" "$PY" bench/invert.py --model "$MERGED" --data "$data" --out "$out" \
      --seed "$seed" "${SAMPLING[@]}" "$@" >> "$LOG" 2>&1
  local rc=$?
  note "invert seed=${seed} rc=${rc}:  $(tail -n 400 "$LOG" | grep -E '^([0-9]+ rows  prompt tokens|WARNING: longest|generate wall|generated )' | tail -4 | tr '\n' ' ')"
  return $rc
}

[[ -f "$ADAPTER/adapter_model.safetensors" ]] || { echo "no adapter at $ADAPTER"; exit 1; }
[[ -f "$PROMPTS" ]] || { echo "no prompts at $PROMPTS — run bench/phase4_format.py first"; exit 1; }
echo "===== [$(date '+%F %T')] launch: ${TAG}  adapter=${CK}  limit=${LIMIT} =====" >> "$LOG"
note "df: $(df -h / | tail -1 | awk '{print $4" free"}')  code=${ROOT}  data=${DATA}/phase4  adapter=${ADAPTER}"

if [[ -f "$MERGED/merge-check.json" ]] && grep -q "\"$ADAPTER\"" "$MERGED/merge-check.json"; then
  note "reusing ${MERGED} — its merge-check names ${ADAPTER}: $(tr -d '\n ' < "$MERGED/merge-check.json")"
else
  rm -rf "$MERGED"
  note "merge ${ADAPTER} -> ${MERGED}"
  "$PYT" bench/phase2_train.py --arm "$ARM" --setting "$SET" --merge --adapter "$ADAPTER" >> "$LOG" 2>&1 \
      || { note "MERGE FAILED"; tail -20 "$LOG"; rm -rf "$MERGED"; exit 1; }
  note "merge check: $(tr -d '\n ' < "$MERGED/merge-check.json")"
fi
cp "$MERGED/merge-check.json" "$DATA/phase4/merge-check-${TAG}.json"

if (( LIMIT > 0 )); then
  OUT="$DATA/phase4/forged-${TAG}-probe.jsonl"
  invert 1234 "$PROMPTS" "$OUT" --limit "$LIMIT"; rc_inv=$?
  note "probe stats — the ORACLE read (paired lengths, read-only, main thread)"
  "$PY" bench/phase1_stats.py "$OUT" --mode inverted --holdout '' --oracle "$ORACLE" --tag "${TAG} probe" 2>&1 | tee -a "$LOG"
  rc_st=${PIPESTATUS[0]}
  note "probe done (invert rc=${rc_inv}, stats rc=${rc_st}); merged weights KEPT at ${MERGED} for the full run"
  exit $(( rc_inv || rc_st ))
fi

trap 'rm -rf "$MERGED"; note "deleted ${MERGED}  df: $(df -h / | tail -1 | awk "{print \$4\" free\"}")"' EXIT
T0=$(date +%s)
D1="$DATA/phase4/forged-${TAG}-draw1.jsonl"
invert 1234 "$PROMPTS" "$D1" || { note "draw 1 FAILED"; exit 1; }
DRAWS=("$D1"); PREV="$D1"; n=1
for seed in 1235 1236; do
  n=$(( n + 1 ))
  SUB="$DATA/phase4/forged-${TAG}-draw${n}-prompts.jsonl"; DN="$DATA/phase4/forged-${TAG}-draw${n}.jsonl"
  "$PY" bench/phase4_draws.py subset --draw "$PREV" --prompts "$PROMPTS" --out "$SUB" 2>&1 | tee -a "$LOG"
  (( $(wc -l < "$SUB") > 0 )) || { note "draw $(( n - 1 )) capped nothing — no draw ${n}"; break; }
  invert "$seed" "$SUB" "$DN" || { note "draw ${n} FAILED"; exit 1; }
  DRAWS+=("$DN"); PREV="$DN"
done
note "generation wall $(( ( $(date +%s) - T0 ) / 60 )) min over ${#DRAWS[@]} draw(s) (merge excluded); assemble"
"$PY" bench/phase4_draws.py assemble --draws "${DRAWS[@]}" --prompts "$PROMPTS" --out "$FINAL" 2>&1 | tee -a "$LOG"
rc_as=${PIPESTATUS[0]}
note "stats — the ORACLE read (paired lengths, read-only, main thread)"
"$PY" bench/phase1_stats.py "$FINAL" --mode inverted --holdout '' --oracle "$ORACLE" --final --tag "${TAG}" 2>&1 | tee -a "$LOG"
rc_st=${PIPESTATUS[0]}
if [[ "$ROOT" != "$MAIN" ]]; then
  mkdir -p "$ROOT/bench/results/phase4"
  cp "$DATA/phase4/forged-${TAG}-consistency.json" "$DATA/phase4/forged-${TAG}-draws.json" \
     "$DATA/phase4/merge-check-${TAG}.json" "$ROOT/bench/results/phase4/"
fi
note "done ${TAG}: assemble rc=${rc_as}  stats rc=${rc_st}  wall $(( ( $(date +%s) - T0 ) / 60 )) min"
exit $(( rc_as || rc_st ))
