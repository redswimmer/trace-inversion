#!/usr/bin/env bash
# Held-out inversion for ONE adapter checkpoint that is not the run's final one — e.g. the epoch-2
# adapter, so the checkpoint choice for Phase 4 can be made on generation behaviour, not eval_loss
# alone. Same loop as the tail of run_phase2.sh: merge -> invert the holdout -> stats -> delete merged.
#
# Usage: bench/run_phase2_invert.sh <arm> <setting> <adapter dir> <label>
#   e.g. bench/run_phase2_invert.sh 7b sum bench/results/phase2/inverter-7b-sum/checkpoint-402 ep2
# Output: bench/results/phase2/holdout-<arm>-<setting>-<label>.jsonl (+ merge-check-<label>.json)
set -uo pipefail
cd /home/asavala/Development/papers/trace-inversion
export PYTHONUNBUFFERED=1

ARM="${1:?arm}"; SET="${2:?setting}"; ADAPTER="${3:?adapter dir}"; LABEL="${4:?label}"
TAG="${ARM}-${SET}"
OUT="bench/results/phase2/inverter-${TAG}"
MERGED="bench/results/phase2/merged-${TAG}"
HOLD="bench/results/phase2/${TAG}-holdout.jsonl"
INV="bench/results/phase2/holdout-${TAG}-${LABEL}.jsonl"
LOG="bench/logs/phase2-${TAG}-${LABEL}-invert.log"
mkdir -p bench/logs
note() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[[ -f "$ADAPTER/adapter_model.safetensors" ]] || { note "no adapter at $ADAPTER"; exit 1; }
note "merge ${ADAPTER}  df: $(df -h / | tail -1 | awk '{print $4" free"}')"
.venv/bin/python bench/phase2_train.py --arm "$ARM" --setting "$SET" --merge --adapter "$ADAPTER" >> "$LOG" 2>&1 \
    || { note "MERGE FAILED"; tail -20 "$LOG"; exit 1; }
cp "$MERGED/merge-check.json" "$OUT/merge-check-${LABEL}.json"
note "invert the holdout (vLLM)"
PATH="$PWD/.venv-vllm/bin:$PATH" .venv-vllm/bin/python bench/invert.py \
    --model "$MERGED" --data "$HOLD" --out "$INV" >> "$LOG" 2>&1
rc_inv=$?
note "invert rc=${rc_inv}; stats"
.venv-vllm/bin/python bench/phase1_stats.py "$INV" --mode inverted --tag "${TAG}, ${LABEL}" 2>&1 | tee -a "$LOG"
rc_st=${PIPESTATUS[0]}
rm -rf "$MERGED"
note "deleted ${MERGED}  (invert rc=${rc_inv}, stats rc=${rc_st})"
exit $(( rc_inv || rc_st ))
