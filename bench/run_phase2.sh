#!/usr/bin/env bash
# Phase 2 driver — ONE inverter, the closed loop (docs/13 §4.6):
#   train (3 epochs, eval + checkpoint per epoch) -> merge -> invert the holdout -> stats -> delete merged
#
# Usage: bench/run_phase2.sh <arm: 7b|1.5b> <setting: sum|nosum> [attn_implementation]
#
# The 20-step probe (phase2_train.py --max-steps 20) runs BEFORE this, by hand, and its
# projection is reported before the long run starts. Never edit this file while it runs.
set -uo pipefail
cd /home/asavala/Development/papers/trace-inversion

ARM="${1:?usage: run_phase2.sh <7b|1.5b> <sum|nosum> [attn]}"
SET="${2:?setting}"
ATTN="${3:-kernels-community/flash-attn2}"
TAG="${ARM}-${SET}"
OUT="bench/results/phase2/inverter-${TAG}"
MERGED="bench/results/phase2/merged-${TAG}"
HOLD="bench/results/phase2/${TAG}-holdout.jsonl"
INV="bench/results/phase2/holdout-${TAG}.jsonl"
LOG="bench/logs/phase2-${TAG}.log"
mkdir -p bench/logs

note() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

note "start ${TAG}  df: $(df -h / | tail -1 | awk '{print $4" free"}')  gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
note "train (attn ${ATTN})"
.venv/bin/python bench/phase2_train.py --arm "$ARM" --setting "$SET" --attn "$ATTN" >> "$LOG" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then note "TRAIN FAILED rc=${rc}"; tail -30 "$LOG"; exit $rc; fi
note "train done; merge"
.venv/bin/python bench/phase2_train.py --arm "$ARM" --setting "$SET" --merge >> "$LOG" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then note "MERGE FAILED rc=${rc}"; tail -30 "$LOG"; exit $rc; fi
note "merge done; invert the holdout (vLLM)"
PATH="$PWD/.venv-vllm/bin:$PATH" .venv-vllm/bin/python bench/invert.py \
    --model "$MERGED" --data "$HOLD" --out "$INV" >> "$LOG" 2>&1
rc_inv=$?
note "invert rc=${rc_inv}; stats"
.venv-vllm/bin/python bench/phase1_stats.py "$INV" --mode inverted --tag "${TAG}, epoch 3" 2>&1 | tee -a "$LOG"
rc_st=${PIPESTATUS[0]}
rm -rf "$MERGED"
note "deleted ${MERGED}; adapter kept at ${OUT}  (invert rc=${rc_inv}, stats rc=${rc_st})"
exit $(( rc_inv || rc_st ))
