#!/usr/bin/env bash
# Phase 5 driver — ONE condition: [probe or] train -> weights-only save -> delete checkpoints
# -> load check (all inside phase5_train.py, which exits non-zero on any gate).
#
# Usage: bench/run_phase5_train.sh <condition> [probe]
#
# Worktree layout (docs/16 §0.1): this runs the BRANCH's code with the MAIN checkout's .venv,
# and writes data/logs under the main checkout's bench/. The 30-step probe runs before every
# full run and its projection is CHECKPOINTed before the run starts. Never edit this while it runs.
# Driver hygiene (docs/16 §6): no path with ORACLE reaches training — structural, since
# phase5_train.py derives its one data path from the condition name (data/<condition>.jsonl).
set -uo pipefail
cd "$(dirname "$0")/.."
MAIN=/home/asavala/Development/papers/trace-inversion
PY="$MAIN/.venv/bin/python"
export PYTHONUNBUFFERED=1

COND="${1:?usage: run_phase5_train.sh <condition> [probe|full] [save-only]}"
MODE="${2:-full}"
EXTRA=""
if [[ "${3:-}" == save-only ]]; then EXTRA="--save-only-model"; fi
SUFFIX=""
if [[ "$MODE" == probe ]]; then SUFFIX="-probe"; fi
LOG="$MAIN/bench/logs/phase5-${COND}${SUFFIX}.log"
mkdir -p "$MAIN/bench/logs"

note() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

FREE_GB=$(( $(df --output=avail -B1G / | tail -1) ))
note "start ${COND} ${MODE}  disk ${FREE_GB}G free  gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
if [[ "$MODE" != probe && $FREE_GB -lt 10 ]]; then
  note "STOP: ${FREE_GB}G free < 10G before a full run (docs/16 §6)"; exit 1
fi

if [[ "$MODE" == probe ]]; then
  $PY bench/phase5_train.py --condition "$COND" --max-steps 30 --data-root "$MAIN/bench/results" $EXTRA >> "$LOG" 2>&1
else
  $PY bench/phase5_train.py --condition "$COND" --data-root "$MAIN/bench/results" $EXTRA >> "$LOG" 2>&1
fi
rc=$?
note "${COND} ${MODE} done rc=${rc}  disk $(df --output=avail -B1G / | tail -1 | tr -d ' ')G free"
[[ $rc -ne 0 ]] && tail -30 "$LOG"
exit $rc
