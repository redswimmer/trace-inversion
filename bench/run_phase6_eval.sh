#!/usr/bin/env bash
# Phase 6 driver (docs/17 §5): render gate -> eval -> audit -> summary append.
#
#   run_phase6_eval.sh probe <run-name> <model-path> [seed]   # gate + --limit 25 (50 tasks), timed
#   run_phase6_eval.sh full  <run-name> <model-path> [seed]   # gate + 1,015 tasks + audits + summary
#
# Worktree rules (docs/16 §0.1): runs THIS branch's bench/ code with the MAIN
# checkout's .venv-vllm by absolute path; jsonl + logs go under the main
# checkout's bench/ (gitignored there); only summary.json lands in this tree.
# The probe/full split exists because a CHECKPOINT to big-boss sits between them.
set -euo pipefail
export PYTHONUNBUFFERED=1

MAIN=/home/asavala/Development/papers/trace-inversion
PY="$MAIN/.venv-vllm/bin/python"
# activation-equivalent: vLLM's engine core subprocess needs the venv's ninja on PATH
export PATH="$MAIN/.venv-vllm/bin:$PATH"
BENCH="$(cd "$(dirname "$0")" && pwd)"
OUT="$MAIN/bench/results/phase6"
LOGS="$MAIN/bench/logs/phase6"
SUMMARY="$BENCH/results/phase6/summary.json"
mkdir -p "$OUT" "$LOGS"

mode="${1:?probe|full}"; run="${2:?run name}"; model="${3:?model path}"; seed="${4:-1234}"

# baseline gets the once-only both-ways diff (docs/17 §4.2); every model gets the flagged assert
diffboth=""
if [ "$run" = "baseline-think" ]; then diffboth="--diff-both"; fi

cd "$BENCH"
"$PY" phase6_render_gate.py --model "$model" $diffboth 2>&1 | tee "$LOGS/$run.$mode.gate.log"

spot=""
if [ "$run" = "baseline-think" ]; then spot="--spot-check-numeric 5"; fi

case "$mode" in
  probe)
    t0=$(date +%s)
    "$PY" eval_baseline.py --model "$model" --out "$OUT/$run.probe.jsonl" \
      --limit 25 --enable-thinking --max-tokens 32768 --max-len 40960 \
      --gpu-frac 0.90 --seed "$seed" 2>&1 | tee "$LOGS/$run.probe.log"
    t1=$(date +%s)
    echo "PROBE WALL: $((t1 - t0)) s for 50 tasks"
    ;;
  full)
    t0=$(date +%s)
    "$PY" eval_baseline.py --model "$model" --out "$OUT/$run.jsonl" \
      --enable-thinking --max-tokens 32768 --max-len 40960 \
      --gpu-frac 0.90 --seed "$seed" 2>&1 | tee "$LOGS/$run.log"
    t1=$(date +%s)
    echo "FULL WALL: $((t1 - t0)) s"
    "$PY" phase6_audit.py --jsonl "$OUT/$run.jsonl" --run "$run" --model "$model" \
      --seed "$seed" --wall-s "$((t1 - t0))" --summary "$SUMMARY" $spot \
      2>&1 | tee "$LOGS/$run.audit.log"
    "$PY" audit_results.py "$OUT/$run.jsonl" 2>&1 | tee -a "$LOGS/$run.audit.log"
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

df -BG / | tail -1
