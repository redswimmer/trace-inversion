#!/usr/bin/env bash
# Detect and recover from a stalled queue.
#
# Twice now the GPU has sat idle with work pending: once for ~9 h after a
# silent OOM at model load, once for ~50 min after two chained runners hung in
# their wait loops. Both were invisible until someone looked. This notices
# within ~15 minutes and restarts the driver.
#
# Stall = GPU under 10% utilization for 3 consecutive checks, expected results
# still missing, and no driver process alive.
set -uo pipefail
cd /home/asavala/Development/papers/trace-inversion

LOG=bench/logs/WATCHDOG.log
INTERVAL=300          # 5 min between checks
IDLE_STRIKES=3        # ~15 min idle before acting
DRIVER="bench/run_remaining.sh"

EXPECTED=(
  "bench/results/Qwen3.8-27B-IQ4_XS-full.jsonl"
  "bench/results/DeepSeek-R1-Distill-Qwen-7B-gguf.jsonl"
)

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

strikes=0
restarts=0
say "watchdog started (interval ${INTERVAL}s, ${IDLE_STRIKES} strikes)"

while true; do
  sleep "$INTERVAL"

  missing=0
  for f in "${EXPECTED[@]}"; do [[ -f "$f" ]] || missing=$((missing+1)); done
  if [[ $missing -eq 0 ]]; then
    say "all expected results present — watchdog exiting"
    exit 0
  fi

  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  driver_alive=0
  pgrep -f "$DRIVER" > /dev/null && driver_alive=1
  server_alive=0
  pgrep -f "llama-server" > /dev/null && server_alive=1

  if [[ "${util:-0}" -lt 10 && $driver_alive -eq 0 && $server_alive -eq 0 ]]; then
    strikes=$((strikes+1))
    say "STALL SUSPECTED (${strikes}/${IDLE_STRIKES}) util=${util}% missing=${missing} driver=0 server=0"
    if [[ $strikes -ge $IDLE_STRIKES ]]; then
      if [[ $restarts -ge 3 ]]; then
        say "ALERT: already restarted ${restarts}x — not restarting again, needs a human"
        exit 1
      fi
      restarts=$((restarts+1))
      say "ALERT: restarting driver (attempt ${restarts}); ${missing} result(s) still missing"
      nohup setsid bash "$DRIVER" >> bench/logs/remaining.log 2>&1 &
      strikes=0
      sleep 120
    fi
  else
    [[ $strikes -gt 0 ]] && say "recovered (util=${util}% driver=${driver_alive} server=${server_alive})"
    strikes=0
  fi
done
