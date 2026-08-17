#!/usr/bin/env bash
# Single driver: victim (full 1015), then 7B surrogate (sweep + full 1015).
# One process, sequential — no inter-runner coordination to get wrong.
set -uo pipefail
cd /home/asavala/Development/papers/trace-inversion
echo "[$(date +%H:%M:%S)] === victim full ==="
bash bench/run_victim_full.sh
echo "[$(date +%H:%M:%S)] === 7B surrogate ==="
bash bench/run_surrogate_7b.sh
echo "[$(date +%H:%M:%S)] === ALL REMAINING BASELINES DONE ==="
