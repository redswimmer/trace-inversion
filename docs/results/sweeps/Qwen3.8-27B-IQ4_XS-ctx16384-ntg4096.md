# Concurrency sweep — Qwen3.8-27B-IQ4_XS

Per-slot context: 16384 · **generation probe: 4096 tokens** · KV q8_0 · flash-attn on · 2026-08-30 09:17

Numbers below are only valid for generations near 4096 tokens. A shorter probe
overstates sustained throughput because per-token attention cost grows with KV depth.

| Slots | Total KV | Gen t/s | Total t/s | Result |
|---:|---:|---:|---:|---|
| 8 | 131072 | 140.64 | 157.03 | ok |
| 12 | 196608 | 164.94 | 183.90 | ok |
| 16 | 262144 | — | — | ❌ OOM (recurrent-state cache) |

**Best: 12 slots at 164.94 gen t/s.**

Use `-np 12 -c 196608`. If throughput was still climbing at the
highest slot count that fit, the ceiling is memory, not compute — consider reducing
per-slot context if the workload's tail allows it.
