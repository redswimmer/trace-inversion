# Concurrency sweep — DeepSeek-R1-Distill-Qwen-7B-F16

Per-slot context: 32768 · KV cache q8_0 · flash-attn on · 2026-08-18 02:23

| Slots | Total KV | Gen t/s | Total t/s | Result |
|---:|---:|---:|---:|---|
| 4 | 131072 | 216.24 | 288.32 | ok |
| 8 | 262144 | 274.20 | 534.92 | ok |
| 12 | 393216 | — | — | ❌ OOM (KV cache) |

**Best: 8 slots at 274.20 gen t/s.**

Use `-np 8 -c 262144`. If throughput was still climbing at the
highest slot count that fit, the ceiling is memory, not compute — consider reducing
per-slot context if the workload's tail allows it.
