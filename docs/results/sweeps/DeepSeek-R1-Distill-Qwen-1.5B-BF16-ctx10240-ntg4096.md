# Concurrency sweep — DeepSeek-R1-Distill-Qwen-1.5B-BF16

Per-slot context: 10240 · **generation probe: 4096 tokens** · KV q8_0 · flash-attn on · 2026-08-28 08:16

Numbers below are only valid for generations near 4096 tokens. A shorter probe
overstates sustained throughput because per-token attention cost grows with KV depth.

| Slots | Total KV | Gen t/s | Total t/s | Result |
|---:|---:|---:|---:|---|
| 48 | 491520 | 3233.32 | 3372.01 | ok |
| 64 | 655360 | 4552.73 | 5006.02 | ok |
| 96 | 983040 | 3379.69 | 3737.90 | ok |
| 128 | 1310720 | 5097.33 | 5590.00 | ok |

**Best: 128 slots at 5097.33 gen t/s.**

Use `-np 128 -c 1310720`. If throughput was still climbing at the
highest slot count that fit, the ceiling is memory, not compute — consider reducing
per-slot context if the workload's tail allows it.
