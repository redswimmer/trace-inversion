# Concurrency sweep — Qwen3.8-27B-IQ4_XS

Per-slot context: 16384 · **generation probe: 4096 tokens** · KV q8_0 · flash-attn on ·
Vulkan backend (llama.cpp b10450, `llama-batched-bench -npp 512`) · 2026-08-30 09:17

Phase 3's shape (docs/14 §4.2): 16,384 per slot = the 14,336 generation cap + prompt. The
`docs/08` sweep was at 2,560 per slot and Phase 0 ran 6 × 32,768; neither is this run's shape.

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

## Peak GPU memory per point (`nvidia-smi`, 5-s samples, 24,564 MiB card)

| Slots | Total KV | Peak MiB | |
|---:|---:|---:|---|
| 8 | 131072 | 20,741 | |
| 12 | 196608 | 23,422 | same KV total Phase 0 ran (6 × 32,768, 21,910 MiB in `llama-server`); the batched-bench figure includes a 12 × 512 prompt batch. `llama-server` at `-np 12 -c 196608` for the Phase 3 run: 22,940 MiB after load, 23,164 MiB while generating |
| 16 | 262144 | — | weights loaded (14,440 MiB sampled), then the recurrent-state cache allocation failed and the process exited within ~10 s |

Two points that both load rank monotonically (140.6 → 164.9), so the table is clean enough to rank
with. It ranks; it does not budget — the realized rate comes from the first 30 minutes of the run
with a full queue (`docs/11` §5).
