# Concurrency sweep — DeepSeek-R1-Distill-Qwen-7B-F16

Per-slot context: 10240 · KV cache q8_0 · flash-attn on · 2026-08-26 07:54

Sized for Phase 1, where `max_new_tokens` is 8192 (not the 32768 used for Phase 0 baselines),
so a slot only needs to cover ~8192 generated + ~2048 prompt tokens.

| Slots | Total KV | Gen t/s | Total t/s | Result |
|---:|---:|---:|---:|---|
| 8 | 81920 | 402.02 | 773.57 | ok |
| 16 | 163840 | 687.53 | 1294.60 | ok |
| 24 | 245760 | 982.72 | 1804.16 | ok |
| 32 | 327680 | 1270.38 | 2278.39 | ok |
| 40 | 409600 | — | — | ❌ OOM (KV cache) |

**Best: 32 slots at 1270.38 gen t/s.** Use `-np 32 -c 327680`.

Throughput was still climbing at 32 slots — the ceiling is the KV cache (~330–410k total
tokens), not compute. There is no more to win without dropping per-slot context below the
8192-token generation cap, which would truncate traces.

## Why this matters

The Phase 0 sweep of the same model at 32768/slot topped out at **8 slots / 274 gen t/s**
and the real run realized 241 t/s (88% of swept). Shrinking per-slot context to what Phase 1
actually needs is a **4.6× throughput gain on identical weights and hardware** — the largest
single speedup available anywhere in this project, and it costs nothing.

Phase 0 sweep for comparison: `DeepSeek-R1-Distill-Qwen-7B-F16-ctx32768.md`.
