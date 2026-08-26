# Concurrency sweep — DeepSeek-R1-Distill-Qwen-7B-F16

Per-slot context: 10240 · **generation probe: 4096 tokens** · KV q8_0 · flash-attn on ·
Vulkan backend (llama.cpp b10450) · 2026-08-26 09:34

Re-probe of the `ctx10240` sweep at a realistic generation length. That sweep used
`-ntg 512`; per-token attention cost grows with KV depth, so a short probe measures the
latency-bound regime and can overstate sustained throughput. This one spends the KV it
allocates.

| Slots | Total KV | Gen t/s | Total t/s | Result |
|---:|---:|---:|---:|---|
| 8 | 81920 | 394.48 | 406.31 | ok |
| 16 | 163840 | 663.71 | 683.13 | ok |
| 24 | 245760 | 926.90 | 953.34 | ok |
| **32** | **327680** | **1192.44** | **1225.56** | ok |

**Best: 32 slots at 1192.44 gen t/s.** Use `-np 32 -c 327680`.

## The depth penalty is small — 2 to 6%

| Slots | `-ntg 512` | `-ntg 4096` | Δ |
|---:|---:|---:|---:|
| 8 | 402.02 | 394.48 | −1.9% |
| 16 | 687.53 | 663.71 | −3.5% |
| 24 | 982.72 | 926.90 | −5.7% |
| 32 | 1270.38 | 1192.44 | −6.1% |

The penalty grows with slot count, as expected — more slots means more KV to read per
decode step — but it stays small. **The original `ctx10240` sweep was substantially
right**, and its 32-slot recommendation stands.

## Why this sweep was run anyway

A 30-row smoke test measured 142 t/s effective and the server log reported 7.85 t/s per
slot, which together looked like an 8× shortfall against the swept 1,270. Both numbers
were artifacts, in opposite directions:

- **142 t/s was too low.** Thirty rows across 32 slots is a single wave, not a steady
  state. Most slots finished early and idled while one capped row generated alone; the
  run's wall clock was just that one generation's duration (8192 × 127.43 ms = 17.4 min,
  which was the whole run's 17.4 min).
- **7.85 t/s per slot was too low to multiply.** It was logged while the pool was
  draining, when few sequences were active. Per-slot wall-clock rate during a drain does
  not scale to a saturated pool.

Neither is evidence about sustained throughput. The lesson is not about sweeping — it is
that **a run that never fills its queue cannot measure throughput at all**, and the same
drain artifact makes in-flight cap-hit read low (short generations finish first) while
in-flight throughput reads high.

Still true, and worth keeping: probe at the generation length the workload uses. Here it
cost 6% rather than the 8× first suspected, but the filename now records `ntg` either
way, so a sweep cannot be quoted for a workload it did not measure.

Prior sweeps: `...-ctx32768.md` (Phase 0, 8 slots / 274 t/s), `...-ctx10240.md`
(`-ntg 512`, the counter-example).
