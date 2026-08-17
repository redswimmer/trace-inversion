# Phase 0 — Baseline Results

Zero-shot accuracy before any training. Establishes headroom, validates the harness, and picks
the student.

**Protocol:** MATH500 (500) + JEEBench (515) · 32,768 max generation / 40,960 context · seed 1234 ·
pass@1 single sample · answer = last `\boxed{...}`, falling back to an `ANSWER: x` line.

**Acceptance gates:** truncation < 10% · zero extraction failures on *completed* generations ·
no benchmark type at exactly 0% · surrogate must calibrate within ~8 pts of the paper's Table 6.

---

## Results

| # | Model | Role | Engine | Precision | MATH500 | JEEBench | Trunc M/J | Median tok (J) |
|---|---|---|---|---|---|---|---|---|
| 0.1 | Qwen3.5-0.8B | student cand. | vLLM | bf16 | 45.6% | 20.2% | 5.8% / 2.1% | 14,098 |
| 0.2 | **Qwen3.5-2B** | **student (chosen)** | vLLM | bf16 | **77.8%** | **47.0%** | 8.2% / 21.9% | 8,734 |
| 0.3 | Qwen3.5-4B | student cand. | vLLM | bf16 | 91.4% | 73.8% | 9.2% / 21.7% | 21,859 |
| 0.4a | R1-Distill-Qwen-1.5B | surrogate | vLLM | bf16 | 83.0% | 32.0% | 2.8% / 5.4% | 8,345 |
| 0.4b | R1-Distill-Qwen-1.5B | **surrogate (accepted)** | **llama.cpp** | GGUF BF16 | **84.0%** | **32.6%** | 1.6% / 3.3% | 8,347 |
| 0.5 | Qwen3.8-27B IQ4_XS | victim | llama.cpp | 4-bit | *100%* | *85.7%* | *0% / 7.1%* | *4,295* |

*0.5 is a 29-problem smoke test — indicative only, full run pending.*
*0.4a ran on vLLM by mistake; 0.4b is the accepted llama.cpp run. Both shown — they agree within ~1 pt, which is our only direct measurement of the engine split.*

### Harness calibration ✅

The surrogate is the paper's exact model, so its published Table 6 scores validate our harness
end to end:

| | vLLM | llama.cpp | Paper Table 6 |
|---|---|---|---|
| MATH500 | 83.0% | **84.0%** | 81.4% (Δ +2.6) |
| JEEBench | 32.0% | **32.6%** | 32.6% (**Δ 0.0**) |

The two engines agree within ~1 point on the same weights at the same precision — the only direct
evidence we have that the split-engine arrangement (victim on llama.cpp, students on vLLM) costs
nothing measurable.

Before the extraction fix these were +1.6 / −2.7 with 9.9% unparsed. Prompting, extraction, and
grading are sound.

---

## Finding 1 — capability shows up as brevity, not length

| | Victim 27B | 4B | 2B | 0.8B |
|---|---|---|---|---|
| JEEBench accuracy | 85.7% | 73.8% | 47.0% | 20.2% |
| **JEEBench median tokens** | **4,295** | 21,859 | 8,734 | 14,098 |

The victim solves JEEBench in **five times fewer tokens than the 4B** while scoring higher. Long
generations were smaller models flailing, not harder problems demanding more context.

Consequences:

- **32k is ample for the victim** (7.1% truncation). Phase 3 generation is safe, and since traces
  run ~4-6k tokens we can use a smaller per-slot context and therefore *more* slots, landing higher
  on the concurrency curve than the 6 slots originally budgeted.
- **Victim traces will be ~4-6k tokens**, matching the paper's 6,130-token average. Our training
  data will be the same shape as theirs.

## Finding 2 — truncation has two distinct causes

Measured by counting how often a model repeats its own final answer:

| Model | Truncated rows | Clear loops | Median repeats of own answer |
|---|---|---|---|
| Qwen3.5-0.8B | 40 | 22% | 1,338 |
| **Qwen3.5-2B** | 154 | **50%** | **3,636** |
| Qwen3.5-4B | 158 | 4% | 1 |
| R1-Distill-1.5B | 42 | 2% | 1 |

- **The 2B is pathological**: half its truncations are degenerate — it emits `\boxed{AC}` then
  repeats it a median of 3,636 times. A termination failure, not a reasoning limit. Fixed by
  `repetition_penalty` (the paper's repo uses 1.05; Qwen3.5's card recommends none — our deviation
  caused this).
- **The 4B does not loop** (4%). Its truncations are genuine unfinished reasoning at a 21,859-token
  median.

So the two models are *both* underestimated, for opposite reasons: the 2B by a generation defect,
the 4B by a real budget limit.

**Deferred, not dropped:** `repetition_penalty=1.05` is required for the Phase 6 protocol, where
trained students are compared across five conditions with margins the paper measured at 0.4-2.4
points. Re-running baselines to a protocol we will change anyway is wasted GPU.

## Finding 3 — the 16k cap was measuring itself

An earlier pass capped generation at 16,384 — a number with no basis in the paper, which sets no
explicit cap (their eval passes only `pretrained`, `tensor_parallel_size`,
`gpu_memory_utilization`, `batch_size 16`, `seed 1234`).

| | 16k cap | **32k cap** |
|---|---|---|
| Qwen3.5-4B MATH500 | 73.6% | **91.4%** |
| Qwen3.5-4B JEEBench | 30.1% | **73.8%** |
| Qwen3.5-0.8B JEEBench | 11.3% | **20.2%** |

The 4B's JEEBench score more than doubled. The cap was hiding most of its capability, not trimming
a tail.

Two diagnostics worth keeping:

- **`acc` converging with `completed-only acc`** signals the cap has stopped binding.
- **`completed-only` is biased upward when truncation is high** — surviving problems are the easy
  ones. The 4B reported 100.0% MATH500-completed at 71% truncation. Do not quote above ~20%.

Superseded 16k results archived at `bench/results/v1-16k/`.

---

## Decisions

### Student: **Qwen3.5-2B, full fine-tuning** ✅

Not the strongest model — the 4B beats it on both benchmarks — but the right experimental subject:

| | 2B | 4B |
|---|---|---|
| Headroom to victim (JEEBench) | **39 pts** (47.0 → 85.7) | 12 pts (73.8 → 85.7) |
| Full fine-tuning on 24 GB | ✅ ~15 GB | ❌ ~28 GB |
| Matches paper's method | ✅ | ✗ forces LoRA |

The 4B at 73.8% JEEBench sits only 12 points below the victim — close to the degenerate regime
where the student already rivals the teacher and inversion has nothing to demonstrate. The paper
worked with a 59-point gap (R1 87.1 vs Qwen2.5-7B 28.3).

Choosing the 2B also **removes our single riskiest deviation**: full fine-tuning fits, so we match
the paper's method rather than substituting LoRA.

### Victim: **Qwen3.8-27B IQ4_XS on llama.cpp** ✅

Decided on measurement (`docs/08`): 47.3 t/s single-stream, 303 t/s at concurrency 32, ~23 h for 5k
traces. Only model forced below full precision — 55 GB at bf16, and GGUF is the only format that
quantizes everything.

### Surrogate: **OPEN** ⏳

R1-Distill-1.5B is the paper's exact model, but against our 2B student it splits:

| | Surrogate 1.5B | 2B student |
|---|---|---|
| MATH500 | 83.0% ✅ above | 77.8% |
| JEEBench | 32.0% ❌ **15 pts below** | 47.0% |

The paper had its surrogate above the student on *both* (81.4/32.6 vs 71.2/28.3). The likely cause
is coverage, not reasoning: R1-Distill was distilled on math, so MATH500 is its home turf, while
JEEBench spans physics and chemistry where a 2026-generation general model simply knows more.

Whether this matters depends on the surrogate's role. As a *format teacher* (the paper's framing),
narrow coverage is tolerable — it only has to demonstrate what a long trace looks like. But it also
defines the `Surrogate-Trace` baseline condition, and beating a baseline that is 15 points below
the student proves less.

**Next:** re-run 1.5B on llama.cpp (in progress), then evaluate **R1-Distill-Qwen-7B** (F16 GGUF,
14.19 GiB) and compare. Running both also yields the surrogate-strength sweep the paper never did —
they tested only 1.5B and 685B, with nothing in between.

---

## Known issues

| Issue | Status |
|---|---|
| Surrogate ran on vLLM, not llama.cpp — skip-marker was self-defeating (GGUF runner deleted it to claim the job, re-enabling the vLLM run) | re-running |
| 2B looping (50% of truncations) — needs `repetition_penalty=1.05` | deferred to Phase 6 |
| Victim baseline OOM'd at 8 slots × 40,960 (327k total KV) — llama.cpp shares one KV budget across slots | fixed: 6 slots |
| `stratified()` dropped the `type` column — pandas 3 excludes grouping columns from `groupby.apply()` | fixed |
| GGUF error handler raised on missing key, masking the real request error | fixed |
| Extraction missed R1-Distill's `ANSWER: x` convention (9.9% of its completed generations) | fixed + re-graded offline |
| **`math_verify` silently degraded to string matching inside worker threads** — it uses SIGALRM timeouts and `signal.signal()` only works in the main thread. Cost **14.6 pts** on MATH500 (68.4 → 84.0) and looked like an engine difference | fixed: grade in main thread after the pool drains |
| Two chained runners hung in `pgrep` wait loops, GPU idle ~50 min — caused by editing a running script (bash reads incrementally) | fixed: single sequential driver + watchdog |
