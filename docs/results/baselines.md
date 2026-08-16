# Phase 0 — Baseline Results

Zero-shot accuracy, before any training. Establishes headroom and validates the harness.

**Protocol (all runs):** MATH500 (500) + JEEBench (515) · **32,768 max generation / 40,960 context**
· seed 1234 · pass@1 single sample · answer extracted from the last `\boxed{}`.

**Acceptance gates** — a run is not reported until it passes all of:
truncation < 10% · zero extraction failures on *completed* generations · no benchmark type at
exactly 0% (n≥20) · and for the surrogate, calibration within ~8 pts of the paper's Table 6.

---

## Results

| # | Model | Role | Engine | Precision | MATH500 | JEEBench | Trunc M/J | Audit |
|---|---|---|---|---|---|---|---|---|
| 0.1 | Qwen3.5-0.8B | student cand. | vLLM | bf16 | **45.6%** | **20.2%** | 5.8% / 2.1% | ✅ PASS |
| 0.2 | Qwen3.5-2B | student cand. | vLLM | bf16 | — | — | — | running |
| 0.3 | Qwen3.5-4B | student cand. | vLLM | bf16 | — | — | — | queued |
| 0.4 | R1-Distill-Qwen-1.5B | surrogate | llama.cpp | GGUF BF16 | — | — | — | queued |
| 0.5 | Qwen3.8-27B IQ4_XS | victim | llama.cpp | GGUF 4-bit | — | — | — | queued |

### 0.1 Qwen3.5-0.8B ✅

```
JEEBench  acc= 20.2%  (completed-only 20.4%)  truncated= 2.1%  no_answer|completed= 0.0%
          tokens med=14098  p95=26833
MATH500   acc= 45.6%  (completed-only 48.2%)  truncated= 5.8%  no_answer|completed= 0.2%
          tokens med= 3130  p95=32768
JEEBench by type:  MCQ 43.6% · Integer 20.7% · MCQ(multiple) 13.4% · Numeric 10.2%
```

---

## The 16k cap was distorting everything

An earlier pass capped generation at 16,384 tokens — a number with no basis in the paper, which
sets no explicit cap at all. It cut into the *body* of the reasoning-length distribution rather
than its tail:

| Qwen3.5-0.8B | 16k cap | **32k cap** |
|---|---|---|
| JEEBench | 11.3% | **20.2%** |
| MATH500 | 43.6% | **45.6%** |
| Truncated (JEEBench) | 36.7% | **2.1%** |

JEEBench accuracy nearly doubled. The p95 explains why: **26,833 tokens** — the hardest 5% of
problems genuinely need 27k+ tokens of reasoning, so a 16k cap was severing them mid-thought.

Two diagnostics worth carrying forward:

- **Convergence of `acc` and `completed-only acc`** signals the cap has stopped binding. At 32k
  they agree (20.2 vs 20.4); at 16k they differed by 5+ points.
- **`completed-only` is a biased estimator when truncation is high** — the surviving problems are
  the easy ones. The 4B reported 100.0% on MATH500-completed at 71% truncation. Do not quote it
  above ~20% truncation.

Superseded 16k results are archived at `bench/results/v1-16k/`.

---

## Open decisions this phase resolves

1. **Student model** — highest JEEBench headroom that still terminates reliably.
2. **FFT vs LoRA** — 0.8B (~7 GB) and 2B (~15 GB) fit full fine-tuning on 24 GB; 4B (~28 GB) does
   not. Choosing 0.8B or 2B removes our single riskiest deviation from the paper.
3. **Surrogate strength** — if 0.4 lands below the chosen student on JEEBench, step up to
   R1-Distill-Qwen-7B (F16 GGUF, 14.19 GiB). At 16k the surrogate sat below the 2B (30.5 vs 45.8),
   so this looks likely.
