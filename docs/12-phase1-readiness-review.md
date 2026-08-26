# Phase 1 Readiness Review

**Date:** 2026-08-26 · **Verdict: GO**, after four corrections below are applied to the plan.

Phase 0's model roles are sound and survive scrutiny. This review's job was the part Phase 0
explicitly deferred: `docs/08` §6 lists *"training throughput / VRAM for the TRL stages"* as **not
yet measured**, and `docs/06` §4.9 marks QLoRA on `qwen3_5` **UNVERIFIED**. Every training-stage
number in the plan was arithmetic. The student choice — 2B over the higher-scoring 4B — rests
entirely on *"full fine-tuning fits ~15 GB"*. If that were wrong, the Phase 0 student decision
unwinds and Phase 1's data would be shaped for the wrong consumer.

It is not wrong. But four other things are.

---

## 1. Training feasibility — measured, first time

RTX 4090 · bf16 · gradient checkpointing · batch 1 · `Qwen3_5ForCausalLM` (text-only) ·
`torch.cuda.max_memory_allocated()`. Optimizer states added analytically at 2 B/param (8-bit)
and 4 B/param (bf16). Usable budget ≈ **23.4 GiB**.

| Model | Role | Mode | @8192 | @16384 |
|---|---|---|---|---|
| Qwen3.5-2B | student | **FFT + 8-bit Adam** | 12.64 GiB ✅ | **15.79 GiB ✅** |
| Qwen3.5-2B | student | FFT + bf16 Adam | 16.15 GiB ✅ | 19.30 GiB ✅ |
| Qwen3.5-4B | inverter | FFT | 27.81 GiB ❌ | OOM ❌ |
| Qwen3.5-4B | inverter | **LoRA, bf16 base** | **15.11 GiB ✅** | 21.47 GiB ⚠️ |

At the inverter's chosen `max_length` **12288**, bf16 LoRA measures **18.30 GiB** — 5.1 GiB of
headroom. Every figure in this table was re-run independently and reproduced to ±0.01 GiB.

Four results, in order of how much they change the plan:

1. **The student fits at the paper's full `cutoff_len=16384`, not merely at 8192.**
   Deviation `09` §3.5 ("sequence length 8192, TBD") closes as *no deviation*.
2. **The inverter does not need QLoRA.** A bf16 base with LoRA fits at 15.11 GiB @8k. Dropping
   NF4 removes the `bitsandbytes` dependency, retires gotcha 9 (*"QLoRA on qwen3_5 is
   UNVERIFIED"*) by making it moot, and avoids quantizing the one model in the project that must
   learn a genuinely new behaviour. Keep QLoRA in reserve only if 16384 is wanted for the
   inverter — 21.47 GiB leaves too little margin for a multi-hour run.
3. **DeltaNet trains on Ada/SM89.** Forward + backward through the 48 gated-delta-net layers with
   gradient checkpointing works. This was the largest unverified architectural risk in the plan,
   and nothing else in the project had a fallback for a hybrid-attention model that simply could
   not be trained.
4. **`chunked_nll` is confirmed load-bearing.** Naive fp32 logits + gradient at 248,320 vocab is
   **15.16 GiB at 8k and 30.31 GiB at 16k** — larger than the card, before weights. `06` §4.7
   called this "the whole ballgame"; it is.

Also confirmed: `Qwen3_5ForCausalLM` loads **text-only**, vision tower absent. The VLM-misdetection
hazard (`06` gotcha 1) is real, but it can be shut off at the model class as well as at
`processing_class`.

**Not verified:** no `trl` / `peft` / `bitsandbytes` is installed anywhere — `.venv` holds
pytest/ruff/ty only. These are raw-transformers lower bounds; TRL adds its own overhead. The
`06` §4.10 smoke test still gates Phase 2.

---

## 2. The 8192 generation cap truncates 27% of the reference distribution

`docs/11` §3 calls `max_new_tokens 8192` "the paper's value" and says *"most traces fit, but
record the truncation rate."* Two problems.

**It is not paper v2's value.** Paper v2's methodology specifies no generation cap. 8192 comes
from the **v1 released code** (`07` §1.1) — the code this project's own rule says *"reproduces v1's
numbers, not the ones in the paper you are reading."* It was also set for R1, a far more
token-efficient reasoner than a distill.

**And "most traces fit" is measured from the mean.** I tokenized sampled
`llamafactory/OpenThoughts-114k` rows — the exact Phase 1 input — with the R1-Distill tokenizer,
measuring R1's own ground-truth `<think>` traces:

| Domain | corpus share | median | mean | **>8192** |
|---|---:|---:|---:|---:|
| math (`numina_math`) | 78.2% | 4,488 | 5,981 | 25.2% |
| code | 17.5% | 5,200 | 7,076 | 32.3% |
| science + puzzle | 4.4% | ~1,300 | ~2,100 | ~4% |
| **all** | | **4,379** | **6,005** | **25.5% ± 2.5** |

**Sampling method, and a correction to it.** The first pass used 800 rows as 8 contiguous 100-row
windows and returned 27.4%. A re-draw with different offsets returned 31.4% — too far apart to
trust either. The cause is that **OpenThoughts is ordered by source**, so a contiguous window is a
near-homogeneous block and 8 windows is effectively n≈8, not n=800. Redrawn as **2,000 rows in 200
dispersed 10-row clusters**, the estimate is **25.5% with a cluster-robust 95% CI of [23.0, 28.0]**,
and an independent 1,140-row draw reproduces it at **25.4%**. The sample's own domain mix
(75.5 / 19.5 / 5.0) also matches the corpus (78.2 / 17.5 / 4.4), which the windowed samples did not.

The mean of 6,005 reproduces the paper's stated R1 average of 6,130.6 within **2%** — better than
the windowed sample managed — which is what validates the sample and the tokenizer choice.

So the cap truncates **about a quarter** of the reference distribution on the paper's own dataset. `07` §1.1
concluded the opposite — *"All sit under 8192, so the cap is rarely binding"* — by comparing the
cap against Table 2's **mean** lengths. That is precisely the error `docs/results/baselines.md`
Finding 3 already caught once, when the 16k eval cap "was measuring itself": a mean below the cap
says nothing about the tail.

### Why this is a Phase 1 decision, not a Phase 6 one

A trace that hits the cap has no `</think>` and no answer. As an inverter training target it
teaches the inverter never to conclude, and that propagates through Phase 4 into the student.

The student is Qwen3.5-2B, whose documented failure mode is **exactly non-termination**: 50% of its
truncations are degenerate loops repeating `\boxed{}` a median 3,521 times (baselines Finding 2).
That was deferred to Phase 6 on the reasoning that *"SFT on cleanly terminating traces may fix
it."* **That premise only holds if the traces terminate.** At 27% they do not, and the deferral
silently becomes a bet against the data.

**Recommendation.** Keep the paper's cap — it is the fidelity-preserving choice and it bounds
generation cost. Then **over-generate 1.34× (6,706 rows to net 5,000) and drop every row that hit the
cap**, so no training target is a severed trace. Report the drop rate; the paper never published one.
Cost is **31.5 M** generated tokens instead of ~23 M, which §4 pays for many times over.

---

## 3. The inverter's `max_length` is too short by ~1,524 tokens

Measured prompt-side budget on the same sample: user prompt median **81** tokens, final answer
median **542**, plus the compression prompt's **600–900**-token summary target. Inverter prompt
overhead is therefore **≈1,524 tokens**, and a trace generated at the 8192 cap needs **~9,716
tokens** of training context.

The plan's tentative `max_length` is 8192 (`09` §3.5). At that setting `truncation_mode="keep_start"`
keeps the prompt and **cuts the tail off roughly a third of the traces** — the same non-termination
poison as §2, introduced a second time, at training rather than generation.

**Set the inverter's `max_length` to 12,288** (**18.30 GiB measured** for a bf16 LoRA). The student's target is `t̂ ; y` against a short problem prompt, so 8192–10,240 is
sufficient there — and 16,384 is affordable if wanted.

---

## 4. Phase 1's time budget is ~4× understated — and the fix is a 4.6× speedup

`docs/11` budgets "~3-4 h" for the 7B and `docs/10` "~4 h" for both surrogates. Phase 0's own 7B
run took **6 h 34 m for 5.7 M tokens — 241 t/s effective**. Phase 1 needs ~31.5 M tokens. At Phase
0's rate that is **36 hours**.

The plan's mandatory concurrency sweep is what closes the gap, and the reason is specific: Phase 0
swept the 7B at **32,768** per slot because baselines generate up to 32k. Phase 1 generates at most
8192. Re-swept at 10,240/slot:

| Slots | Total KV | Gen t/s |
|---:|---:|---:|
| 8 | 81,920 | 402.02 |
| 16 | 163,840 | 687.53 |
| 24 | 245,760 | 982.72 |
| **32** | **327,680** | **1,270.38** |
| 40 | 409,600 | ❌ OOM (KV cache) |

**4.6× on identical weights and identical hardware, for free** — the largest single speedup
available anywhere in this project. Throughput was still climbing at 32 slots; the ceiling is the
KV cache, not compute. Full result: `docs/results/sweeps/DeepSeek-R1-Distill-Qwen-7B-F16-ctx10240.md`.

Revised Phase 1, applying Phase 0's observed 88% swept-to-realized ratio:

| Step | Est. |
|---|---|
| 7B generation, 31.5 M tokens @ ~1,120 t/s | ~7.8 h |
| 1.5B generation | ~4–6 h |
| Compression ×2 arms | ~2–3 h |
| **Phase 1 total** | **~15–18 h** *(plan says 4 h)* |

Project total moves ~120 h → ~135 h. A budget correction, not a blocker.

---

## 5. Two errors in the handoff that would break Phase 1's first step

**5.1 The dataset schemas are swapped.** `07` §3 and `11` §2 both state that the
`llamafactory` mirror uses `system` + `conversations` and the canonical repo uses `messages`.
Verified live, it is the reverse:

| Repo | Columns |
|---|---|
| `llamafactory/OpenThoughts-114k` | **`messages`**, `original_solution`, `domain`, `source` |
| `open-thoughts/OpenThoughts-114k` | `system`, `conversations` |

The *conclusion* stands — use the llamafactory mirror, it is what the paper's code parses — but a
loader written from the handoff's description gets a `KeyError` on row 0, or worse, a defensive
parser that silently yields nothing. Mirror verified: 113,957 rows, 1.18 GB parquet, viewer live.
Domain mix **78.2% math / 17.5% code / 4.4% science+puzzle**, so MATH500 is the better length
proxy of the two Phase 0 benchmarks, not JEEBench.

**5.2 Disk is 26 GB, not 28 GB — and 36 GB of it is recoverable.**
`~/.cache/huggingface/hub/models--unsloth--DeepSeek-R1-Distill-Qwen-14B-GGUF` holds **five quants
(Q2_K, Q2_K_L, Q3_K_M, Q4_K_M, Q5_K_M) of a model that appears in no script, no log, and no phase
of the plan** — a surrogate candidate `05` rejected as "DOES NOT FIT". Deleting it takes free space
**26 GB → 62 GB**, which dissolves the train-evaluate-delete checkpoint policy in `11` §5 *and* its
stated consequence that "re-evaluating means retraining, so the eval protocol must be settled
before Phase 5 begins." One `rm` retires a standing constraint on Phases 2 and 5.

---

## 6. What was checked and found sound

- Role ordering **student 47.8 < surrogate 60.6 < victim 86.2** on JEEBench — the regime the
  paper's argument requires — holds, and the 7B-vs-1.5B two-arm decision is well-reasoned.
- Harness calibration against the paper's Table 6 (JEEBench 32.6 vs 32.6) is genuine end-to-end
  validation, not a coincidence.
- Victim `IQ4_XS` selection, the 32-slot recurrent-state ceiling, and the split-engine
  justification (`09` §7.5) all survive review.
- The compression-prompt problem (`11` §4) is correctly identified as the highest-leverage and
  least-specified artifact in the project. Nothing here changes that assessment.

## 7. Still unverified going into Phase 1

| Item | Blocks |
|---|---|
| TRL / peft installed and `06` §4.10 smoke test run | Phase 2 |
| Trace lengths **our surrogates** produce on OpenThoughts (R1's were the proxy) | measured during Phase 1 step 2 |
| Compression throughput for Qwen3.5-4B on vLLM | Phase 1 step 4 |
| 1.5B concurrency at 10,240/slot | Phase 1 step 5 |
