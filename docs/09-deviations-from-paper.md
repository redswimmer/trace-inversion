# Deviations From the Paper — Running Log

Every place our reproduction differs from Zhang/Morris/Shmatikov (arXiv 2603.07267v2), with
the reason and the expected effect on results. Updated as decisions are made.

**Legend for "Forced?"** — `HW` hardware limit · `COST` time/money · `PAPER` the paper itself
makes the original impossible · `CHOICE` deliberate design decision.

---

## 1. Infrastructure

| # | Item | Paper | Ours | Forced? | Expected impact |
|---|---|---|---|---|---|
| 1.1 | Hardware | 8× A100 80GB (640 GB) | 1× RTX 4090 24 GB | HW | Everything below follows from this |
| 1.2 | Training framework | LLaMA-Factory + DeepSpeed ZeRO-3 | **TRL** `SFTTrainer` | CHOICE | Config must be translated, not copied; defaults differ (see `07`) |
| 1.3 | Inference backend | vLLM throughout | vLLM (students/surrogate/inverter) + **llama.cpp** (27B victim) | HW | Split engine — see §7.5 for why it's acceptable |
| 1.4 | Precision asymmetry | all full precision (API victim) | victim **4-bit GGUF**, everything else **bf16 safetensors** | HW | Victim is a fixed input, never retrained |

## 2. Models

| # | Role | Paper | Ours | Forced? | Expected impact |
|---|---|---|---|---|---|
| 2.1 | Victim (open-weight) | DeepSeek-R1 (685 B, 688 GB) | **Qwen3.8-27B** @ IQ4_XS | HW | R1 is unrunnable at any quant. Different victim ⇒ absolute numbers not comparable |
| 2.2 | Victim precision | full (via API) | **IQ4_XS 4-bit** | HW | Slightly worse victim ⇒ slightly worse traces. Constant across conditions, so not a confound |
| 2.3 | Victim (black-box) | `gpt-5.4-mini-2026-03-17` | an OpenAI reasoning model (TBD) | COST | Optional second track |
| 2.4 | **Surrogate** | **R1-Distill-Qwen-1.5B** | **same — no deviation** ✅ | — | Deliberately matched to the paper |
| 2.5 | Student | Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct | **Qwen3.5-0.8B / 2B / 4B** | HW + CHOICE | **See §5 — this is the most consequential deviation** |
| 2.6 | Compressor | Qwen2.5-7B-Instruct | Qwen3.5-4B | HW | Zero-shot only; low risk |
| 2.7 | Inverter base | Qwen2.5-7B-Instruct | Qwen3.5-4B | HW | Smaller inverter ⇒ possibly lower trace fidelity |

## 3. Training

| # | Item | Paper | Ours | Forced? | Expected impact |
|---|---|---|---|---|---|
| 3.1 | Method | **full-parameter SFT** | student **FFT** ✅ no deviation; inverter **bf16 LoRA** | HW (inverter only) | **Decided on measurement** — see §5.1 |
| 3.2 | Learning rate | 1e-5 | student **1e-5** (matches paper); inverter 1e-4 – 2e-4 (LoRA) | follows 3.1 | Student matches; inverter cannot |
| 3.3 | LR schedule | cosine, warmup 0.1 | same | — | — |
| 3.4 | Epochs | 3 | 2–3 (TBD) | COST | Possible slight undertraining |
| 3.5 | Sequence length | `cutoff_len` 16384 | student **16384** ✅ no deviation; inverter **12288** | HW (inverter only) | Measured, not estimated — `12` §1. Inverter needs ≥9,716 (1,524 prompt + 8,192 trace); 12288 gives margin |
| 3.6 | Effective batch | 24 (inverter) / 96 (student) | matched via grad accumulation | — | Must not collapse both to one value |
| 3.7 | Packing | `packing: true`, `neat_packing: false` (leaky) | TRL packing (correct masking) | CHOICE | We match TRL's *correct* behavior, not the reference's leaky one |

## 4. Experiment scope

| # | Item | Paper | Ours | Forced? | Expected impact |
|---|---|---|---|---|---|
| 4.1 | Victim query budget | 10k (Fig 3 to 25k) | **5k** | COST | Paper's own Fig 3: 5k gives 67.0 vs 77.6 MATH500 at 10k |
| 4.2 | Surrogate queries | 10k | 5k | COST | Less inverter training data |
| 4.3 | Query-budget sweep (Fig 3) | 5 points, 5k→25k | likely skipped | COST | 5 extra student trainings |
| 4.4 | Benchmarks | MATH500, JEEBench, LCB (+HE+) | MATH500, JEEBench first; LCB if time | COST | LCB is the paper's weakest result anyway |
| 4.5 | Seeds / variance | **none — single run, no CIs** | **3 seeds on ≥1 condition** | CHOICE | An *improvement*: paper's key claim rests on 0.4-2.4 pt margins |

## 5. The two deviations that could change conclusions

### 5.1 Fine-tuning method — DECIDED ON MEASUREMENT (3.1)

**Status: resolved 2026-08-26.** Measured, not estimated — full method and numbers in `12` §1.
RTX 4090 · bf16 · gradient checkpointing · batch 1 · `Qwen3_5ForCausalLM` (text-only) ·
`torch.cuda.max_memory_allocated()`, with optimizer states added at 2 B/param (8-bit) and
4 B/param (bf16). Usable budget ≈ **23.4 GiB**.

| Model | Role | Mode | @8192 | @12288 | @16384 |
|---|---|---|---:|---:|---:|
| Qwen3.5-2B | student | **FFT + 8-bit Adam** | 12.64 GiB ✅ | 14.2 GiB ✅ | **15.79 GiB ✅** |
| Qwen3.5-2B | student | FFT + bf16 Adam | 16.15 GiB ✅ | — | 19.30 GiB ✅ |
| Qwen3.5-4B | inverter | FFT | 27.81 GiB ❌ | — | OOM ❌ |
| Qwen3.5-4B | inverter | **LoRA, bf16 base** | 15.11 GiB ✅ | **18.30 GiB ✅** | 21.47 GiB ⚠️ |

**Student: full-parameter SFT at LR 1e-5, `max_length` 16384 — no deviation from the paper at all.**
The earlier estimate ("~15 GB @8k") was right but conservative; 16384, the paper's own `cutoff_len`,
also fits, so row 3.5 closes as well.

**Inverter: LoRA on a bf16 base, not QLoRA.** 4B full fine-tuning is confirmed impossible. But NF4 is
not needed either — a bf16 base with LoRA fits at 15.11 GiB @8k and **18.30 GiB measured at the chosen 12288**.
Dropping NF4 removes the `bitsandbytes` dependency, makes `06` gotcha 9 ("QLoRA on `qwen3_5` is
UNVERIFIED") moot rather than unresolved, and avoids quantizing the one model that must learn a
genuinely new behaviour. Keep QLoRA in reserve only if 16384 is wanted for the inverter — 21.47 GiB
leaves too little margin for a multi-hour run.

**LoRA remains the first suspect** if inverter trace fidelity comes in below the paper's: it adapts
style well but is weaker at large behavioural shifts, and "emit a 5,000-token chain of thought" is
exactly such a shift. Mitigations: rank 32–64, all linear projections except `lm_head`, LR 1e-4–2e-4.

**Still run the 2B student both ways.** It is the largest student that fits FFT, so one paired
comparison measures what LoRA costs on this task instead of leaving it an unquantified risk.

Two things the same measurement settled:

- **DeltaNet trains on Ada/SM89.** Forward + backward through the 48 gated-delta-net layers with
  gradient checkpointing works. This was the largest unverified architectural risk in the plan.
- **`chunked_nll` is load-bearing, confirmed.** Naive fp32 logits + gradient at 248,320 vocab is
  **15.16 GiB at 8k and 30.31 GiB at 16k** — larger than the card, before weights. Every number above
  assumes it stays enabled.

### 5.2 A student that already reasons (2.5)

The paper's students are pre-reasoning-era models that do short CoT. Every Qwen3.5 Instruct model
thinks by default.

**Why it matters:** it changes the claim from *"inversion instills reasoning"* to
*"inversion improves reasoning."* Effect sizes will be smaller because the student starts higher.

**Why we accept it:** the claim still rests on **baselines, not on a naive student**. If
`Synthesized-Trace` beats `Answer-only`, `Summary+Answer`, and `Surrogate-Trace` on the same
student, inversion added value regardless of the starting point. And "can you still extract value
when your student isn't naive?" is a live 2026 question the paper doesn't answer.

**Mitigation:** Qwen3.5 supports `enable_thinking=False`, giving a non-reasoning baseline of the
*same* model — so the instill-vs-improve distinction can be measured directly rather than assumed.

**Secondary confound:** same-family victim/student (3.8-27B → 3.5-4B) shares tokenizer and
pretraining lineage, which may ease transfer relative to the paper's cross-family setup. Keep one
cross-family student as a check.

## 6. Forced on us by the paper itself

| # | Item | Problem | Our response |
|---|---|---|---|
| 6.1 | Compression prompt `π` | v2's Appendix B says its two few-shot exemplars are "released with our code" — but the released code predates the v2 prompt entirely and uses a **structurally opposite** format (numbered bullets vs bold-header prose). The exemplars do not exist anywhere. | Write our own; validate against Table 1's four style statistics |
| 6.2 | Inversion prompt | v2's inversion prompt expects "numbered reasoning bubbles" that v2's compression prompt explicitly forbids producing — a v1 leftover, live in the published paper | Fix the mismatch; **also re-run the zero-shot baseline format-matched**, since the mismatch may have depressed it |
| 6.3 | Reference numbers | Released code reproduces **v1**, not the v2 tables | Compare against the paper's tables, never the repo's outputs |

## 7. Minor / procedural

| # | Item | Paper (from repo) | Ours | Why |
|---|---|---|---|---|
| 7.1 | Eval sampling | generation pinned at temp 0.7 / top_p 0.9 / rep 1.05; **eval sampling unspecified** | **paper's generation values throughout** | See §7.7 — an earlier deviation to vendor defaults was corrected |
| 7.2 | `max_new_tokens` | 8192 | 16384 for baselines | Smoke test truncated 4/5 JEEBench items at 2048 — capping would measure the cap |
| 7.3 | Victim concurrency | n/a (8 GPUs) | 32 (hard ceiling — recurrent-state cache OOMs at 40) | Measured, see `08` |
| 7.4 | Dataset mirror | `llamafactory/OpenThoughts-114k` | same | Canonical `open-thoughts/` repo has an incompatible schema |
| 7.9 | Dropping capped traces | `preprocess_r1_distill.py`: `if '</think>' not in prediction: continue` | **same — no deviation** ✅ | The reference already discards non-terminating traces. Our over-generate-and-drop policy matches it; only the over-generation ratio is ours |
| 7.8 | Surrogate system prompt | repo injects the R1-Distill prompt; **paper never mentions it** | **same as the repo** — not the near-identical one the OpenThoughts rows carry | `07` §3: it is part of what makes the surrogate emit long traces, so it must not be left to chance. See `11` §3 |
| 7.10 | `D₂`'s prompt distribution is surrogate-filtered | same drop policy, **never measured** | **same policy, measured** ✅ | **PROVISIONAL, n=1,142 — being re-measured on the full ~7,875-row 7B run.** Source: `~/trace-inversion-bench/archive/traces-7b-partial-1130rows.jsonl` (the name says 1130; it holds 1,142 — the count was taken before the last flush). Same cap, system prompt and sampling as the live run; reproduced independently 2026-08-26. Dropping capped traces removes **34.2%** of prompts, and paired against R1's ground truth on the *same* prompts, **33% of our drops (11.2% of all prompts) are ones R1 completes fine** (we drop 390, R1 would drop 314, both 262, ours-only 128, R1-only 52). An earlier figure of 31.6% / 33% / 10.3% circulated with no recorded sample and is not recoverable — do not cite it. **Treat all three numbers as ±4 pts.** Two independent samples of the *same* 244 prompts at temperature 0.7 give cap-hit 32.0% and 36.1%, disagreeing on capped-vs-kept for 15.6% of rows: whether a given prompt hits the cap is substantially a resampling outcome, not a fixed property of the prompt. The 7B probe's 36.5% / 40% / 14.5% (n=200) sits inside that noise, so it is not evidence of a shift — they are where *our* 7B ran away, not hard prompts. So the inverter trains on a surrogate-filtered distribution and Phase 4 serves it on split B, which is essentially unfiltered (victim truncation was 0.0-0.4%). **A train/serve shift created by our own drop policy.** The paper's code drops identically and carries the same bias unmeasured. Not a Phase 1 problem and not fixed here — reported in `docs/results/phase1.md`, and the two-arm design measures it for free: overlap between the 7B's and 1.5B's drops separates "genuinely hard prompt" from "this surrogate loops here". |
| 7.11 | The two arms cover **different amounts of prompt space** | not measured | **measured per arm** ✅ | Follows from 7.10. Cap-hit is surrogate-specific, so each arm's `D₂` is built from a different subset of split A. The 7B drops ~33%; the 1.5B is more verbose on every Phase 0 measurement (median 4,849 vs 3,875 at a 32k cap; 35.0% vs 19.2% would exceed 8192) and is projected far higher. **So the arms differ not only in surrogate strength — the intended variable — but in how much of the prompt space each training set covers.** That is a confound when reading the surrogate-strength comparison, and it must be reported alongside it. It is also a genuine finding: a weak surrogate may work partly *because* it trains on the subset it handles cleanly, which the paper never measured. Split A was extended to 16,000 indices so the weaker arm is not silently starved. |
| 7.5 | Split inference engine | vLLM only | victim on llama.cpp, everything else on vLLM | see below |
| 7.6 | Generation cap | none set (Evalchemy defaults) | **32,768 tokens / 40,960 context**, seed 1234 | see below |

### 7.5 Why the split engine is acceptable

The victim runs on llama.cpp (GGUF IQ4_XS); students, surrogate and inverter run on vLLM
(bf16 safetensors). Different engines differ in sampler implementation, attention numerics, and
chat-template handling, so this is a real asymmetry. It is acceptable because:

1. **No comparison we make crosses the engine boundary.** The claim rests on student-before vs
   student-after (`Synthesized-Trace` vs `Answer-only` vs `Summary+Answer` vs `Surrogate-Trace`) —
   all on the same student, all on vLLM. The victim's benchmark score is only a reference ceiling
   answering "is the victim far enough above the students?", where a few points of engine noise
   changes nothing.
2. **The victim's real job is generating traces, not being benchmarked.** llama.cpp is the measured,
   tuned path for that workload (303 t/s at concurrency 32, see `08`).
3. **The students are the artifacts we produce.** Every fine-tuned checkpoint comes out of TRL as
   safetensors; ~10+ of them across the five training conditions and two students. Keeping students
   on vLLM means zero conversion steps, and no chance of a template or tokenizer mismatch silently
   corrupting a result during GGUF conversion.

Alternatives considered and rejected: all-llama.cpp (would require converting every fine-tuned
checkpoint to GGUF); all-vLLM with a community AWQ build of the victim (unverified quantization
quality — 91 downloads — and would discard the throughput measurements we already trust).

### 7.6 Why we cap generation at all

The paper sets no explicit cap, but one always exists — vLLM sizes its KV cache from
`max_model_len`, and Evalchemy supplies a per-task default. Two reasons a bound is unavoidable:

1. **KV preallocation drives parallelism.** For Qwen3.5-4B (32 layers, 8 full-attention, 4 KV
   heads, head_dim 256 → 32 KiB/token): a 16k cap costs 0.5 GiB/sequence (~28 concurrent), 32k costs
   1.0 GiB (~14), 64k costs 2.0 GiB (~7). Doubling the cap roughly halves throughput, and you pay
   the worst-case memory on every request.
2. **Reasoning models loop.** Without a bound one stuck problem hangs the run indefinitely.

So the real question is not capped-vs-uncapped but **how much throughput to spend covering the
tail**. Our first attempt at 16,384 was wrong — it cut into the *body* of the distribution (71% of
the 4B's JEEBench runs hit it exactly), meaning we measured the cap rather than the model. 32,768 is
chosen to be **non-binding**; the truncation rate in the re-run is the test of whether it is. If
truncation stays high at 32k, that is a finding about the model, not an artifact.

Caveat on our own metric: "completed-only accuracy" is a **biased estimator** when truncation is
high, because the surviving problems are the easy ones (the 4B reported 100.0% on MATH500 completed
at 71% truncation). Do not quote it above ~20% truncation.

---

## Deviations we are *not* making

Worth stating explicitly, since matching the paper where possible is what makes the rest credible:

- **Surrogate is the paper's exact model** (`DeepSeek-R1-Distill-Qwen-1.5B`)
- Same dataset (OpenThoughts-114k), same disjoint 2-split design
- Same benchmarks, same five training conditions
  (Answer-only / Summary+Answer / Surrogate-Trace / Synthesized-Trace / Victim-Trace oracle)
- Same three-stage pipeline and training objectives
- Same LR schedule and warmup ratio


### 7.7 Sampling: corrected deviation, and a finding that contradicted the fix

**What went wrong.** The eval harness was written before the paper's repo was analysed, so sampling
came from the Qwen model card (`temp 1.0 / top_p 0.95 / top_k 20`, no repetition penalty). The
paper's values (`0.7 / 0.9 / rep 1.05`) were later recovered into `docs/07` but never wired into
the code. This was an ordering accident, not a considered tradeoff — no reason existed to prefer a
vendor default when the goal is comparability with the paper's tables.

Note the paper pins sampling only for **generation**; its eval invocation passes just
`pretrained`, `tensor_parallel_size`, `gpu_memory_utilization`, `batch_size 16`, `seed 1234`, so
eval sampling falls through to Evalchemy defaults. We now use the paper's generation values for
eval too — not because they match (there is nothing to match) but so the whole project runs one
protocol instead of two.

**The finding.** The switch was expected to fix the 2B's degenerate looping. It did the opposite:

| Qwen3.5-2B, JEEBench | card sampling (1.0 / 0.95, no penalty) | paper sampling (0.7 / 0.9 / 1.05) |
|---|---|---|
| Truncated | 154 | **221** |
| Clear loops (>5 repeats of own answer) | 84 | **144** |
| Median repeats | 3,636 | 3,521 |
| JEEBench accuracy | 47.0% | **47.8%** |
| MATH500 accuracy | 77.8% | **79.0%** |

**Why:** the dominant variable is temperature, not the penalty. Lowering 1.0 → 0.7 makes decoding
greedier, and greedy decoding is *more* prone to degenerate repetition — once inside a `\boxed{AC}`
cycle, lower temperature makes escape less likely. A 1.05 penalty shifts logits ~5% against a loop
being reinforced every step; far too weak to break it.

The causal story behind the original diagnosis ("the paper uses a repetition penalty, we don't,
we loop") was therefore backwards. Temperature governs loop entry; 1.05 barely touches it.

**Resolution:** keep the paper's sampling — accuracy improved on both benchmarks (completed-only
51.2 → 55.3 JEEBench, 82.8 → 86.0 MATH500) and it matches their protocol. Treat looping as a
separate problem, and **revisit only after the first trained checkpoint**: SFT on cleanly
terminating traces is exactly the behaviour that should fix it, so tuning sampling now risks
solving a problem that will not exist.

**Watch for asymmetry.** If training reduces looping, part of any measured gain is the model
learning to terminate rather than to reason. Legitimate as an effect of distillation, but it must
be reported as such — compare truncation rates pre/post, not just accuracy.

Results under card sampling archived at `bench/results/v2-cardsampling/`.
