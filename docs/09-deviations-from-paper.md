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
| 3.1 | Method | **full-parameter SFT** | **UNDECIDED** — FFT where it fits, LoRA only where it doesn't | HW (partly) | **Possibly no deviation at all** — see §5.1 |
| 3.2 | Learning rate | 1e-5 | 1e-5 if FFT (matches paper); 1e-4 – 2e-4 if LoRA | follows 3.1 | Depends entirely on 3.1 |
| 3.3 | LR schedule | cosine, warmup 0.1 | same | — | — |
| 3.4 | Epochs | 3 | 2–3 (TBD) | COST | Possible slight undertraining |
| 3.5 | Sequence length | `cutoff_len` 16384 | 8192 (TBD) | HW | Truncates the longest traces; paper's traces run 5-6k so most survive |
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

### 5.1 Fine-tuning method — OPEN DECISION (3.1)

**Status: not yet decided.** Deferred until baselines lock the student model. This may turn out
not to be a deviation at all.

The paper full-fine-tunes. Whether we can depends entirely on which student we pick — at
~6 bytes/param with 8-bit Adam, bf16, and gradient checkpointing:

| Student | FFT states | + activations @8k | FFT viable? |
|---|---:|---:|---|
| Qwen3.5-0.8B | 4.8 GB | ~7 GB | ✅ comfortable |
| Qwen3.5-2B | 12 GB | ~15 GB | ✅ fits |
| Qwen3.5-4B | 24 GB | ~28 GB | ❌ LoRA required |

So **if the baselines point at 0.8B or 2B, we match the paper's method exactly** — full-parameter
SFT at LR 1e-5 — and this row disappears from the deviations list. Only a 4B student forces LoRA.

**If LoRA does become necessary:** it adapts style well but is weaker at large behavioral shifts,
and "begin emitting 5,000-token chains of thought" is exactly such a shift. If effect sizes then
come in below the paper's, LoRA is the first suspect. Mitigations: rank 32–64, target all linear
projections, LR 1e-4–2e-4.

**Either way, run the 2B both ways.** It's the largest student that fits FFT, so one paired
comparison measures what LoRA costs on this task instead of leaving it as an unquantified risk.

*Caveat on the table above:* Qwen3.5's 248,320-token vocab makes embed/`lm_head` a large share of
a small model's parameters, and the fp32 logits tensor is ~8 GB at 8k context before TRL's
`chunked_nll` reduces it. These numbers assume `chunked_nll` stays enabled.

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
| 7.1 | Eval sampling | uniform temp 0.7 / top_p 0.9 / rep 1.05 | per-family recommended (Qwen3.5 1.0/0.95/k20; R1-Distill 0.6/0.95) | Off-spec sampling penalizes models unfairly; recorded so it can be re-run uniformly if needed |
| 7.2 | `max_new_tokens` | 8192 | 16384 for baselines | Smoke test truncated 4/5 JEEBench items at 2048 — capping would measure the cap |
| 7.3 | Victim concurrency | n/a (8 GPUs) | 32 (hard ceiling — recurrent-state cache OOMs at 40) | Measured, see `08` |
| 7.4 | Dataset mirror | `llamafactory/OpenThoughts-114k` | same | Canonical `open-thoughts/` repo has an incompatible schema |
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
