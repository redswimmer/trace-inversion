# Experiments and Results — Complete Inventory

**Paper:** *How to Steal Reasoning Without Reasoning Traces* — Tingwei Zhang, John X. Morris, Vitaly Shmatikov (Cornell Tech). arXiv:2603.07267v2 [cs.CR], 12 May 2026.
**Code (per paper):** https://github.com/Tingwei-Zhang/Trace_Inversion_Attack
**Scope of this document:** §5 (Evaluation) in full, every table (1–6) and figure (1–6), plus Appendices A–D.

All numbers below were transcribed from the arXiv HTML source (`<table>` cells), not from text extraction, and Figure 3 was recovered numerically from the source vector plot (`query_plot.svg`) by calibrating against the axis ticks. Discrepancies between the paper's prose and its own tables are flagged explicitly in each section — these matter when reproducing.

---

## Master table of experiments

| ID | Paper artifact | One-line description |
|---|---|---|
| `EXP-T1-summary-style-match` | Table 1 (§5.2) | Does the attacker's own compression prompt produce summaries whose length/style distribution matches GPT-5.4 mini's real reasoning summaries? |
| `EXP-T2-trace-fidelity` | Table 2 (§5.2) | How closely do inverted traces match R1's *ground-truth* traces (Len/BLEU/TF1/ROUGE), for zero-shot vs fine-tuned inversion, strong vs weak surrogate, summary vs no-summary? |
| `EXP-T3-steal-r1` | Table 3 (§5.3) | Downstream student accuracy when the victim is the open-weight R1 — inverted traces vs answer-only / summary+answer / surrogate-trace / oracle victim traces. |
| `EXP-T4-steal-gpt54mini` | Table 4 (§5.4) | Downstream student accuracy when the victim is the black-box commercial GPT-5.4 mini (no oracle available). |
| `EXP-F3-query-budget` | Figure 3 (§5.4) | How does student accuracy scale with the victim query budget, 5k → 25k queries? |
| `EXP-T5-augmentation-ablation` | Table 5 (§5.4) | Ablation: surrogate-data augmentation and per-domain (code) supervision gating on top of Synthesized-Trace. |
| `EXP-T6-model-baselines` | Table 6 (Appendix A) | Zero-shot benchmark accuracy of the victims and surrogates themselves (R1, R1-Distill, GPT-5.4 mini) — the capability ceiling / regime check. |
| `EXP-COST-query-economics` | §5.4 (prose, no table) | What does the attack cost in API dollars at a 10k query budget? |
| `EXP-F1-teaser-example` | Figure 1 (§1) | Qualitative: one end-to-end Trace Inversion example against GPT-5.4 mini (input + answer + summary → synthesized trace). |
| `EXP-F2-pipeline-diagram` | Figure 2 (§4) | Not an experiment — the three-stage pipeline diagram (`inversion_pipeline.png`). |
| `EXP-F4-zeroshot-vs-ft-qualitative` | Figure 4 (Appendix C) | Qualitative: zero-shot inversion output vs fine-tuned inversion output (surrogate = R1-Distill), victim = R1. |
| `EXP-F5-gt-vs-synth-qualitative` | Figure 5 (Appendix C) | Qualitative: R1 ground-truth trace vs synthesized trace (surrogate = R1-Distill), victim = R1. |
| `EXP-F6-denoising-case-study` | Figure 6 (Appendix D) | Qualitative evidence for the "synthesized beats ground truth" claim: R1's backtracking trace vs the inversion model's clean forward reasoning on one inequality problem. |

---

## Shared experimental setup (§5.1)

Applies to every quantitative experiment unless overridden.

**Roles (threat model, §3):**

| Role | Symbol | Instantiation |
|---|---|---|
| Victim (open-weight) | `V` | DeepSeek-R1 ("R1"), accessed via commercial API |
| Victim (black-box) | `V` | `gpt-5.4-mini-2026-03-17` ("GPT-5.4 mini"), commercial API |
| Surrogate reasoning model | `R` | R1 (strong) **or** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (called **R1-Weak** / "R1-Distill") |
| Compression / summarization model | `C'` | Qwen2.5-7B-Instruct, prompted only (**no fine-tuning**), prompt template `π` in Appendix B |
| Inversion model | `I` | Qwen2.5-7B-Instruct backbone, fine-tuned (separate model per setting: `I_sum`, `I_nosum`) |
| Student models | `S` | Qwen2.5-7B-Instruct ("Qwen") and Llama-3.1-8B-Instruct ("Llama") |

**Data:** OpenThoughts-114k. Two **disjoint** splits of **10k inputs each**:
- *surrogate split* → queried against `R` to get `(x', t', y')` = dataset `D1`; traces optionally compressed by `C'` into `b'` giving `D2_sum = {(x',y',b',t')}` and `D2_nosum = {(x',y',t')}` for inversion training.
- *victim split* → queried against `V` to get `(x, t, y)` (open-weight victim) or `(x, b*, y)` (black-box victim); inverted into `t̂` and used for student fine-tuning.

*Reproduction gap:* §5.1 states 10k per split, but Figure 3 sweeps the victim budget up to **25k** queries. The 25k point therefore requires a larger victim split than the 10k stated as the default. Treat "10k" as the default budget for Tables 3–5 and the 5k–25k sweep as a separate data collection.

**Two settings, two inversion models:** *summary* (victim exposes `b*`) and *no-summary* (only `(x, y)`). "Each setting requires a separate inversion model" (§4).

**Five student training-data conditions (§5.1):**
1. **Answer-only** — victim answers.
2. **Summary+Answer** — victim summaries + answers.
3. **Surrogate-Trace** — the surrogate's own traces + answers (i.e., plain distillation from `R`).
4. **Synthesized-Trace (ours)** — inverted traces `t̂` + victim answers.
5. **Victim-Trace (oracle)** — victim's real traces + answers; upper bound, only possible for the open-weight victim (R1).

Student supervision target is the concatenation `y⁺ = t̂ ⊕ y` (Eq. 3), trained with teacher-forced cross-entropy. Inversion training (Eqs. 1–2) is likewise teacher forcing with token-level cross-entropy.

**Benchmarks:** MATH500 (Hendrycks et al. 2021), JEEBench (Arora et al. 2023), LiveCodeBench / LCB (Jain et al. 2025, execution-based). **HumanEval+** (Liu et al. 2023) appears only in Table 5.

### Metric definitions (§5.1, "Inversion metrics")

These are the paper's *complete* definitions — there is **no** appendix expanding them. This is a genuine reproduction gap; the exact tokenizer, casing, and stopword handling are not specified anywhere in the paper.

| Metric | Paper's definition, verbatim in substance | Notes for reimplementation |
|---|---|---|
| **Len** | "token count; a coarse proxy for completeness" | Token count of the synthesized trace `t̂`. Tokenizer unspecified (most likely the Qwen2.5 tokenizer, since §5.1 says "we use each model's default optimizer and **tokenizer**"). Reference point: R1 ground-truth traces average **6,130.6** tokens. Lower is *not* better — closeness to 6,130.6 is what "81–89% recovery" refers to. |
| **BLEU** | "n-gram precision, capturing surface lexical similarity" | n unspecified (assume standard BLEU-4 with smoothing). Reported ×100. |
| **TF1** | "**token-overlap F1**, i.e., the harmonic mean of token-level precision and recall" | **Unigram/bag-of-tokens F1**, not n-gram, not sequence-aligned. Precision = |tokens shared between `t̂` and `t`| / |tokens in `t̂`|; recall = same over |tokens in `t`|; TF1 = 2PR/(P+R), reported ×100. The paper does not say whether counts are multiset (clipped counts) or set-based; multiset/clipped is the conventional reading and is what makes TF1 differ from ROUGE-1 in the table (e.g. 52.76 vs 56.59). |
| **R-1 / R-2 / R-L** | ROUGE "unigram, bigram, and longest-common-subsequence overlap" | Standard ROUGE. The paper does not state whether F-measure or recall is reported; the values track TF1 closely, consistent with F-measure. Reported ×100. |
| **Downstream accuracy** | Benchmark accuracy (%) of the fine-tuned student | The *primary* metric. The inversion metrics are explicitly labelled "auxiliary diagnostics." |

### Evaluation protocol — what is and isn't specified

The paper says only: *"Unless otherwise specified, we use each model's the default optimizer and tokenizer and follow the original evaluation protocols of each benchmark."* (§5.1)

**Not reported anywhere in the paper** (must be recovered from the released code or chosen by the reproducer):
- Decoding temperature, top-p, top-k, repetition penalty — for the victim queries, the surrogate queries, the inversion model's generation, and the student's benchmark generation.
- Greedy vs sampling; pass@1 vs pass@k; number of samples per problem (LCB and HumanEval+ are pass@k benchmarks by convention — the paper reports a single accuracy number, implying pass@1).
- Max generation length at eval time (training `cutoff_len` is 16384).
- MATH500 / JEEBench answer grading method (exact match on `\boxed{}` vs an equivalence checker vs LLM judge). "Original evaluation protocols" implies the benchmarks' own harnesses.
- LCB release/version and problem subset; JEEBench subject split.
- Random seeds; single run vs averaged runs (no error bars or variance anywhere in the paper).

### Training hyperparameters (§5.1, "Hyperparameters")

Reported for **all** fine-tuning (inversion models and student models alike):

| Hyperparameter | Value |
|---|---|
| `num_train_epochs` | **3** |
| `learning_rate` | **1e-5** |
| `warmup_ratio` | **0.1** |
| `cutoff_len` (max sequence length) | **16384** |
| Optimizer | "each model's default" (not named; no betas/weight decay/scheduler reported) |
| Tokenizer | "each model's default" |
| Batch size / grad accumulation | **Not reported** |
| LR schedule after warmup | **Not reported** |
| Full fine-tune vs LoRA/PEFT | **Not reported** |
| Precision (bf16/fp16), DeepSpeed/FSDP config | **Not reported** |

*Note for reproducers — framework inference:* `cutoff_len` is **not** a HuggingFace/TRL parameter name; it is LLaMA-Factory's key for max sequence length, and `num_train_epochs` / `learning_rate` / `warmup_ratio` are the other three keys in a standard LLaMA-Factory SFT YAML. The authors almost certainly trained with **LLaMA-Factory**, not TRL. If reimplementing in **TRL**, the mapping is:

| Paper (LLaMA-Factory) | TRL `SFTConfig` |
|---|---|
| `cutoff_len = 16384` | `max_seq_length = 16384` |
| `num_train_epochs = 3` | `num_train_epochs = 3` |
| `learning_rate = 1e-5` | `learning_rate = 1e-5` |
| `warmup_ratio = 0.1` | `warmup_ratio = 0.1` |

Also beware the defaults that differ between the two frameworks and that the paper never pins: LLaMA-Factory defaults to a cosine LR schedule, `adamw_torch`, and prompt-token loss masking (`train_on_prompt: false`), whereas TRL's defaults differ on schedule (linear) and completion-only loss (opt-in via a collator). Since the paper says only "each model's default optimizer," these choices are unrecoverable from the text and should be matched against the released code.

`cutoff_len = 16384` is load-bearing: R1 traces average 6,130.6 tokens and inverted traces reach ~6k, so a shorter context silently truncates supervision targets.

**Hardware:** *"We runn [sic] training and evaluation on 8 NVIDIA Tesla A100s (80GB)."* No wall-clock times, GPU-hours, or node counts reported.

---

## `EXP-T1-summary-style-match` — Table 1 (§5.2)

**Research question:** Does the attacker's hand-designed compression prompt `π`, applied by an un-tuned Qwen2.5-7B-Instruct to R1's traces, produce summaries whose distribution matches the real summaries GPT-5.4 mini emits? (This matters because `I` is trained on `b'` but deployed on `b*` — a distribution mismatch would break the attack.)

| Field | Value |
|---|---|
| Victim | GPT-5.4 mini (`gpt-5.4-mini-2026-03-17`) — reference distribution |
| Surrogate | R1 (traces being compressed) |
| Compression model | Qwen2.5-7B-Instruct, prompt-only, prompt in Appendix B |
| Student / inversion model | n/a |
| Dataset | OpenThoughts-114k surrogate split (R1 traces) vs GPT-5.4 mini summaries on victim split |
| Query budget | Implicitly the 10k victim split (not stated separately) |
| Metrics | Median token count; % of summaries with bold-header sections; % first-person prose; % containing LaTeX — all measured by "lightweight regex heuristics" (regexes not given) |

**Results (Table 1):**

| Feature | Ours (`C'` on R1 traces) | GPT-5.4 mini |
|---|---|---|
| Median tokens | 537 | 592 |
| Bold-header sections | 94.1% | 92.9% |
| First-person prose | 97.3% | 97.0% |
| LaTeX | 79.1% | 71.9% |

**Additional numbers from §5.2 prose (not in the table):**
- Non-empty GPT-5.4 mini summaries: median length **≈592 tokens**, median compression ratio **≈4.2×** relative to the hidden trace.
- GPT-5.4 mini's hidden trace length is recovered by **subtracting visible-output tokens from the API's billed reasoning-token count**. This is the only channel by which the paper observes anything about GPT-5.4 mini's true traces.

**Baselines compared against:** none — this is a distribution-matching sanity check, GPT-5.4 mini's own summaries are the reference.

---

## `EXP-T2-trace-fidelity` — Table 2 (§5.2)

**Research question:** How faithfully do inverted traces reconstruct the victim's *actual* hidden reasoning? Answerable only against the open-weight victim R1, whose real traces are visible.

| Field | Value |
|---|---|
| Victim | **R1** (open-weight; ground-truth traces `t` available) |
| Surrogate | R1 (upper bound), R1-Weak (realistic attack), or none (zero-shot rows) |
| Inversion model | Qwen2.5-7B-Instruct — "Qwen (0-shot)" = prompted only (Appendix B prompts), "Qwen (FT)" = fine-tuned per Eq. 1/2 |
| Student | n/a (no fine-tuning in this experiment) |
| Dataset | OpenThoughts-114k; inversion trained on surrogate split, evaluated on the victim split's R1 traces |
| Query budget | 10k surrogate + 10k victim (default) |
| Benchmarks | none — intrinsic similarity only |
| Metrics | Len, BLEU, TF1, ROUGE-1, ROUGE-2, ROUGE-L |

**Results (Table 2) — full transcription:**

| Setting | Inversion | Surrogate | Len | BLEU | TF1 | R-1 | R-2 | R-L |
|---|---|---|---|---|---|---|---|---|
| Summary | Qwen (0-shot) | – | 978.5 | 3.04 | 35.36 | 26.77 | 15.73 | 17.60 |
| Summary | Qwen (FT) | R1 | 5,767.3 | 26.64 | 64.42 | 66.84 | 40.51 | 29.17 |
| **Summary** | **Qwen (FT)** | **R1-Weak** | **4,971.9** | **16.64** | **52.76** | **56.59** | **29.72** | **24.84** |
| No-summary | Qwen (0-shot) | – | 1,040.0 | 3.31 | 37.58 | 28.30 | 16.39 | 17.97 |
| No-summary | Qwen (FT) | R1 | 6,020.9 | 25.07 | 57.08 | 65.43 | 38.91 | 27.97 |
| **No-summary** | **Qwen (FT)** | **R1-Weak** | **5,434.0** | **11.52** | **49.01** | **47.45** | **21.85** | **21.73** |

Bold rows = the realistic attack (weak surrogate). Higher is better for every column **except Len**, where the target is R1's ground-truth average of **6,130.6 tokens**.

**Derived length recovery** (paper: "≈81–89% of R1's average"):
- Summary / R1-Weak: 4,971.9 / 6,130.6 = **81.1%** ← this is the "~81% token-length recovery" claim from the abstract/intro.
- No-summary / R1-Weak: 5,434.0 / 6,130.6 = **88.6%**
- Summary / R1: 5,767.3 / 6,130.6 = 94.1%; No-summary / R1: 6,020.9 / 6,130.6 = 98.2%
- Zero-shot collapses to ≈1,000 tokens = ~16–17% of the target length.

**Baselines:** (a) **zero-shot inversion** — prompting un-tuned Qwen with the Appendix B prompts (rows 1 and 4); (b) **strong-surrogate inversion (R1)** as an achievability upper bound, since train and eval trace distributions then coincide.

**⚠ Prose/table discrepancy (verify before reproducing):** §5.2 states the R1-surrogate upper bound is *"TF1 58.00 / 57.08 and R-L 29.01 / 27.97 in the summary and no-summary settings."* The no-summary pair (57.08 / 27.97) matches Table 2 exactly, but the summary pair does **not** — Table 2 gives **TF1 64.42** and **R-L 29.17**. One of the two is stale. The abstract's headline "52.76 token-overlap F1 for DeepSeek-R1 traces" refers to the *R1-Weak summary* row and is consistent with Table 2.

---

## `EXP-T3-steal-r1` — Table 3 (§5.3)

**Research question:** Does fine-tuning a student on inverted traces actually improve its reasoning, and how does it compare to every baseline an attacker could realistically use — plus the oracle that only exists because this victim is open-weight?

| Field | Value |
|---|---|
| Victim | **R1** (open-weight) |
| Surrogate | R1-Weak (realistic) or R1 (upper bound); Surrogate-Trace baseline row uses R1-Weak |
| Inversion model | Qwen2.5-7B-Instruct (FT), separate models for summary and no-summary |
| Students | Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct |
| Dataset | OpenThoughts-114k, 10k victim split for student fine-tuning; 10k surrogate split for inversion training |
| Query budget | 10k victim queries |
| Benchmarks | MATH500, JEEBench, LiveCodeBench |
| Metric | Accuracy (%) — single number per cell, no variance reported |

**Results (Table 3) — full transcription.** `*` marks entries that **outperform the Victim-Trace oracle**.

| Setting | Training Sources | Surrogate | MATH500 Qwen | MATH500 Llama | JEEBench Qwen | JEEBench Llama | LCB Qwen | LCB Llama |
|---|---|---|---|---|---|---|---|---|
| – | No finetuning | – | 71.2 | 42.1 | 28.3 | 16.6 | 28.9 | 13.1 |
| – | Answer-only | – | 61.0 | 44.0 | 21.6 | 17.2 | 25.4 | 9.6 |
| – | Surrogate-Trace | R1-Weak | 63.2 | 48.8 | 19.7 | 11.3 | 22.9 | 9.1 |
| Summary | Summary+Answer | – | 63.0 | 45.8 | 24.0 | 18.2 | 25.6 | 11.7 |
| Summary | **Synthesized-Trace (ours)** | R1-Weak | 71.8 | 50.2 | 36.3 | 24.2* | 30.9 | 8.8 |
| Summary | **Synthesized-Trace (ours)** | R1 | 74.4* | 59.2 | 37.1 | 26.1* | 32.2 | 9.9* |
| No-sum. | **Synthesized-Trace (ours)** | R1-Weak | 66.4 | 44.8 | 29.7 | 16.3 | 26.2 | 9.2 |
| No-sum. | **Synthesized-Trace (ours)** | R1 | 74.1* | 62.0* | 33.2 | 25.2* | 29.1 | 8.6 |
| – | Victim-Trace (oracle) | – | 72.2 | 59.6 | 43.7 | 23.8 | 33.2 | 9.8 |

**Baselines compared against:** No-finetuning (base model), Answer-only, Surrogate-Trace (R1-Weak), Summary+Answer, and Victim-Trace (oracle).

**Key findings as the authors state them:**
- Fine-tuning on the most obvious attack data can *hurt*: Qwen MATH500 71.2 → 61.0 (Answer-only). The authors attribute Qwen2.5-7B-Instruct's high base MATH500 to prior optimization for that benchmark.
- Realistic attack (R1-Weak, summary): Qwen MATH500 63.0 → 71.8 and Llama 45.8 → 50.2 over Summary+Answer; JEEBench Qwen 24.0 → 36.3, Llama 18.2 → 24.2.
- No-summary still works: Qwen JEEBench 19.7 → 29.7 vs Surrogate-Trace.
- **Beating the oracle** (the paper's headline claim) happens when surrogate = victim = R1: Qwen/MATH500 74.1 vs 72.2, Llama/MATH500 62.0 vs 59.6, Llama/JEEBench 25.2 vs 23.8 (no-summary), plus Qwen/MATH500 74.4, Llama/JEEBench 26.1, Llama/LCB 9.9 in the summary setting, and Llama/JEEBench 24.2 even with the *weak* surrogate. Explanation offered: R1's raw traces contain backtracking and dead ends; the inversion model, conditioned on the final answer, emits clean forward reasoning that "denoises" the supervision signal (see `EXP-F6-denoising-case-study`).
- LCB is the weak spot: Llama drops 13.1 → 8.8 on Synthesized-Trace and even to 9.8 on the oracle; Qwen roughly flat (28.9 → 30.9). Attributed to math-flavored reasoning style and output-format mismatch with LCB.

**⚠ Prose/table discrepancies in §5.3:**
- Prose: *"Llama closes the JEEBench gap from 11.3% ... to 24.2% vs. **23.1%** oracle"* — Table 3 gives the Llama JEEBench oracle as **23.8**.
- Prose: *"even on the Victim-Trace oracle (**10.4%**)"* for Llama LCB — Table 3 gives **9.8**.
- Prose: *"Qwen **matches** oracle JEEBench (36.3% vs. 43.7%)"* — 36.3 does not match 43.7; the claim overstates the table.

---

## `EXP-T4-steal-gpt54mini` — Table 4 (§5.4)

**Research question:** Does the attack transfer to a genuinely black-box commercial victim that never exposes traces at all?

| Field | Value |
|---|---|
| Victim | **GPT-5.4 mini** (`gpt-5.4-mini-2026-03-17`), exposes only `(x, b*, y)` |
| Surrogate | R1 (strong) or R1-Weak |
| Inversion model | Qwen2.5-7B-Instruct (FT) |
| Students | Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct |
| Dataset | OpenThoughts-114k, 10k victim split (default) |
| Query budget | 10k victim queries (cost analysis below assumes 10k) |
| Benchmarks | MATH500, JEEBench, LiveCodeBench |
| Metric | Accuracy (%) |

**Results (Table 4) — full transcription.** No oracle row exists here: GPT-5.4 mini's traces are unobtainable.

| Setting | Training Sources | Surrogate | MATH500 Qwen | MATH500 Llama | JEEBench Qwen | JEEBench Llama | LCB Qwen | LCB Llama |
|---|---|---|---|---|---|---|---|---|
| – | Answer-only | – | 68.4 | 13.0 | 27.6 | 6.3 | 0.8 | 1.4 |
| – | Surrogate-Trace | R1 | 72.2 | 59.6 | 43.7 | 23.8 | 33.2 | 9.8 |
| – | Surrogate-Trace | R1-Weak | 63.2 | 48.8 | 19.7 | 11.3 | 22.9 | 9.1 |
| Summary | Summary+Answer | – | 19.8 | 16.4 | 1.6 | 5.5 | 0.0 | 0.4 |
| Summary | **Synthesized-Trace (ours)** | R1 | 76.0 | 52.4 | 43.7 | 19.9 | 28.9 | 10.0 |
| Summary | **Synthesized-Trace (ours)** | R1-Weak | 73.2 | 48.3 | 31.6 | 16.8 | 25.6 | 10.2 |
| No-sum. | **Synthesized-Trace (ours)** | R1 | 71.8 | 47.2 | 41.6 | 16.6 | 27.7 | 9.7 |

**⚠ ONE EXPERIMENT APPEARING TWICE — do not count it as two independent results.** Both Surrogate-Trace rows here are numerically *identical* to rows in Table 3:
- Table 4 `Surrogate-Trace / R1` (72.2 / 59.6 / 43.7 / 23.8 / 33.2 / 9.8) **≡** Table 3 `Victim-Trace (oracle)`.
- Table 4 `Surrogate-Trace / R1-Weak` (63.2 / 48.8 / 19.7 / 11.3 / 22.9 / 9.1) **≡** Table 3 `Surrogate-Trace / R1-Weak`.

This is internally consistent — R1's traces are the "oracle" when R1 is the victim and the "surrogate" traces when GPT-5.4 mini is the victim, so the *same fine-tuning runs* are reused across both tables. A reproducer only needs to run these two student fine-tunes once, and should not treat the agreement between the tables as independent corroboration.

**Baselines compared against:** Answer-only, Summary+Answer, Surrogate-Trace with both R1 and R1-Weak. (No zero-shot inversion baseline and no oracle in this table.)

**Key findings:**
- Against Answer-only with the R1-trained inverter: Qwen MATH500 68.4 → 76.0, Qwen JEEBench 27.6 → 43.7; Llama MATH500 13.0 → 52.4, Llama JEEBench 6.3 → 19.9.
- **Summary+Answer is actively harmful** here — often worse than Answer-only (Qwen JEEBench 27.6 → 1.6; Qwen LCB 0.8 → 0.0; Qwen MATH500 68.4 → 19.8). Direct distillation on summaries does not work.
- Inversion's advantage is largest when the surrogate is weak: R1-Weak Synthesized-Trace beats R1-Weak Surrogate-Trace everywhere it matters (Qwen JEEBench 19.7 → 31.6, Llama 11.3 → 16.8).
- With the *strong* R1 surrogate, Surrogate-Trace can equal or beat inversion, because Table 6 shows R1 ≈ GPT-5.4 mini in capability. The authors scope the claim: inversion is most advantageous when the victim is substantially stronger than the attacker's best surrogate.

**⚠ Prose/table discrepancy in §5.4:** *"With the strong R1 surrogate ... (Qwen JEEBench 43.7% → 43.7%, Llama **33.2% → 28.9%**)."* The Llama pair does not appear in any JEEBench column — 33.2 and 28.9 are the **Qwen LCB** Surrogate-Trace-R1 and Synthesized-Trace-R1 values. The intended Llama JEEBench comparison from the table is 23.8 → 19.9.

---

## `EXP-F3-query-budget` — Figure 3 (§5.4)

**Research question:** How does downstream student accuracy scale with the number of victim queries the attacker pays for?

| Field | Value |
|---|---|
| Victim | GPT-5.4 mini |
| Surrogate | R1 (R1-based inversion model) |
| Inversion model | Qwen2.5-7B-Instruct (FT), summary setting |
| Student | **Qwen2.5-7B-Instruct only** (Llama not swept) |
| Dataset | OpenThoughts-114k victim split |
| Query budget | **5k, 10k, 15k, 20k, 25k** |
| Benchmarks | MATH500, JEEBench, LiveCodeBench |
| Metric | Accuracy (%) |

**Results — DERIVED, not printed in the paper.** Figure 3 is a plot with no data labels and §5.4's prose quotes only the 5k/10k/15k MATH500 and JEEBench points. The full series below was recovered from the published source vector figure (`https://arxiv.org/html/2603.07267v2/query_plot.svg`) by reading each series' marker y-coordinates and calibrating them against the 0/20/40/60/80/100 y-axis tick positions:

```
value = (y_marker - 41.41875) / (254.69875 - 41.41875) * 100
```

Every recovered point lands on a clean 0.1 grid and the three prose-quoted MATH500 values (67.0, 77.6, 80.8) reproduce **exactly**, which validates the calibration. Treat the JEEBench and LCB series as *derived-and-plausible* rather than paper-stated — the paper's own prose disagrees with the plot on two JEEBench points (see below).

| Victim queries | MATH500 | JEEBench | LiveCodeBench |
|---|---|---|---|
| 5,000 | 67.0 | 38.8 | 29.5 |
| 10,000 | 77.6 | 42.3 | 28.9 |
| 15,000 | 80.8 | 44.3 | 24.9 |
| 20,000 | 79.2 | 46.4 | 24.9 |
| 25,000 | 80.6 | 47.3 | 26.5 |

Trends: MATH500 rises steeply 5k→15k then plateaus around 80; JEEBench rises monotonically all the way to 47.3 at 25k; LCB *declines* with scale (29.5 → 24.9) before a small recovery, consistent with the style/format mismatch discussed in §5.3.

**⚠ Prose vs figure:** §5.4 says *"from 38.8% to **43.7%** on JEEBench; 15k queries reach 80.8% and **44.5%**."* The MATH500 numbers (67.0, 77.6, 80.8) match the plot exactly, but the JEEBench values at 10k and 15k plot as **42.3** and **44.3**. The prose's 43.7 is the Table 4 value for the 10k main run; the two do not agree. When reproducing, expect JEEBench ≈42–44 at 10k, not 43.7 exactly.

**⚠ Budget vs data split:** the 20k and 25k points exceed the 10k victim split described in §5.1. The additional inputs must come from further OpenThoughts sampling; the paper does not say how disjointness from the surrogate split was maintained at those sizes.

**Baselines:** none — this is a scaling curve, not a comparison. The 10k column is the point of contact with Table 4 (Summary / Synthesized-Trace / R1 / Qwen: 76.0, 43.7, 28.9 there vs 77.6, 42.3, 28.9 here — LCB agrees, MATH500 and JEEBench differ by 1.4–1.6 points, so these are apparently separate runs).

---

## `EXP-T5-augmentation-ablation` — Table 5 (§5.4)

**Research question:** Can the attacker do better using only data it *already has*, without spending another victim query — (a) folding in the surrogate traces `D1` generated in Stage 1, and (b) domain-gating the supervision format so code prompts get bare answers instead of a reasoning prefix?

| Field | Value |
|---|---|
| Victim | GPT-5.4 mini |
| Surrogate | R1 (the augmentation data is R1's `D1` traces) |
| Inversion model | Qwen2.5-7B-Instruct (FT), summary setting |
| Student | **Llama-3.1-8B-Instruct only** |
| Dataset | Inverted victim traces `{(x, t̂, y)}`, optionally unioned with surrogate dataset `D1` |
| Query budget | No additional victim queries — "for free" (§4.3 optional augmentation) |
| Benchmarks | MATH500, JEEBench, LCB, **HumanEval+** (only table where HE+ appears) |
| Metric | Accuracy (%) |

**Results (Table 5) — full transcription:**

| Training Sources | MATH500 | JEEBench | LCB | HE+ |
|---|---|---|---|---|
| Synthesized-Trace | 50.2 | 15.5 | 10.0 | 20.7 |
| + surrogate augmentation | 51.6 | 19.9 | 8.6 | 20.7 |
| + domain gating | 48.0 | 15.5 | 5.3 | 51.2 |
| + both | **57.6** | **21.6** | 6.1 | 48.2 |

**Interventions defined:**
- *Surrogate augmentation* — train on `{(x, t̂, y)} ∪ D1` (§4.3 says this can be a single mixed corpus **or** a curriculum with surrogate traces first, inverted traces second; Table 5 does not say which was used — a reproduction gap).
- *Domain gating* — on code prompts, the assistant target is the bare answer with **no** reasoning prefix.

**Baselines:** the un-ablated Synthesized-Trace row is the internal baseline. No answer-only/oracle comparison in this table.

**⚠ Baseline-row ambiguity (important for reproduction):** Table 5's base row (Llama: 50.2 / 15.5 / 10.0 / 20.7) does not match any Llama row in Table 4. Table 4's Llama Synthesized-Trace/R1/summary row is 52.4 / 19.9 / 10.0 and the R1-Weak row is 48.3 / 16.8 / 10.2. Only LCB (10.0) lines up with the R1 row. Meanwhile "50.2" is exactly Table 3's Llama R1-Weak-summary MATH500, and "19.9" (the augmented JEEBench result) is exactly Table 4's Llama R1-summary JEEBench. The surrogate for Table 5's base row is therefore not determinable from the paper; assume R1 (the augmentation requires R1's `D1`) and expect the base numbers to shift on reproduction.

---

## `EXP-T6-model-baselines` — Table 6 (Appendix A)

**Research question:** How strong are the victims and surrogates themselves? This calibrates which regime each attack sits in — Table 4's caveat ("R1 is comparable to GPT-5.4 mini") rests entirely on this table.

| Field | Value |
|---|---|
| Models evaluated | R1, R1-Distill (= R1-Weak, DeepSeek-R1-Distill-Qwen-1.5B), GPT-5.4-mini |
| Students / inversion | n/a |
| Protocol | **Zero-shot** accuracy, benchmarks' own protocols |
| Benchmarks | MATH500, JEEBench, LCB |

**Results (Table 6):**

| Model | MATH500 | JEEBench | LCB |
|---|---|---|---|
| R1 | 91.6 | 87.1 | 74.9 |
| R1-Distill | 81.4 | 32.6 | 27.6 |
| GPT-5.4-mini | 91.4 | 85.1 | 66.9 |

Reading: R1 ≥ GPT-5.4 mini on all three (so GPT-5.4 mini is *not* "much stronger than the attacker's best surrogate" when that surrogate is R1). R1-Distill is dramatically weaker on JEEBench (32.6) and LCB (27.6) — this is the regime where inversion pays off most, matching the Table 4 finding.

---

## `EXP-COST-query-economics` — §5.4 (prose only)

**Research question:** Is the attack economically practical?

| Input | Value |
|---|---|
| GPT-5.4 mini pricing used | **$0.75 / M input tokens**, **$4.50 / M output tokens** |
| Query budget priced | 10k `(x, b*, y)` queries |
| **Total cost** | **$173.28** |

Inversion-model training and student fine-tuning require **zero** additional victim queries. Context given by the authors: their budgets (≤25k) are far below production distillation corpora — OpenThoughts is 114k and OpenThoughts3 is 1.2M examples.

---

## Qualitative / illustrative figures

These carry no numbers but are part of the reproduction surface (the released code should regenerate the underlying examples).

### `EXP-F1-teaser-example` — Figure 1 (§1)
Source asset: `example_0.svg`. One Trace Inversion instance against GPT-5.4 mini: given input + final answer + reasoning summary, the synthesized detailed trace usable for student SFT.

### `EXP-F2-pipeline-diagram` — Figure 2 (§4)
Source asset: `inversion_pipeline.png`. The three-stage pipeline: (1) train inversion model on surrogate data, (2) invert victim outputs, (3) distill into the student. Not an experiment.

### `EXP-F4-zeroshot-vs-ft-qualitative` — Figure 4 (Appendix C)
Victim = R1. Side-by-side of **zero-shot inversion** vs **trace inversion with R1-Distill as surrogate**. The qualitative counterpart to Table 2's zero-shot rows (the ~1,000-token collapse).

### `EXP-F5-gt-vs-synth-qualitative` — Figure 5 (Appendix C)
Victim = R1. Side-by-side of the **ground-truth R1 trace** vs a **trace synthesized with R1-Distill as surrogate** — the realistic-attack fidelity illustration.

### `EXP-F6-denoising-case-study` — Figure 6 (Appendix D)
The evidence behind "synthesized traces can be better than ground-truth traces."

- **Query:** "The positive reals `a` and `b` satisfy [equation]. Show that [inequality]." (the specific symbols are dropped by the HTML math rendering; recover from the PDF).
- **Left — R1 ground-truth trace:** visibly thrashes — *"Wait, but we already have ... Wait, let's step back"*, tries Jensen's inequality (*"But not sure if helpful here"*), tries a substitution (*"But this seems complicated. Maybe not the best approach"*), and ends a branch with *"Hmm, not sure if helpful."*
- **Right — synthesized trace (inversion model trained with R1 as surrogate):** direct forward derivation — substitution, expansion, factoring, a discriminant argument (*"Since the discriminant is negative, the quadratic has no real roots and since the coefficient is positive, it is always positive"*), concluding cleanly.

This is the mechanism the authors propose for the starred oracle-beating entries in Table 3: conditioning on the final answer removes backtracking and dead ends, producing a lower-noise SFT signal.

---

## Prompts used (Appendix B) — needed for exact reproduction

Three prompts are given in full in Appendix B:

1. **Reasoning Summary Compression Prompt** (`π`, used by `C'` = Qwen2.5-7B-Instruct). System prompt specifies: 3–6 sections, each opening with a short bold markdown header on its own line followed by a 2–5 sentence paragraph; no numbered lists, no bullets; first-person present tense, tentative/exploratory voice; one meaningful reasoning move per section; target length **roughly 600–900 tokens**; inline LaTeX where the original used math; no meta-commentary. User prompt: `Summarize this thinking process as a first-person inner-monologue recap: <think>{thinking_content}</think>`.
   **⚠ Not fully published:** the prompt "additionally includes two few-shot exemplars of the target style (one algebra/number theory and one geometry, ~600 tokens each), drawn from GPT-5-mini's own summaries," which are omitted from the paper and only released with the code. Table 1's style match cannot be reproduced without them.
2. **Zero-shot Inversion Prompt (With Summaries)** — expands numbered "reasoning bubbles" into a full trace wrapped in `<think>…</think>`, up to 20,000 characters; explicitly instructed not to invent steps outside the bubbles. Placeholders: `{user_prompt}`, `{assistant_answer}`, `{reasoning_summary}`.
3. **Zero-shot Inversion Prompt (No Summaries)** — reconstructs a trace from `(input, output)` only, `<think>…</think>`-wrapped, "span thousands of tokens if needed." Placeholders: `{user_prompt}`, `{assistant_answer}`.

Prompts 2 and 3 are used **only** for the "Qwen (0-shot)" baseline rows of Table 2; the fine-tuned inversion models are trained with Eq. 1 / Eq. 2 objectives.

---

## Consolidated reproduction risk list

1. **No decoding parameters anywhere** — temperature, top-p, sampling vs greedy, pass@k, sample counts are unspecified for every generation step (victim, surrogate, inversion, student eval).
2. **No batch size, scheduler, optimizer name, or precision** — only epochs (3), LR (1e-5), warmup ratio (0.1), cutoff_len (16384).
3. **TF1 is defined in one clause** — bag-of-tokens F1; tokenizer, clipping, and normalization unspecified.
4. **Grading procedures deferred** to "the original evaluation protocols of each benchmark"; LCB version/subset not pinned.
5. **Table 2 summary-setting R1 numbers disagree between prose (TF1 58.00, R-L 29.01) and the table (64.42, 29.17).**
6. **Table 3 prose disagrees with the table** on the Llama JEEBench oracle (23.1 vs 23.8) and the Llama LCB oracle (10.4 vs 9.8).
7. **Table 4 prose** cites a "Llama 33.2% → 28.9%" JEEBench comparison that corresponds to Qwen LCB values.
8. **Figure 3 JEEBench** prose values (43.7 @10k, 44.5 @15k) differ from the plotted values (42.3, 44.3); Figure 3's 10k point also differs from Table 4's 10k run.
9. **Table 5's base row cannot be matched** to any Table 4 row; its surrogate and whether augmentation used mixed-corpus or curriculum training are unstated.
10. **Query budget exceeds the stated data splits** — §5.1 defines 10k/10k splits while Figure 3 sweeps to 25k victim queries.
11. **Compression prompt's two few-shot exemplars are not in the paper** (code-only), and they are load-bearing for the Table 1 style match.
12. **No seeds, no repeated runs, no error bars** on any accuracy number in the paper.
13. **Framework mismatch** — the reported hyperparameter names are LLaMA-Factory's; reimplementing in TRL inherits different defaults for LR schedule, optimizer, and prompt-token loss masking, none of which the paper pins down.
14. **Table 4's two Surrogate-Trace rows are reused from Table 3**, not independent runs — 2 of the 16 student fine-tunes across the two tables are shared.

**Mitigation for most of the above:** the authors release code at https://github.com/Tingwei-Zhang/Trace_Inversion_Attack (§1, footnote 1). Every gap flagged here (decoding params, batch size, TF1 implementation, grading harnesses, the two withheld few-shot exemplars in the compression prompt) should be resolved against that repo before attempting a from-scratch reimplementation.
