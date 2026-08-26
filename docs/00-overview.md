# Trace Inversion — Paper Overview & Reproduction Roadmap

**Paper:** How to Steal Reasoning Without Reasoning Traces
**Authors:** Tingwei Zhang, John X. Morris, Vitaly Shmatikov (Cornell Tech)
**arXiv:** [2603.07267v2](https://arxiv.org/abs/2603.07267) [cs.CR] — v1 2026-03-07, v2 2026-05-12
**Code:** https://github.com/Tingwei-Zhang/Trace_Inversion_Attack (Apache-2.0, verified live)

> ⚠️ **The released code is v1-only.** Last push 2026-03-06; arXiv v2 landed 2026-05-12.
> `chatgpt_inference.py` hard-codes `gpt-5-mini-2025-08-07`, but v2's victim is
> `gpt-5.4-mini-2026-03-17`. The v2 experiments were re-run against a newer victim and the code was
> never updated — so the repo reproduces v1's numbers, not the ones in the paper you are reading.
> It also depends on two pinned submodules (LLaMA-Factory `9501c33`, Evalchemy `6ed6741`) and needs
> Python **3.11** despite the README saying 3.10.

---

## 1. The one-paragraph version

Commercial reasoning APIs (OpenAI o-series, Gemini, Claude extended thinking) hide the raw chain of
thought and expose only the final answer plus a short "reasoning summary." The implicit security
assumption is that hiding the trace prevents competitors from distilling the model's reasoning
ability. This paper breaks that assumption. The attacker trains a **trace inversion model** on a
*surrogate* reasoning model they control — learning the mapping `(input, answer, summary) → full
trace`. They then apply that inverter to the *victim's* black-box outputs, synthesizing traces the
victim never revealed, and fine-tune a student on them. The synthesized traces do not need to match
the victim's real reasoning; they only need to be good training data. They are.

## 2. Why the result is interesting (not just "distillation works")

Three findings carry the paper, and each is a separate reproducible claim:

1. **Summaries are worse than useless for the defender.** Fine-tuning directly on the victim's
   answers *degrades* the student (Qwen MATH500 71.2% → 61.0%). Fine-tuning on summaries+answers is
   catastrophic against GPT-5.4 mini (Qwen JEEBench 27.6% → 1.6%). But run those *same* outputs
   through the inverter and you get 76.0% / 43.7%. The summary is a free conditioning signal for the
   attacker while being nearly worthless as direct supervision.

2. **A weak surrogate is nearly as good as a strong one.** The inverter trained on
   R1-Distill-Qwen-**1.5B** traces recovers TF1 52.76 against R1, vs 64.42 for an inverter trained on
   R1 itself — and zero-shot prompting gets only 35.36. Most of the value comes from *any* trained
   inverter, not from a strong surrogate. This is what makes the attack cheap.

3. **Inverted traces can beat the ground truth.** In 6 cells of Table 3, synthesized traces
   outperform the Victim-Trace oracle. The paper's explanation: R1's real traces contain backtracking
   and dead ends; the inverter, conditioned on the *known correct answer*, emits clean forward
   reasoning. Inversion is accidentally a denoiser.

The mechanism behind #3 is the conceptual crux: **the answer is conditioning input at Stage 2 but the
prediction target at Stage 3.** The inverter is allowed to peek at the answer, which is exactly why
the trace it writes never wanders.

## 3. The pipeline

```
Stage 1 — TRAIN INVERTER (no victim queries)
  x' ~ OpenThoughts (surrogate split, 10k)
      → surrogate V'  ⇒ (t', y')
      → compressor C' (zero-shot, fixed prompt π)  ⇒ b'
  train I:  (x', y', b') → t'     [teacher-forced token CE, Eq. 1]
  train I:  (x', y')     → t'     [no-summary variant,     Eq. 2]

Stage 2 — INVERT VICTIM (this is the only step that costs money)
  x ~ OpenThoughts (victim split, 10k, DISJOINT from above)
      → victim V  ⇒ (y, b*)
      → I(x, y, b*)  ⇒  t̂        [no filtering, no verification]

Stage 3 — DISTILL STUDENT (no victim queries)
  fine-tune S on  x → y⁺ = [t̂ ; y]   [teacher-forced CE, Eq. 3]
  optional: mix in Stage-1 surrogate data D₁ "for free"
```

Two settings, **two separately trained inverters**: `summary` and `no-summary`.

`C'` exists only because the attacker cannot observe victim (trace, summary) pairs — they must
manufacture training pairs using their own summarizer, then hope the summary *distributions* match.
Table 1 shows the authors reverse-engineered the prompt until they did (median 537 vs 592 tokens,
bold headers 94.1% vs 92.9%, first-person 97.3% vs 97.0%, LaTeX 79.1% vs 71.9%).

## 4. Paper's configuration vs. ours

| Role | Paper | Our plan |
|---|---|---|
| Victim (open-weight) | DeepSeek-R1 — **684.5B, 688.6 GB** | Impossible. Substitute **Qwen3.8-27B** (GGUF **IQ4_XS**, 14.62 GiB) |
| Victim (black-box) | `gpt-5.4-mini-2026-03-17` | Optional second track. $0.75/M in, $4.50/M out |
| Surrogate `V'` | R1, or R1-Distill-Qwen-1.5B ("R1-Weak") | **R1-Distill-7B** primary, **1.5B** as a second arm |
| Compressor `C'` | Qwen2.5-7B-Instruct, zero-shot | `Qwen3.5-4B`, local |
| Inverter `I` | Qwen2.5-7B-Instruct, fine-tuned | `Qwen3.5-4B`, TRL + **bf16 LoRA** |
| Student `S` | Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct | `Qwen3.5-2B`, TRL + **full fine-tuning** |
| Data | OpenThoughts-114k, 2×10k disjoint splits | local (`llamafactory/OpenThoughts-114k`) |
| Benchmarks | MATH500, JEEBench, LiveCodeBench (+HumanEval+) | local |
| Hardware | 8× A100 80GB | 1× RTX 4090 24GB, 30GB RAM |
| Training | full-param SFT, 3 ep, lr 1e-5, warmup 0.1, `cutoff_len` 16384 | TRL `SFTTrainer`; student **full FT @ 16384, lr 1e-5 — matches** · inverter LoRA @ 12288 |

Two translation problems this creates:

1. **`cutoff_len` is LLaMA-Factory's parameter name, not TRL's.** The authors used LLaMA-Factory; we
   are reimplementing in TRL, so every training setting needs mapping into `SFTConfig` — including
   defaults that silently differ (LR schedule, prompt-token loss masking). See `06`.
2. **Swapping the open-weight victim from R1 to Qwen3.8-27B is a deviation, not a shortcut.** R1 at
   688 GB was never locally runnable. The substitution is what makes trace-fidelity (Table 2)
   reproducible at all, since that metric requires reading the victim's ground-truth trace — which no
   API victim can ever provide.

**Bonus the substitution buys us:** Qwen3.8-27B (GPQA Diamond 89.2) against a 1.5B surrogate finally
tests the *victim ≫ surrogate* regime. The paper claims inversion matters most there but never
evaluates it — its own Table 6 shows R1 (91.6/87.1/74.9) ≳ GPT-5.4 mini (91.4/85.1/66.9). This is
the most valuable experiment available to us and it is not in the paper.

### 4.1 Role assignments — SUPERSEDED, see `10-run-plan.md`

> The table below is the **pre-Phase-0 plan**. Phase 0 measured every candidate and moved four
> roles. The authoritative list is `10-run-plan.md`; results are in `docs/results/baselines.md`.
>
> | Role | Chosen | MATH500 | JEEBench |
> |---|---|---|---|
> | Victim `V` | `Qwen3.8-27B` **IQ4_XS** (not Q5_K_M) | 98.8% | 86.2% |
> | Surrogate `V'` | **R1-Distill-7B** primary (not the 1.5B), 1.5B as arm 2 | 92.6% | 60.6% |
> | Compressor `C'` | `Qwen3.5-4B` — unchanged | — | — |
> | Inverter `I` | `Qwen3.5-4B`, **bf16 LoRA** (not QLoRA) | — | — |
> | Student `S` | **`Qwen3.5-2B`, full fine-tuning** (not Qwen2.5-7B QLoRA) | 79.0% | 47.8% |
>
> Ordering achieved on JEEBench: student 47.8 < surrogate 60.6 < victim 86.2.



Note there are **four** attacker-side roles, not three — the compression model `C'` is easy to
overlook and is required to build the summary-setting data at all.

| Role | Pick | Size | Notes |
|---|---|---|---|
| Victim `V` | `unsloth/Qwen3.8-27B-GGUF` @ Q5_K_M, llama.cpp, no `--mmproj` | 18.47 GiB | quality matters most here |
| Surrogate `V'` | `DeepSeek-R1-Distill-Qwen-1.5B` | 3.5 GB | the paper's "R1-Weak" |
| Compressor `C'` | `Qwen/Qwen3.5-4B`, `enable_thinking=False` | 9.3 GB | zero-shot, never trained |
| Inverter `I` | `Qwen/Qwen3.5-4B` QLoRA | ~7.5 GB @ 8k | TRL |
| Student `S` | `Qwen/Qwen2.5-7B-Instruct` QLoRA | ~9–10 GB @ 8k | TRL |

> **Why the student is *not* a Qwen3.5/3.8 model.** Every Qwen3.5/3.8 Instruct model reasons by
> default. You cannot demonstrate that inversion *taught* a model to reason if the model already
> reasons — the treatment effect is unmeasurable. The student must be a non-reasoning instruct
> model, which is exactly why the paper used Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct.
> Llama-3.1-8B is gated; `allenai/OLMo-2-1124-7B-Instruct` is the ungated substitute.

**Correction to an earlier estimate:** 4-bit weights for this model are **not** ~13.9 GB. The
embedding and `lm_head` stay 16-bit and are untied at 248,320 × 5,120, adding ~5.09 GB — a real
W4A16 AWQ checkpoint measures **18.70 GB**. The GGUF figures quoted elsewhere in these docs are
measured file sizes and are unaffected.

## 5. Reported results at a glance

Victim = R1 (Table 3), best realistic attack row (`R1-Weak` surrogate, summary setting):

| | MATH500 | JEEBench | LCB |
|---|---|---|---|
| No finetuning | 71.2 / 42.1 | 28.3 / 16.6 | 28.9 / 13.1 |
| Answer-only | 61.0 / 44.0 | 21.6 / 17.2 | 25.4 / 9.6 |
| Summary+Answer | 63.0 / 45.8 | 24.0 / 18.2 | 25.6 / 11.7 |
| Surrogate-Trace | 63.2 / 48.8 | 19.7 / 11.3 | 22.9 / 9.1 |
| **Synthesized (ours)** | **71.8 / 50.2** | **36.3 / 24.2** | **30.9 / 8.8** |
| Victim-Trace (oracle) | 72.2 / 59.6 | 43.7 / 23.8 | 33.2 / 9.8 |

*(Qwen / Llama. Full tables incl. GPT-5.4 mini victim in `02-experiments-and-results.md`.)*

LiveCodeBench is the consistent weak spot — Llama *regresses* on it under every condition, including
the oracle. The paper attributes this to style/format mismatch between math-flavored traces and
coding tasks.

**Query-budget scaling (Figure 3).** The paper's prose quotes only three points; the full series was
recovered from the plot SVG and the quoted values reproduce exactly, so the rest is trustworthy:

| Queries | MATH500 | JEEBench | LCB |
|---|---|---|---|
| 5k | 67.0 | 38.8 | 29.5 |
| 10k | 77.6 | 42.3 | 28.9 |
| 15k | 80.8 | 44.3 | 24.9 |
| 20k | 79.2 | 46.4 | 24.9 |
| 25k | 80.6 | 47.3 | 26.5 |

Note MATH500 essentially saturates after 15k while JEEBench keeps climbing and LCB decays. **For our
purposes the important reading is that 5k queries already delivers most of the MATH500 benefit** —
which matters enormously, because trace generation is the single most expensive step of the local
reproduction.

## 6. Where the paper is soft

Flagged here because these determine what a reproduction can honestly claim:

- **No seeds, no variance, no confidence intervals anywhere.** The "beats the oracle" claim rests on
  margins of 0.4–2.4 points across 6 cells with n=1 run each.
- **No length-matched control.** Nothing separates "recovered the victim's reasoning" from "trained
  on long CoT-shaped text." A reproduction should add this control — it is the cheapest high-value
  addition available.
- **The victim-stronger-than-surrogate regime is never actually tested.** Table 6 shows R1 (91.6 /
  87.1 / 74.9) ≳ GPT-5.4 mini (91.4 / 85.1 / 66.9). The paper's own framing says inversion should
  matter most when the victim outclasses the surrogate — that experiment is missing.
- **Qwen2.5-7B-Instruct is acknowledged to be MATH500-contaminated**, which is why fine-tuning can
  hurt it. That makes MATH500 deltas on Qwen hard to interpret.
- **Fidelity is unverifiable for the closed victim** by construction, so Table 2 only exists for R1.
- Table 4's `Surrogate-Trace / R1` row is the same run as Table 3's oracle row — one experiment
  appearing in two tables, not two results.

**Internal inconsistencies found during extraction** (prose vs. tables — check against the tables,
which are self-consistent):

| Where | Prose says | Table/figure says |
|---|---|---|
| §5.2 | R1-surrogate summary upper bound TF1 58.00 / R-L 29.01 | Table 2: **64.42 / 29.17** |
| §5.4 | JEEBench 43.7 @10k, 44.5 @15k | Figure 3: **42.3 / 44.3** |
| §5.4 | "Llama 33.2→28.9" on JEEBench | those are **Qwen LCB** values |
| §5.3 | Llama oracle JEEBench 23.1, LCB 10.4 | Table 3: **23.8 / 9.8** |
| Table 5 | base row 50.2 / 15.5 / 10.0 / 20.7 | matches **no row** in Table 4 — surrogate undeterminable |

That last one matters most: the Table 5 ablation baseline can't be tied to a specific Table 4
configuration, so expect that ablation's starting point to shift on reproduction.

See `04-defenses-and-limitations.md` for the full treatment.

## 7. Document map

| Doc | Contents |
|---|---|
| `01-method-and-threat-model.md` | §1–4: threat model, three stages, equations, prompts, underspecified details |
| `02-experiments-and-results.md` | §5 + all tables/figures: full experiment inventory with numbers |
| `03-artifacts-and-availability.md` | Code repo, HF datasets/models, sizes, gating, API access |
| `04-defenses-and-limitations.md` | §6–7: defenses (none tested), limitations, independent critique |
| `05-rtx4090-feasibility.md` | VRAM math, throughput estimates, what's impossible locally |
| `06-model-selection-and-trl.md` | Qwen3.5/3.8 survey, role assignments, TRL configs |
| `07-reference-implementation.md` | What the released code specifies, and where it contradicts the paper |
| `08-measured-hardware-results.md` | **Measured** throughput/VRAM on this 4090 — supersedes `05`'s estimates |
| `09-deviations-from-paper.md` | **Running log of every deviation from the paper, and why** |
| `10-run-plan.md` | **Authoritative** role assignments, per-phase plan and time budget |
| `11-phase1-handoff.md` | Operational handoff for Phase 1 |
| `12-phase1-readiness-review.md` | Pre-Phase-1 audit: measured training VRAM, the 8192-cap finding, four plan corrections |

**Read `07` before writing any code.** It recovers the decoding and training hyperparameters the
paper omits, and documents three places where the released code disagrees with paper v2 — including
a compression prompt that is the *structural opposite* of the one in Appendix B, and a
format mismatch between v2's own compression and inversion prompts.

## 8. Status

- **Paper comprehension and documentation** (`00`–`09`) — complete.
- **Phase 0, baselines** — complete. Every model role fixed on measurement; see
  `docs/results/baselines.md` and `10-run-plan.md`.
- **Pre-Phase-1 readiness review** (`12`) — complete. Training VRAM measured, four plan
  corrections applied.
- **Phase 1, surrogate data** — next. Operational handoff in `11-phase1-handoff.md`.
