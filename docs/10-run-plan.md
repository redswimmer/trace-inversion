# Run Plan — Models, Engines, Quantization

Every model we will load, in what format, on which engine, and why.

## The rule

> **Fixed inputs** (never trained) → GGUF on llama.cpp, at the highest precision that fits.
> **Trainable artifacts** (produced by TRL) → safetensors bf16 on vLLM, so every checkpoint
> evaluates with no conversion step.

Quantize only where the model does not otherwise fit. The victim is the sole model forced below
full precision.

---

## Role assignments

| Role | Model | Format | Engine | Precision | Trained? |
|---|---|---|---|---|---|
| **Victim** `V` | `unsloth/Qwen3.8-27B-GGUF` → `IQ4_XS` | GGUF 14.62 GiB | llama.cpp | **4-bit** (forced — 55 GB at bf16) | no |
| **Surrogate** `V'` (primary) | `DeepSeek-R1-Distill-Qwen-7B` | F16 14.19 GiB | llama.cpp | F16 | no |
| **Surrogate** `V'` (arm 2) | `DeepSeek-R1-Distill-Qwen-1.5B` | BF16 3.32 GiB | llama.cpp | bf16 | no |
| **Compressor** `C'` | `Qwen/Qwen3.5-4B` | safetensors | vLLM | bf16 | no (zero-shot) |
| **Inverter** `I` | `Qwen/Qwen3.5-4B` | safetensors | TRL → vLLM | bf16 | **yes** |
| **Student** `S` | `Qwen/Qwen3.5-2B` *(pending baselines)* | safetensors | TRL → vLLM | bf16 | **yes** |

**Both surrogates run** — decided on Phase 0 measurements. The 1.5B sits 15 pts *below* the student
on JEEBench (32.6 vs 47.8), inverting the paper's ordering; the 7B clears it by 12.8 and sits
midway to the victim. Running both turns a forced choice into the surrogate-strength sweep the
paper never did (they tested only 1.5B and 685B).

---

## Phase 0 — Baselines *(in progress)*

Establishes headroom and validates the harness before anything is trained.

| # | Model | Engine | Precision | Tasks | Status |
|---|---|---|---|---|---|
| 0.1 | Qwen3.5-0.8B | vLLM | bf16 | MATH500 + JEEBench | running |
| 0.2 | Qwen3.5-2B | vLLM | bf16 | " | queued |
| 0.3 | Qwen3.5-4B | vLLM | bf16 | " | queued |
| 0.4 | R1-Distill-1.5B | vLLM | bf16 | " | queued — **harness calibration** |
| 0.5 | Qwen3.8-27B IQ4_XS | llama.cpp | 4-bit | 250-problem subset | after GPU frees |

Shared: 32,768 max generation / 40,960 context, seed 1234, pass@1.
Sampling: Qwen3.5 `1.0 / 0.95 / k20`; R1-Distill `0.6 / 0.95 / no top-k`.

**Gate:** 0.4 must reproduce the paper's Table 6 (MATH500 81.4, JEEBench 32.6) within ~8 points, or
every other number is suspect. **Decides:** which student, and whether FFT is possible.

---

## Phase 1 — Surrogate data *(paper Stage 1)*

| Step | Model | Engine | Output |
|---|---|---|---|
| 1.1 | Surrogate | llama.cpp | 5k reasoning traces `t'` + answers `y'` on OpenThoughts split A |
| 1.2 | Compressor Qwen3.5-4B | vLLM | summaries `b'` from each `t'` (zero-shot, prompt `π`) |

Yields `D₂ = {(x', y', b', t')}` — the inverter's training set.

`π` must be **written from scratch**: the paper's v2 compression prompt exists only in the PDF and
its two few-shot exemplars were never released. Validate against Table 1's style statistics
(median ~592 tokens, bold headers ~93%, first-person ~97%, LaTeX ~72%).

Est. ~2 h.

## Phase 2 — Train the inverter *(paper Stage 1 cont.)*

| Step | Model | Framework | Method |
|---|---|---|---|
| 2.1 | Qwen3.5-4B | TRL `SFTTrainer` | `(x', y', b') → t'` — summary setting |
| 2.2 | Qwen3.5-4B | TRL `SFTTrainer` | `(x', y') → t'` — no-summary setting |

Two separately trained inverters, per §4. FFT if it fits, else LoRA r=32-64.
Est. ~14 h for both.

## Phase 3 — Query the victim *(paper Stage 2)*

| Step | Model | Engine | Output |
|---|---|---|---|
| 3.1 | Victim IQ4_XS | llama.cpp, `-np 32`, q8_0 KV | 5k `(y, t)` on OpenThoughts split B (**disjoint** from A) |
| 3.2 | Compressor | vLLM | victim summaries `b*` from victim traces |

The victim's real traces `t` are **withheld from the attack** and used only for (a) the
`Victim-Trace` oracle baseline and (b) Table 2 fidelity scoring. This is what a local victim buys
that no API victim can.

Est. **~23 h** — the single largest cost. At 303 t/s measured, 5k × ~5k tokens.

## Phase 4 — Invert *(paper Stage 2 cont.)*

| Step | Model | Engine | Output |
|---|---|---|---|
| 4.1 | Inverter (merged) | vLLM | synthetic traces `t̂` from `(x, y, b*)` |

Est. ~4 h. Then score `t̂` against withheld `t` → **Table 2 reproduction**.

## Phase 5 — Train students *(paper Stage 3)*

Five conditions, matching the paper exactly:

| Condition | Supervision target |
|---|---|
| Answer-only | `y` |
| Summary+Answer | `b*`, `y` |
| Surrogate-Trace | `t'`, `y'` |
| **Synthesized-Trace (ours)** | `t̂`, `y` |
| Victim-Trace (oracle) | `t`, `y` — withheld ground truth |

Plus, if the 2B is chosen, the same condition run **FFT and LoRA** to measure what LoRA costs.
Est. ~20 h.

## Phase 6 — Evaluate

All fine-tuned students on MATH500 + JEEBench (+ LiveCodeBench if time), vLLM, same protocol as
Phase 0 so pre/post is directly comparable. Est. ~8 h.

---

## Totals

| Phase | Est. |
|---|---|
| 0 Baselines | ~11 h |
| 1 Surrogate data | ~2 h |
| 2 Inverter training | ~14 h |
| 3 **Victim queries** | **~23 h** |
| 4 Inversion | ~4 h |
| 5 Student training | ~20 h |
| 6 Evaluation | ~8 h |
| **Total** | **~82 h** (~3.5 days of GPU time) |

Every individual stage fits an overnight run. Phase 3 dominates; if it needs cutting, the paper's
own Figure 3 shows 5k queries already delivers most of the MATH500 benefit, and 2k would halve it
again at some cost to the result.

## Engine boundary

Only one comparison in the whole plan crosses engines: victim benchmark (llama.cpp) versus student
benchmark (vLLM), in Phase 0. That comparison is a sanity check on headroom, not a measured claim.
Every result that carries the paper's argument — student before vs after, across the five training
conditions — is vLLM on both sides. See `09` §7.5.
