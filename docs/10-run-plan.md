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
| **Inverter** `I` | `Qwen/Qwen3.5-4B` | safetensors | TRL → vLLM | bf16 + **LoRA** (not QLoRA) | **yes** |
| **Student** `S` | `Qwen/Qwen3.5-2B` | safetensors | TRL → vLLM | bf16 (**full FT**, `max_length` 16384) | **yes** |

**Both surrogates run** — decided on Phase 0 measurements. The 1.5B sits 15 pts *below* the student
on JEEBench (32.6 vs 47.8), inverting the paper's ordering; the 7B clears it by 12.8 and sits
midway to the victim. Running both turns a forced choice into the surrogate-strength sweep the
paper never did (they tested only 1.5B and 685B).

---

## Phase 0 — Baselines ✅ COMPLETE

Full results and analysis in `docs/results/baselines.md`.

| # | Model | Role | Engine | Precision | MATH500 | JEEBench | Audit |
|---|---|---|---|---|---|---|---|
| 0.1 | Qwen3.5-0.8B | student cand. | vLLM | bf16 | 45.6% | 20.2% | ✅ (card sampling) |
| 0.2 | **Qwen3.5-2B** | **student** | vLLM | bf16 | **79.0%** | **47.8%** | ⚠️ looping, deferred |
| 0.3 | Qwen3.5-4B | student cand. | vLLM | bf16 | 91.4% | 73.8% | ✅ (card sampling) |
| 0.4 | R1-Distill-1.5B | surrogate arm 2 | llama.cpp | GGUF BF16 | 84.0% | 32.6% | ✅ **calibration** |
| 0.5 | **Qwen3.8-27B IQ4_XS** | **victim** | llama.cpp | GGUF 4-bit | **98.8%** | **86.2%** | ✅ |
| 0.6 | **R1-Distill-7B** | **surrogate** | llama.cpp | GGUF F16 | **92.6%** | **60.6%** | ✅ |

Protocol: 1015 tasks · 32,768 max gen / 40,960 ctx · seed 1234 · pass@1 ·
paper sampling `temp 0.7 / top_p 0.9 / rep 1.05`.

**Gate passed:** the surrogate reproduces the paper's Table 6 — JEEBench 32.6 vs their 32.6,
MATH500 84.0 vs 81.4. Harness validated end to end.

**Measured headroom, student → victim:** MATH500 +19.8 pts, JEEBench +38.4 pts. Both benchmarks
have room. The 4B would have left only ~12 pts on JEEBench, which is why it lost despite scoring
higher.

**Ordering achieved:** student 47.8 < surrogate 60.6 < victim 86.2 on JEEBench — the regime the
paper's argument requires.

---

## Phase 1 — Surrogate data *(paper Stage 1)*

| Step | Model | Engine | Output |
|---|---|---|---|
| 1.1 | Surrogate **7B**, then **1.5B** | llama.cpp | 5k traces `t'` + answers `y'` each, on OpenThoughts split A |
| 1.2 | Compressor Qwen3.5-4B | vLLM | summaries `b'` from each `t'` (zero-shot, prompt `π`) |

Yields `D₂ = {(x', y', b', t')}` — the inverter's training set.

`π` must be **written from scratch**: the paper's v2 compression prompt exists only in the PDF and
its two few-shot exemplars were never released. Validate against Table 1's style statistics
(median ~592 tokens, bold headers ~93%, first-person ~97%, LaTeX ~72%).

**Over-generate 1.34× and drop capped rows.** 25.5% ± 2.5 of R1's own ground-truth traces on this
dataset exceed `max_new_tokens 8192`; a capped trace has no `</think>` and poisons the inverter
(`12` §2). 6,706 rows in, ~5,000 clean out, 31.5 M generated tokens.

Est. **~29-31 h** for both surrogates including compression — revised up from ~4 h, then from
~15-18 h once the 7B run was measured. **Sweep concurrency at 10,240/slot, not 32,768**
(`bench/sweep_concurrency.sh`); for the 7B that is 32 slots / 1,270 t/s **swept**. Treat that as a
*ranking* of slot counts, not an operating rate: the 7B realized ~445 t/s in the real run, so budget
from the first 30 minutes of the run itself (`11` §5).

## Phase 2 — Train the inverter *(paper Stage 1 cont.)*

| Step | Model | Framework | Method |
|---|---|---|---|
| 2.1 | Qwen3.5-4B | TRL `SFTTrainer` | `(x', y', b') → t'` — summary setting |
| 2.2 | Qwen3.5-4B | TRL `SFTTrainer` | `(x', y') → t'` — no-summary setting |

Two separately trained inverters, per §4. **LoRA r=32-64 on a bf16 base** — 4B FFT is confirmed
impossible (27.81 GiB @8k) and NF4 is not needed (bf16 LoRA is 15.11 GiB @8k). `max_length` **12288**:
the inverter prompt is ~1,524 tokens (problem 81 + answer 542 + 900-token summary), so a cap-8192
trace needs ~9,716. Measured in `12` §1 and §3. Est. ~14 h for both.

## Phase 3 — Query the victim *(paper Stage 2)*

| Step | Model | Engine | Output |
|---|---|---|---|
| 3.1 | Victim IQ4_XS | llama.cpp, `-np 32`, q8_0 KV | 5k `(y, t)` on OpenThoughts split B (**disjoint** from A) |
| 3.2 | Compressor | vLLM | victim summaries `b*` from victim traces |

The victim's real traces `t` are **withheld from the attack** and used only for (a) the
`Victim-Trace` oracle baseline and (b) Table 2 fidelity scoring. This is what a local victim buys
that no API victim can.

> **Make the withholding STRUCTURAL here, in 3.1 — not a convention.** The sentence above states the
> rule, which is exactly what makes it feel handled when nothing enforces it. `t` is simultaneously
> the oracle condition and the fidelity reference, so it must never reach the attack path — and
> today that rests on remembering, across three phases and however many sessions.
>
> **Write `t` to a different file from the `(y, b*)` the attack consumes**, so leaking it requires
> opening a file the attack path has no reason to touch. One wrong join otherwise puts the oracle
> into the attack, and the result is silently meaningless while *looking like a spectacular success*
> — the worst available failure shape, and one no gate downstream can catch, because a leaked oracle
> fails no sanity check.
>
> **This is a Phase 3 decision and cannot be deferred to Phase 4**: by then the file layout is
> fixed. Raise it before 3.1 runs.

Est. **~10-15 h** — revised down from 23 h. The victim's measured median is only 3,484 tokens
(JEEBench) / 593 (MATH500), not the ~5k assumed. Its p95 is 17,795, so per-slot context can drop
below 32k to buy more slots. **Sweep first at 16k/slot.**

## Phase 4 — Invert *(paper Stage 2 cont.)*

| Step | Model | Engine | Output |
|---|---|---|---|
| 4.1 | Inverter (merged) | vLLM | synthetic traces `t̂` from `(x, y, b*)` |

Est. ~4 h. Then score `t̂` against withheld `t` → **Table 2 reproduction**.

> **The scorer does not exist.** `grep -rlniE "TF1|token_overlap|rouge|bleu|fidelity" bench/` returns
> nothing, and `pyproject.toml` declares no scoring library. The one sentence above is the entire
> implementation. The metrics *are* specified — `docs/02` §73-74: **TF1** is unigram bag-of-tokens
> F1, the harmonic mean of token-level precision and recall, explicitly **not** n-gram and **not**
> sequence-aligned; ROUGE-1/2/L standard. So the spec exists and the code does not, which is the
> shape most likely to be mistaken for done. The three-arm comparison in `docs/11` §2 depends
> entirely on this scorer. Note the paper never says whether its ROUGE figures are F or recall —
> **compute and report both** rather than picking one and hoping.

## Phase 5 — Train students *(paper Stage 3)*

Five conditions, matching the paper exactly:

| Condition | Supervision target |
|---|---|
| Answer-only | `y` |
| Summary+Answer | `b*`, `y` |
| Surrogate-Trace | `t'`, `y'` |
| **Synthesized-Trace (ours)** | `t̂`, `y` |
| Victim-Trace (oracle) | `t`, `y` — withheld ground truth |

Plus the `Synthesized-Trace` condition run **both FFT and LoRA** on the 2B, to measure what LoRA costs.
Est. ~20 h.

## Phase 6 — Evaluate

All fine-tuned students on MATH500 + JEEBench (+ LiveCodeBench if time), vLLM, same protocol as
Phase 0 so pre/post is directly comparable. Est. ~8 h.

---

## Totals

| Phase | Est. |
|---|---|
| 0 Baselines | ~30 h *(actual, incl. re-runs)* |
| 1 Surrogate data (**2 surrogates**) | **~29-31 h** *(measured mid-run; see `12` §7)* |
| 2 Inverter training (**2 surrogates × 2 settings**) | ~28 h |
| 3 **Victim queries** | **~10-15 h** |
| 4 Inversion | ~4 h |
| 5 Student training | ~20 h |
| 6 Evaluation | ~8 h |
| **Total** | **~135-150 h** (~6 days of GPU time) |

Every individual stage fits an overnight run. **Phase 2 now dominates** (~28 h), with Phase 5 next;
Phase 3 dropped to ~10-15 h once the victim's real trace lengths were measured. If generation needs
cutting, the paper's own Figure 3 shows 5k queries already delivers most of the MATH500 benefit, and
2k would halve it again at some cost to the result.

## Engine boundary

Only one comparison in the whole plan crosses engines: victim benchmark (llama.cpp) versus student
benchmark (vLLM), in Phase 0. That comparison is a sanity check on headroom, not a measured claim.
Every result that carries the paper's argument — student before vs after, across the five training
conditions — is vLLM on both sides. See `09` §7.5.
