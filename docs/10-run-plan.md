# Run Plan — Models, Engines, Quantization

Every model we will load, in what format, on which engine, and why.

## The rule

> **Fixed inputs** (never trained) → GGUF on llama.cpp, at the highest precision that fits.
> **Trainable artifacts** (produced by TRL) → safetensors bf16 on vLLM, so every checkpoint
> evaluates with no conversion step.

Quantize only where the model does not otherwise fit. The victim is the sole model forced below
full precision.

---

## Before proposing an experiment

Answer these four **in order**. If any fails, do not propose it.

| # | Question | What does not count as an answer |
|---|---|---|
| 1 | **What specific question does it answer?** | "It would be interesting." "For comparison." "As a reference point." The answer must have a possible *outcome that changes what we do*. |
| 2 | **What is compared to what?** | Name both sides concretely: which artifact, scored by which metric, against which reference. |
| 3 | **Does anything ELSE vary alongside the variable of interest?** | Provenance, harness, sampling settings, serving stack, system prompt. If so the result is uninterpretable *no matter how cleanly it runs*. |
| 4 | **Does the machinery exist?** | If not, building it is part of the cost and the proposal carries it. "Cheap to run" is not "cheap." |

> **Cost is never a reason to run an experiment. It is only ever a reason not to.**

**Worked example — the R1-surrogate arm** (`docs/11` §2, considered and rejected). The sequence is the
instructive part, because every round narrowed a vague proposal toward a concrete one and the concrete
version was invalid:

| | |
|---|---|
| proposed | because generation was free — the traces are already on disk. That is **question 4 answered first, and 1-3 not asked at all**, and it was labelled an "optional reference point", which *defers* justification rather than supplying it. |
| round 1 | *what is it for?* → an abstract answer about testing flatness across 450× rather than 4.7× |
| round 2 | *what would you compare the traces to?* → TF1 against the victim's withheld real traces |
| round 3 | *does the project do that anywhere?* → **no.** No TF1/BLEU/ROUGE code exists. Question 4 was actually *unanswered*: what had been checked was that generation was free, not that scoring existed. |
| round 4 | *sounds like we're not doing it* → **question 3, finally.** The bundled traces differ from ours in provenance as well as surrogate strength, so the sweep confounds the variable it measures. Fatal — and present from the very beginning. |

The four questions asked up front would have killed it in a minute instead of an hour, and **question 3
would have killed it outright**.

**The word is the tell.** *"Optional", "reference", "nice to have", "while we're at it", "since it's
free"* — these are the labels a proposal wears when it has not answered question 1. Treat them as a
prompt to ask the four questions, not as a category of work.

**The corollary:** a proposal that survives only because nobody asked what it would measure is
indistinguishable from a good one until someone asks. Here, the user asked. Neither of the two sessions
running the work did.

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

## Phase 2 — Train the inverter *(paper Stage 1 cont.)* ✅ COMPLETE

| Step | Model | Framework | Method |
|---|---|---|---|
| 2.1 | Qwen3.5-4B | TRL `SFTTrainer` | `(x', y', b') → t'` — summary setting |
| 2.2 | Qwen3.5-4B | TRL `SFTTrainer` | `(x', y') → t'` — no-summary setting |

Two separately trained inverters, per §4. **LoRA r=32-64 on a bf16 base** — 4B FFT is confirmed
impossible (27.81 GiB @8k) and NF4 is not needed (bf16 LoRA is 15.11 GiB @8k). `max_length` **12288**:
the inverter prompt is ~1,524 tokens (problem 81 + answer 542 + 900-token summary), so a cap-8192
trace needs ~9,716. Measured in `12` §1 and §3. Est. ~14 h for both.

**Done 2026-08-30** — four inverters (2 arms × 2 settings), r=64, **33.8 h** of training at ~2,100
tok/s, peak 15.4 GiB; every long run probed first. Record: `docs/results/phase2.md`; handoff `13`.
Adapters on disk (gitignored): `bench/results/phase2/inverter-{7b,1.5b}-{sum,nosum}/checkpoint-*`,
three per inverter. Held-out forged traces run 0.89–0.99× the surrogate's median length and land on
their given answer ~95–97 % of the time; the 1.5B-arm inverters cap 10–17 % of forgeries at 8,192
against the 7B arm's 3–5 %. The three Phase 4 decisions that follow from it are under Phase 4.

## Phase 3 — Query the victim *(paper Stage 2)*

| Step | Model | Engine | Output |
|---|---|---|---|
| 3.1 | Victim IQ4_XS | llama.cpp, `-np 32`, q8_0 KV | 5k `(y, t)` on OpenThoughts split B (**disjoint** from A) |
| 3.2 | Compressor | vLLM | victim summaries `b*` from victim traces |

The victim's real traces `t` are **withheld from the attack** and used only for (a) the
`Victim-Trace` oracle baseline. (They were also the Table 2 fidelity reference until that was
dropped — see Phase 4.) This is what a local victim buys
that no API victim can.

> **Make the withholding STRUCTURAL here, in 3.1 — not a convention.** The sentence above states the
> rule, which is exactly what makes it feel handled when nothing enforces it. `t` is simultaneously
> the **`Victim-Trace` oracle training condition** — the ceiling row of Table 3 — so it must never
> reach the attack path, and today that rests on remembering, across three phases and however many
> sessions. The oracle alone is sufficient reason; dropping the fidelity scorer does not weaken this.
>
> **Write `t` to a different file from the `(y, b*)` the attack consumes**, so leaking it requires
> opening a file the attack path has no reason to touch. One wrong join otherwise puts the oracle
> into the attack, and the result is silently meaningless while *looking like a spectacular success*
> — the worst available failure shape, and one no gate downstream can catch, because a leaked oracle
> fails no sanity check. It would silently invalidate the headline result, which is Table 3.
>
> **This is a Phase 3 decision and cannot be deferred to Phase 4**: by then the file layout is
> fixed. Raise it before 3.1 runs.

Est. **~10-15 h** — revised down from 23 h. The victim's measured median is only 3,484 tokens
(JEEBench) / 593 (MATH500), not the ~5k assumed. Its p95 is 17,795, so per-slot context can drop
below 32k to buy more slots. **Sweep first at 16k/slot.**

## Phase 4 — Invert *(paper Stage 2 cont.)*

| Step | Model | Engine | Output |
|---|---|---|---|
| 4.1 | Inverter — the **epoch-2 adapter** of each of the four, merged | vLLM (`bench/invert.py`) | synthetic traces `t̂` from `(x, y, b*)` (summary setting) and from `(x, y)` (no-summary), on split B, once per inverter |

Est. ~4 h per inverter — size it from a probe, as every phase has. **Smoke-test ~30 rows first**, read a few of the resulting traces, check their length
looks sane against the victim's, then run the rest. **No new tooling.** If you want the length as a
number, point `bench/phase1_stats.py`'s existing length reporting at the output file — a flag on code
we already have.

### Decided 2026-08-30, from the Phase 2 results — read before 4.1 runs

| Decision | What | Why (`results/phase2.md`) |
|---|---|---|
| **Which adapter** | **Epoch 2, all four inverters** — `checkpoint-402` (7B arms), `checkpoint-404` (1.5B arms). Not epoch 3, not per-arm. | Held-out eval loss is lowest at epoch 2 on every arm and is the only deterministic signal; the generation differences between epochs (cap-hit, consistency) are single draws at temperature 0.7 inside the ~15 % flip-rate noise Phase 1 measured, and they point opposite ways on the two arms. One rule keeps the arms comparable checkpoint-for-checkpoint (§4, §5.5). |
| **Capped forgeries** (`finish_reason == "length"` at 8,192 — 3–17 % of held-out rows depending on arm and epoch) | **Regenerate at a new seed, up to three draws; drop what still caps; report the count.** Same policy on every arm and setting. Keep the first-draw outputs and report their cap-hit as the inverter's property; the rescued set is the training artifact. Log as `09` row 7.14 — the paper regenerates nothing. | A severed `t̂` in the student target `[t̂; y]` teaches non-termination, the 2B student's documented failure mode (`12` §2) — the same reason Phase 1 dropped capped surrogate traces. Dropping alone would shrink the 1.5B-arm student's data ~10–15 % more than the 7B's and put data quantity into the surrogate-strength comparison. Capping is mostly a property of the draw, not the prompt: no prompt capped on all four inverters, and the same inverter's epoch-2 and epoch-3 capped sets overlap 1 in 10 on the 7B arm (§5.4), so a re-draw rescues most; the ~4–10-row prompt-level core is dropped and reported. |
| **Answer-inconsistent forgeries** (~3 % of gradable held-out rows argue themselves out of the answer they were conditioned on) | **No filtering in the main condition; report the rate per arm.** `09` row 7.15. | Matches the paper, which filters nothing; filtering would blur the comparison to it. A consistency-filtered student condition is a separate Phase 5 proposal that must pass the four questions above on its own. |

> **Considered and rejected: a Table 2 fidelity suite (TF1 / BLEU / ROUGE).** Not a missing tool — the
> metrics do not measure what their names imply.
>
> **They are largely length in disguise.** Across the paper's own six Table 2 rows, Len correlates with
> ROUGE-1 at **r = +0.945**, TF1 at **+0.916**, ROUGE-L **+0.890**, BLEU **+0.889**, ROUGE-2 **+0.838**
> (n=6, so this corroborates the structural argument rather than carrying it). Sort Table 2 by length
> and the fidelity columns sort with it.
>
> **The structure is why.** TF1's recall term is `|shared tokens| / |tokens in the real trace|`, so a
> short trace cannot recall a long one *regardless of reasoning quality*, while precision is nearly free
> — two traces on the same problem share `the/so/then/we/x/=` in bulk. A trace concluding `x=5` and one
> concluding `x=7` have near-identical token bags. **The metric cannot separate correct reasoning from
> incorrect, and it heavily rewards length.**
>
> The paper calls these "auxiliary diagnostics" itself, and its headline finding cuts against them:
> synthesized traces sometimes *beat* the oracle, so lower similarity with better outcomes is the
> interesting case, not a failure. **Table 3 — downstream student accuracy — carries the entire result.**
>
> **Also rejected: a formal length gate.** Either the inverter learns or it does not, and student
> training reveals that either way. A 1,000-token trace where 4,000 was expected is not a hidden
> property needing instrumentation — it is the most visible thing about the output file. A gate would
> earn its place only if Phase 4 fed Phase 5 unattended, and it does not: there is a checkpoint between
> them. Smoke test, spot-check, proceed.

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
